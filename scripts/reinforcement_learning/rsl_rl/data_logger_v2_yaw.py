# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import time
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
# 座標変換用ユーティリティ
import isaaclab.utils.math as math_utils 

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


def get_yaw_from_quat(quat):
    """
    クォータニオン (w, x, y, z) から Yaw 角度 (rad) を計算するヘルパー関数
    """
    # quat shape: (w, x, y, z)
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    
    # atan2(2(wz + xy), 1 - 2(y^2 + z^2)) は Z-Y-X 変換などの一般的なYaw計算
    # Isaac Sim の座標系に合わせて計算
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return yaw.item()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # ---------------------------------------------------------
    # リセット設定 (ゴールするまで止めない)
    # ---------------------------------------------------------
    env_cfg.episode_length_s = 1000.0  # タイムアウトを防ぐ
    if hasattr(env_cfg, "terminations"):
        env_cfg.terminations = None    # 転倒等によるリセットを無効化
        print("[INFO] Terminations disabled for continuous data collection.")

    # ---------------------------------------------------------
    # コマンド設定: その場旋回 (X=0, Y=0, Rot=0.5 rad/s)
    # ---------------------------------------------------------
    target_speed_yaw = 0.5  # 目標旋回速度 (rad/s)

    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity"):
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (target_speed_yaw, target_speed_yaw)
        print(f"[INFO] Initial velocity commands fixed: X=0.0, Y=0.0, AngZ={target_speed_yaw}")

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # ログディレクトリの設定
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir
    print(f"[INFO] Output files will be saved to: {log_dir}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # =========================================================================
    # カメラ位置の設定 (旋回が見やすい位置へ変更)
    # =========================================================================
    # 斜め上から見下ろす
    env.unwrapped.sim.set_camera_view(eye=[3.0, 3.0, 3.0], target=[0.0, 0.0, 0.0])
    # =========================================================================

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # ---------------------------------------------------------
    # センサーとロボット情報の取得
    # ---------------------------------------------------------
    scene = env.unwrapped.scene
    robot = scene["robot"]
    
    # ロボットの質量を取得 (エネルギー計算用)
    robot_mass = torch.sum(robot.data.default_mass[0]).item()
    print(f"[INFO] Robot Mass: {robot_mass:.2f} kg")

    # ---------------------------------------------------------
    # 計測対象の関節インデックスを取得
    # ---------------------------------------------------------
    left_joint_names = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint"
    ]
    right_joint_names = [
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"
    ]
    target_joint_names = left_joint_names + right_joint_names
    
    all_joint_names = robot.data.joint_names
    target_joint_indices = []
    
    print("[INFO] Searching for target joint indices...")
    for name in target_joint_names:
        if name in all_joint_names:
            idx = all_joint_names.index(name)
            target_joint_indices.append(idx)
        else:
            print(f"  [WARNING] Joint '{name}' not found in robot model!")
    
    # ---------------------------------------------------------
    # 接触力センサーの取得
    # ---------------------------------------------------------
    contact_sensor = None
    if "contact_forces" in scene.sensors:
        contact_sensor = scene["contact_forces"]
    else:
        print("[WARNING] 'contact_forces' sensor not found in scene.")

    # extract and export policy
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # ---------------------------------------------------------
    # 初期化と (0,0) へのリセット
    # ---------------------------------------------------------
    obs = env.get_observations()
    
    print("[INFO] Forcing robot to start at (0, 0) facing forward...")
    root_state = robot.data.default_root_state.clone()
    root_state[:, :2] = 0.0 
    root_state[:, 3] = 1.0 # Quaternion W
    root_state[:, 4:] = 0.0 
    
    robot.write_root_state_to_sim(root_state)
    robot.reset()
    scene.reset() 

    timestep = 0
    sim_time = 0.0
    
    # =========================================================
    # [変更点] 旋回判定用の変数
    # =========================================================
    target_rotation = 2 * np.pi  # 2π = 360度
    accumulated_yaw = 0.0
    previous_yaw = 0.0
    
    # 初期Yawを取得
    init_quat = robot.data.root_quat_w[0]
    previous_yaw = get_yaw_from_quat(init_quat)
    print(f"[INFO] Initial Yaw: {previous_yaw:.4f} rad")

    # エネルギー計算用
    total_energy = 0.0
    measurement_time = 0.0
    measurement_started = False

    # ---------------------------------------------------------
    # CSVファイル設定
    # ---------------------------------------------------------
    # ファイル名も少し変更して区別しやすくします
    file_forces = os.path.join(log_dir, "rot_data_forces.csv")
    file_vel = os.path.join(log_dir, "rot_data_velocity.csv")
    file_yaw = os.path.join(log_dir, "rot_data_yaw.csv")
    file_power = os.path.join(log_dir, "rot_data_power.csv")

    try:
        f_forces = open(file_forces, "w", newline="", encoding="utf-8")
        f_vel = open(file_vel, "w", newline="", encoding="utf-8")
        f_yaw = open(file_yaw, "w", newline="", encoding="utf-8")
        f_power = open(file_power, "w", newline="", encoding="utf-8")

        w_forces = csv.writer(f_forces)
        w_vel = csv.writer(f_vel)
        w_yaw = csv.writer(f_yaw)
        w_power = csv.writer(f_power)

        w_forces.writerow(["Timestep", "Left_Foot_Force_N", "Right_Foot_Force_N"])
        w_vel.writerow(["Timestep", "Target_Ang_Vel_Z", "Actual_Ang_Vel_Z"])
        w_yaw.writerow(["Timestep", "Current_Yaw_Rad", "Accumulated_Yaw_Rad", "Accumulated_Yaw_Deg"])
        w_power.writerow(["Timestep", "Instant_Power_W", "Total_Energy_J"])

        print("[INFO] CSV files opened for recording.")
        
        # -----------------------------------------------------
        # データリスト初期化
        # -----------------------------------------------------
        log_yaw_accum = []
        log_yaw_curr = []
        log_vel_target = []
        log_vel_actual = []

        try:
            vel_command_term = env.unwrapped.command_manager._terms.get("base_velocity")
        except:
            vel_command_term = None
            print("[WARNING] Could not access 'base_velocity' term.")

        print(f"[INFO] Starting simulation... 0 speed for 1s, then spin until {target_rotation:.2f} rad.")
        
        while simulation_app.is_running():
            start_time = time.time()
            
            # --- 角度計算 (Yaw積算) ---
            current_quat = robot.data.root_quat_w[0]
            current_yaw = get_yaw_from_quat(current_quat)
            
            # 差分計算 (ラップアラウンド対応)
            # -pi ~ pi の範囲でジャンプする場合を補正
            diff_yaw = current_yaw - previous_yaw
            if diff_yaw > np.pi:
                diff_yaw -= 2 * np.pi
            elif diff_yaw < -np.pi:
                diff_yaw += 2 * np.pi
            
            # 旋回開始（コマンド入力後）してから積算する
            if measurement_started:
                accumulated_yaw += abs(diff_yaw) # 絶対値で積算（どちら周りでもカウント）
            
            previous_yaw = current_yaw

            # 判定: 累積角度が 2π (360度) を超えたら終了
            if accumulated_yaw >= target_rotation:
                print(f"[INFO] Target rotation {accumulated_yaw:.4f} rad (>= {target_rotation:.4f}) reached. Stopping.")
                break

            # -----------------------------------------------------
            # コマンド制御: 1秒待機 -> 旋回開始
            # -----------------------------------------------------
            cmd_x = 0.0 
            cmd_y = 0.0
            cmd_yaw = target_speed_yaw
            
            if vel_command_term is not None:
                # 最初の1秒間は速度0
                if sim_time < 1.0:
                    cmd_yaw = 0.0
                
                # コマンド適用
                vel_command_term.vel_command_b[:, 0] = cmd_x
                vel_command_term.vel_command_b[:, 1] = cmd_y
                vel_command_term.vel_command_b[:, 2] = cmd_yaw

            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)

                # =============================================================
                # 計測処理 (コマンドが有効になってから開始)
                # =============================================================
                if abs(cmd_yaw) >= (target_speed_yaw * 0.9): 
                    
                    if not measurement_started:
                        print(f"[INFO] Command Yaw is {cmd_yaw} rad/s. Measurement STARTED at time {sim_time:.2f}s")
                        measurement_started = True

                    # --- データ取得 ---
                    # 速度
                    root_quat = robot.data.root_quat_w[0]
                    ang_vel_b = math_utils.quat_apply_inverse(root_quat, robot.data.root_ang_vel_w[0])
                    actual_ang_z = ang_vel_b[2].item()

                    # 力・トルク
                    all_torques = robot.data.applied_torque[0]
                    all_vels = robot.data.joint_vel[0]
                    inst_power = torch.sum(torch.abs(all_torques * all_vels)).item()
                    
                    # エネルギーと計測時間を加算
                    total_energy += inst_power * dt
                    measurement_time += dt

                    # --- CSV書き込み ---
                    w_vel.writerow([timestep, target_speed_yaw, actual_ang_z])
                    w_yaw.writerow([timestep, current_yaw, accumulated_yaw, np.degrees(accumulated_yaw)])
                    w_power.writerow([timestep, inst_power, total_energy])

                    # --- リスト保存 ---
                    log_yaw_accum.append(accumulated_yaw)
                    log_yaw_curr.append(current_yaw)
                    log_vel_target.append(target_speed_yaw)
                    log_vel_actual.append(actual_ang_z)

            timestep += 1
            sim_time += dt

            if args_cli.video and timestep >= args_cli.video_length: pass
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0: time.sleep(sleep_time)

    except IOError as e:
        print(f"[ERROR] Failed to handle CSV files: {e}")
    finally:
        if 'f_vel' in locals(): f_vel.close()
        if 'f_yaw' in locals(): f_yaw.close()
        if 'f_power' in locals(): f_power.close()
        print("[INFO] All CSV data saved.")

    # ---------------------------------------------------------
    # メトリクス計算結果出力
    # ---------------------------------------------------------
    avg_power = total_energy / measurement_time if measurement_time > 0 else 0.0

    print("="*40)
    print(f"METRICS REPORT (Rotation Test)")
    print(f"Target Angle: 360 deg (2pi rad)")
    print(f"Actual Yaw:   {np.degrees(accumulated_yaw):.2f} deg")
    print(f"Meas. Time:   {measurement_time:.2f} s")
    print(f"Total Energy: {total_energy:.2f} J")
    print(f"Avg Power:    {avg_power:.2f} W")
    print("="*40)

    # ---------------------------------------------------------
    # グラフ生成処理
    # ---------------------------------------------------------
    if len(log_yaw_accum) > 0:
        log_yaw_accum = np.array(log_yaw_accum)
        log_yaw_curr = np.array(log_yaw_curr)
        log_vel_target = np.array(log_vel_target)
        log_vel_actual = np.array(log_vel_actual)
        time_axis = np.arange(len(log_yaw_accum)) * dt

        # 1. Yaw角度の推移
        fig_yaw, ax_yaw = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # 累積角度 (これが増えていく)
        ax_yaw[0].plot(time_axis, np.degrees(log_yaw_accum), 'g-', linewidth=2, label="Accumulated Yaw")
        ax_yaw[0].axhline(y=360.0, color='r', linestyle='--', label="Target (360 deg)")
        ax_yaw[0].set_ylabel("Accumulated Angle [deg]")
        ax_yaw[0].set_title("Rotation Progress")
        ax_yaw[0].legend()
        ax_yaw[0].grid(True)

        # 現在のYaw (-180 ~ 180)
        ax_yaw[1].plot(time_axis, np.degrees(log_yaw_curr), 'b-', alpha=0.6, label="Current Yaw")
        ax_yaw[1].set_ylabel("Current Heading [deg]")
        ax_yaw[1].set_xlabel("Time [s]")
        ax_yaw[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "rotation_yaw_tracking.png"))
        plt.close(fig_yaw)

        # 2. 角速度追従グラフ
        fig_vel, ax_vel = plt.subplots(figsize=(10, 5))
        ax_vel.plot(time_axis, log_vel_target, 'r--', label="Target Vel Z")
        ax_vel.plot(time_axis, log_vel_actual, 'b-', alpha=0.7, label="Actual Vel Z")
        ax_vel.set_xlabel("Time [s]")
        ax_vel.set_ylabel("Angular Velocity [rad/s]")
        ax_vel.set_title("Yaw Velocity Tracking")
        ax_vel.grid(True)
        ax_vel.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "rotation_velocity_tracking.png"))
        plt.close(fig_vel)
        
    else:
        print("[WARNING] No data collected.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()