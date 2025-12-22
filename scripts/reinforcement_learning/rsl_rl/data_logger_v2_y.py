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
    # 初期コマンド設定 (X=0.0, Y=0.5m/s, Rot=0)
    # ※ 横移動速度
    # ---------------------------------------------------------
    target_speed_y = 0.5

    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity"):
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (target_speed_y, target_speed_y)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        print(f"[INFO] Initial velocity commands fixed: X=0.0, Y={target_speed_y}, AngZ=0.0")

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
    # カメラ位置の設定 (横移動が見やすい位置へ変更)
    # =========================================================================
    # X軸方向(正面)から見る
    env.unwrapped.sim.set_camera_view(eye=[5.0, 5.0, 3.0], target=[0.0, 5.0, 0.0])
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
    
    # ロボットの質量を取得 (CoT計算用)
    robot_mass = torch.sum(robot.data.default_mass[0]).item()
    gravity = 9.81
    print(f"[INFO] Robot Mass: {robot_mass:.2f} kg (for CoT calculation)")

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
        print("[WARNING] 'contact_forces' sensor not found in scene. Forces will be 0.")

    body_names = robot.body_names
    try:
        left_foot_names = [name for name in body_names if "left_ankle_roll_link" in name]
        right_foot_names = [name for name in body_names if "right_ankle_roll_link" in name]
        
        if not left_foot_names or not right_foot_names:
            l_foot_idx_robot = [i for i, n in enumerate(body_names) if "left" in n and "ankle" in n][-1]
            r_foot_idx_robot = [i for i, n in enumerate(body_names) if "right" in n and "ankle" in n][-1]
        else:
            l_foot_idx_robot = body_names.index(left_foot_names[0])
            r_foot_idx_robot = body_names.index(right_foot_names[0])
            
        l_foot_idx_sensor = 0
        r_foot_idx_sensor = 0
        if contact_sensor is not None:
            sensor_bodies = contact_sensor.body_names
            l_foot_name = body_names[l_foot_idx_robot]
            r_foot_name = body_names[r_foot_idx_robot]
            try:
                l_foot_idx_sensor = sensor_bodies.index(l_foot_name)
                r_foot_idx_sensor = sensor_bodies.index(r_foot_name)
            except ValueError:
                l_foot_idx_sensor = [i for i, n in enumerate(sensor_bodies) if "left" in n and "ankle" in n][-1]
                r_foot_idx_sensor = [i for i, n in enumerate(sensor_bodies) if "right" in n and "ankle" in n][-1]
            
    except Exception as e:
        print(f"[ERROR] Could not determine foot indices: {e}")
        return

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
    
    # ロボットの位置を強制的に (0, 0) スタートにする
    print("[INFO] Forcing robot to start at (0, 0) facing forward...")
    
    root_state = robot.data.default_root_state.clone()
    root_state[:, :2] = 0.0 
    root_state[:, 3] = 1.0 
    root_state[:, 4:] = 0.0 
    
    # シミュレーションに反映
    robot.write_root_state_to_sim(root_state)
    robot.reset()
    scene.reset() 

    start_pos = torch.tensor([0.0, 0.0, 0.0], device=env.unwrapped.device)

    timestep = 0
    sim_time = 0.0
    distance_traveled = 0.0
    
    # =========================================================
    # [変更点] 目標距離を 5m に設定
    # =========================================================
    target_distance = 5.0 

    # エネルギー計算用
    total_energy = 0.0
    measurement_time = 0.0
    measurement_started = False
    start_measure_y = 0.0 # Y座標の開始地点

    # ---------------------------------------------------------
    # CSVファイル設定
    # ---------------------------------------------------------
    file_forces = os.path.join(log_dir, "data_forces.csv")
    file_foot_z = os.path.join(log_dir, "data_foot_z.csv")
    file_com = os.path.join(log_dir, "data_com.csv")
    file_vel = os.path.join(log_dir, "data_velocity.csv")
    file_joints = os.path.join(log_dir, "data_joints_all.csv")
    file_power = os.path.join(log_dir, "data_power.csv")

    try:
        f_forces = open(file_forces, "w", newline="", encoding="utf-8")
        f_foot_z = open(file_foot_z, "w", newline="", encoding="utf-8")
        f_com = open(file_com, "w", newline="", encoding="utf-8")
        f_vel = open(file_vel, "w", newline="", encoding="utf-8")
        f_joints = open(file_joints, "w", newline="", encoding="utf-8")
        f_power = open(file_power, "w", newline="", encoding="utf-8")

        w_forces = csv.writer(f_forces)
        w_foot_z = csv.writer(f_foot_z)
        w_com = csv.writer(f_com)
        w_vel = csv.writer(f_vel)
        w_joints = csv.writer(f_joints)
        w_power = csv.writer(f_power)

        w_forces.writerow(["Timestep", "Left_Foot_Force_N", "Right_Foot_Force_N"])
        w_foot_z.writerow(["Timestep", "Left_Foot_Height_Z", "Right_Foot_Height_Z"])
        w_com.writerow(["Timestep", "CoM_X", "CoM_Y", "CoM_Z"])
        w_vel.writerow(["Timestep", "Target_Vel_X", "Actual_Vel_X", "Target_Vel_Y", "Actual_Vel_Y", "Target_Ang_Vel_Z", "Actual_Ang_Vel_Z"])
        w_power.writerow(["Timestep", "Instant_Power_W", "Total_Energy_J"])

        joint_header = ["Timestep"]
        for name in target_joint_names: joint_header.append(f"{name}_Torque")
        for name in target_joint_names: joint_header.append(f"{name}_Vel")
        w_joints.writerow(joint_header)

        print("[INFO] CSV files opened for recording.")
        
        # -----------------------------------------------------
        # データリスト初期化
        # -----------------------------------------------------
        traj_com = []
        traj_l_foot = []
        traj_r_foot = []
        forces_log_l = []
        forces_log_r = []
        vel_log_target = []
        vel_log_actual = []

        try:
            vel_command_term = env.unwrapped.command_manager._terms.get("base_velocity")
        except:
            vel_command_term = None
            print("[WARNING] Could not access 'base_velocity' term. Dynamic heading correction disabled.")

        print(f"[INFO] Starting simulation... 0m/s for 1s, then target Y={target_distance}m.")
        
        while simulation_app.is_running():
            start_time = time.time()
            
            # --- 距離判定 (Y方向) ---
            current_pos = robot.data.root_pos_w[0]
            current_y = current_pos[1].item() # Y座標を取得
            distance_traveled = abs(current_y) # 絶対値で判定

            if distance_traveled >= target_distance:
                print(f"[INFO] Target Y distance {distance_traveled:.2f}m reached. Stopping.")
                break

            # -----------------------------------------------------
            # コマンド制御: 1秒待機 -> 移動開始 & 補正
            # -----------------------------------------------------
            cmd_x = 0.0 
            cmd_y = 0.0
            
            if vel_command_term is not None:
                # デフォルト: 横移動コマンド
                cmd_x = 0.0
                cmd_y = target_speed_y # 設定したY速度
                cmd_yaw = 0.0

                # =============================================================
                # 最初の1秒間は速度0を指令 (足踏み/バランス維持)
                # =============================================================
                if sim_time < 1.0:
                    cmd_x = 0.0
                    cmd_y = 0.0
                    cmd_yaw = 0.0
                
                # コマンド適用
                vel_command_term.vel_command_b[:, 0] = cmd_x
                vel_command_term.vel_command_b[:, 1] = cmd_y
                vel_command_term.vel_command_b[:, 2] = cmd_yaw


            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)

                # =============================================================
                # 計測処理 (Yコマンドが指定値以上になってから記録開始)
                # =============================================================
                # コマンドが設定値に達したら記録を開始する
                if abs(cmd_y) >= (target_speed_y * 0.9): 
                    
                    # 計測開始時の初期位置を保存
                    if not measurement_started:
                        print(f"[INFO] Command Y is {cmd_y} m/s. Measurement STARTED at time {sim_time:.2f}s")
                        measurement_started = True
                        start_measure_y = current_y

                    # --- データ取得 ---
                    com_pos = robot.data.root_pos_w[0]
                    all_body_pos = robot.data.body_pos_w[0]
                    l_foot_pos_vec = all_body_pos[l_foot_idx_robot]
                    r_foot_pos_vec = all_body_pos[r_foot_idx_robot]

                    l_force_val = 0.0
                    r_force_val = 0.0
                    if contact_sensor is not None:
                        net_forces = contact_sensor.data.net_forces_w[0]
                        l_force_val = torch.norm(net_forces[l_foot_idx_sensor]).item()
                        r_force_val = torch.norm(net_forces[r_foot_idx_sensor]).item()

                    cmd = env.unwrapped.command_manager.get_command("base_velocity")[0] 
                    target_vel_x = cmd[0].item(); target_vel_y = cmd[1].item(); target_ang_z = cmd[2].item()

                    root_quat = robot.data.root_quat_w[0]
                    lin_vel_b = math_utils.quat_apply_inverse(root_quat, robot.data.root_lin_vel_w[0])
                    ang_vel_b = math_utils.quat_apply_inverse(root_quat, robot.data.root_ang_vel_w[0])
                    actual_vel_x = lin_vel_b[0].item(); actual_vel_y = lin_vel_b[1].item(); actual_ang_z = ang_vel_b[2].item()

                    all_torques = robot.data.applied_torque[0]
                    all_vels = robot.data.joint_vel[0]
                    
                    inst_power = torch.sum(torch.abs(all_torques * all_vels)).item()
                    
                    # エネルギーと計測時間を加算
                    total_energy += inst_power * dt
                    measurement_time += dt

                    current_torques = [all_torques[idx].item() for idx in target_joint_indices]
                    current_vels = [all_vels[idx].item() for idx in target_joint_indices]

                    # --- CSV書き込み ---
                    w_forces.writerow([timestep, -l_force_val, r_force_val])
                    w_foot_z.writerow([timestep, l_foot_pos_vec[2].item(), r_foot_pos_vec[2].item()])
                    w_com.writerow([timestep, com_pos[0].item(), com_pos[1].item(), com_pos[2].item()])
                    w_vel.writerow([timestep, target_vel_x, actual_vel_x, target_vel_y, actual_vel_y, target_ang_z, actual_ang_z])
                    w_joints.writerow([timestep] + current_torques + current_vels)
                    w_power.writerow([timestep, inst_power, total_energy])

                    # --- リスト保存 ---
                    traj_com.append(com_pos.cpu().numpy())
                    traj_l_foot.append(l_foot_pos_vec.cpu().numpy())
                    traj_r_foot.append(r_foot_pos_vec.cpu().numpy())
                    forces_log_l.append(l_force_val)
                    forces_log_r.append(r_force_val)
                    vel_log_target.append([target_vel_x, target_vel_y, target_ang_z])
                    vel_log_actual.append([actual_vel_x, actual_vel_y, actual_ang_z])

            timestep += 1
            sim_time += dt

            if args_cli.video and timestep >= args_cli.video_length: pass
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0: time.sleep(sleep_time)

    except IOError as e:
        print(f"[ERROR] Failed to handle CSV files: {e}")
    finally:
        if 'f_forces' in locals(): f_forces.close()
        if 'f_foot_z' in locals(): f_foot_z.close()
        if 'f_com' in locals(): f_com.close()
        if 'f_vel' in locals(): f_vel.close()
        if 'f_joints' in locals(): f_joints.close()
        if 'f_power' in locals(): f_power.close()
        print("[INFO] All CSV data saved.")

    # ---------------------------------------------------------
    # メトリクス計算結果出力 (Yベースに変更)
    # ---------------------------------------------------------
    # 計測区間の移動距離 (Y方向)
    measured_distance = abs(distance_traveled - abs(start_measure_y))
    
    if measured_distance > 0:
        cot = total_energy / (robot_mass * gravity * measured_distance)
    else:
        cot = float('inf')
        
    avg_power = total_energy / measurement_time if measurement_time > 0 else 0.0

    print("="*40)
    print(f"METRICS REPORT (Measurement Phase Only) - LATERAL")
    print(f"Meas. Dist Y: {measured_distance:.2f} m")
    print(f"Meas. Time:   {measurement_time:.2f} s")
    print(f"Total Energy: {total_energy:.2f} J")
    print(f"Avg Power:    {avg_power:.2f} W")
    print(f"CoT:          {cot:.4f}")
    print("="*40)

    # ---------------------------------------------------------
    # グラフ生成処理
    # ---------------------------------------------------------
    if len(vel_log_target) > 0:
        vel_log_target = np.array(vel_log_target)
        vel_log_actual = np.array(vel_log_actual)
        traj_com = np.array(traj_com)
        traj_l_foot = np.array(traj_l_foot)
        traj_r_foot = np.array(traj_r_foot)
        forces_l_arr = np.array(forces_log_l)
        forces_r_arr = np.array(forces_log_r)

        # 1. 速度追従グラフ
        fig_vel, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        time_axis = np.arange(len(vel_log_target)) * dt
        axs[0].plot(time_axis, vel_log_target[:, 0], 'r--', label="Target"); axs[0].plot(time_axis, vel_log_actual[:, 0], 'b', alpha=0.7, label="Actual"); axs[0].set_ylabel("Vel X"); axs[0].legend(); axs[0].grid(True)
        axs[1].plot(time_axis, vel_log_target[:, 1], 'r--', label="Target"); axs[1].plot(time_axis, vel_log_actual[:, 1], 'b', alpha=0.7, label="Actual"); axs[1].set_ylabel("Vel Y (Lateral)"); axs[1].grid(True)
        axs[2].plot(time_axis, vel_log_target[:, 2], 'r--', label="Target"); axs[2].plot(time_axis, vel_log_actual[:, 2], 'b', alpha=0.7, label="Actual"); axs[2].set_ylabel("Ang Vel Z"); axs[2].grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "velocity_tracking.png"))
        plt.close(fig_vel)

        # -----------------------------------------------------
        # 2. 2D 側面軌跡プロット (Front View: Y-Z)
        # 横移動なので、Y軸とZ軸をプロットします
        # -----------------------------------------------------
        print("[INFO] Generating 2D Front View plot (Y-Z plane)...")
        fig_side, ax_side = plt.subplots(figsize=(12, 6))

        # 重心 (CoM) の軌跡 (Y vs Z)
        ax_side.plot(traj_com[:, 1], traj_com[:, 2], 'k-', linewidth=2, label='CoM Height')

        # 足首の軌跡 (Y vs Z)
        ax_side.plot(traj_l_foot[:, 1], traj_l_foot[:, 2], 'b--', linewidth=1.0, alpha=0.6, label='Left Ankle')
        ax_side.plot(traj_r_foot[:, 1], traj_r_foot[:, 2], 'r--', linewidth=1.0, alpha=0.6, label='Right Ankle')
        
        # 地面 (Z=0) のラインを描画
        ax_side.axhline(y=0.0, color='gray', linestyle='-', linewidth=1.0, alpha=0.5)

        # スタートとゴール位置を点で表示
        ax_side.scatter(traj_com[0, 1], traj_com[0, 2], color='green', s=50, zorder=5, label='Start')
        ax_side.scatter(traj_com[-1, 1], traj_com[-1, 2], color='magenta', s=50, zorder=5, label='End')
        
        ax_side.set_xlabel("Y Position (Lateral Distance) [m]")
        ax_side.set_ylabel("Z Position (Height) [m]")
        ax_side.set_title("Front View Trajectory (Coronal Plane: Y-Z)")
        ax_side.grid(True)
        ax_side.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "trajectory_side_view.png")) 
        plt.close(fig_side)

        # 3. 2D 足跡プロット (着地時のみ)
        print("[INFO] Generating 2D Footprint plot (Stance Phase only)...")
        contact_threshold = 1.0 
        l_stance_idx = forces_l_arr > contact_threshold
        r_stance_idx = forces_r_arr > contact_threshold
        l_footprints = traj_l_foot[l_stance_idx]
        r_footprints = traj_r_foot[r_stance_idx]
        
        fig2d, ax2d = plt.subplots(figsize=(10, 8))
        ax2d.plot(traj_com[:, 0], traj_com[:, 1], 'k-', linewidth=1.5, alpha=0.6, label='CoM Trajectory')
        if len(l_footprints) > 0:
            ax2d.scatter(l_footprints[:, 0], l_footprints[:, 1], c='blue', s=10, alpha=0.5, label='Left Stance')
        if len(r_footprints) > 0:
            ax2d.scatter(r_footprints[:, 0], r_footprints[:, 1], c='red', s=10, alpha=0.5, label='Right Stance')
            
        # [変更点] 理想ライン (0,0)-(0, 5) を描画 (Y軸方向)
        ax2d.plot([0, 0], [0, target_distance], 'g--', label='Target Line (Y)')
        
        ax2d.set_xlabel("X Position (m)")
        ax2d.set_ylabel("Y Position (m)")
        ax2d.set_title("2D Footprint Map (Top-Down View)")
        ax2d.axis('equal') 
        ax2d.grid(True)
        ax2d.legend()
        plt.savefig(os.path.join(log_dir, "footprint_2d.png"))
        plt.close(fig2d)
        
    else:
        print("[WARNING] No data collected.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()