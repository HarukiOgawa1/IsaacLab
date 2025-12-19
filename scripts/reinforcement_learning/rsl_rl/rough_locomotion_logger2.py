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
    # リセット設定 (10m歩くまで止めない)
    # ---------------------------------------------------------
    env_cfg.episode_length_s = 1000.0  # タイムアウトを防ぐ
    if hasattr(env_cfg, "terminations"):
        env_cfg.terminations = None    # 転倒等によるリセットを無効化
        print("[INFO] Terminations disabled for continuous data collection.")

    # 初期コマンド設定 (X=1.0m/s, Y=0, Rot=0)
    # ※ ループ内で補正を行いますが、初期値としても設定します
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity"):
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        print("[INFO] Initial velocity commands fixed: X=1.0, Y=0.0, AngZ=0.0")

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

    # -------------------------------------------------------------------------
    # ### 【追加】 カメラ位置の設定
    # IsaacLabのGym環境では env.unwrapped.sim で SimulationContext にアクセスできます
    # -------------------------------------------------------------------------
    # ※注意: 後述のコードでロボットを Y = -40.0 に配置しているため、
    # ロボットを映したい場合は eye=[3.0, -40.0, 2.25], target=[0.0, -40.0, 1.0] 
    # のようにY座標を合わせる必要があるかもしれません。
    # ここではリクエスト通りの値を設定します。
    env.unwrapped.sim.set_camera_view(eye=[3.0, 0.0, 2.25], target=[0.0, 0.0, 1.0])
    # -------------------------------------------------------------------------

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
    # また、Yaw（向き）も 0 にリセットします
    print("[INFO] Forcing robot to start at (0, 0) facing forward...")
    
    # デフォルト状態を取得
    root_state = robot.data.default_root_state.clone()
    
    # 設定したいスタート座標 (X, Y)
    # Y) 階段up:, 階段down:, 凸凹:10, 突起:30, 坂道up:60, 坂道down:80
    START_X = 0.0
    START_Y = -40.0

    # 位置のリセット (X=START_X, Y=START_Y)
    # root_state[:, :2] = 0.0  # 元のコード
    root_state[:, 0] = START_X # X座標を設定
    root_state[:, 1] = START_Y # Y座標を設定
    
    # 向きのリセット (Quaternion = [1, 0, 0, 0] -> Yaw=0)
    # Isaac Simのクォータニオンは [w, x, y, z]
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
    
    target_distance = 10.0 # 10m進むまで

    # エネルギー計算用
    total_energy = 0.0

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

        # ヘッダー書き込み
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

        # コマンド制御用の変数を準備
        # コマンドマネージャの内部項を取得して、動的に書き換えられるようにする
        try:
            vel_command_term = env.unwrapped.command_manager._terms.get("base_velocity")
        except:
            vel_command_term = None
            print("[WARNING] Could not access 'base_velocity' term. Dynamic heading correction disabled.")

        print(f"[INFO] Starting simulation... Target: {target_distance}m distance along X-axis.")
        
        while simulation_app.is_running():
            start_time = time.time()
            
            # --- 距離判定 (10m) ---
            current_pos = robot.data.root_pos_w[0]
            current_x = current_pos[0].item()
            distance_traveled = current_x  # (0,0)スタートなのでX座標そのものが距離

            if distance_traveled >= target_distance:
                print(f"[INFO] Target distance {distance_traveled:.2f}m reached. Stopping.")
                break

            # -----------------------------------------------------
            # コース補正 (Heading Correction)
            # -----------------------------------------------------
            # Y軸のズレ (current_pos[1]) をゼロにしたい
            # P制御で角速度(ang_vel_z)を調整して向きを変える
            if vel_command_term is not None:
                # ユーザーコードのSTART_Yが基準になるように修正した方が良いですが、
                # 元のコードは「Y誤差=current_pos[1]」としています。
                # START_Y=-40の場合、常に誤差がある状態になってしまいます。
                # ここでは元コードのままにしています。
                y_error = current_pos[1].item() # 正なら左にずれている -> 右(-Z回転)に修正したい
                
                # ゲイン設定 (適宜調整)
                kp_heading = 1.0 
                
                # 補正角速度 (Yがプラスならマイナス方向へ回転させたいので -Kp * y)
                correction_ang_vel = -kp_heading * y_error
                
                # リミット (急激な回転を防ぐ)
                correction_ang_vel = np.clip(correction_ang_vel, -0.5, 0.5)
                
                # コマンドを上書き (vel_command_b は [num_envs, 3] のテンソル)
                # X=1.0, Y=0.0, Z=補正値
                vel_command_term.vel_command_b[:, 0] = 1.0
                vel_command_term.vel_command_b[:, 1] = 0.0
                vel_command_term.vel_command_b[:, 2] = correction_ang_vel

            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)

                # --- データ取得 ---
                com_pos = robot.data.root_pos_w[0]
                all_body_pos = robot.data.body_pos_w[0]
                l_foot_pos_vec = all_body_pos[l_foot_idx_robot]
                r_foot_pos_vec = all_body_pos[r_foot_idx_robot]

                # 力の取得
                l_force_val = 0.0
                r_force_val = 0.0
                if contact_sensor is not None:
                    net_forces = contact_sensor.data.net_forces_w[0]
                    l_force_val = torch.norm(net_forces[l_foot_idx_sensor]).item()
                    r_force_val = torch.norm(net_forces[r_foot_idx_sensor]).item()

                # 速度取得
                cmd = env.unwrapped.command_manager.get_command("base_velocity")[0] 
                target_vel_x = cmd[0].item(); target_vel_y = cmd[1].item(); target_ang_z = cmd[2].item()

                root_quat = robot.data.root_quat_w[0]
                lin_vel_b = math_utils.quat_apply_inverse(root_quat, robot.data.root_lin_vel_w[0])
                ang_vel_b = math_utils.quat_apply_inverse(root_quat, robot.data.root_ang_vel_w[0])
                actual_vel_x = lin_vel_b[0].item(); actual_vel_y = lin_vel_b[1].item(); actual_ang_z = ang_vel_b[2].item()

                # 関節データ & 仕事率計算
                all_torques = robot.data.applied_torque[0]
                all_vels = robot.data.joint_vel[0]
                
                # 全関節の 仕事率 = sum(|トルク * 角速度|)
                # 瞬時パワー (W)
                inst_power = torch.sum(torch.abs(all_torques * all_vels)).item()
                # 積算エネルギー (J)
                total_energy += inst_power * dt

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
    # メトリクス計算結果出力
    # ---------------------------------------------------------
    # CoT = E / (mgd)
    if distance_traveled > 0:
        cot = total_energy / (robot_mass * gravity * distance_traveled)
    else:
        cot = float('inf')
        
    avg_power = total_energy / sim_time if sim_time > 0 else 0.0

    print("="*40)
    print(f"METRICS REPORT")
    print(f"Distance:     {distance_traveled:.2f} m")
    print(f"Time:         {sim_time:.2f} s")
    print(f"Total Energy: {total_energy:.2f} J")
    print(f"Avg Power:    {avg_power:.2f} W")
    print(f"CoT:          {cot:.4f}")
    print("="*40)

    # ---------------------------------------------------------
    # グラフ生成処理
    # ---------------------------------------------------------
    if len(vel_log_target) > 0:
        # NumPy配列化
        vel_log_target = np.array(vel_log_target)
        vel_log_actual = np.array(vel_log_actual)
        traj_com = np.array(traj_com)
        traj_l_foot = np.array(traj_l_foot)
        traj_r_foot = np.array(traj_r_foot)
        forces_l_arr = np.array(forces_log_l)
        forces_r_arr = np.array(forces_log_r)

        # 1. 速度追従グラフ
        errors = vel_log_target - vel_log_actual
        rmse = np.sqrt(np.mean(errors**2, axis=0))
        
        fig_vel, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        time_axis = np.arange(len(vel_log_target)) * dt
        axs[0].plot(time_axis, vel_log_target[:, 0], 'r--', label="Target"); axs[0].plot(time_axis, vel_log_actual[:, 0], 'b', alpha=0.7, label="Actual"); axs[0].set_ylabel("Vel X"); axs[0].legend(); axs[0].grid(True)
        axs[1].plot(time_axis, vel_log_target[:, 1], 'r--', label="Target"); axs[1].plot(time_axis, vel_log_actual[:, 1], 'b', alpha=0.7, label="Actual"); axs[1].set_ylabel("Vel Y"); axs[1].grid(True)
        axs[2].plot(time_axis, vel_log_target[:, 2], 'r--', label="Target"); axs[2].plot(time_axis, vel_log_actual[:, 2], 'b', alpha=0.7, label="Actual"); axs[2].set_ylabel("Ang Vel Z"); axs[2].grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "velocity_tracking.png"))
        plt.close(fig_vel)

        # 2. 3D軌跡プロット
        print("[INFO] Generating 3D plot...")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(traj_com[:, 0], traj_com[:, 1], traj_com[:, 2], 'k-', linewidth=2, label='CoM')
        ax.plot(traj_l_foot[:, 0], traj_l_foot[:, 1], traj_l_foot[:, 2], 'b--', linewidth=1.5, label='Left Ankle')
        ax.plot(traj_r_foot[:, 0], traj_r_foot[:, 1], traj_r_foot[:, 2], 'r--', linewidth=1.5, label='Right Ankle')
        
        # スタートとゴールを強調
        ax.scatter(0, 0, traj_com[0, 2], color='green', s=100, label='Start (0,0)')
        ax.scatter(10, 0, traj_com[-1, 2], color='magenta', s=100, label='Goal (10,0)')
        
        ax.set_title('3D Trajectory')
        ax.legend()
        plt.savefig(os.path.join(log_dir, "trajectory_3d.png"))
        plt.close()

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
            
        # 理想ライン (0,0)-(10,0) を描画
        ax2d.plot([0, 10], [0, 0], 'g--', label='Target Line')
        
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

    # close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()