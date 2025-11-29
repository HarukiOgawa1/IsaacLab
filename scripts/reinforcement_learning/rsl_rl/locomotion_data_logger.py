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
import matplotlib.pyplot as plt # グラフ描画用

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
    # 30秒間リセットせずに動作させる設定
    # ---------------------------------------------------------
    env_cfg.episode_length_s = 1000.0  # タイムアウトを防ぐ
    if hasattr(env_cfg, "terminations"):
        env_cfg.terminations = None    # 転倒等によるリセットを無効化
        print("[INFO] Terminations disabled for continuous data collection.")

    # 速度コマンドを固定 (必要な場合)
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity"):
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        print("[INFO] Velocity commands fixed: X=1.0, Y=0.0, AngZ=0.0")

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

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
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

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # ---------------------------------------------------------
    # センサーとリンク情報の取得
    # ---------------------------------------------------------
    scene = env.unwrapped.scene
    robot = scene["robot"]

    # ---------------------------------------------------------
    # ### 追加・変更 ### 計測対象の関節インデックスを取得（左足＋右足）
    # ---------------------------------------------------------
    
    # 左足の関節リスト
    left_joint_names = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint"
    ]
    
    # 右足の関節リスト（追加）
    right_joint_names = [
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint"
    ]

    # 計測対象を結合
    target_joint_names = left_joint_names + right_joint_names
    
    # ロボットの全関節名を取得してインデックスを特定
    all_joint_names = robot.data.joint_names
    target_joint_indices = []
    
    print("[INFO] Searching for target joint indices...")
    for name in target_joint_names:
        if name in all_joint_names:
            idx = all_joint_names.index(name)
            target_joint_indices.append(idx)
            print(f"  Found {name} at index {idx}")
        else:
            print(f"  [WARNING] Joint '{name}' not found in robot model!")
    
    # ---------------------------------------------------------

    # 接触力センサーの取得
    contact_sensor = None
    if "contact_forces" in scene.sensors:
        contact_sensor = scene["contact_forces"]
    else:
        print("[WARNING] 'contact_forces' sensor not found in scene. Forces will be 0.")

    body_names = robot.body_names
    # 足のインデックスを検索
    try:
        left_foot_names = [name for name in body_names if "left_ankle_roll_link" in name]
        right_foot_names = [name for name in body_names if "right_ankle_roll_link" in name]
        
        if not left_foot_names or not right_foot_names:
            print("[WARNING] Specific ankle names not found. Trying generic search...")
            l_foot_idx_robot = [i for i, n in enumerate(body_names) if "left" in n and "ankle" in n][-1]
            r_foot_idx_robot = [i for i, n in enumerate(body_names) if "right" in n and "ankle" in n][-1]
        else:
            l_foot_idx_robot = body_names.index(left_foot_names[0])
            r_foot_idx_robot = body_names.index(right_foot_names[0])
            
        print(f"[INFO] Robot Foot Indices: Left={l_foot_idx_robot}, Right={r_foot_idx_robot}")

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
            
            print(f"[INFO] Sensor Foot Indices: Left={l_foot_idx_sensor}, Right={r_foot_idx_sensor}")

    except Exception as e:
        print(f"[ERROR] Could not determine foot indices: {e}")
        return

    # extract the neural network module
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

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    sim_time = 0.0
    target_duration = 15.0 # 15秒間計測

    # ---------------------------------------------------------
    # CSVファイルを開く
    # ---------------------------------------------------------
    try:
        f_forces = open("data_forces.csv", "w", newline="", encoding="utf-8")
        f_foot_z = open("data_foot_z.csv", "w", newline="", encoding="utf-8")
        f_com = open("data_com.csv", "w", newline="", encoding="utf-8")
        f_vel = open("data_velocity.csv", "w", newline="", encoding="utf-8")
        
        # ### 追加・変更 ### 関節データ用CSV (左右両方)
        f_joints = open("data_joints_all.csv", "w", newline="", encoding="utf-8")

        w_forces = csv.writer(f_forces)
        w_foot_z = csv.writer(f_foot_z)
        w_com = csv.writer(f_com)
        w_vel = csv.writer(f_vel)
        w_joints = csv.writer(f_joints)

        # ヘッダー
        w_forces.writerow(["Timestep", "Left_Foot_Force_N", "Right_Foot_Force_N"])
        w_foot_z.writerow(["Timestep", "Left_Foot_Height_Z", "Right_Foot_Height_Z"])
        w_com.writerow(["Timestep", "CoM_X", "CoM_Y", "CoM_Z"])
        w_vel.writerow([
            "Timestep", 
            "Target_Vel_X", "Actual_Vel_X", 
            "Target_Vel_Y", "Actual_Vel_Y", 
            "Target_Ang_Vel_Z", "Actual_Ang_Vel_Z"
        ])

        # ### 追加・変更 ### 関節データヘッダー (前半トルク -> 後半速度)
        joint_header = ["Timestep"]
        # 先に全関節のトルクカラムを追加
        for name in target_joint_names:
            joint_header.append(f"{name}_Torque")
        # 次に全関節の速度カラムを追加
        for name in target_joint_names:
            joint_header.append(f"{name}_Vel")
            
        w_joints.writerow(joint_header)

        print("[INFO] CSV files opened for recording.")
    except IOError as e:
        print(f"[ERROR] Failed to open CSV files: {e}")
        return

    # 3Dプロット用のデータを保存するリスト
    traj_com = []
    traj_l_foot = []
    traj_r_foot = []

    # 速度比較用リスト
    vel_log_target = [] # [x, y, ang_z]
    vel_log_actual = [] # [x, y, ang_z]

    # simulate environment
    print(f"[INFO] Starting simulation for {target_duration} seconds...")
    
    while simulation_app.is_running():
        start_time = time.time()
        
        # 30秒経過したら終了
        if sim_time >= target_duration:
            print("[INFO] 30 seconds reached. Stopping.")
            break

        # run everything in inference mode
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

            # -----------------------------------------------------
            # データ取得
            # -----------------------------------------------------
            
            # 1. 重心 (CoM)
            com_pos = robot.data.root_pos_w[0]

            # 2. 足の位置 (X,Y,Z全て取得 - 3Dプロット用)
            all_body_pos = robot.data.body_pos_w[0]
            l_foot_pos_vec = all_body_pos[l_foot_idx_robot]
            r_foot_pos_vec = all_body_pos[r_foot_idx_robot]

            # 3. 足の力 (Force N)
            l_force_val = 0.0
            r_force_val = 0.0
            
            if contact_sensor is not None:
                net_forces = contact_sensor.data.net_forces_w[0]
                l_force_vec = net_forces[l_foot_idx_sensor]
                r_force_vec = net_forces[r_foot_idx_sensor]
                l_force_val = torch.norm(l_force_vec).item()
                r_force_val = torch.norm(r_force_vec).item()

            # 4. 速度 (Target vs Actual)
            # コマンドの取得 (Target)
            cmd_mgr = env.unwrapped.command_manager
            cmd = cmd_mgr.get_command("base_velocity")[0] 
            target_vel_x = cmd[0].item()
            target_vel_y = cmd[1].item()
            target_ang_z = cmd[2].item()

            # 実際の速度 (Actual) - World FrameからBase Frameに変換が必要
            root_quat = robot.data.root_quat_w[0]
            root_lin_vel_w = robot.data.root_lin_vel_w[0]
            root_ang_vel_w = robot.data.root_ang_vel_w[0]
            
            # 【修正点】クォータニオンを使ってワールド速度をベース速度へ回転変換
            lin_vel_b = math_utils.quat_apply_inverse(root_quat, root_lin_vel_w)
            ang_vel_b = math_utils.quat_apply_inverse(root_quat, root_ang_vel_w)
            
            actual_vel_x = lin_vel_b[0].item()
            actual_vel_y = lin_vel_b[1].item()
            actual_ang_z = ang_vel_b[2].item() # Z軸周りの角速度

            # -----------------------------------------------------
            # ### 追加・変更 ### 関節トルクと角速度の取得と書き込み
            # -----------------------------------------------------
            all_torques = robot.data.applied_torque[0]
            all_vels = robot.data.joint_vel[0]
            
            # データを保存するためのリスト
            current_torques = []
            current_vels = []

            # ターゲット関節のデータを抽出
            for idx in target_joint_indices:
                current_torques.append(all_torques[idx].item())
                current_vels.append(all_vels[idx].item())

            # CSV行データの作成: [Timestep] + [全トルク] + [全速度]
            joint_row = [timestep] + current_torques + current_vels

            # -----------------------------------------------------
            # CSV書き込み
            # -----------------------------------------------------
            w_forces.writerow([timestep, -l_force_val, r_force_val])
            w_foot_z.writerow([timestep, l_foot_pos_vec[2].item(), r_foot_pos_vec[2].item()])
            w_com.writerow([timestep, com_pos[0].item(), com_pos[1].item(), com_pos[2].item()])
            w_vel.writerow([
                timestep,
                target_vel_x, actual_vel_x,
                target_vel_y, actual_vel_y,
                target_ang_z, actual_ang_z
            ])
            # ### 追加・変更 ### 関節データ書き込み
            w_joints.writerow(joint_row)

            # -----------------------------------------------------
            # プロット用データ保存
            # -----------------------------------------------------
            traj_com.append(com_pos.cpu().numpy())
            traj_l_foot.append(l_foot_pos_vec.cpu().numpy())
            traj_r_foot.append(r_foot_pos_vec.cpu().numpy())
            
            vel_log_target.append([target_vel_x, target_vel_y, target_ang_z])
            vel_log_actual.append([actual_vel_x, actual_vel_y, actual_ang_z])

        timestep += 1
        sim_time += dt

        # video recording用
        if args_cli.video and timestep >= args_cli.video_length:
             pass

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # シミュレーション終了後にファイルを閉じる
    f_forces.close()
    f_foot_z.close()
    f_com.close()
    f_vel.close()
    # ### 追加・変更 ###
    f_joints.close() 

    print("[INFO] All CSV data saved.")

    # ---------------------------------------------------------
    # 追従精度の計算 (RMSE)
    # ---------------------------------------------------------
    vel_log_target = np.array(vel_log_target)
    vel_log_actual = np.array(vel_log_actual)
    
    # データが空でないか確認
    if len(vel_log_target) > 0:
        errors = vel_log_target - vel_log_actual
        rmse = np.sqrt(np.mean(errors**2, axis=0))
        
        print("-" * 50)
        print("Velocity Tracking Accuracy (RMSE - Lower is better):")
        print(f"  Linear Velocity X: {rmse[0]:.4f} m/s")
        print(f"  Linear Velocity Y: {rmse[1]:.4f} m/s")
        print(f"  Angular Velocity Z: {rmse[2]:.4f} rad/s")
        print("-" * 50)

        # ---------------------------------------------------------
        # 速度追従グラフの生成
        # ---------------------------------------------------------
        fig_vel, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        time_axis = np.arange(len(vel_log_target)) * dt
        
        axs[0].plot(time_axis, vel_log_target[:, 0], label="Target", color="red", linestyle="--")
        axs[0].plot(time_axis, vel_log_actual[:, 0], label="Actual", color="blue", alpha=0.7)
        axs[0].set_ylabel("Vel X (m/s)")
        axs[0].set_title("Linear Velocity X Tracking")
        axs[0].legend()
        axs[0].grid(True)
        
        axs[1].plot(time_axis, vel_log_target[:, 1], label="Target", color="red", linestyle="--")
        axs[1].plot(time_axis, vel_log_actual[:, 1], label="Actual", color="blue", alpha=0.7)
        axs[1].set_ylabel("Vel Y (m/s)")
        axs[1].set_title("Linear Velocity Y Tracking")
        axs[1].grid(True)

        axs[2].plot(time_axis, vel_log_target[:, 2], label="Target", color="red", linestyle="--")
        axs[2].plot(time_axis, vel_log_actual[:, 2], label="Actual", color="blue", alpha=0.7)
        axs[2].set_ylabel("Ang Vel Z (rad/s)")
        axs[2].set_xlabel("Time (s)")
        axs[2].set_title("Angular Velocity Z Tracking")
        axs[2].grid(True)

        plt.tight_layout()
        plt.savefig("velocity_tracking.png")
        print("[INFO] Velocity tracking plot saved to velocity_tracking.png")
        plt.close(fig_vel)
    else:
        print("[WARNING] No data collected.")

    # ---------------------------------------------------------
    # 3Dプロットの生成と保存
    # ---------------------------------------------------------
    if len(traj_com) > 0:
        print("[INFO] Generating 3D plot...")
        traj_com = np.array(traj_com)
        traj_l_foot = np.array(traj_l_foot)
        traj_r_foot = np.array(traj_r_foot)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(traj_com[:, 0], traj_com[:, 1], traj_com[:, 2], 
                label='CoM Trajectory', color='black', linestyle='-', linewidth=2)

        ax.plot(traj_l_foot[:, 0], traj_l_foot[:, 1], traj_l_foot[:, 2], 
                label='Left Ankle', color='blue', linestyle='--', linewidth=1.5)

        ax.plot(traj_r_foot[:, 0], traj_r_foot[:, 1], traj_r_foot[:, 2], 
                label='Right Ankle', color='red', linestyle='--', linewidth=1.5)

        ax.scatter(traj_com[0, 0], traj_com[0, 1], traj_com[0, 2], 
                   color='green', s=100, label='Start Point (CoM)')
        ax.scatter(traj_com[-1, 0], traj_com[-1, 1], traj_com[-1, 2], 
                   color='magenta', s=100, label='End Point (CoM)')

        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate')
        ax.set_title('3D Plot of CoM and Ankle Trajectories')
        ax.legend()

        plot_filename = "trajectory_plot.png"
        plt.savefig(plot_filename)
        print(f"[INFO] 3D plot saved to {plot_filename}")
        plt.close()

    # close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()