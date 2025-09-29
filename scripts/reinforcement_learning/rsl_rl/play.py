# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

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
import os
import time
import torch
import csv

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

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
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

    # set the log directory for the environment (works for all environment types)
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

    # ロボットオブジェクトを取得し、足首リンクのボディインデックスを見つける
    robot = env.unwrapped.scene["robot"]
    body_names = robot.body_names
    # リンク名がUSDファイルで定義されているものと一致しているか確認してください
    left_ankle_idx = body_names.index("left_ankle_roll_link")
    right_ankle_idx = body_names.index("right_ankle_roll_link")
    print(f"[INFO] Found ankle link indices: Left='{left_ankle_idx}', Right='{right_ankle_idx}'")

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
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

    # CSVファイルを開き、書き込み準備
    csv_filename = "contact_forces.csv"
    try:
        csv_file = open(csv_filename, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        # ヘッダーを書き込む
        #csv_writer.writerow(["Timestep", "Left_Foot_Force_N", "Right_Foot_Force_N", "CoM_X", "CoM_Y", "CoM_Z", "Left_Foot_Height_Z", "Right_Foot_Height_Z"])
        csv_writer.writerow([
            "Timestep",
            "CoM_X", "CoM_Y", "CoM_Z",
            "Left_Ankle_X", "Left_Ankle_Y", "Left_Ankle_Z",
            "Right_Ankle_X", "Right_Ankle_Y", "Right_Ankle_Z"
        ])
        print(f"[INFO] Opened {csv_filename} for writing contact force data.")
    except IOError as e:
        print(f"[ERROR] Failed to open {csv_filename} for writing: {e}")
        # ファイルが開けなかった場合は、書き込み処理をスキップするためにNoneを設定
        csv_writer = None

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)

            # センサーデータを取得して表示
            # env.unwrappedでラップされた元の環境にアクセス
            scene = env.unwrapped.scene

            # ロボットのデータを取得 (最初の環境のみ)
            robot_data = scene["robot"].data
      
            # ベースリンクの位置をCoMとして取得 (ワールド座標系)
            com_position = robot_data.root_pos_w[0]

            # 全てのボディの位置を取得
            all_body_positions = robot_data.body_pos_w[0]
            # インデックスを使って足首リンクのZ座標（高さ）を取得
            left_foot_height = all_body_positions[left_ankle_idx][2]
            right_foot_height = all_body_positions[right_ankle_idx][2]

            left_ankle_position = all_body_positions[left_ankle_idx]
            right_ankle_position = all_body_positions[right_ankle_idx]

            # 左足の接触力を取得 (最初の環境のみ表示)
            contact_force_lf = scene["contact_forces_LF"].data.net_forces_w[0]
            # 右足の接触力を取得 (最初の環境のみ表示)
            contact_force_rf = scene["contact_forces_RF"].data.net_forces_w[0]

            # 接触力のノルム（大きさ）を計算
            force_norm_lf = torch.linalg.norm(contact_force_lf)
            force_norm_rf = torch.linalg.norm(contact_force_rf)

            print(
                f"T: {timestep:4} | "
                f"CoM (X,Y,Z): {com_position[0]:.2f}, {com_position[1]:.2f}, {com_position[2]:.2f} | "
                f"L Ankle (X,Y,Z): {left_ankle_position[0]:.2f}, {left_ankle_position[1]:.2f}, {left_ankle_position[2]:.2f} | "
                f"R Ankle (X,Y,Z): {right_ankle_position[0]:.2f}, {right_ankle_position[1]:.2f}, {right_ankle_position[2]:.2f}"
            )

            #print(
            #f"T: {timestep:4} | "
            #f"CoM (X,Y,Z): {com_position[0]:6.2f}, {com_position[1]:6.2f}, {com_position[2]:6.2f} | "
            #f"Foot Height (L,R): {left_foot_height:5.3f}, {right_foot_height:5.3f} | "
            #f"Forces (L,R): {force_norm_lf:7.2f} N, {force_norm_rf:7.2f} N"
            #)

            # データをCSVファイルに書き込む
            #if csv_writer is not None:
            #    # .item() を使ってTensorからPythonの数値に変換
            #    csv_writer.writerow([timestep, force_norm_lf.item(), force_norm_rf.item()])

            if csv_writer is not None:
                # .item() を使ってTensorからPythonの数値に変換
                csv_writer.writerow([
                    timestep,
                    com_position[0].item(),
                    com_position[1].item(),
                    com_position[2].item(),
                    left_ankle_position[0].item(),
                    left_ankle_position[1].item(),
                    left_ankle_position[2].item(),
                    right_ankle_position[0].item(),
                    right_ankle_position[1].item(),
                    right_ankle_position[2].item(),
                ])
            
        timestep += 1
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep >= args_cli.video_length:
                break

        #if args_cli.video:
        #    timestep += 1
        #    # Exit the play loop after recording one video
        #    if timestep == args_cli.video_length:
        #        break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # シミュレーション終了後にファイルを閉じる
    if 'csv_file' in locals() and not csv_file.closed:
        csv_file.close()
        print(f"[INFO] Contact force data saved to {csv_filename}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()