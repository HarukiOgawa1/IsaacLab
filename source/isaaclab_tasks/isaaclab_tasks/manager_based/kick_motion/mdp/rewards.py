# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from .observations import ball_vel as get_ball_vel
from isaaclab.envs.mdp import projected_gravity
from .state import getTouchState
from .setting import getLegJointLimits,LEG_IDS,getArmJointLimits,ARM_IDS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# 正規化用に作ったの逆シグモイド
# 今は使っていない。
def normalize_func(x : torch.tensor)-> torch.tensor:
    return 1.1 - (torch.tanh((x / 2) - 1) + 1) / 2
# 正規化するには
# ガウシアンカーネルとか。シグモイドとかclipするのも手。
# 姿勢の誤差とかなら90度以上は意味ないから90度でclipするとかは可能。物理的な意味とか考えながらやる必要がある

# 0.2と0.5だと、xが0の時に約0.6で傾きがx=15くらいまで存在する
def sigmoid_normalize(x : torch.tensor) -> torch.tensor:
    return 1 / (1 + torch.exp(0.2 * (x - 0.5)))

# logistic kernel関数
def logistic_kernel(x :float,sensitivity:float)-> torch.tensor :
    return 2 / ((torch.exp(-x * sensitivity) + torch.exp(x * sensitivity)))

# ボールがコマンド(目標x座標、目標y座標)に追従しているほど報酬が大きくなる。
# この内部に角度の誤差と距離の誤差に対して重みを設定する箇所があるが、その重みも重要そう。
def ball_command_tracking(env: ManagerBasedRLEnv, command_name: str = "target_pos" ,asset_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball")) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)[:,:2]
    ball_vel = get_ball_vel(env)
    # ball_distance = torch.norm(ball_pos_rel,dim=1,p=2).unsqueeze(1) #l2ノルム計算してボールの距離を出す
    # command_norm = torch.norm(command,dim=1,p=2).unsqueeze(1)
    # 単位難しい。ラジアンなので基本0~3.14の間
    # radian = (command * ball_pos_rel).sum(dim=1) / (ball_distance * command_norm).squeeze()
    # radian = torch.abs(torch.acos(radian))
    # radian = torch.nan_to_num(radian,nan=0.0) # nanになったものを置き換える
    # ball_command_err = torch.abs(ball_distance - command_norm)
    
    #bd = 0~10くらい、rad = 0~3.14、bcd = 0~10mくらい？
    # radを*3くらいすれば3つのスケールは合いそう
    # rad_weight = 3.3
    # return  ((radian.unsqueeze(1) * rad_weight) + ball_command_err).squeeze()
    # return radian # 角度だけに報酬を与えるようにしてみた

    # 内積を取る形式に変えた。こちらにすると距離よりも角度が近い方が点数が高くなる。こちらの場合は重みを+にする必要あり
    # return (ball_vel*command).sum(dim=1)
    
    # コサイン類似度で計算するようにした
    cosine_si = (F.cosine_similarity(ball_vel,command,dim=1) + 1.0) / 2.0 # [0,1]の範囲にする
    return cosine_si


# ボールの飛距離に応じて報酬を与える。
def ball_distance(env: ManagerBasedRLEnv,asset_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball")) -> torch.Tensor:
    ball_vel = get_ball_vel(env) # 積分してる事になったのでこちらの方が正しく距離である。
    ball_distance = torch.norm(ball_vel,dim=1,p=2) #l2ノルム計算してボールの距離を出す
    return ball_distance

# ちょっとの誤差であまり反応しない方がよい
# コサインとか使うと、90度で0になったりするから良い感じになる。
# 上半身(Trunk)の姿勢が垂直に近いほど報酬が貰える。これが大きすぎるとダイナミックなキックができない気もする。
def reguralize_orientaion(env: ManagerBasedRLEnv,before_reward_scale_div: float = 0.1,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    posture = projected_gravity(env,asset_cfg)
    rew = torch.norm(posture,dim=1,p=2)
    return torch.where(getTouchState(env,touch_offset_step=0),rew, rew / before_reward_scale_div)

# ボールに触っていると報酬を得る。これを最後までオンにしていると蹴るよりも触ってしまうため、
# カリキュラムで途中で重みを途中で下げる等した方がよさそう。
def touch_ball(env: ManagerBasedRLEnv) -> torch.Tensor:
    contact_threshould = 0.1
    sensor_right = env.scene.sensors["contact_balls_right"]
    sensor_left = env.scene.sensors["contact_balls_left"]
    force_right = torch.norm(sensor_right.data.force_matrix_w[:,0,0],p=2,dim=1)
    force_left = torch.norm(sensor_left.data.force_matrix_w[:,0,0],p=2,dim=1)
    has_contact = torch.where(force_right > contact_threshould,1.0,0.0) + torch.where(force_left > contact_threshould,1.0,0.0)
    return torch.clip(has_contact,min=0.0,max=1.0)

# ジャンプに対してペナルティを掛ける関数
def penalize_jump(env: ManagerBasedRLEnv,asset_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_foot")) -> torch.Tensor:
    sensors = env.scene.sensors[asset_cfg.name]
    air_time_right = sensors.data.current_air_time[:,0]
    air_time_left = sensors.data.current_air_time[:,1]
    is_both_foot_in_air = (air_time_left > 0) & (air_time_right > 0)
    return is_both_foot_in_air.float()

# ボールを蹴った後に初期姿勢に戻すための報酬
# そのエピソードでボールに触ってからtouch_offset_step経った後から報酬の計算が始まり、各関節が初期姿勢に近くなるほど良い評価がされる。
# impose_stanceを下げるためにはボールを触らなくなる可能性もあるのでそこは注意
def impose_stance(env: ManagerBasedRLEnv,touch_offset_step: int = 0,before_reward_scale_div: float = 10.0,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.tensor:
    robot = env.scene[asset_cfg.name]
    err_pose = torch.abs(robot.data.joint_pos - robot.data.default_joint_pos)
    err_pose = err_pose.sum(dim=1)
    reward = sigmoid_normalize(err_pose)
    reward = torch.where(getTouchState(env,touch_offset_step),reward,reward / before_reward_scale_div)
    return reward

# spotが使っていた報酬関数.前回アクションとの差が大きいほどペナルティ
def action_smoothness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large instantaneous changes in the network action output"""
    return torch.linalg.norm((env.action_manager.action - env.action_manager.prev_action), dim=1)

# spotが使っていた報酬関数.ジョイントの加速度が大きいとペナルティ
def joint_acceleration_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint accelerations on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.joint_acc), dim=1)

# spotが使っていた報酬関数.ジョイントのトルクが高いとペナルティ
def joint_torques_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint torques on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.applied_torque), dim=1)

# spotが使っていた報酬関数.足の滑りにペナルティを与える
def foot_slip_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Penalize foot planar (xy) slip when in contact with the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    foot_planar_velocity = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)

    reward = is_contact * foot_planar_velocity
    return torch.sum(reward, dim=1)

# 蹴る足の軌道を決める報酬
def foot_swing(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.tensor:
    pass

# 安定性に寄与する報酬があった方が良いのでは？上半身を安定させる報酬よりもこちらの方が重要そうな気がする
def requralize_com(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.tensor:
    pass

# 姿勢整える系は蹴った後に発動するようにすれば良いのでは？
def logger(env: ManagerBasedRLEnv) -> torch.Tensor:
    # print(env.episode_length_buf[:3]) # episode_length_bufはステップ数が返ってくる。
    robot = env.scene["robot"]
    print(robot.find_joints([".*Shoulder.*",".*Elbow.*"]))
    raise RuntimeError("finish!!!")
    return torch.zeros(env.num_envs,device=env.device)

# kickした後にどれだけ生き残るかに報酬を与える
def alive_after_kick(env: ManagerBasedRLEnv,before_reward_div_scale :float = 10.0)-> torch.tensor :
    reward = (~env.termination_manager.terminated).float()
    return torch.where(getTouchState(env,touch_offset_step=0) ,reward, reward / before_reward_div_scale)
 
def joint_pos_outof_limits(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    min_j,max_j = getLegJointLimits(env.device)
    # compute out of limits constraints
    out_of_limits = -(
        robot.data.joint_pos[:,LEG_IDS] - min_j
    ).clip(max=0.0)
    out_of_limits += (
        robot.data.joint_pos[:,LEG_IDS] - max_j
    ).clip(min=0.0)

    # armの計算
    min_j,max_j = getArmJointLimits(env.device)
    out_of_limits = -(
        robot.data.joint_pos[:,ARM_IDS] - min_j
    ).clip(max=0.0)
    out_of_limits += (
        robot.data.joint_pos[:,ARM_IDS] - max_j
    ).clip(min=0.0)

    return torch.sum(out_of_limits, dim=1)

def fix_stance_foot_pos(env: ManagerBasedRLEnv,asset_cfg: SceneEntityCfg ) -> torch.tensor:
    robot = env.scene[asset_cfg.name]
    stance_foot_vel = robot.data.body_com_vel_w[:,asset_cfg.body_ids,:2].squeeze(1) # x,y
    stance_foot_vel_sum = torch.linalg.norm(stance_foot_vel,dim=1)
    return stance_foot_vel_sum.squeeze()


def force_touch_ball_downhalf(env: ManagerBasedRLEnv,asset_cfg: SceneEntityCfg) -> torch.tensor:
    robot = env.scene[asset_cfg.name]
    swing_foot_z = robot.data.body_link_pos_w[:,asset_cfg.body_ids,2].squeeze(1) # z position
    out_touch_range = (swing_foot_z > 0.20) | (swing_foot_z < 0.1) # 20cmよりも上、もしくは10cmよりも下でボールに触れたらペナルティ
    penalty =  torch.where(out_touch_range,1.0,0.0) # 20cmよりも上でボールに触れたらペナルティ
    # 衝突判定
    contact_threshould = 0.1
    sensor_right = env.scene.sensors["contact_balls_right"]
    sensor_left = env.scene.sensors["contact_balls_left"]
    force_right = torch.norm(sensor_right.data.force_matrix_w[:,0,0],p=2,dim=1)
    force_left = torch.norm(sensor_left.data.force_matrix_w[:,0,0],p=2,dim=1)    
    touch = (force_right > contact_threshould) | (force_left > contact_threshould)

    return torch.where(touch,penalty,0.0)   # 触れているやつはペナルティの値を返す。触れていないなら0