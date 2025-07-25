
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ボールとロボットの相対位置を計算する
def ball_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    ball: Articulation = env.scene["soccer_ball"]
    ball_vel = ball.data.root_com_vel_w[:,:2]
    return ball_vel

def ball_pos_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    ball: Articulation = env.scene["soccer_ball"]
    ball_pos_w = ball.data.root_pos_w[:,:2]
    robot: Articulation = env.scene[asset_cfg.name]
    robot_pos_w = robot.data.root_pos_w[:,:2]
    ball_pos_rel = ball_pos_w - robot_pos_w
    return ball_pos_rel