# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Booster Robotics robots.

The following configurations are available:

* :obj:`T1_CFG`: T1 humanoid robot

Reference: https://github.com/BoosterRobotics
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
#from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

T1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/workspace/isaaclab/source/isaaclab_assets/data/RFC-Tsudanuma/T1_serial.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.72),
        joint_pos={
            "AAHead_yaw": 0.0,
            "Head_pitch": 0.0,
            ".*_Shoulder_Pitch": 1.04,
            ".*_Shoulder_Roll": 0.0,
            ".*_Elbow_Pitch": 0.52,
            ".*_Elbow_Yaw": 0.0,
            "Waist": 0.0,
            ".*_Hip_Pitch": 0.0,
            ".*_Hip_Roll": 0.0,
            ".*_Hip_Yaw": 0.0,
            ".*_Knee_Pitch": 0.0,
            ".*_Ankle_Pitch": 0.0,
            ".*_Ankle_Roll": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "Waist",
                ".*_Hip_Pitch",
                ".*_Hip_Roll",
                ".*_Hip_Yaw",
                ".*_Knee_Pitch",
            ],
            effort_limit=130,
            velocity_limit=100.0,
            stiffness={
                "Waist": 130.0,
                ".*_Hip_Pitch": 130.0,
                ".*_Hip_Roll": 130.0,
                ".*_Hip_Yaw": 130.0,
                ".*_Knee_Pitch": 130.0,
            },
            damping={
                "Waist": 5.0,
                ".*_Hip_Pitch": 5.0,
                ".*_Hip_Roll": 5.0,
                ".*_Hip_Yaw": 5.0,
                ".*_Knee_Pitch": 5.0,
            },
            armature={
                "Waist": 0.01,
                ".*_Hip_Pitch": 0.01,
                ".*_Hip_Roll": 0.01,
                ".*_Hip_Yaw": 0.01,
                ".*_Knee_Pitch": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit=20,
            joint_names_expr=[".*_Ankle_Pitch",".*_Ankle_Roll"],
            stiffness=50.0,
            damping=1.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "AAHead_yaw",
                "Head_pitch",
                ".*_Shoulder_Pitch",
                ".*_Shoulder_Roll",
                ".*_Elbow_Pitch",
                ".*_Elbow_Yaw",
            ],
            effort_limit=130,
            velocity_limit=100.0,
            stiffness=40.0,
            damping=10.0,
            armature={
                "AAHead_yaw": 0.01,
                "Head_pitch": 0.01,
                ".*_Shoulder_Pitch": 0.01,
                ".*_Shoulder_Roll": 0.01,
                ".*_Elbow_Pitch": 0.01,
                ".*_Elbow_Yaw": 0.01,
            },
        ),
    },
)
"""Configuration for the BoosterRobotics T1 Humanoid robot."""


#T1_MINIMAL_CFG = T1_CFG.copy()
#T1_MINIMAL_CFG.spawn.usd_path = "/workspace/isaaclab/source/isaaclab_assets/data/RFC-Tsudanuma/t1_minimal.usd"
"""Configuration for the BoosterRobotics T1 Humanoid robot with fewer collision meshes.

This configuration removes most collision meshes to speed up simulation.
"""