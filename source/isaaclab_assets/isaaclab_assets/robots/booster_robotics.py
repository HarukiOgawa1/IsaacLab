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
            "Left_Shoulder_Pitch": 0.0,
            "Right_Shoulder_Pitch": 0.0,
            "Left_Shoulder_Roll": 0.0,
            "Right_Shoulder_Roll": 0.0,
            "Left_Elbow_Pitch": 0.0,
            "Right_Elbow_Pitch": 0.0,
            "Left_Elbow_Yaw": 0.0,
            "Right_Elbow_Yaw": 0.0,
            "Waist": 0.0,
            "Left_Hip_Pitch": 0.0,
            "Right_Hip_Pitch": 0.0,
            "Left_Hip_Roll": 0.0,
            "Right_Hip_Roll": 0.0,
            "Left_Hip_Yaw": 0.0,
            "Right_Hip_Yaw": 0.0,
            "Left_Knee_Pitch": 0.0,
            "Right_Knee_Pitch": 0.0,
            "Left_Ankle_Pitch": 0.0,
            "Right_Ankle_Pitch": 0.0,
            "Left_Ankle_Roll": 0.0,
            "Right_Ankle_Roll": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip_pitch": ImplicitActuatorCfg(
            joint_names_expr=[
                "Left_Hip_Pitch",
                "Right_Hip_Pitch",
            ],
            effort_limit=45,
            velocity_limit=12.5,
            stiffness={
                "Left_Hip_Pitch": 200.0,
                "Right_Hip_Pitch": 200.0,
            },
            damping={
                "Left_Hip_Pitch": 5.0,
                "Right_Hip_Pitch": 5.0,
            },
            armature={
                "Left_Hip_Pitch": 0.01,
                "Right_Hip_Pitch": 0.01,
            },
        ),
        "hip_roll_yaw_waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "Left_Hip_Roll",
                "Right_Hip_Roll",
                "Left_Hip_Yaw",
                "Right_Hip_Yaw",
                "Waist",
            ],
            effort_limit=30,
            velocity_limit=10.9,
            stiffness={
                "Left_Hip_Roll": 200.0,
                "Right_Hip_Roll": 200.0,
                "Left_Hip_Yaw": 200.0,
                "Right_Hip_Yaw": 200.0,
                "Waist": 200.0,
            },
            damping={
                "Left_Hip_Roll": 5.0,
                "Right_Hip_Roll": 5.0,
                "Left_Hip_Yaw": 5.0,
                "Right_Hip_Yaw": 5.0,
                "Waist": 5.0,
            },
            armature={
                "Left_Hip_Roll": 0.01,
                "Right_Hip_Roll": 0.01,
                "Left_Hip_Yaw": 0.01,
                "Right_Hip_Yaw": 0.01,
                "Waist": 0.01,
            },
        ),
        "knee_pitch": ImplicitActuatorCfg(
            joint_names_expr=[
                "Left_Knee_Pitch",
                "Right_Knee_Pitch",
            ],
            effort_limit=60,
            velocity_limit=11.7,
            stiffness={
                "Left_Knee_Pitch": 200.0,
                "Right_Knee_Pitch": 200.0,
            },
            damping={
                "Left_Knee_Pitch": 5.0,
                "Right_Knee_Pitch": 5.0,
            },
            armature={
                "Left_Knee_Pitch": 0.01,
                "Right_Knee_Pitch": 0.01,
            },
        ),
        "ankle_pitch": ImplicitActuatorCfg(
            joint_names_expr=[
                "Left_Ankle_Pitch",
                "Right_Ankle_Pitch",
            ],
            effort_limit=24,
            velocity_limit=18.8,
            stiffness={
                "Left_Ankle_Pitch": 50.0,
                "Right_Ankle_Pitch": 50.0,
            },
            damping={
                "Left_Ankle_Pitch": 1.0,
                "Right_Ankle_Pitch": 1.0,
            },
            armature={
                "Left_Ankle_Pitch": 0.01,
                "Right_Ankle_Pitch": 0.01,
            },
        ),
        "ankle_roll": ImplicitActuatorCfg(
            joint_names_expr=[
                "Left_Ankle_Roll",
                "Right_Ankle_Roll",
            ],
            effort_limit=15,
            velocity_limit=12.4,
            stiffness={
                "Left_Ankle_Roll": 50.0,
                "Right_Ankle_Roll": 50.0,
            },
            damping={
                "Left_Ankle_Roll": 1.0,
                "Right_Ankle_Roll": 1.0,
            },
            armature={
                "Left_Ankle_Roll": 0.01,
                "Right_Ankle_Roll": 0.01,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "Left_Shoulder_Pitch",
                "Right_Shoulder_Pitch",
                "Left_Shoulder_Roll",
                "Right_Shoulder_Roll",
                "Left_Elbow_Pitch",
                "Right_Elbow_Pitch",
                "Left_Elbow_Yaw",
                "Right_Elbow_Yaw",
            ],
            effort_limit=18,
            velocity_limit=18.8,
            stiffness={
                "Left_Shoulder_Pitch": 15.0,
                "Right_Shoulder_Pitch": 15.0,
                "Left_Shoulder_Roll": 15.0,
                "Right_Shoulder_Roll": 15.0,
                "Left_Elbow_Pitch": 15.0,
                "Right_Elbow_Pitch": 15.0,
                "Left_Elbow_Yaw": 15.0,
                "Right_Elbow_Yaw": 15.0,
            },
            damping={
                "Left_Shoulder_Pitch": 5.0,
                "Right_Shoulder_Pitch": 5.0,
                "Left_Shoulder_Roll": 5.0,
                "Right_Shoulder_Roll": 5.0,
                "Left_Elbow_Pitch": 5.0,
                "Right_Elbow_Pitch": 5.0,
                "Left_Elbow_Yaw": 5.0,
                "Right_Elbow_Yaw": 5.0,
            },
            armature={
                "Left_Shoulder_Pitch": 0.01,
                "Right_Shoulder_Pitch": 0.01,
                "Left_Shoulder_Roll": 0.01,
                "Right_Shoulder_Roll": 0.01,
                "Left_Elbow_Pitch": 0.01,
                "Right_Elbow_Pitch": 0.01,
                "Left_Elbow_Yaw": 0.01,
                "Right_Elbow_Yaw": 0.01,
            },
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=[
                "AAHead_yaw",
                "Head_pitch",
            ],
            effort_limit=7,
            velocity_limit=12.6,
            stiffness={
                "AAHead_yaw": 15.0,
                "Head_pitch": 15.0,
            },
            damping={
                "AAHead_yaw": 5.0,
                "Head_pitch": 5.0,
            },
            armature={
                "AAHead_yaw": 0.01,
                "Head_pitch": 0.01,
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