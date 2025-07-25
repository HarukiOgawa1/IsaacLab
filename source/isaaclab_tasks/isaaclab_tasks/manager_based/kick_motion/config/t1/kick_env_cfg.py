# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg,RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.actuators import  IdealPDActuatorCfg
from isaaclab.sim import UsdFileCfg
from isaaclab.sensors import ContactSensorCfg,ImuCfg
from isaaclab.sim.schemas import CollisionPropertiesCfg,MassPropertiesCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR


import isaaclab_tasks.manager_based.kick_motion.mdp as mdp

##
# ArticulationCfg for booster T1
##

BOOSTER_T1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/workspace/isaaclab/source/isaaclab_assets/data/T1_serial.usd",
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
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # pos=(0.0, 0.0, 0.72 - 0.0841),
        pos=(0.0, 0.0, 0.72),
        joint_pos={
            "AAHead_yaw" : 0.0, 
            "Head_pitch" : 0.0, 
            "Waist" : 0.0, 
            "Left_Shoulder_Pitch" : 0.0, 
            "Right_Shoulder_Pitch" : 0.0,   # ここはラジアンで指定するっぽい
            "Left_Shoulder_Roll" : -0.7853981634,  # 地面に対して水平が0度なので、内側に動かすで正しい？ 
            "Left_Elbow_Pitch" : 0.0, 
            "Left_Elbow_Yaw" : 0.0, 
            "Right_Shoulder_Roll" : 0.7853981634, 
            "Right_Elbow_Pitch" : 0.0, 
            "Right_Elbow_Yaw" : 0.0, 

            "Left_Hip_Pitch" : -0.2, 
            "Left_Hip_Roll" : 0.0, 
            "Left_Hip_Yaw" : 0.0, 
            "Left_Knee_Pitch" : 0.4, 
            "Left_Ankle_Pitch" : -0.25, 
            "Left_Ankle_Roll" : 0.0, 
            
            "Right_Hip_Pitch" : -0.2, 
            "Right_Hip_Roll" : 0.0, 
            "Right_Hip_Yaw" : 0.0, 
            "Right_Knee_Pitch" : 0.4, 
            "Right_Ankle_Pitch" : -0.25, 
            "Right_Ankle_Roll" : 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[".*_Hip_.*", ".*_Knee_.*", ".*_Ankle_.*"],
            effort_limit_sim={
                ".*Hip_Pitch.*" : 45.0 * 1.5,
                ".*Hip_Roll": 30.0 * 1.5,
                ".*Hip_Yaw.*": 30.0 * 1.5,
                ".*_Knee_Pitch": 60.0 * 1.5,
                ".*_Ankle_Pitch": 24.0 * 1.5,
                ".*_Ankle_Roll": 15.0 * 1.5,
            },
            velocity_limit_sim={
                ".*Hip_Pitch.*" : 12.5, #rad/s
                ".*Hip_Roll": 10.9,
                ".*Hip_Yaw.*": 10.9,
                ".*_Knee_Pitch": 11.7,
                ".*_Ankle_Pitch": 18.8,
                ".*_Ankle_Roll": 12.4,
            },
            stiffness={
                ".*Hip_Yaw.*": 200.0,
                ".*Hip_Roll": 200.0,
                ".*Hip_Pitch.*": 200.0,
                ".*_Knee_Pitch": 200.0,
                ".*_Ankle_.*": 50.0,
            },
            damping={
                ".*Hip_Yaw.*": 5.0,
                ".*Hip_Roll": 5.0,
                ".*Hip_Pitch.*": 5.0,
                ".*_Knee_Pitch": 5.0,
                ".*_Ankle_.*": 1.0,
            },
        ),
        "arms": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_Shoulder_Pitch",
                ".*_Shoulder_Roll",
                ".*_Elbow_Pitch",
                ".*_Elbow_Yaw",
            ],
            effort_limit=100,
            velocity_limit=50.0,
            stiffness=40.0,
            damping=10.0,
        ),
        "bodies": IdealPDActuatorCfg(
            joint_names_expr=["Waist","AAHead_yaw", "Head_pitch"],
            effort_limit=100.0,
            velocity_limit=100.0,
            stiffness=100.0,
            damping=10.0,
        )
    },
)


##
# Scene definition
##


@configclass
class KickEnvSceneCfg(InteractiveSceneCfg):

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=2.0,
            dynamic_friction=2.0,
        ),
        debug_vis=False,
    )


    soccer_ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/SoccerBall",
        spawn=sim_utils.SphereCfg(
            radius=0.11,  # 11cm
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
            ),
            mass_props=MassPropertiesCfg(
                mass=0.45,  # 450g
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 1.0, 1.0),
                metallic=0.0,
                roughness=0.7,
            ),
            collision_props=CollisionPropertiesCfg(
                collision_enabled=True
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.34,0.20,0.055)),
    )


    # robot
    robot: ArticulationCfg = BOOSTER_T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    contact_forces_foot = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*_foot_link",
                                            history_length=3, 
                                            track_air_time=True,
                                            update_period=0.0)
    contact_balls_right = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/right_foot_link", # 足に当たった場合を検知する
                                        update_period=0.0,
                                        history_length=1,
                                        track_air_time=True,
                                        filter_prim_paths_expr=[
                                        "{ENV_REGEX_NS}/SoccerBall",
                                    ])
    contact_balls_left = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/left_foot_link", # 足に当たった場合を検知する
                                        update_period=0.0,
                                        history_length=1,
                                        track_air_time=True,
                                        filter_prim_paths_expr=[
                                        "{ENV_REGEX_NS}/SoccerBall",
                                    ])
    base_imu = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/Trunk",gravity_bias=(0,0,0),debug_vis=True)


##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # なんか3~10に設定しているはずなのに12とかが引かれるんだが...。
    target_pos = mdp.commands.commands_cfg.UniformPose2dCommandCfg(
                        ranges=mdp.commands.commands_cfg.UniformPose2dCommandCfg.Ranges((1.0,1.0),
                                                                                        (-0.,0.0),
                                                                                        (0,0)),
                        simple_heading=True,
                        debug_vis=True,
                        resampling_time_range=(12.,12.), 
                        asset_name="robot"
                    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # elbowと首の動きが入っていない。Waistを消す手もあるかも。
    joint_effort = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*Knee.*",
                                                                             ".*Ankle.*",
                                                                             ".*Hip.*",
                                                                             "Waist",
                                                                             ".*Shoulder.*",] , scale=1.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        target_pos_obs = ObsTerm(func=mdp.generated_commands, params={"command_name": "target_pos"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        imu_orientation = ObsTerm(func=mdp.imu_orientation,
                                    params={"asset_cfg": SceneEntityCfg("base_imu")},     #, body_names="Trunk"
                                  )
        imu_angular_velocity = ObsTerm(func=mdp.imu_ang_vel,
                                        params={"asset_cfg": SceneEntityCfg("base_imu")},     #, body_names="Trunk"
                                       )
        imu_lin_acc = ObsTerm(func=mdp.imu_lin_acc,
                              params={"asset_cfg": SceneEntityCfg("base_imu")},
                              )
        actions = ObsTerm(func=mdp.last_action)
        ball_pos_rel = ObsTerm(func=mdp.ball_pos_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

# 最初はDR無くすのも手段。
# ボールの位置とかもランダム化無しにしてしまうのも良いね。それこそカリキュラムにするとか。
# まずは同じ初期条件から始めた方が良いんじゃない。
@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "mass_distribution_params": (0.98, 1.05),
            "operation": "scale",
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    # reset
    reset_joint_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            #"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "velocity_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.), "yaw": (0.0, 0.0)},
            "pose_range": {
                "x": (0., 0.),
                "y": (0.0, 0.0),
                "yaw": (-0.03, 0.03),
            },
        },
    )

    reset_ball = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "velocity_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.), "yaw": (0.0, 0.0)},
            "pose_range": {
                "x": (-0.02, 0.02),   # ここのx,y,zはdefault_posに追加される。
                "y": (-0.02, 0.00),
                "z": (0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
            "asset_cfg" :  SceneEntityCfg("soccer_ball")
        },
    )

    # interval


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    alive_after_kick = RewTerm(func=mdp.alive_after_kick,
                               weight=0.0,
                               params={"before_reward_div_scale" : 1.0}
                               )
    terminating = RewTerm(func=mdp.is_terminated, weight=-1000.0)
    # スパースな報酬(いつも貰える訳じゃ無い報酬は学習が難しい)
    ball_command_tracking = RewTerm(func=mdp.ball_command_tracking,weight=7.0)
    ball_distance = RewTerm(func=mdp.ball_distance,weight=2.0) # distanceまであると距離に対して過剰に配慮して適当な方向に蹴ってしまうのでは？
    reguralize_orientaion=RewTerm(func=mdp.reguralize_orientaion,
                                  weight=-1.2,
                                  params={"before_reward_scale_div":1.0},
                                  ) # 足だけで蹴るキックになりそう？
    touch_ball=RewTerm(func=mdp.touch_ball,weight=200.0)
    force_touch_ball_downhalf=RewTerm(func=mdp.force_touch_ball_downhalf,
                                      weight= -touch_ball.weight,   # 触るのを帳消しにする
                                      params={"asset_cfg":SceneEntityCfg("robot",body_names=["left_foot_link"])}
                                    )
    impose_stance=RewTerm(func=mdp.impose_stance,weight=1.6,
                          params={"touch_offset_step":0,
                                  "before_reward_scale_div": 1.3}
                        )
    action_smoothness_penalty=RewTerm(func=mdp.action_smoothness_penalty,weight=-0.3)
    joint_acceleration_penalty=RewTerm(func=mdp.joint_acceleration_penalty,weight=-2e-4,
                                        params={"asset_cfg":SceneEntityCfg("robot")})
    joint_torques_penalty=RewTerm(func=mdp.joint_torques_penalty,weight=-0.01,
                                    params={"asset_cfg":SceneEntityCfg("robot")})
    penalize_jump=RewTerm(func=mdp.penalize_jump,weight=-50.0)
    joint_pos_outof_limits=RewTerm(func=mdp.joint_pos_outof_limits,weight=0.0)
    fix_stance_foot_pos=RewTerm(func=mdp.fix_stance_foot_pos,
                                params={"asset_cfg":SceneEntityCfg("robot",body_names=["right_foot_link"])},
                                weight=-1.5)
    foot_slip_penalty=RewTerm(func=mdp.foot_slip_penalty,
                              params={"asset_cfg":SceneEntityCfg("robot",body_names="right_foot_link"),
                                      "sensor_cfg":SceneEntityCfg("contact_forces_foot",body_names="right_foot_link"),
                                      "threshold": 50.0
                                      },
                              weight=-1.0)
    # logger=RewTerm(func=mdp.logger,weight=1.0)
    # 地面とロボットとの摩擦を確認する
    # 蹴った後に元に戻るのに報酬を渡す
    # 参照軌道を使うにしてもそれからどれだけ離すのかというパラメータが入ってる

@configclass
class CurriculumCfg:
    touch_ball_cur = CurTerm(func=mdp.modify_reward_weight,
                                    params={
                                        "term_name" : "touch_ball",
                                        "num_steps" : 3000,
                                        "weight" : 20.0
                                    })
    force_touch_ball_downhalf_cur = CurTerm(func=mdp.modify_reward_weight,
                                        params={
                                            "term_name" : "force_touch_ball_downhalf",
                                            "num_steps" : 3000,
                                            "weight" : touch_ball_cur.params["weight"] * -1.0
                                        })


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    root_height_below_minimum = DoneTerm(mdp.root_height_below_minimum,params={"minimum_height": 0.6})
    bad_orientation = DoneTerm(func=mdp.bad_orientation,params={"limit_angle": 1.396}) # 80度で死ぬ

##
# Environment configuration
##


@configclass
class KickEnvEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: KickEnvSceneCfg = KickEnvSceneCfg(num_envs=4096, env_spacing=2.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 12  # ロボットのリセット無しでボールを蹴らせるとか。そもそももう少し長くした方が良さそう。
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15



@configclass
class KickEnvEnvCfg_Play(KickEnvEnvCfg):
    scene: KickEnvSceneCfg = KickEnvSceneCfg(num_envs=20, env_spacing=2.5)
    
    def __post__init__(self):
        super().__post_init__()
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class KickEnvCfg_Incremental(KickEnvEnvCfg):
    
    def __post__init__(self):
        super().__post_init__()
        self.rewards.touch_ball = None
        self.curriculum.touch_ball_cur.params["num_steps"] = 0.0 # 最初からカリキュラム開始

@configclass
class KickEnvCfg_StrongFriction(KickEnvEnvCfg_Play):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.terrain.physics_material.static_friction = 2.0
        self.scene.terrain.physics_material.dynamic_friction = 2.0