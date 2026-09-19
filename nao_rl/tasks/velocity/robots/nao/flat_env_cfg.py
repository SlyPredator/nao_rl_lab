"""Tuned Flat Ground Locomotion Environment for NAO."""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import nao_rl.tasks.velocity.mdp as custom_mdp
from .env_cfg import NAOEnvCfgBase, JOINT_NAMES


@configclass
class ObservationsCfg:
    """Observation specifications for NAO policy and critic (clean 20 limb joints)."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group (pure proprioception, no base_lin_vel)."""

        base_ang_vel = ObsTerm(
            func=custom_mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObsTerm(
            func=custom_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=custom_mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(
            func=custom_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=custom_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(func=custom_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged observations for critic group."""

        # Privileged true root states
        base_lin_vel = ObsTerm(func=custom_mdp.base_lin_vel)
        base_pos_z = ObsTerm(func=custom_mdp.base_pos_z)

        # Uncorrupted proprioceptive states
        base_ang_vel = ObsTerm(func=custom_mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=custom_mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=custom_mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(
            func=custom_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)},
        )
        joint_vel = ObsTerm(
            func=custom_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)},
        )
        actions = ObsTerm(func=custom_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for NAO flat locomotion."""

    # Terminations
    termination_penalty = RewTerm(func=custom_mdp.is_terminated, weight=-200.0)

    # Command tracking
    track_lin_vel_xy_exp = RewTerm(
        func=custom_mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.25},
    )
    track_ang_vel_z_exp = RewTerm(
        func=custom_mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.25},
    )

    feet_air_time = RewTerm(
        func=custom_mdp.feet_air_time,
        weight=0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle"),
            "command_name": "base_velocity",
            "threshold": 0.15,
        },
    )

    # Foot slipping penalty
    feet_slide = RewTerm(
        func=custom_mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle"),
        },
    )

    # Posture and stability
    flat_orientation_l2 = RewTerm(func=custom_mdp.flat_orientation_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=custom_mdp.ang_vel_xy_l2, weight=-0.05)
    dof_pos_limits = RewTerm(
        func=custom_mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*Ankle.*")},
    )
    joint_deviation_hip = RewTerm(
        func=custom_mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*HipYawPitch", ".*HipRoll", ".*HipPitch"])},
    )
    joint_deviation_arm = RewTerm(
        func=custom_mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*Elbow.*", ".*Shoulder.*", ".*Wrist.*"])},
    )

    # Motion smoothness
    action_rate_l2 = RewTerm(func=custom_mdp.action_rate_l2, weight=-0.005)
    dof_acc_l2 = RewTerm(func=custom_mdp.joint_acc_l2, weight=-1.25e-7)


@configclass
class NaoVelocityFlatEnvCfg(NAOEnvCfgBase):
    """NAO tuned flat ground velocity tracking environment."""

    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()


@configclass
class NaoVelocityFlatEnvPlayCfg(NaoVelocityFlatEnvCfg):
    """Evaluation / Play variant of the flat environment."""

    def __post_init__(self):
        super().__post_init__()

        # Smaller scene for visualization
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        # Fixed forward velocity command for evaluation
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # Disable noise and push perturbations
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
