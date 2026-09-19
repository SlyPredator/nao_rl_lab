"""Base environment configuration for NAO robot locomotion."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from nao_rl.assets.nao import NAO_CFG, NAO_SPEC
import nao_rl.tasks.velocity.mdp as custom_mdp

JOINT_NAMES = [
    ".*Hip.*",
    ".*Ankle.*",
    ".*Knee.*",
    ".*Elbow.*",
    ".*Shoulder.*",
]


@configclass
class NAOSceneCfg(InteractiveSceneCfg):
    """Configuration for the interactive scene."""

    # Ground terrain (plane by default)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # Dome lighting
    light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # Robot articulation (spawns from local mass-corrected USD)
    robot = NAO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Contact sensor for feet air time & collision tracking
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=3,
        track_air_time=True,
    )


@configclass
class NAOActionsCfg:
    """Action term specifications."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES,
        scale=NAO_SPEC.action_scale,
        use_default_offset=True,
    )


@configclass
class NAOCommandsCfg:
    """Command generation specifications."""

    base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.2, 0.6),
            lin_vel_y=(-0.15, 0.15),
            ang_vel_z=(-0.8, 0.8),
            heading=(-3.14159, 3.14159),
        ),
    )


@configclass
class NAOEventsCfg:
    """Event / Domain Randomization specifications."""

    # Physics material friction randomization
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.5, 1.25),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # Joint initialization jitter
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # Base pose initialization jitter
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    # Push velocity perturbation every 10-15s
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}},
    )


@configclass
class NAOTerminationsCfg:
    """Episode termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["torso", ".*ForeArm"]),
            "threshold": 1.0,
        },
    )


@configclass
class NAOEnvCfgBase(ManagerBasedRLEnvCfg):
    """Base Manager-Based RL environment for NAO."""

    scene: NAOSceneCfg = NAOSceneCfg(num_envs=4096, env_spacing=2.5)
    actions: NAOActionsCfg = NAOActionsCfg()
    commands: NAOCommandsCfg = NAOCommandsCfg()
    events: NAOEventsCfg = NAOEventsCfg()
    terminations: NAOTerminationsCfg = NAOTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = NAO_SPEC.isaac_decimation
        self.sim.dt = NAO_SPEC.isaac_sim_dt
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0
