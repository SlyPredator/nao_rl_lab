"""Rough Terrain Locomotion Environment for NAO."""

from __future__ import annotations

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from .flat_env_cfg import NaoVelocityFlatEnvCfg, NaoVelocityFlatEnvPlayCfg, ObservationsCfg


@configclass
class RoughObservationsCfg(ObservationsCfg):
    """Observation specifications for NAO rough terrain with history."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.history_length = 5
            self.flatten_history_dim = True

    policy: PolicyCfg = PolicyCfg()


ROUGH_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.30),
        "mild_roughness": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.40, noise_range=(0.005, 0.015), noise_step=0.005, border_width=0.25
        ),
        "gentle_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.03, 0.10), platform_width=2.0, border_width=0.25
        ),
        "gentle_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.03, 0.10), platform_width=2.0, border_width=0.25
        ),
    },
)


@configclass
class NaoVelocityRoughEnvCfg(NaoVelocityFlatEnvCfg):
    """NAO rough terrain velocity tracking environment."""

    observations: RoughObservationsCfg = RoughObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=ROUGH_TERRAIN_CFG,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            debug_vis=False,
        )

        # Reward adjustments for terrain adaptation
        self.rewards.flat_orientation_l2.weight = -0.3       # relaxed from -1.0 so torso can pitch on slopes
        self.rewards.feet_air_time.weight = 0.5              # increased from 0.25 for decisive step clearance
        self.rewards.feet_air_time.params["threshold"] = 0.18  # step threshold for stepping over small bumps
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.35  # widened from 0.25 to tolerate bump slowing


@configclass
class NaoVelocityRoughEnvPlayCfg(NaoVelocityRoughEnvCfg):
    """Evaluation / Play variant for rough terrain."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 256
        self.episode_length_s = 40.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
