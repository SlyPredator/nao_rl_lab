"""Aldebaran NAO robot configuration for IsaacLab."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from .nao_actuators import NAO_ACTUATOR_CFG

# Resolve path to local assembled USD inside robots/nao/usd/nao.usd
_LOCAL_USD_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../robots/nao/usd/nao.usd"))


def resolve_nao_usd_path() -> str:
    override = os.environ.get("NAO_USD_PATH")
    if override and os.path.isfile(override):
        return override
    if os.path.isfile(_LOCAL_USD_PATH):
        return _LOCAL_USD_PATH
    # Fallback to source repo path if accessed before copy
    return _LOCAL_USD_PATH


@dataclass(frozen=True, slots=True)
class NAORobotSpec:
    """Robot physical specifications and constants."""
    joint_names: list[str]
    foot_link_names: list[str]
    action_scale: float
    nominal_base_height: float
    isaac_sim_dt: float
    isaac_decimation: int


NAO_SPEC = NAORobotSpec(
    joint_names=[
        "HeadYaw", "HeadPitch",
        "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw", "LHand",
        "RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw", "RHand",
        "LHipYawPitch", "LHipRoll", "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll",
        "RHipYawPitch", "RHipRoll", "RHipPitch", "RKneePitch", "RAnklePitch", "RAnkleRoll",
    ],
    foot_link_names=["l_ankle", "r_ankle"],
    action_scale=0.5,
    nominal_base_height=0.35,
    isaac_sim_dt=0.005,
    isaac_decimation=4,
)

NAO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=resolve_nao_usd_path(),
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
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),
        joint_pos={
            "HeadPitch": 0.0,
            "HeadYaw": 0.0,
            ".*ShoulderPitch": 1.3963,
            "LShoulderRoll": 0.3491,
            "RShoulderRoll": -0.3491,
            "LElbowYaw": -1.3963,
            "LElbowRoll": -1.0472,
            "RElbowYaw": 1.3963,
            "RElbowRoll": 1.0472,
            ".*WristYaw": 0.0,
            ".*Hand": 0.0,
            ".*HipYawPitch": 0.0,
            ".*HipRoll": 0.0,
            ".*HipPitch": -0.3491,
            ".*KneePitch": 0.6981,
            ".*AnklePitch": -0.3491,
            ".*AnkleRoll": 0.0,
        },
    ),
    actuators=NAO_ACTUATOR_CFG,
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration for the Aldebaran NAO robot with mass-corrected USD physics."""
