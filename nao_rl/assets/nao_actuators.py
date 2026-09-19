"""NAO actuator configurations for IsaacLab."""

from __future__ import annotations

from isaaclab.actuators import ImplicitActuatorCfg

NAO_ACTUATOR_CFG = {
    "legs": ImplicitActuatorCfg(
        joint_names_expr=[".*Hip.*", ".*Ankle.*", ".*Knee.*"],
        stiffness={
            ".*Hip.*": 20.0,
            ".*Knee.*": 10.0,
            ".*Ankle.*": 2.0,
        },
        damping={
            ".*Hip.*": 5.0,
            ".*Knee.*": 5.0,
            ".*Ankle.*": 0.1,
        },
        effort_limit_sim={
            ".*Hip.*": 8.0,
            ".*Ankle.*": 5.0,
            ".*Knee.*": 5.0,
        },
        velocity_limit_sim={
            ".*Hip.*": 12.0,
            ".*Ankle.*": 8.0,
            ".*Knee.*": 8.0,
        },
    ),
    "arms": ImplicitActuatorCfg(
        joint_names_expr=[".*Elbow.*", ".*Shoulder.*"],
        stiffness={
            ".*Shoulder.*": 10.0,
            ".*Elbow.*": 2.0,
        },
        damping={
            ".*Shoulder.*": 5.0,
            ".*Elbow.*": 1.0,
        },
        effort_limit_sim=5.0,
        velocity_limit_sim=7.0,
    ),
    "head": ImplicitActuatorCfg(
        joint_names_expr=["Head.*"],
        stiffness={
            ".*": 2.0,
        },
        damping={
            ".*": 2.0,
        },
        effort_limit_sim=8.0,
        velocity_limit_sim=12.0,
    ),
    "hands": ImplicitActuatorCfg(
        joint_names_expr=[".*Finger.*", ".*Thumb.*", ".*Wrist.*", ".*Hand.*"],
        stiffness={
            ".*": 2.0,
        },
        damping={
            ".*": 0.1,
        },
    ),
}
