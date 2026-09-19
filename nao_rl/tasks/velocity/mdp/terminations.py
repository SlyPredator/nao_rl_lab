"""Termination conditions for NAO bipedal locomotion."""

from __future__ import annotations

from isaaclab.envs.mdp.terminations import (
    time_out,
    illegal_contact,
    bad_orientation,
    root_height_below_minimum,
)
