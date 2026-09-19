"""NAO task registrations with Gymnasium."""

import gymnasium as gym

from .flat_env_cfg import NaoVelocityFlatEnvCfg, NaoVelocityFlatEnvPlayCfg
from .rough_env_cfg import NaoVelocityRoughEnvCfg, NaoVelocityRoughEnvPlayCfg

# -----------------------------------------------------------------------------
# Flat Ground Tasks
# -----------------------------------------------------------------------------

gym.register(
    id="Nao-Velocity-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": NaoVelocityFlatEnvCfg,
        "rsl_rl_cfg_entry_point": "nao_rl.tasks.velocity.agents.ppo_cfg:NaoFlatPPORunnerCfg",
    },
)

gym.register(
    id="Nao-Velocity-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": NaoVelocityFlatEnvPlayCfg,
        "rsl_rl_cfg_entry_point": "nao_rl.tasks.velocity.agents.ppo_cfg:NaoFlatPPORunnerCfg",
    },
)

# -----------------------------------------------------------------------------
# Rough Terrain Tasks
# -----------------------------------------------------------------------------

gym.register(
    id="Nao-Velocity-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": NaoVelocityRoughEnvCfg,
        "rsl_rl_cfg_entry_point": "nao_rl.tasks.velocity.agents.ppo_cfg:NaoRoughPPORunnerCfg",
    },
)

gym.register(
    id="Nao-Velocity-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": NaoVelocityRoughEnvPlayCfg,
        "rsl_rl_cfg_entry_point": "nao_rl.tasks.velocity.agents.ppo_cfg:NaoRoughPPORunnerCfg",
    },
)
