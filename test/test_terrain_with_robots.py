import argparse
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize extreme terrain with 1024 robots.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from nao_rl.assets.nao import NAO_CFG
from nao_rl.tasks.velocity.robots.nao.rough_env_cfg import ROUGH_TERRAIN_CFG

@configclass
class CustomSceneCfg(InteractiveSceneCfg):
    num_envs = 1024
    env_spacing = 2.5
    
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAIN_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
    )

    robot = NAO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

def main():
    sim_cfg = sim_utils.SimulationCfg(
        device="cuda:0",
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[25.0, 25.0, 25.0], target=[0.0, 0.0, 0.0])
    
    # light
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    # load scene
    scene_cfg = CustomSceneCfg()
    scene = InteractiveScene(scene_cfg)
    
    sim.reset()
    scene.reset()
    
    print("[INFO]: Setup complete. Spawned 1024 Nao robots on the terrain taking 0 actions.")
    
    while simulation_app.is_running():
        # zero action - just keep them at default joint targets so they try to stand still
        # instead of collapsing immediately
        scene["robot"].set_joint_position_target(scene["robot"].data.default_joint_pos)
        
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_cfg.dt)
        
if __name__ == "__main__":
    main()
    simulation_app.close()
