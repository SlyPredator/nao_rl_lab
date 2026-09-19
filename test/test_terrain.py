import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize extreme terrain.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainImporterCfg, TerrainImporter

# NOTE: Import your custom terrain configurations here
# from your_module import HfPyramidSlopeWithNoiseCfg, HfDiscreteObstaclesTerrainCfg

# Mocking custom configs if they aren't imported (Remove these if you have them imported)
# class HfPyramidSlopeWithNoiseCfg(terrain_gen.TerrainGeneratorCfg):
#     pass
# class HfDiscreteObstaclesTerrainCfg(terrain_gen.TerrainGeneratorCfg):
#     pass

EXTREME_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    seed=42,
    size=(8.0, 8.0),
    border_width=25.0,
    num_rows=12,
    num_cols=12,
    horizontal_scale=0.10,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=True,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.10),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.20,
            noise_range=(0.005, 0.12),
            noise_step=0.005,
            downsampled_scale=0.2,
            border_width=0.0,
        ),
        # "hf_slope_with_noise": HfPyramidSlopeWithNoiseCfg(
        #     proportion=0.15,
        #     slope_range=(0.0, 0.4),
        #     platform_width=3.0,
        #     border_width=0.0,
        #     noise_amplitude_range=(0.005, 0.10),
        #     noise_step=0.005,
        #     downsampled_scale=0.2,
        # ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05,
            slope_range=(0.0, 0.4),
            platform_width=3.0,
            border_width=0.0,
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.05,
            slope_range=(0.0, 0.4),
            platform_width=3.0,
            border_width=0.0,
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.01, 0.20),
            step_width=0.35,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.01, 0.20),
            step_width=0.35,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # "discrete_obstacles": HfDiscreteObstaclesTerrainCfg(
        #     proportion=0.15,
        #     max_height_range=(0.01, 0.35),
        #     obstacle_size_range=(0.6, 1.5),
        #     num_obstacles=20,
        #     platform_width=3.0,
        # ),
    },
)


def main():
    sim_cfg = sim_utils.SimulationCfg(
        device="cuda:0",
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[15.0, 15.0, 15.0], target=[0.0, 0.0, 0.0])
    
    # light
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    # load terrain
    terrain_importer_cfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=EXTREME_TERRAIN_CFG,
        max_init_terrain_level=11,
        collision_group=-1,
    )
    
    terrain = TerrainImporter(terrain_importer_cfg)
    
    sim.reset()
    
    print("[INFO]: Setup complete. You can now view the terrain in the viewer.")
    
    while simulation_app.is_running():
        sim.step()
        
if __name__ == "__main__":
    main()
    simulation_app.close()
