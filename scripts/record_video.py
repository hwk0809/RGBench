import os
import os.path as osp
import hydra

from omegaconf import DictConfig,OmegaConf
import sys
from loguru import logger

# --- Main Framework Imports ---
from rgbench.envs import get_env
from rgbench.envs.base import BaseEnvWrapper

try:
    PROJECT_ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__)))
    if PROJECT_ROOT_DIR not in sys.path:
        sys.path.append(PROJECT_ROOT_DIR)
except NameError:
    # Handle cases where __file__ is not defined (e.g., interactive environments)
    PROJECT_ROOT_DIR = os.getcwd()
    if PROJECT_ROOT_DIR not in sys.path:
        sys.path.append(PROJECT_ROOT_DIR)


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(cfg: DictConfig):
    try:
        logger.info("Performing configuration check...")
        OmegaConf.resolve(cfg)
        logger.info(" Configuration interpolation check passed.")
    except Exception as e:
        logger.error("CONFIGURATION ERROR! Please check your YAML files or command-line overrides.")
        logger.error(f"--> Error details: {e}")
        logger.error(f"--- Failing Configuration ---\n{OmegaConf.to_yaml(cfg)}")
        sys.exit(1)

        # --- 1. Setup paths and logging ---
    raw_cfg = cfg
    cfg = cfg.active_run
    if cfg.action.get('type') == "fling" and cfg.sim_model.get('type') != 'fixed_point':
        logger.error("Fling action is only supported with fixed point simulation model.")
        sys.exit(1)

    logger.info("Initializing environment: '{}'...", cfg.env.name)
    env: BaseEnvWrapper = get_env(
        cfg)  # Somthing important, import cv2 must after pybullet env initialization (pybullet bug)
    logger.success("Environment initialized successfully.")

    output_filename = os.path.join(PROJECT_ROOT_DIR, "outputs", "video", f"{cfg.cloth.name}_{cfg.sim_model.type}_{cfg.action.type}_{raw_cfg.params.sample_index}_{cfg.env.name}_video.mp4")
    env.record_video(output_filename,camera_name="video_third_view")

if __name__ == "__main__":
    main()