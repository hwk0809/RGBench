import os
import os.path as osp
import sys
import time
import argparse
import torch
import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm

from loguru import logger
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING) 
logging.getLogger('PIL').setLevel(logging.WARNING)  # silence DEBUG/INFO logs
import hydra
from omegaconf import DictConfig, ValidationError, OmegaConf
from typing import Dict

# --- Main Framework Imports ---
from rgbench.envs import get_env
from rgbench.envs.base import BaseEnvWrapper

# --- Generic Utility Imports ---
# These utilities are simulator-agnostic and work with any environment.
from rgbench.metrics import calculate_cloth_sim_stability, chamfer_distance
from rgbench.visualization import visualize_pcd_comparison, analyze_and_visualize_metrics
from rgbench.transforms import TransformsManager
from rgbench.output import OutputManager
from rgbench.csv_data import load_processed_data, TrajectoryInterpolator
import point_cloud_utils as pcu

# --- System Path Setup ---
try:
    PROJECT_ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), ".."))
    if PROJECT_ROOT_DIR not in sys.path:
        sys.path.append(PROJECT_ROOT_DIR)
except NameError:
    # Handle cases where __file__ is not defined (e.g., interactive environments)
    PROJECT_ROOT_DIR = os.getcwd()
    if PROJECT_ROOT_DIR not in sys.path:
        sys.path.append(PROJECT_ROOT_DIR)


def load_target_pcds(pcd_dir: str, master_start_time: float, transforms_manager: TransformsManager) -> list:
    """
    Loads all segmented PCD files, transforms them to the world frame, and sorts them.

    Args:
        pcd_dir (str): Path to the 'segment_pcds' directory.
        master_start_time (float): The absolute start time of the entire process.
        transforms_manager (TransformsManager): The manager providing the coordinate transform.

    Returns:
        list: A sorted list of tuples (normalized_timestamp, transformed_point_cloud).
    """
    target_pcds = []
    if not osp.isdir(pcd_dir):
        logger.warning(f"Target PCD directory not found at {pcd_dir}")
        return []

    for f in os.listdir(pcd_dir):
        if f.endswith("_segmented.pcd"):
            try:
                timestamp_str = f.replace("pointcloud_", "").replace("_segmented.pcd", "")
                absolute_timestamp = float(timestamp_str)
                normalized_time = absolute_timestamp - master_start_time

                if normalized_time >= 0:
                    pcd_path = osp.join(pcd_dir, f)
                    # Load the PCD from camera frame
                    pcd_camera = o3d.io.read_point_cloud(pcd_path)
                    if not pcd_camera.has_points():
                        continue

                    # --- TRANSFORMATION STEP ---
                    # Transform the PCD to the world frame
                    pcd_world = pcd_camera.transform(transforms_manager.camera_to_world_transform)
                    target_pcds.append((normalized_time, pcd_world))
            except (ValueError, IndexError):
                logger.warning(f"Could not parse timestamp from filename: {f}")
                continue

    target_pcds.sort(key=lambda x: x[0])
    logger.success(f"Found, transformed, and sorted {len(target_pcds)} target point clouds.")
    return target_pcds

def calculate_camera_dynamic_delay(
    current_pcd: o3d.geometry.PointCloud,
    current_time: float,
    left_trajectory_interpolator: TrajectoryInterpolator,
    right_trajectory_interpolator: TrajectoryInterpolator,
    config: Dict
) -> float:

    cfg_auto = config['camera']['auto_correct_delay']
    fallback_delay = config['camera']['system_delay_time']
    max_delay = cfg_auto['max_expected_delay_s']


    axis_map = {'x': 0, 'y': 1, 'z': 2}
    z_axis_idx = 2
    object_axis_idx = axis_map[cfg_auto['object_axis'].lower()]

    if config['action']['type'] == 'fling' or config['action']['type'] == 'fold':
        movement_axis_idx = axis_map[cfg_auto['movement_axis'].lower()]
    elif config['action']['type'] == 'grasp':
        movement_axis_idx = axis_map['z']
    else:
        logger.error("Unsupported action type for dynamic delay calculation: {}".format(config['action']['type']))
        return fallback_delay

    logger.info("Calculating dynamic delay for action type: {}, movement_axis: {}, object_axis: {}".format(
        config['action']['type'], cfg_auto['movement_axis'], cfg_auto['object_axis']
    ))

    # 1.Check if movement is too slow
    left_average_velocity = left_trajectory_interpolator.get_velocity(current_time + max_delay/2, movement_axis_idx , dt=max_delay/2 )
    right_average_velocity = right_trajectory_interpolator.get_velocity(current_time + max_delay/2, movement_axis_idx,dt=max_delay/2 )
    if max(abs(left_average_velocity), abs(right_average_velocity)) < cfg_auto['motion_threshold_velocity']:
        # logger.debug("Movement too slow, using fallback delay, left_velocity: {:.6f}, right_velocity: {:.6f}, current_time: {:.6f}".format(
        #     left_average_velocity, right_average_velocity, current_time))
        return fallback_delay

    # 2. Extract left and right feature values from the point cloud
    points = np.asarray(current_pcd.points)
    if points.shape[0] < 10:
        logger.warning("Not enough points to calculate dynamic delay. Using fallback delay.")
        return fallback_delay

    axis_mean = np.mean(points[:, object_axis_idx])
    left_points = points[points[:, object_axis_idx] < axis_mean]
    right_points = points[points[:, object_axis_idx] > axis_mean]
    if left_points.shape[0] == 0 or right_points.shape[0] == 0:
        logger.warning("Left or right points are empty . Using fallback delay.")
        return fallback_delay

    pcd_left_feature_val = left_points[np.argmax(left_points[:, z_axis_idx])][movement_axis_idx]
    pcd_right_feature_val = right_points[np.argmax(right_points[:, z_axis_idx])][movement_axis_idx]

    # 3. search for the best matching trajectory time
    search_start_time = current_time
    search_end_time = current_time + max_delay
    search_times = np.linspace(search_start_time, search_end_time, 100)

    left_traj_vals = left_trajectory_interpolator.get_position(search_times)[:, movement_axis_idx]
    right_traj_vals = right_trajectory_interpolator.get_position(search_times)[:, movement_axis_idx]

    diffs_left = np.abs(left_traj_vals - pcd_left_feature_val)
    diffs_right = np.abs(right_traj_vals - pcd_right_feature_val)
    combined_diffs = diffs_left + diffs_right
    best_match_idx = np.argmin(combined_diffs)

    if combined_diffs[best_match_idx] > cfg_auto['feature_match_threshold_m']:
        logger.warning(f"Best match error ({combined_diffs[best_match_idx]:.3f}) exceeds threshold. Using fallback.")
        return fallback_delay

    matching_traj_time = search_times[best_match_idx]
    calculated_delay = matching_traj_time - current_time
    logger.debug(f"left_feature_val: {pcd_left_feature_val:.3f},right_feature_val: {pcd_right_feature_val:.3f}, left_traj_val: {left_traj_vals[best_match_idx]:.3f}, right_traj_val: {right_traj_vals[best_match_idx]:.3f}")
    logger.debug(f"calculated delay: {calculated_delay:.6f}")
    return calculated_delay


def detect_unstable_vertices(
        sim_vertices: np.ndarray,
        z_threshold: float = 0.0
) -> np.ndarray:
    """
        Filters out simulation vertices that are considered unstable (e.g., have fallen through the floor).

        Args:
            sim_vertices (np.ndarray): The (N, 3) array of simulated vertices.
            z_threshold (float): The Z-coordinate threshold. Vertices below this will be removed.

        Returns:
            np.ndarray: The filtered (M, 3) array of vertices, where M <= N.
        """
    if sim_vertices.size == 0:
        return sim_vertices

    good_points_mask = sim_vertices[:, 2] >= z_threshold
    num_original_points = sim_vertices.shape[0]
    num_bad_points = num_original_points - np.sum(good_points_mask)
    if num_bad_points > 0:
        logger.warning(
            f"detect {num_bad_points} unstable points (Z < {z_threshold})"
            f"Removed from the point cloud of the current frame"
        )
    # return sim_vertices[good_points_mask]
    return sim_vertices


def compute_cost(sim_vertices : np.ndarray ,
                 target_vertices: np.ndarray,
                 iteration_number: int) -> dict:
    """
        Computes various cost metrics between simulated and target point clouds.

        Args:
            sim_vertices (np.ndarray): The simulated point cloud.
            target_vertices (np.ndarray): The target (real-world) point cloud.
            iteration_number (int): The pcd item number

        Returns:
            dict: A dictionary containing computed metrics like Chamfer and Hausdorff distances.
    """
    sim_vertices_np = np.asarray(sim_vertices, dtype=np.float32)
    target = np.asarray(target_vertices, dtype=np.float32)

    # Compute the Chamfer distance — dispatcher picks PyKeOps+CUDA when
    # both are available, falls back to SciPy CPU otherwise.
    cd_cost_l1 = chamfer_distance(sim_vertices_np, target, norm=1)
    cd_cost_l1_real_to_sim = chamfer_distance(target, sim_vertices_np, norm=1)
    cd_cost_l2 = chamfer_distance(sim_vertices_np, target, norm=2)


    # Hausdorff distance
    hd_p1_to_p2 = pcu.one_sided_hausdorff_distance(
        sim_vertices_np, target, return_index=False
    )
    hd_real_to_sim = pcu.one_sided_hausdorff_distance(
        target, sim_vertices_np, return_index=False
    )

    # sim stability
    sim_stability_score = calculate_cloth_sim_stability(
        sim_vertices_np
    )

    z_mean = np.mean(sim_vertices_np[:, 2])
    z_mean_error = np.abs(z_mean - target[:, 2].mean())

    errors_dict = {
        "chamfer_l1_sim_to_real": cd_cost_l1,
        "chamfer_l2_sim_to_real": cd_cost_l2,
        "chamfer_l1_real_to_sim": cd_cost_l1_real_to_sim,
        "one_sided_hausdorff_sim_to_real": hd_p1_to_p2,
        "one_sided_hausdorff_real_to_sim": hd_real_to_sim,
        "sim_stability_score": sim_stability_score,
        "z_mean_error": z_mean_error,
    }

    return errors_dict


@hydra.main(config_path="../configs", config_name="main", version_base=None)
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
    cfg = cfg.active_run
    if cfg.action.get('type') == "fling" and cfg.sim_model.get('type') != 'fixed_point':
        logger.error("Fling action is only supported with fixed point simulation model.")
        sys.exit(1)

    # --- 2. Initialize Simulation Environment ---
    logger.info("Initializing environment: '{}'...", cfg.env.name)
    env: BaseEnvWrapper = get_env(cfg) # Somthing important, import cv2 must after pybullet env initialization (pybullet bug)
    logger.success("Environment initialized successfully.")

    output_manager = OutputManager(cfg.output, cfg.visualization)

    # --- 3. Initialize Transforms and Load Target Data ---
    master_start_time = env.get_master_start_time()
    logger.info(f"Master start time from environment: {master_start_time:.4f}")
    logger.info(f"Start Evaluation Time: {cfg.evaluate.start_calculate_time:.4f}, End Evaluation Time: {cfg.evaluate.end_calculate_time:.4f}")

    transforms_manager = TransformsManager(option=cfg)
    target_pcds_dir = cfg.data.pcd
    target_pcds_with_times = load_target_pcds(target_pcds_dir, master_start_time, transforms_manager)

    # --- 4. Time Synchronization Setup ---
    sim_prepare_time = 0.0
    if cfg.action.get('type') == 'fling':
        prepare_time = cfg.action.get('fling_prepare_time', 0.0)
        wait_time = cfg.action.get('fling_wait_time', 0.0)
        sim_prepare_time = prepare_time + wait_time
        logger.info(f"Fling Total preparation time: {sim_prepare_time:.4f}s. Comparison will begin after this time.")

    if not target_pcds_with_times:
        sys.exit("FATAL: No target point clouds found.")

    # camera system delay time
    left_trajectory_interpolator, right_trajectory_interpolator = None, None
    if cfg.camera.enable_auto_correct_delay:
        left_end_pose = load_processed_data(csv_path=cfg.data.robot_joints.left_arm_csv_path,
                                        mode='pose', start_time_offset=cfg.data.sim_start_time)
        right_end_pose = load_processed_data(csv_path=cfg.data.robot_joints.right_arm_csv_path,
                                             mode='pose', start_time_offset=cfg.data.sim_start_time)
        left_trajectory_interpolator = TrajectoryInterpolator(left_end_pose)
        right_trajectory_interpolator = TrajectoryInterpolator(right_end_pose)

    # --- 5. Run Simulation and Compute Metrics ---
    metrics_results = []

    logger.info("Starting simulation and comparison loop...")
    for i, (target_time, target_pcd) in enumerate(tqdm(target_pcds_with_times, desc="Comparing Sim to Real")):

        # ---- time synchronization, 1. camera system delay time, 2. sim prepare time ----
        # calculate camera system delay time
        sys_delay_time = cfg.camera.system_delay_time
        if cfg.camera.enable_auto_correct_delay:
            dynamic_system_delay_time = calculate_camera_dynamic_delay(
                current_pcd=target_pcd, current_time=target_time + master_start_time ,
                left_trajectory_interpolator=left_trajectory_interpolator,
                right_trajectory_interpolator=right_trajectory_interpolator,
                config=cfg
            )
            sys_delay_time = dynamic_system_delay_time
            logger.info("Dynamic system auto delay time calculated: {:.4f}s".format(sys_delay_time))
        compensated_target_time = target_time + sys_delay_time + sim_prepare_time

        # Step the simulation to the precise time of the current target point cloud.
        env.step_to_time(compensated_target_time)

        sim_vertices = env.get_sim_vertices()
        sim_vertices = detect_unstable_vertices(sim_vertices, z_threshold=cfg.env.get("sim_filter_z", 0.0))

        if sim_vertices.size == 0:
            logger.warning("Skipping frame: no valid sim vertices.")
            continue

        target_vertices = np.asarray(target_pcd.points)

        # Calculate Metrics and optionally save visualization frames
        metrics = None
        current_sim_time = env.get_current_sim_time() - sim_prepare_time
        if cfg.evaluate.start_record_time <= current_sim_time <= cfg.evaluate.end_record_time:
            metrics = compute_cost(
                sim_vertices=sim_vertices,
                target_vertices=target_vertices,
                iteration_number= i
            )

            output_manager.save_pcd(sim_vertices, 'sim', i)
            output_manager.save_pcd(target_pcd, 'target', i)
            output_manager.save_comparison_frame(sim_vertices, target_vertices, i)

            processed_metrics = {key: value.item() if hasattr(value, 'item') else value for key, value in
                                 metrics.items()}
            metrics_entry = processed_metrics.copy()
            metrics_entry['sim_time'] = current_sim_time
            metrics_entry['target_time'] = target_time
            metrics_results.append(metrics_entry)

            # --- Visualization ---
            vis_cfg = cfg.visualization

            if vis_cfg.visualize_every_n_frames > 0 and i % vis_cfg.visualize_every_n_frames == 0:
                if (vis_cfg.visualize_pcd_start_time < current_sim_time <= cfg.evaluate.end_calculate_time
                        and current_sim_time >= cfg.evaluate.start_calculate_time):
                    logger.info(f"Visualizing frame {i} at sim_time {current_sim_time:.4f}...")
                    if metrics:
                        cost_str = ", ".join([f"{key}: {value:.4f}" for key, value in metrics.items()])
                        logger.info(f"Computed Costs: [{cost_str}]")
                    visualize_pcd_comparison(sim_vertices, target_pcd)

    # --- 5. Save Results and GIF ---
    if metrics_results:
        summary_df = analyze_and_visualize_metrics(
            results_df=pd.DataFrame(metrics_results).copy(),  # Pass a copy to be safe
            output_path=cfg.output.path,
            logger=logger,
            start_time = cfg.evaluate.start_calculate_time,
            end_time = cfg.evaluate.end_calculate_time
        )
        output_manager.save_metric_results(metrics_results, summary_df)
        output_manager.create_gif()
    else:
        logger.warning("No metrics were computed.")

    logger.info("Run finished successfully.")

if __name__ == "__main__":
    main()