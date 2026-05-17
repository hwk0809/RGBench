import os
import os.path as osp
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from .visualization import create_gif_from_folder
import numpy as np
from typing import List, Dict, Union


class OutputManager:
    """
    Manages all file output operations for a run, including logging,
    saving results, and creating visualizations like PCDs and GIFs.
    """

    def __init__(self, output_cfg: DictConfig, vis_cfg: DictConfig):
        """
        Initializes the OutputManager, creates directories, and sets up logging.
        Args:
            output_cfg (DictConfig): Configuration for output paths.
            vis_cfg (DictConfig): Configuration for visualization options.
        """
        self.output_path = output_cfg.path
        self.vis_cfg = vis_cfg

        # Define all output paths
        self.paths = {
            "log": osp.join(self.output_path, "run.log"),
            "metrics_csv": osp.join(self.output_path, "metrics.csv"),
            "sim_pcd_frames": osp.join(self.output_path, "sim_pcd_frames"),
            "target_pcd_frames": osp.join(self.output_path, "target_pcd_frames"),
            "gif_frames": osp.join(self.output_path, "gif_frames"),
            "gif_output": osp.join(self.output_path, "simulation_comparison.gif")
        }

        # Create all necessary directories
        os.makedirs(self.output_path, exist_ok=True)
        if self.vis_cfg.save_sim_pcd:
            os.makedirs(self.paths['sim_pcd_frames'], exist_ok=True)
        if self.vis_cfg.save_target_pcd:
            os.makedirs(self.paths['target_pcd_frames'], exist_ok=True)
        if self.vis_cfg.save_gifs:
            os.makedirs(self.paths['gif_frames'], exist_ok=True)

        # Setup logger
        logger.add(self.paths['log'])
        logger.info("OutputManager initialized. Outputs will be saved to: {}", self.output_path)


    def save_pcd(self, vertices: Union[np.ndarray, o3d.geometry.PointCloud], frame_type: str, iteration: int):
        """
        Saves a point cloud to a .pcd file.
        Args:
            vertices (np.ndarray): The point cloud vertices (N, 3).
            frame_type (str): The type of PCD ('sim' or 'target').
            iteration (int): The current frame/iteration number.
        """
        if frame_type == 'sim' and not self.vis_cfg.save_sim_pcd:
            return
        if frame_type == 'target' and not self.vis_cfg.save_target_pcd:
            return

        if isinstance(vertices, np.ndarray):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vertices)
        elif isinstance(vertices, o3d.geometry.PointCloud):
            pcd = vertices
        else:
            raise TypeError("vertices must be either np.ndarray or o3d.geometry.PointCloud")

        dir_path = self.paths.get(f"{frame_type}_pcd_frames")
        if dir_path:
            file_path = osp.join(dir_path, f"{frame_type}_pcd_{iteration:04d}.pcd")
            o3d.io.write_point_cloud(file_path, pcd)

    def save_comparison_frame(self, sim_vertices: np.ndarray, target_vertices: np.ndarray, iteration: int):
        """
        Plots and saves a single frame comparing simulated and target point clouds.
        Args:
            sim_vertices (np.ndarray): The simulated vertices.
            target_vertices (np.ndarray): The target vertices.
            iteration (int): The current frame/iteration number.
        """
        if not self.vis_cfg.save_gifs:
            return

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        # Plot simulation points
        ax.scatter(sim_vertices[:, 0], sim_vertices[:, 1], sim_vertices[:, 2], marker="*", color="r", label="Sim")

        # Plot target points
        ax.scatter(target_vertices[:, 0], target_vertices[:, 1], target_vertices[:, 2], marker="^", color="k",
                   alpha=0.2, label="Real")

        plt.legend()
        ax.view_init(elev=self.vis_cfg.get('gif_elev', 8), azim=self.vis_cfg.get('gif_azim', 13))
        ax.set_xlim([-0.4, 0.4])
        ax.set_ylim([-0.8, 0.8])
        ax.set_zlim([0.00, 1.1])

        frame_path = osp.join(self.paths['gif_frames'], f"comparison_frame_{iteration:04d}.png")
        plt.savefig(frame_path)
        plt.close()

    def save_metric_results(self, metrics_results: List[Dict], summary_df: pd.DataFrame):
        """
        Saves the detailed and summary metrics to a CSV file.
        """
        if not metrics_results:
            logger.warning("No metrics were computed, skipping CSV saving.")
            return

        results_df = pd.DataFrame(metrics_results)
        if not summary_df.empty:
            results_df.loc['mean'] = summary_df['mean']
            results_df.loc['std_dev'] = summary_df['std_dev']

        results_df.to_csv(self.paths['metrics_csv'], index=False)
        logger.success(f"Metrics successfully saved to: {self.paths['metrics_csv']}")

    def create_gif(self):
        """
        Creates a GIF from the saved comparison frames.
        """
        if not self.vis_cfg.save_gifs:
            return
        gif_frames_dir = self.paths['gif_frames']
        if os.path.exists(gif_frames_dir) and len(os.listdir(gif_frames_dir)) > 0:
            logger.info("Creating GIF from saved frames...")
            create_gif_from_folder(
                image_directory=gif_frames_dir,
                output_gif_path=self.paths['gif_output'],
                fps=self.vis_cfg.gif_fps,
                file_pattern="comparison_frame_*.png"
            )
            logger.success("GIF successfully created at: {}", self.paths['gif_output'])
        else:
            logger.warning("GIF creation skipped: No frames were found in {}.", gif_frames_dir)