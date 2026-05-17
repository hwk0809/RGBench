import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import pandas as pd
import os
import os.path as osp
import glob
import re
from loguru import  logger
from typing import Any, Optional
import imageio
from tqdm import tqdm
import time




def natural_sort_key(s):
    """Helper for natural sorting."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def visualize_pcds_in_directory(directory: str, sort_order: str = "natural", autoplay: bool = False,
                                delay: float = 0.05, reverse: bool = False, loop: bool = False):
    """
    Visualizes a sequence of .pcd files from a directory.

    Args:
        directory (str): Path to the directory containing .pcd files.
        sort_order (str): Sorting method ('name', 'time', 'natural').
        autoplay (bool): Enables autoplay mode.
        delay (float): Delay between frames in autoplay.
        reverse (bool): Reverses the playback order.
        loop (bool): Enables looping in autoplay.
    """
    view_params = {
        "lookat": [0, 0, 1],
        "up": [0, -1, 0],  # Y axis up
        "front": [0, 0, -1],  # from Z axis towards the origin
        "zoom": 0.5  # Zoom level
    }
    pcd_files = sorted(
        [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pcd")],
        key=lambda f: os.path.getmtime(f) if sort_order == "time" else (
            natural_sort_key(f) if sort_order == "natural" else f)
    )

    if reverse:
        pcd_files.reverse()

    if not pcd_files:
        print(f"⚠️ Warning: No .pcd files found in {directory}.")
        return

    pcd = o3d.io.read_point_cloud(pcd_files[0])

    vis = o3d.visualization.VisualizerWithKeyCallback()

    class PlayerState:
        def __init__(self, files):
            self.files = files
            self.count = len(files)
            self.idx = 0
            self.is_paused = not autoplay

    state = PlayerState(pcd_files)

    def update_geometry(new_idx):
        """Loads a new point cloud and updates the visualizer."""
        state.idx = new_idx
        # --- FIX: Print frame info to the console instead of updating window title ---
        print(f"\rDisplaying: [{state.idx + 1}/{state.count}] {os.path.basename(state.files[state.idx])}", end="")

        new_pcd = o3d.io.read_point_cloud(state.files[state.idx])
        pcd.points = new_pcd.points
        pcd.colors = new_pcd.colors
        vis.update_geometry(pcd)

    def toggle_pause(v):
        if state.idx == state.count - 1 and not loop and state.is_paused:
            print("\n▶️ End of sequence. Restarting from the beginning.")
            update_geometry(0)
        state.is_paused = not state.is_paused
        # Add a newline to avoid overwriting the frame info line
        print("\n|| Paused" if state.is_paused else "▶️ Resumed")

    def advance_frame(v):
        state.is_paused = True
        next_idx = min(state.idx + 1, state.count - 1)
        if next_idx != state.idx:
            update_geometry(next_idx)

    def previous_frame(v):
        state.is_paused = True
        prev_idx = max(state.idx - 1, 0)
        if prev_idx != state.idx:
            update_geometry(prev_idx)

    vis.register_key_callback(ord(" "), toggle_pause)
    vis.register_key_callback(ord("N"), advance_frame)
    vis.register_key_callback(ord("B"), previous_frame)

    vis.create_window(window_name="PCD Sequence Viewer")
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0]))
    vis.add_geometry(pcd)

    view_control = vis.get_view_control()
    view_control.set_front(view_params["front"])
    view_control.set_lookat(view_params["lookat"])
    view_control.set_up(view_params["up"])
    view_control.set_zoom(view_params["zoom"])

    update_geometry(0)  # Display the first frame's info at the start

    while True:
        if not state.is_paused:
            if loop:
                next_idx = (state.idx + 1) % state.count
                update_geometry(next_idx)
            else:
                if state.idx < state.count - 1:
                    update_geometry(state.idx + 1)
                else:
                    state.is_paused = True
                    print("\n⏹️ Playback finished.")

            time.sleep(delay)

        if not vis.poll_events():
            break
        vis.update_renderer()

    print("\nVisualizer closed.")
    vis.destroy_window()

def visualize_pcd_comparison(sim_vertices: np.ndarray, target_pcd_with_color: o3d.geometry.PointCloud):
    """
    Visualizes the simulation and target point clouds for comparison.
    """
    sim_pcd = o3d.geometry.PointCloud()
    sim_pcd.points = o3d.utility.Vector3dVector(sim_vertices)
    sim_pcd.paint_uniform_color([0, 0.651, 0.929])  # Blue

    # target_pcd = o3d.geometry.PointCloud()
    # target_pcd.points = o3d.utility.Vector3dVector(target_pcd_with_color)
    # target_pcd.paint_uniform_color([1, 0.706, 0])  # Orange

    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries(
        [sim_pcd, target_pcd_with_color, world_frame],
        window_name="Sim vs. Target Comparison"
    )

def plot_sim_and_target(sim_cloud, target_cloud, it_nr, output_path, elev=8, azim=13):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    sim_cloud_npy = sim_cloud.numpy()[0]

    ax.scatter(
        sim_cloud_npy[:, 0],
        sim_cloud_npy[:, 1],
        sim_cloud_npy[:, 2],
        marker="*",
        color="r",
        label="Sim",
    )
    if target_cloud is not None:
        target_vertices = target_cloud.numpy()[0]
        ax.scatter(
            target_vertices[:, 0],
            target_vertices[:, 1],
            target_vertices[:, 2],
            marker="^",
            color="k",
            alpha=0.1,
            label="Real",
        )

    plt.legend()
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim([-0.4, 0.4])
    ax.set_ylim([-0.8, 0.8])
    ax.set_zlim([0.00, 1.1])

    if it_nr < 10:
        it_str = f"00{it_nr}"
    elif it_nr < 100:
        it_str = f"0{it_nr}"
    else:
        it_str = f"{it_nr}"

    plt.savefig(f"{output_path}/target_and_source_elev_{elev}_azim_{azim}_{it_str}.png")
    plt.close()

def create_gif_from_folder(image_directory: str, output_gif_path: str, fps: int = 30, file_pattern: str = "*.png"):
    logger.info(f"Searching for images in '{image_directory}' with pattern '{file_pattern}'...")

    # 1. Find every matching image file
    search_path = osp.join(image_directory, file_pattern)
    image_files = glob.glob(search_path)

    if not image_files:
        logger.error(f"No images found in '{image_directory}'. Cannot create GIF.")
        return

    try:
        image_files.sort(key=lambda f: int(re.search(r'(\d+)', f).group()))
        logger.info(f"Found and sorted {len(image_files)} image frames.")
    except (AttributeError, TypeError):
        image_files.sort()
        logger.warning("Could not sort files by number, using alphabetical sort instead.")

    logger.info(f"Creating GIF at '{output_gif_path}' with {fps} FPS...")
    try:
        with imageio.get_writer(output_gif_path, mode='I', fps=fps) as writer:
            for filename in tqdm(image_files, desc="Building GIF"):
                image = imageio.imread(filename)
                writer.append_data(image)
        logger.success(f"GIF successfully created: {output_gif_path}")
    except Exception as e:
        logger.error(f"Failed to create GIF. Error: {e}")


def analyze_and_visualize_metrics(
        results_df: pd.DataFrame,
        output_path: str,
        logger: Any,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
) -> pd.DataFrame:
    """
    Analyzes a metrics DataFrame within a specific time window, prints/visualizes
    stats, and returns the summary.

    Args:
        results_df (pd.DataFrame): DataFrame containing per-frame metrics data.
        output_path (str): The folder path to save the visualization chart.
        logger (Any Logger): The logger instance for printing information.
        start_time (Optional[float]): The absolute start time for the evaluation.
                                      If None, uses the beginning of the data.
        end_time (Optional[float]): The absolute end time for the evaluation.
                                    If None, uses the end of the data.

    Returns:
        pd.DataFrame: A DataFrame containing the mean and standard deviation for each metric.
                      Returns an empty DataFrame if analysis fails.
    """
    if results_df.empty:
        logger.warning("Metrics DataFrame is empty. Skipping analysis and visualization.")
        return pd.DataFrame()

    # --- NEW: Filter DataFrame based on the time window ---
    eval_df = results_df.copy()  # Work on a copy to avoid modifying the original data

    # Build a log message for the filter condition
    time_window_msg = "all time"
    if start_time is not None and end_time is not None:
        time_window_msg = f"sim_time window [{start_time}, {end_time}]"
        eval_df = eval_df[(eval_df['sim_time'] >= start_time) & (eval_df['sim_time'] <= end_time)]
    elif start_time is not None:
        time_window_msg = f"sim_time >= {start_time}"
        eval_df = eval_df[eval_df['sim_time'] >= start_time]
    elif end_time is not None:
        time_window_msg = f"sim_time <= {end_time}"
        eval_df = eval_df[eval_df['sim_time'] <= end_time]

    if eval_df.empty:
        logger.warning(f"No data available in the specified time window {time_window_msg}. Skipping analysis.")
        return pd.DataFrame()

    # --- 1. Calculate Statistics ---
    logger.info(f"Analyzing metrics for mean and standard deviation ({time_window_msg})...")
    # Note: All subsequent operations use the filtered 'eval_df'
    metric_columns = [col for col in eval_df.columns if col not in ['sim_time', 'target_time']]

    if not metric_columns:
        logger.warning("No metric columns found to analyze. Skipping.")
        return pd.DataFrame()

    mean_values = eval_df[metric_columns].mean()
    std_values = eval_df[metric_columns].std()

    summary_df = pd.DataFrame({'mean': mean_values, 'std_dev': std_values})

    # --- 2. Print the Summary ---
    logger.info(f"--- Metric Summary Analysis ({time_window_msg}) ---")
    logger.info("\n" + summary_df.to_string())
    logger.info("-------------------------------------------")

    # --- 3. Visualize the Results ---
    logger.info("Generating summary visualization...")
    # (The visualization part should also use 'eval_df')

    # --- 4. Return the summary statistics ---
    return summary_df