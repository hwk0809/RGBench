import os
import os.path as osp
import open3d as o3d
from jupyter_core.version import pattern
from natsort import natsorted
import logging
import numpy as np
from loguru import logger
import json
from pathlib import Path
import seaborn as sns

# --- Basic Setup ---
# Configure logging to print informational messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_camera_to_world_transform(json_path):
    """
    Loads the world-to-camera transform from a JSON file and returns the inverse
    (camera-to-world) transform.

    Args:
        json_path (str): The full path to the 'world_to_camera_transform.json' file.

    Returns:
        np.ndarray: The 4x4 camera-to-world transformation matrix. Returns an identity
                    matrix if the file is not found.
    """
    if not osp.exists(json_path):
        logging.warning(
            f"Calibration file not found at '{json_path}'. Using identity matrix. Point clouds will not be transformed.")
        return np.eye(4)

    try:
        with open(json_path, 'r') as f:
            world_to_camera_transform = np.array(json.load(f))

        # We need the inverse to go from Camera coordinates to World coordinates
        camera_to_world_transform = np.linalg.inv(world_to_camera_transform)
        logging.info(f"Successfully loaded and inverted transform from {json_path}")
        return camera_to_world_transform
    except Exception as e:
        logging.error(f"Failed to load or process calibration file {json_path}: {e}")
        return np.eye(4)

def find_pcd_files(directory):
    """Finds all .pcd files in a directory and sorts them naturally."""
    if not osp.isdir(directory):
        logging.warning(f"Directory not found: {directory}")
        return []

    files = [osp.join(directory, f) for f in os.listdir(directory) if f.endswith('.pcd')]
    # natsorted handles filenames like 'file_1.pcd', 'file_10.pcd' correctly
    return natsorted(files)


def load_and_sort_real_pcds_by_timestamp(pcd_dir):
    """
    Loads all '_segmented.pcd' files, sorts them based on the timestamp in the
    filename, and returns the sorted list of file paths. This mimics your
    original data loading process to ensure perfect frame correspondence.
    """
    pcds_with_time = []
    if not osp.isdir(pcd_dir):
        logging.error(f"Real PCD directory not found: {pcd_dir}")
        return []

    for f in os.listdir(pcd_dir):
        if f.endswith("_segmented.pcd"):
            try:
                # Extracts the timestamp string for sorting
                timestamp_str = f.replace("pointcloud_", "").replace("_segmented.pcd", "")
                absolute_timestamp = float(timestamp_str)
                pcds_with_time.append((absolute_timestamp, osp.join(pcd_dir, f)))
            except (ValueError, IndexError):
                logging.warning(f"Could not parse timestamp from filename: {f}")
                continue

    # Sort the list based on the numeric timestamp (the first element of the tuple)
    pcds_with_time.sort(key=lambda x: x[0])

    # Return only the sorted file paths
    sorted_paths = [item[1] for item in pcds_with_time]
    logging.info(f"Found and sorted {len(sorted_paths)} real-world PCDs by timestamp.")
    return sorted_paths



def visualize_preprocessed_comparison(processed_real_pcd_dir, sim_pcd_dirs, sim_labels):
    """
    Load and visualize preprocessed real-world and simulated point clouds (color logic fixed).

    This function assumes that files in 'processed_real_pcd_dir' have already been transformed
    to the world frame and that their filenames correspond one-to-one with files in
    'sim_pcd_dirs' in sorted order.

    Args:
        processed_real_pcd_dir (str): Path to the directory of preprocessed real-world PCDs.
        sim_pcd_dirs (list[str]): List of simulation result directories.
        sim_labels (list[str]): List of simulation labels (for logging).
    """
    if len(sim_pcd_dirs) != len(sim_labels):
        logging.error("The number of simulation directories must match the number of labels. Exiting.")
        return

    # --- 1. Define colors and visual strategy ---
    BACKGROUND_COLOR = [0.1, 0.1, 0.12]
    REAL_COLOR = [0.0, 0.0, 0.0]
    # REAL_COLOR = [0.7, 0.7, 0.7]
    palette = sns.color_palette("Set1", n_colors=len(sim_pcd_dirs) * 40)
    OUR_SIM_COLOR = [0.0, 0.45, 0.7]
    OUR_SIM_COLOR = palette[1]  # bright blue for our simulator
    OTHER_SIM_COLORS = [
        [0.44, 0.5, 0.56],
        palette[9],  # other simulator: dark red
    ]


    # Label for our own simulator, used to assign its dedicated color
    OUR_SIM_LABEL = "GarmentDynamics"
    # Build color map
    color_map = {}
    other_color_idx = 0
    for label in sim_labels:
        if label == OUR_SIM_LABEL:
            color_map[label] = OUR_SIM_COLOR
        else:
            color_map[label] = OTHER_SIM_COLORS[other_color_idx % len(OTHER_SIM_COLORS)]
            other_color_idx += 1


    # --- 2. Load all file lists ---
    real_pcd_files = find_pcd_files(processed_real_pcd_dir)
    if not real_pcd_files:
        logging.error(f"No preprocessed PCD files found in '{processed_real_pcd_dir}'. Please check the path.")
        return

    sim_files_lists = [find_pcd_files(d) for d in sim_pcd_dirs]
    num_frames = len(real_pcd_files)
    logging.info(f"Found {num_frames} preprocessed real-world frames to visualize.")

    # --- 3. Visualization loop ---
    for i, real_pcd_path in enumerate(real_pcd_files):
        geometries_to_draw = []
        real_pcd_filename = osp.basename(real_pcd_path)
        logging.info(f"\n--- Loading frame {i} (real file: {real_pcd_filename}) ---")

        # Load real-world point cloud
        real_pcd = o3d.io.read_point_cloud(real_pcd_path)
        if not real_pcd.has_points():
            logging.warning(f"  - Skipping frame {i}: real-world point cloud is empty.")
            continue

        real_pcd.paint_uniform_color(REAL_COLOR)
        geometries_to_draw.append(real_pcd)

        # Load matching simulation point clouds
        for j, sim_label in enumerate(sim_labels):
            sim_files = sim_files_lists[j]
            if i < len(sim_files):
                sim_pcd_path = sim_files[i]
                sim_pcd = o3d.io.read_point_cloud(sim_pcd_path)
                if sim_pcd.has_points():
                    sim_pcd.paint_uniform_color(color_map[sim_label])
                    geometries_to_draw.append(sim_pcd)
                    logging.info(f"  - Loaded {sim_label}: {osp.basename(sim_pcd_path)}")
            else:
                logging.warning(f"  - No matching file for frame {i} in {sim_label} directory.")

        if len(geometries_to_draw) > 1:
            logging.info(f"  --> Showing frame {i}. Close window to continue.")
            o3d.visualization.draw_geometries(
                geometries_to_draw,
                window_name=f"Frame {i} - Comparing {', '.join(sim_labels)} vs. real",
                width=1280,
                height=720
            )
        else:
            logging.warning(f"  --> Skipping frame {i}: no simulation files available to compare.")

    logging.info("\nVisualization finished.")

def get_latest_run_dir(base_path):
    """
    Find the most recently modified subdirectory at `base_path`.
    Returns the latest directory name (typically a timestamp), or None if none found.
    """
    if not os.path.isdir(base_path):
        print(f"Warning: directory does not exist -> {base_path}")
        return None

    # Collect all subdirectories
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    if not subdirs:
        print(f"Warning: no subdirectories found under {base_path}.")
        return None

    # Pick the latest one by modification time
    latest_dir = max(subdirs, key=lambda d: os.path.getmtime(os.path.join(base_path, d)))
    print(f"Latest run in {base_path}: {latest_dir}")
    return latest_dir

# ==============================================================================
# --- MAIN EXECUTION BLOCK ---
# ==============================================================================
if __name__ == "__main__":
    # --- 1. EDIT THESE PATHS ---
    # Update these paths to point to your data folders.
    PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    CLOTH = "blue_dress"
    ACTION_TYPE = "fold" # grasp, fling, fold
    CONTROL_TYPE = "fixed_point" # fixed_point, robot
    SAMPLE_ID = "01"
    ROBOT_TYPE = "piper"  # piper, k1
    DATASET_ROOT = os.environ.get("RGBENCH_DATA_ROOT", os.path.join(PROJECT_ROOT_DIR, "data", "sample"))
    DATA_ROOT = os.path.join(DATASET_ROOT, "Piper_Data")

    # Path to the folder with the real-world (ground truth) point clouds.

    OUTPUT_PATH = os.path.join(PROJECT_ROOT_DIR, "outputs", CLOTH, ACTION_TYPE)

    mujoco_sample_base_dir = os.path.join(OUTPUT_PATH, "garment_dynamics", CONTROL_TYPE, ROBOT_TYPE,
                                          f"sample_{SAMPLE_ID}")
    isaac_sample_base_dir = os.path.join(OUTPUT_PATH, "isaacsim", CONTROL_TYPE, ROBOT_TYPE, f"sample_{SAMPLE_ID}")
    pybullet_sample_base_dir = os.path.join(OUTPUT_PATH, "pybullet", CONTROL_TYPE, ROBOT_TYPE, f"sample_{SAMPLE_ID}")

    latest_mujoco_run_time = get_latest_run_dir(mujoco_sample_base_dir)
    latest_isaac_run_time = get_latest_run_dir(isaac_sample_base_dir)
    latest_pybullet_run_time = get_latest_run_dir(pybullet_sample_base_dir)

    # Directory of the real-world ground-truth point clouds
    GROUND_TRUTH_DIR = os.path.join(mujoco_sample_base_dir, latest_mujoco_run_time, "target_pcd_frames")


    SIMULATION_DIRS = [
        os.path.join(mujoco_sample_base_dir, latest_mujoco_run_time, "sim_pcd_frames"),
        os.path.join(pybullet_sample_base_dir, latest_pybullet_run_time, "sim_pcd_frames"),
        # os.path.join(isaac_sample_base_dir, latest_isaac_run_time, "sim_pcd_frames"),

    ]

    # List of names for your simulations. Must match the order above.
    SIMULATION_LABELS = [
        "GarmentDynamics",
        "PyBullet",
        # "IsaacSim",
    ]

    # --- 2. RUN THE VISUALIZATION ---
    # No need to edit below this line.
    # visualize_pcd_comparison(
    #     real_pcd_dir=GROUND_TRUTH_DIR,
    #     sim_pcd_dirs=SIMULATION_DIRS,
    #     sim_labels=SIMULATION_LABELS,
    #     calibration_json_path=CALIBRATION_JSON_PATH
    # )q
    visualize_preprocessed_comparison(processed_real_pcd_dir=GROUND_TRUTH_DIR,sim_pcd_dirs=SIMULATION_DIRS, sim_labels=SIMULATION_LABELS)