# -*- coding: utf-8 -*-
"""
Experiment Setup and Calibration Tool

=====================================================================================
This script is a pre-processing utility for robotics and simulation experiments.
Its primary goal is to establish and calculate the coordinate transformations between a
simulation world and the physical world, ensuring they are precisely aligned at the
start of an experiment.

The tool uses point cloud registration to solve for transformations in two core scenarios:

Scenario 1: "Camera Setup" (setup_camera)
---------------------------------------------
Use this mode when your camera is not precisely calibrated, but you want to align
your simulation to a known, fixed initial pose of a physical object.
- **Input**:
    1. The KNOWN initial pose of the object in your simulation (e.g., at the world origin or
       loaded from a file).
    2. A point cloud scan of the real object placed in that corresponding physical pose.
- **Output**:
    - `world_to_camera_transform.json`: The calculated transformation from the world frame
      to the camera frame.
    - `landmarks.json` (cache): The manually picked points are saved to speed up
      subsequent runs.

Scenario 2: "Object Setup" (setup_cloth)
---------------------------------------------
Use this mode when your camera IS already calibrated (e.g., via hand-eye calibration),
and you want to determine the world pose of an object that has been placed freely in
the scene.
- **Input**:
    1. An existing, accurate `world_to_camera_transform.json` file.
    2. A point cloud scan of the real object in its desired initial position.
- **Output**:
    - `initial_object_pose.json`: The calculated initial pose of the object in the world
      frame. You should use this pose to initialize the object in your simulation.
    - `landmarks.json` (cache)

Typical Workflow
---------------------------------------------
1.  Configure your desired mode (`--mode`) and options either via command-line
    arguments or in the "DEBUG/IDE-MODE SETTINGS" block at the bottom of the script.
2.  (First run) Follow the prompts to manually select corresponding feature points on the
    model and scan point clouds in the pop-up windows.
3.  The script automatically saves your picked points to `landmarks.json` for future use.
4.  The script generates the appropriate `..._transform.json` or `..._pose.json` file
    based on the selected mode.
5.  Load the generated configuration file in your main simulation program.
6.  (Subsequent runs) If using the same data, add the `--use-cache` flag (or set
    `DEBUG_USE_CACHE = True`) to skip the manual picking step for faster recalculations.

=====================================================================================
"""
import open3d as o3d
import trimesh
import numpy as np
import copy
import json
import os
import sys
import glob
import argparse
from scipy.spatial.transform import Rotation
from typing import List, Tuple
from types import SimpleNamespace

# --- 0. Utility Functions ---
# ... (save/load_transform_from_json, create_transform_from_pos_quat from previous response)
def save_transform_to_json(transform: np.ndarray, output_path: str):
    """Saves a 4x4 transformation matrix to a JSON file."""
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n--- Saving transformation to: {output_path} ---")
    try:
        with open(output_path, 'w') as f:
            json.dump(transform.tolist(), f, indent=4)
        print(f"   Successfully saved transformation.")
    except Exception as e:
        print(f"   ERROR: Failed to save transformation: {e}")

def load_transform_from_json(json_path: str) -> np.ndarray:
    """Loads a 4x4 transformation matrix from a JSON file."""
    print(f"\n--- Loading transformation from: {json_path} ---")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Calibration file not found at {json_path}")
    with open(json_path, 'r') as f:
        transform_list = json.load(f)
    transform = np.array(transform_list)
    print("   Successfully loaded transformation.")
    return transform

def create_transform_from_pos_quat(pos: list = [0, 0, 0], quat: list = [1, 0, 0, 0]) -> np.ndarray:
    """Creates a 4x4 transformation matrix from position and quaternion (w, x, y, z)."""
    transform = np.eye(4)
    transform[:3, 3] = np.array(pos)
    # Note: scipy uses (x, y, z, w) format for quaternions
    r = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
    transform[:3, :3] = r.as_matrix()
    return transform

# ---  Landmark Caching Utilities ---
def save_landmarks(model_points, scan_points, path):
    """Saves picked landmark points to a JSON file."""
    output_dir = os.path.dirname(path)
    os.makedirs(output_dir, exist_ok=True)
    data = {
        "model_points": [p.tolist() for p in model_points],
        "scan_points": [p.tolist() for p in scan_points]
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"\n--- Saved picked landmarks to: {path} ---")

def load_landmarks(path):
    """Loads picked landmark points from a JSON file."""
    if not os.path.exists(path):
        return None, None
    with open(path, 'r') as f:
        data = json.load(f)
    model_points = [np.array(p) for p in data["model_points"]]
    scan_points = [np.array(p) for p in data["scan_points"]]
    print(f"\n--- Loaded cached landmarks from: {path} ---")
    return model_points, scan_points

# --- 1. Data Loading Functions ---
def load_obj_as_pointcloud(obj_file_path, number_of_points=None):
    """
    Loads an OBJ file and converts its vertices into an Open3D PointCloud object.
    If number_of_points is specified, it will sample points from the mesh surface.
    Otherwise, it uses the mesh vertices directly.
    """
    print(f":: Loading OBJ model from: {obj_file_path}")
    mesh = trimesh.load_mesh(obj_file_path)
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) > 0:
            print(":: OBJ contains a scene, concatenating geometries...")
            mesh = trimesh.util.concatenate(
                [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
            )
            if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
                raise ValueError("OBJ file scene could not be concatenated into a valid Trimesh object.")
        else:
            raise ValueError("OBJ file contains a scene with no geometry.")

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Loaded OBJ is not a Trimesh object. Type: {type(mesh)}")

    pcd = o3d.geometry.PointCloud()
    if number_of_points is None:
        print(":: Using OBJ vertices as point cloud.")
        vertices = mesh.vertices
        pcd.points = o3d.utility.Vector3dVector(vertices)
    else:
        print(f":: Sampling {number_of_points} points from OBJ mesh surface.")
        if not mesh.is_watertight:
            print(":: Warning: Mesh is not watertight, sampling might be suboptimal. Attempting to fill holes.")
            mesh.fill_holes()
        if mesh.area == 0:
            print(":: Warning: Mesh area is zero. Using vertices instead of sampling.")
            pcd.points = o3d.utility.Vector3dVector(mesh.vertices)
        else:
            try:
                points_sampled = mesh.sample(number_of_points)
                pcd.points = o3d.utility.Vector3dVector(points_sampled)
            except Exception as e:
                print(f":: Error during mesh sampling: {e}. Falling back to using vertices.")
                pcd.points = o3d.utility.Vector3dVector(mesh.vertices)

    if not pcd.has_points():
        raise ValueError("Failed to create point cloud from OBJ model.")
    print(f":: OBJ model loaded as point cloud with {len(pcd.points)} points.")
    return pcd, mesh


def load_scan_pointcloud(pcd_file_path):
    """
    Loads the point cloud file of the real object (e.g., .ply, .pcd).
    """
    print(f":: Loading scan point cloud from: {pcd_file_path}")
    pcd = o3d.io.read_point_cloud(pcd_file_path)
    if not pcd.has_points():
        raise ValueError(f"Could not read point cloud from {pcd_file_path}")
    print(f":: Scan point cloud loaded with {len(pcd.points)} points.")
    return pcd


# --- 2. Preprocessing Functions ---
def preprocess_point_cloud_for_registration(pcd, voxel_size, pcd_name="Point Cloud"):
    """
    Preprocesses a point cloud for registration: downsampling, normal estimation, and FPFH feature computation.
    """
    print(f"\n:: Preprocessing {pcd_name} ::")
    print(f"   Input points: {len(pcd.points)}")
    pcd_down = pcd.voxel_down_sample(voxel_size)
    print(f"   Downsampled to {len(pcd_down.points)} points with voxel_size {voxel_size:.4f}")
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    print(f"   Normals estimated with radius {radius_normal:.4f}")
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    print(f"   FPFH features computed with radius {radius_feature:.4f}")
    return pcd_down, pcd_fpfh


# --- 3. Registration Functions ---
def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """
    Executes global registration (RANSAC on FPFH features).
    """
    distance_threshold_ransac = voxel_size * 4
    print("\n:: Executing Global Registration (RANSAC on FPFH) ::")
    print(f"   Distance threshold for RANSAC: {distance_threshold_ransac:.4f}")
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold_ransac,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold_ransac)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    print(f"   Global Registration Fitness: {result.fitness:.4f}")
    print(f"   Global Registration Inlier RMSE: {result.inlier_rmse:.4f}")
    return result


def execute_icp_refinement(source, target, initial_transform, voxel_size_icp):
    """
    Executes ICP (Iterative Closest Point) refinement.
    """
    distance_threshold_icp = voxel_size_icp * 2
    print("\n:: Executing ICP Refinement (Point-to-Point) ::")
    print(f"   Distance threshold for ICP: {distance_threshold_icp:.4f}")
    if not source.has_normals():
        source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size_icp * 2, max_nn=30))
    if not target.has_normals():
        target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size_icp * 2, max_nn=30))
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold_icp, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-7, relative_rmse=1e-7, max_iteration=200))
    print(f"   ICP Fitness: {result.fitness:.4f}")
    print(f"   ICP Inlier RMSE: {result.inlier_rmse:.4f}")
    return result


def pick_points_interactive(pcd, window_name="Pick points (Shift+Click), Q to finish",view_params=None):
    """
    Interactively pick points from a point cloud.
    """
    print(f"\nWindow: {window_name}")
    print("  Instructions: [Shift] + Left-Click to pick, [Q] to finish.")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=window_name, width=1000, height=800)
    vis.add_geometry(pcd)

    if view_params:
        ctr = vis.get_view_control()
        ctr.set_lookat(view_params.get("lookat", [0, 0, 0]))
        ctr.set_up(view_params.get("up", [0, -1, 0]))
        ctr.set_front(view_params.get("front", [0, 0, -1]))
        ctr.set_zoom(view_params.get("zoom", 0.5))
        vis.update_renderer()

    vis.run()
    picked_indices = vis.get_picked_points()
    vis.destroy_window()
    if not picked_indices:
        print("  No points picked.")
        return [], []
    picked_coords = [pcd.points[i] for i in picked_indices]
    print(f"  Picked {len(picked_coords)} points.")
    return picked_coords, picked_indices


# --- 4. Visualization Functions ---
def draw_registration_result(source, target, transformation, window_title="Registration Result"):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_transformed = copy.deepcopy(source)
    source_transformed.transform(transformation)
    o3d.visualization.draw_geometries([target_temp, source_transformed], window_name=window_title, width=800,
                                      height=600)


# --- 5. Registration pipeline with caching ---
def run_registration_pipeline(pcd_model_original, pcd_scan_original, cfg):
    """Encapsulates the full registration process to get T_model_to_camera."""
    pcd_model_down, _ = preprocess_point_cloud_for_registration(
        copy.deepcopy(pcd_model_original), cfg['voxel_size'], "Model"
    )
    pcd_scan_down, _ = preprocess_point_cloud_for_registration(
        copy.deepcopy(pcd_scan_original), cfg['voxel_size'], "Scan"
    )

    model_landmarks, scan_landmarks = None, None
    landmark_cache_file = os.path.join(cfg['output_dir'], 'landmarks.json')

    if cfg['use_cache']:
        model_landmarks, scan_landmarks = load_landmarks(landmark_cache_file)

    if model_landmarks is None or scan_landmarks is None:
        if cfg['use_cache']:
            print("   WARNING: Cache requested but not found. Falling back to manual picking.")
        print("\n--- Manual Landmark Selection for Initial Alignment ---")
        model_landmarks, _ = pick_points_interactive(pcd_model_original, "Pick on MODEL (Blue)")
        if len(model_landmarks) < 3:
            raise ValueError("At least 3 landmarks must be picked on the model.")

        view_params = {
            "lookat": [0, 0, 1],
            "up": [0, -1, 0],  # Y axis up
            "front": [0, 0, -1],  # from Z axis towards the origin
            "zoom": 0.5  # Zoom level
        }
        scan_landmarks, _ = pick_points_interactive(pcd_scan_original,
                                                    f"Pick on SCAN (Orange) - {len(model_landmarks)} points",
                                                    view_params=view_params)
        if len(scan_landmarks) != len(model_landmarks):
            raise ValueError("Number of landmarks must match.")

        # Save the newly picked points for future runs
        save_landmarks(model_landmarks, scan_landmarks, landmark_cache_file)

    # --- Proceed with the loaded or newly picked landmarks ---
    corresp = o3d.utility.Vector2iVector(np.array([[i, i] for i in range(len(model_landmarks))]))
    lm_pcd_model = o3d.geometry.PointCloud()
    lm_pcd_model.points = o3d.utility.Vector3dVector(np.array(model_landmarks))
    lm_pcd_scan = o3d.geometry.PointCloud()
    lm_pcd_scan.points = o3d.utility.Vector3dVector(np.array(scan_landmarks))

    est = o3d.pipelines.registration.TransformationEstimationPointToPoint(False)
    transform_initial = est.compute_transformation(lm_pcd_model, lm_pcd_scan, corresp)

    # ICP Refinement
    icp_result = execute_icp_refinement(pcd_model_down, pcd_scan_down, transform_initial, cfg['voxel_size'])

    T_model_to_camera = icp_result.transformation
    print("\nFinal Transformation Matrix (Model -> Camera/Scan):")
    print(T_model_to_camera)

    o3d.visualization.draw_geometries(
        [pcd_scan_original, copy.deepcopy(pcd_model_original).transform(T_model_to_camera),pcd_model_original],
        window_name="Final Alignment")

    return T_model_to_camera


# --- 6. Workflow-specific Logic (Unchanged) ---
def run_camera_setup(cfg: dict):
    """
    Executes 'setup_camera' mode. Generates `world_to_camera_transform.json`.
    """
    print("===== Running Mode: [setup_camera] =====")
    pcd_model, _ = load_obj_as_pointcloud(cfg['obj_file'])
    pcd_scan = load_scan_pointcloud(cfg['scan_file'])

    # Registration gives us: model -> camera
    T_model_to_camera = run_registration_pipeline(pcd_model, pcd_scan, cfg)

    # User provides the pose of the model in the world: model -> world
    print("\n--- Defining initial model pose in simulation (T_model_to_world) ---")
    T_model_to_world = None
    if cfg.get('initial_pose_matrix') is not None:
        print("   Using initial pose defined directly by the 'initial_pose_matrix' variable.")
        T_model_to_world = np.array(cfg['initial_pose_matrix'])
    elif cfg.get('initial_pose_file') is not None:
        print(f"   Loading initial pose from file: {cfg['initial_pose_file']}")
        T_model_to_world = load_transform_from_json(cfg['initial_pose_file'])
    else:
        print("   WARNING: No initial pose provided. Assuming model starts at World Origin (Identity Transform).")
        T_model_to_world = np.eye(4)

    # Calculate the transform from world to camera: T_world_to_camera
    # Formula: T_world_to_camera = T_model_to_camera @ inv(T_model_to_world)
    T_world_to_camera = T_model_to_camera @ np.linalg.inv(T_model_to_world)

    output_file = os.path.join(cfg['output_dir'], 'world_to_camera_transform.json')
    save_transform_to_json(T_world_to_camera, output_file)
    print("\n✅ Camera setup complete. The file `world_to_camera_transform.json` has been generated.")


def run_object_setup(cfg: dict):
    """
    Executes 'setup_object' mode. Generates `initial_object_pose.json`.
    """
    print("===== Running Mode: [setup_object] =====")
    pcd_model, _ = load_obj_as_pointcloud(cfg['obj_file'])
    pcd_scan = load_scan_pointcloud(cfg['scan_file'])

    # Load the known extrinsic calibration: world -> camera
    camera_calib_file = os.path.join(cfg['output_dir'], 'world_to_camera_transform.json')
    T_world_to_camera = load_transform_from_json(camera_calib_file)

    # Registration gives us: model -> camera
    T_model_to_camera = run_registration_pipeline(pcd_model, pcd_scan, cfg)

    # Calculate the desired object pose in the world: T_model_to_world
    # To get there, we chain transforms: model -> camera -> world
    # Formula: T_model_to_world = inv(T_world_to_camera) @ T_model_to_camera
    T_camera_to_world = np.linalg.inv(T_world_to_camera)
    T_model_to_world = T_camera_to_world @ T_model_to_camera

    output_file = os.path.join(cfg['output_dir'], 'initial_object_pose.json')
    save_transform_to_json(T_model_to_world, output_file)
    print("\n✅ Object setup complete. The file `initial_object_pose.json` has been generated.")


# --- 5. Main Execution Block (Unchanged) ---
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # --- Command-Line Interface (CLI) Mode ---
        print("--- Running in Command-Line Mode ---")
        parser = argparse.ArgumentParser(description="Experiment Setup and Calibration Tool.")
        parser.add_argument('--mode', type=str, required=True, choices=['setup_camera', 'setup_object'],
                            help="The operation mode to run.")
        parser.add_argument('--use-cache', action='store_true', help="Use cached landmarks to skip manual picking.")
        parser.add_argument('--initial-pose', type=str, default=None,
                            help="[For setup_camera mode] Path to a .json file for the model's initial pose in the world (T_model_to_world).")
        parser.add_argument('--scan-file', type=str, default=None,
                            help="Path to a specific scan point cloud file (.pcd). If not provided, the first PCD in 'segment_pcds' will be used.")
        args = parser.parse_args()
        args.initial_pose_matrix = None
    else:
        # --- IDE / Debug Mode ---
        print("--- Running in IDE/Debug Mode ---")

        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        # --- EDIT DEBUG PARAMETERS HERE ---
        DEBUG_MODE = 'setup_camera'
        DEBUG_USE_CACHE = False

        # pcd file
        DEBUG_SCAN_FILE = None

        # This defines T_model_to_world (the pose of the model in the world frame)
        # It is ONLY used in 'setup_camera' mode.
        # Method 1: Define pose directly as a matrix. Takes priority if not None.
        DEBUG_INITIAL_POSE_MATRIX = [
            [1, 0, 0, 0.1],  # This represents T_model_to_world
            [0, 1, 0, 0.0],  # i.e., the model is at x=0.1 in the world.
            [0, 0, 1, 0.0],
            [0, 0, 0, 1.0]
        ]

        # Method 2: Define pose via file. Used only if MATRIX is None.
        DEBUG_INITIAL_POSE_FILE = None
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        args = SimpleNamespace(
            mode=DEBUG_MODE,
            use_cache=DEBUG_USE_CACHE,
            initial_pose=DEBUG_INITIAL_POSE_FILE,
            initial_pose_matrix=DEBUG_INITIAL_POSE_MATRIX,
            scan_file=DEBUG_SCAN_FILE,
        )
        print(f"   Mode: {args.mode}, Use Cache: {args.use_cache}")
        if args.initial_pose_matrix is not None:
            print("   Initial Pose (T_model_to_world): Defined directly in script.")
        elif args.initial_pose is not None:
            print(f"   Initial Pose File (T_model_to_world): {args.initial_pose}")

    # --- Central Configuration Dictionary ---
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    DEFAULT_SAMPLE_ROOT = os.path.join(REPO_ROOT, "data", "sample")
    DATA_ROOT = os.environ.get("RGBENCH_DATA_ROOT", DEFAULT_SAMPLE_ROOT)
    CLOTH_MESH_ROOT = os.environ.get("RGBENCH_CLOTH_MESH_ROOT", os.path.join(DEFAULT_SAMPLE_ROOT, "cloth_meshes"))
    config = {
        "data_dir": os.environ.get("RGBENCH_PIPER_DATA", os.path.join(DATA_ROOT, "Piper_Data")),
        "obj_file": os.environ.get(
            "RGBENCH_CLOTH_MESH",
            os.path.join(CLOTH_MESH_ROOT, "LargeCoat_Flat_Simple_33k.obj"),
        ),
        "voxel_size": 0.015,
        "use_cache": args.use_cache,
        "initial_pose_file": args.initial_pose,
        "initial_pose_matrix": args.initial_pose_matrix
    }

    scan_file_path = None
    if args.scan_file:
        # 1. Use the file specified by the user
        scan_file_path = args.scan_file if os.path.isabs(args.scan_file) else os.path.join(config["data_dir"],
                                                                                           args.scan_file)
        print(f"--- Using user-specified scan file: {scan_file_path} ---")
    else:
        # 2. If not specified, search automatically
        print("--- No scan file specified, searching automatically in 'segment_pcds'... ---")
        scan_dir = os.path.join(config["data_dir"], "segment_pcds")
        if not os.path.isdir(scan_dir):
            raise FileNotFoundError(
                f"Automatic search failed: 'segment_pcds' directory not found in {config['data_dir']}")

        pcd_files = sorted(glob.glob(os.path.join(scan_dir, '*.pcd')))
        if not pcd_files:
            raise FileNotFoundError(f"Automatic search failed: No .pcd files found in {scan_dir}")

        scan_file_path = pcd_files[0]  # Select the first file after sorting
        print(f"   --> Automatically selected: {os.path.basename(scan_file_path)}")

    config["scan_file"] = scan_file_path
    config["output_dir"] = os.path.join(config["data_dir"], "calibration")

    # --- Run the selected mode ---
    if args.mode == 'setup_camera':
        run_camera_setup(config)
    elif args.mode == 'setup_object':
        run_object_setup(config)