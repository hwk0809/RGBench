import open3d as o3d
import trimesh
import numpy as np
import copy  # Used for deep copying objects
import json  # Used for saving JSON files
import os  # Used for handling file paths


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


# --- 5. Saving Function ---
def save_transformation_matrix(transformation, output_dir, output_filename):
    """
    Saves a 4x4 transformation matrix to a JSON file.

    Args:
        transformation (np.ndarray): The 4x4 transformation matrix to save.
        output_dir (str): The target folder for the saved file.
        output_filename (str): The name of the file to save.
    """
    print(f"\n--- Saving Transformation to {os.path.join(output_dir, output_filename)} ---")

    # Ensure the output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"   Output directory '{output_dir}' ensured.")
    except OSError as e:
        print(f"Error creating directory {output_dir}: {e}")
        return  # Exit the function if creation fails

    # Construct the full file path
    output_path = os.path.join(output_dir, output_filename)

    # Save as a JSON file
    try:
        with open(output_path, 'w') as f:
            # Convert the numpy array to a python list for JSON serialization
            json.dump(transformation.tolist(), f, indent=4)
        print(f"   Successfully saved transformation.")
    except Exception as e:
        print(f"   Error saving transformation: {e}")


# --- Main Program ---
if __name__ == "__main__":
    # --- Configuration Parameters ---
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    DEFAULT_SAMPLE_ROOT = os.path.join(REPO_ROOT, "data", "sample")
    DATA_ROOT = os.environ.get("RGBENCH_DATA_ROOT", DEFAULT_SAMPLE_ROOT)
    CLOTH_MESH_ROOT = os.environ.get("RGBENCH_CLOTH_MESH_ROOT", os.path.join(DEFAULT_SAMPLE_ROOT, "cloth_meshes"))
    data_dir = os.environ.get("RGBENCH_PCD_ALIGN_DATA", os.path.join(DATA_ROOT, "realsense_data"))
    scan_file = os.environ.get(
        "RGBENCH_PCD_ALIGN_SCAN",
        os.path.join(data_dir, "segment_pcds", "scan.pcd"),
    )  # Replace with the path to your scan point cloud
    obj_file = os.environ.get(
        "RGBENCH_PCD_ALIGN_OBJ",
        os.path.join(CLOTH_MESH_ROOT, "cloth_model.obj"),
    )  # Replace with the path to your OBJ file

    # Configuration for saving the transformation matrix
    SAVE_TRANSFORMATION = True
    OUTPUT_DIR = data_dir +  "/calibration"
    OUTPUT_FILENAME = "model_to_pcd_transform.json"

    VOXEL_SIZE_REGISTRATION = 0.015
    NUM_POINTS_FROM_OBJ = None
    USE_MANUAL_LANDMARKS = True

    # --- 1. Load Data ---
    try:
        pcd_model_original, _ = load_obj_as_pointcloud(obj_file, NUM_POINTS_FROM_OBJ)
        pcd_scan_original = load_scan_pointcloud(scan_file)
    except Exception as e:
        print(f"Error during data loading: {e}")
        exit()

    # --- 2. Preprocessing ---
    pcd_model_down, fpfh_model = preprocess_point_cloud_for_registration(
        copy.deepcopy(pcd_model_original), VOXEL_SIZE_REGISTRATION, "Model"
    )
    pcd_scan_down, fpfh_scan = preprocess_point_cloud_for_registration(
        copy.deepcopy(pcd_scan_original), VOXEL_SIZE_REGISTRATION, "Scan"
    )

    # --- 3. Initial Alignment ---
    transform_global = np.identity(4)
    view_params = {
        "lookat": [0, 0, 1],
        "up": [0, -1, 0],  # Y axis up
        "front": [0, 0, -1],  # from Z axis towards the origin
        "zoom": 0.5  # Zoom level
    }
    if USE_MANUAL_LANDMARKS:
        print("\n--- Manual Landmark Selection for Initial Alignment ---")
        model_landmarks_coords, _ = pick_points_interactive(pcd_model_original,
                                                            "Pick Landmarks on MODEL (Blue) - At least 3")
        if len(model_landmarks_coords) < 3:
            print("Error: At least 3 landmarks must be picked on the model. Exiting.")
            exit()
        scan_landmarks_coords, _ = pick_points_interactive(pcd_scan_original,
                                                           f"Pick Corresponding Landmarks on SCAN (Orange) - Exactly {len(model_landmarks_coords)}",view_params=view_params)
        if len(scan_landmarks_coords) != len(model_landmarks_coords):
            print(f"Error: Number of landmarks mismatch. Exiting.")
            exit()

        source_lm_pcd = o3d.geometry.PointCloud()
        source_lm_pcd.points = o3d.utility.Vector3dVector(np.array(model_landmarks_coords))
        target_lm_pcd = o3d.geometry.PointCloud()
        target_lm_pcd.points = o3d.utility.Vector3dVector(np.array(scan_landmarks_coords))
        correspondences_lm = o3d.utility.Vector2iVector([[i, i] for i in range(len(model_landmarks_coords))])

        print(":: Calculating initial transform from landmarks...")
        estimation_method_lm = o3d.pipelines.registration.TransformationEstimationPointToPoint(False)
        transform_global = estimation_method_lm.compute_transformation(source_lm_pcd, target_lm_pcd, correspondences_lm)
        draw_registration_result(pcd_model_original, pcd_scan_original, transform_global,
                                 "Landmark-based Initial Alignment")
    else:
        print("\n--- Automatic Global Registration ---")
        global_reg_result = execute_global_registration(pcd_model_down, pcd_scan_down, fpfh_model, fpfh_scan,
                                                        VOXEL_SIZE_REGISTRATION)
        transform_global = global_reg_result.transformation

    # --- 4. ICP Refinement ---
    icp_reg_result = execute_icp_refinement(pcd_model_down, pcd_scan_down, transform_global,
                                            VOXEL_SIZE_REGISTRATION)
    transform_final_icp = icp_reg_result.transformation

    print("\nFinal Transformation Matrix (from Model to Scan):")
    print(transform_final_icp)

    # --- 5. Visualize Final Result ---
    print("\nVisualizing Final Alignment (Target: Scan, Source: Aligned Model)...")
    draw_registration_result(pcd_model_original, pcd_scan_original, transform_final_icp, "Final Alignment")

    # see the raw result
    pcd_scan_display = copy.deepcopy(pcd_scan_original)
    pcd_model_at_origin_display = copy.deepcopy(pcd_model_original)
    pcd_model_aligned_display = copy.deepcopy(pcd_model_original)
    pcd_model_aligned_display.transform(transform_final_icp)
    o3d.visualization.draw_geometries(
        [pcd_scan_display, pcd_model_at_origin_display, pcd_model_aligned_display],
        window_name="Final Alignment: Scan vs Original Model vs Aligned Model",
        width=1000, height=700
    )

    # --- 6. Save Initial Estimate Transformation ---
    if SAVE_TRANSFORMATION:
        # The script calculates T (model -> scan).
        model_to_scan_transform = transform_final_icp
        save_transformation_matrix(
            model_to_scan_transform,
            OUTPUT_DIR,
            "model_to_scan_transform.json"
        )
        sacn_to_model_transform = np.linalg.inv(model_to_scan_transform)
        save_transformation_matrix(
            sacn_to_model_transform,
            OUTPUT_DIR,
            "scan_to_model_transform.json"
        )

        # Call the new function to save the matrix

        print("   This file can now be used as '.json' for your TransformsManager.")

    print("\nRegistration process complete.")
