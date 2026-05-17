import sys
import os
import os.path as osp
import time

import numpy as np
import open3d as o3d
import cv2
from loguru import logger
import csv
from sympy import false

sys.path.append(osp.join(osp.dirname(__file__),".."))

from third_party.grounded_sam.grounded_sam import GroundedSAM
from third_party.realsense.realsense import RealSenseCamera

from rgbench.config import config_completion

from omegaconf import OmegaConf, DictConfig, open_dict, ListConfig
from typing import Union,Dict,Optional,Tuple
from easydict import EasyDict


class PointCloudProcessor:
    def __init__(self, camera: RealSenseCamera, config: Union[Dict, DictConfig, str] = None):
        """
        Initializes the processor. If a config is provided, it's used.
        Otherwise, a set of hard-coded default values are used.
        """
        self.camera = camera

        if config:
            self.option: DictConfig = config_completion(config)
        else:
            # If no config is provided, create an empty one to allow default values to be used.
            self.option: DictConfig = OmegaConf.create()

            # --- Load all processing parameters from config with clear defaults ---
        # This centralizes all parameter handling.

        # Postprocessing parameters
        post_cfg = self.option.get('postprocessing', {})
        self.visualize_postprocess = post_cfg.get('visualize_postprocess', False)  # Default to False
        self.enable_denoising = post_cfg.get('enable_denoising', False)  # Default to False
        self.enable_hole_filling = post_cfg.get('enable_hole_filling', True)  # Default to True
        self.voxel_size = post_cfg.get('voxel_size', 0.005)

        # Outlier removal parameters
        outlier_cfg = post_cfg.get('outlier_removal', {})
        self.outlier_method = outlier_cfg.get('method', 'statistical')
        # Get params for the chosen method, using its specific sub-config or an empty dict.
        self.outlier_params = outlier_cfg.get(self.outlier_method, {})

        # Hole filling parameters (now with Ball Pivoting)
        self.hole_fill_cfg = post_cfg.get('hole_filling', {})

        # Segmentation core parameters
        core_cfg = self.option.get('segment_core', {})
        self.max_mask_ratio = core_cfg.get('max_mask_ratio', 0.8)
        self.save_segmentation_images = core_cfg.get('save_segmentation_images', False)  # Combined flag
        self.segmentation_image_output_folder = core_cfg.get('segmentation_image_output_folder',
                                                             'segmentation_images')

        # Initialize Segmentation Model
        segmentation_config = self.option.get('segmentation', {})
        self.segmentation_model = GroundedSAM(**segmentation_config)

        print("PointCloudProcessor initialized with the following parameters:")
        print(f"  - Denoising Enabled: {self.enable_denoising}")
        print(f"  - Hole Filling Enabled: {self.enable_hole_filling}")
        print(f"  - Voxel Size: {self.voxel_size}")
        print(f"  - Outlier Method: {self.outlier_method}")
        print(f"  - Outlier Params: {self.outlier_params}")
        print(f"  - Hole Fill Config: {self.hole_fill_cfg}")
        print(f"  - Mask Ratio: {self.max_mask_ratio}")

    def fill_holes(self, pcd: o3d.geometry.PointCloud, visualize: bool = False) -> o3d.geometry.PointCloud:
        """
        Fills holes in a point cloud by creating a surface mesh using the Alpha Shape algorithm,
        which is robust for open, non-watertight surfaces.
        """
        try:
            print("  Attempting to create surface mesh with Alpha Shape...")

            # Determine the alpha parameter. A larger alpha bridges larger holes.
            # If not specified in config, calculate a reasonable default.
            alpha = self.hole_fill_cfg.get('alpha')
            if alpha is None:
                # Calculate average distance between points to inform alpha
                distances = pcd.compute_nearest_neighbor_distance()
                avg_dist = np.mean(distances)
                # A good starting alpha is often a multiple of the average distance.
                alpha = avg_dist * 2
                print(f"    Alpha not specified in config. Using auto-calculated alpha: {alpha:.4f}")
            else:
                print(f"    Using alpha from config: {alpha}")

            # Create the Alpha Shape mesh
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)

            # The result of an alpha shape is often a volume. We want to extract the largest connected surface.
            print("    Clustering connected components of the mesh...")
            triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
            triangle_clusters = np.asarray(triangle_clusters)
            cluster_n_triangles = np.asarray(cluster_n_triangles)

            # Select the mesh component with the largest surface area
            largest_cluster_idx = cluster_n_triangles.argmax()
            triangles_to_remove = triangle_clusters != largest_cluster_idx
            mesh.remove_triangles_by_mask(triangles_to_remove)
            mesh.remove_unreferenced_vertices()

            if visualize:
                print("  Visualizing the generated mesh...")
                o3d.visualization.draw_geometries([mesh], window_name="2a. Generated Mesh (Alpha Shape)")

            # Sample points from the final mesh.
            num_points_to_sample = len(pcd.points) * 2
            pcd_filled = mesh.sample_points_uniformly(number_of_points=num_points_to_sample)

            if pcd_filled.has_points():
                print(f"  Surface reconstruction successful. New point count: {len(pcd_filled.points)}")
                return pcd_filled
            else:
                print(
                    "  Warning: Surface reconstruction resulted in an empty point cloud. Reverting to pre-filled PCD.")
                return pcd

        except Exception as e:
            print(f"  Warning: Surface reconstruction failed with error: {e}. Reverting to pre-filled PCD.")
            return pcd

    def postprocess_pcd(self, input_pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Post-processes a point cloud using parameters initialized in the constructor.
        The new order is: Hole Filling -> Voxel Downsampling -> Denoising.
        """
        if not input_pcd.has_points():
            logger.error("Input PCD for postprocessing is empty.")
            return input_pcd

        pcd_to_process = input_pcd
        if self.visualize_postprocess:
            print("raw pcd size:", len(pcd_to_process.points))
            o3d.visualization.draw_geometries([pcd_to_process], window_name="1. After Segmentation")

        # Step 1: Hole Filling (on the original dense cloud)
        if self.enable_hole_filling:
            pcd_to_process = self.fill_holes(pcd_to_process, visualize=self.visualize_postprocess)
            if self.visualize_postprocess and pcd_to_process.has_points():
                o3d.visualization.draw_geometries([pcd_to_process], window_name="2. After Hole Filling")
        else:
            print("  Skipping hole filling.")

        # Step 2: Denoising (on the potentially filled and downsampled cloud)
        if self.enable_denoising:
            print("  Applying outlier removal (denoising)...")
            pcd_to_process, _ = self.remove_outliers(pcd_to_process, method=self.outlier_method,
                                                     **self.outlier_params)
            if self.visualize_postprocess and pcd_to_process.has_points():
                print("pcd_size decrease after outlier removal:", len(pcd_to_process.points))
                o3d.visualization.draw_geometries([pcd_to_process], window_name="4. After Denoising (Final Result)")
        else:
            print("  Skipping outlier removal (denoising).")
            if self.visualize_postprocess and pcd_to_process.has_points():
                o3d.visualization.draw_geometries([pcd_to_process],
                                                  window_name="4. Final Result (Denoising Skipped)")

        # Step 3: Voxel Downsampling (on the potentially filled cloud)
        if self.voxel_size:
            pcd_to_process = pcd_to_process.voxel_down_sample(self.voxel_size)
            if self.visualize_postprocess:
                print("pcd_size decrease after outlier removal:", len(pcd_to_process.points))
                o3d.visualization.draw_geometries([pcd_to_process], window_name="3. After Voxel Downsampling")



        return pcd_to_process

    @staticmethod
    def remove_outliers(pcd: o3d.geometry.PointCloud, method: str = "statistical", **kwargs) -> Tuple[
        o3d.geometry.PointCloud, np.ndarray]:
        """Static utility method for outlier removal."""
        method = method.lower()
        try:
            if method == "statistical":
                return pcd.remove_statistical_outlier(nb_neighbors=kwargs.get("nb_neighbors", 30),
                                                      std_ratio=kwargs.get("std_ratio", 1.0))
            elif method == "radius":
                return pcd.remove_radius_outlier(nb_points=kwargs.get("nb_points", 30),
                                                 radius=kwargs.get("radius", 0.02))
            elif method == "dbscan":
                labels = np.array(
                    pcd.cluster_dbscan(eps=kwargs.get("eps", 0.1), min_points=kwargs.get("min_points", 10)))
                if len(labels) == 0: return pcd, np.array([])
                counts = np.bincount(labels[labels >= 0])
                if len(counts) == 0: return pcd.select_by_index([], invert=True), np.array([])
                largest_cluster_id = counts.argmax()
                ind = np.where(labels == largest_cluster_id)[0]
                return pcd.select_by_index(ind), ind
            else:
                print(f"Warning: Unknown outlier removal method '{method}'. Skipping.")
                return pcd, np.arange(len(pcd.points))
        except Exception as e:
            raise RuntimeError(f"Outlier removal failed: {str(e)}")

    def _segment_core(self, rgb_img, pcd_with_color, vis=False):
        pc_xyz = np.asarray(pcd_with_color.points).copy()

        # Call the new method to get all outputs
        # The color conversion is necessary because OpenCV loads images as BGR,
        # but the segmentation model expects RGB.
        all_masks, annotated_frame, annotated_sam_image = self.segmentation_model.predict_and_get_annotated_images(
            cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        )

        # Re-implement the "select largest mask" logic
        mask_sum = all_masks.sum(axis=-1).sum(axis=-1)
        mask_sum_filter = mask_sum.copy()
        h, w = all_masks.shape[1:]
        mask_sum_filter[mask_sum_filter > h * w * self.max_mask_ratio] = 0

        # for debug
        surviving_masks_values = mask_sum_filter[mask_sum_filter > 0]
        num_survived = len(surviving_masks_values)
        if num_survived == 0: # all filter
            original_ratios = [val / (h * w) for val in mask_sum]
            logger.error(
                f"All masks were filtered out, ratios were: {[f'{r:.5f}' for r in original_ratios]}"
            )
        elif num_survived >= 2: # 2+ mask
            surviving_ratios = [val / (h * w) for val in surviving_masks_values]
            logger.warning(
                f"Found {num_survived} masks after filtering, ratios are: {[f'{r:.5f}' for r in surviving_ratios]}"
            )


        final_mask_for_projection = np.zeros((h, w), dtype=bool)
        if mask_sum_filter.any():  # Check if there are any masks left after filtering
            max_mask_idx = np.argmax(mask_sum_filter)
            final_mask_for_projection = all_masks[max_mask_idx]

        # Create a visual B&W mask image for saving
        final_mask_img_visual = np.zeros(rgb_img.shape, dtype=np.uint8)
        final_mask_img_visual[final_mask_for_projection] = [255, 255, 255]

        mask_values = self.camera.project_image_to_point_cloud(final_mask_img_visual, pc_xyz, dtype=np.uint8)
        valid_idxs = mask_values[:, 0] > 0
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(pc_xyz[valid_idxs, :])
        if pcd_with_color.has_colors():
            target_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd_with_color.colors)[valid_idxs])

        clean_pcd = self.postprocess_pcd(target_pcd)

        if vis:
            print("Visualizing the segmentation results...")
            cv2.imshow("Annotated DINO", annotated_frame)
            cv2.imshow("Annotated SAM", annotated_sam_image)
            cv2.imshow("Final Mask Image", final_mask_img_visual)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            o3d.visualization.draw_geometries([clean_pcd], window_name="Segmented Point Cloud")

        return clean_pcd, final_mask_img_visual, annotated_frame, annotated_sam_image

    def segment_single_pcd_rgb(self, rgb_filepath, pcd_filepath, vis=False):
        """Processes a single pair of files."""
        rgb_img_bgr = cv2.imread(rgb_filepath)
        if rgb_img_bgr is None: raise FileNotFoundError(f"Could not read RGB image at {rgb_filepath}")
        rgb_img_rgb = cv2.cvtColor(rgb_img_bgr, cv2.COLOR_BGR2RGB)
        pcd = o3d.io.read_point_cloud(pcd_filepath)
        if not pcd.has_points(): raise ValueError(f"Could not read point cloud at {pcd_filepath}")
        return self._segment_core(rgb_img_rgb, pcd, vis=vis)

    def segment_capture_pcd_with_rgb_mask(self, vis=False):
        """Original method for live capture, now streamlined."""
        rgb_img_bgr = self.camera.capture_rgb()
        camera_pcd = self.camera.capture_pcd()
        rgb_img_rgb = cv2.cvtColor(rgb_img_bgr, cv2.COLOR_BGR2RGB)
        return self._segment_core(rgb_img_rgb, camera_pcd, vis=vis)


def run_batch_segmentation(
        processor: PointCloudProcessor,
        extracted_data_dir: str,
        pcd_output_dir: str,
        rgb_output_dir: str,
        visualize_every_n: int = 0
):
    """
      Runs batch segmentation on synchronized data from the extractor script.

      Args:
          processor: Instance of PointCloudProcessor initialized with camera and config.
          extracted_data_dir: Directory containing 'rgb/', 'pcd/', and 'intrinsics/' folders.
          pcd_output_dir: Directory where segmented PCDs will be saved.
          rgb_output_dir: Directory where segmented images will be saved.
          visualize_every_n: If 0, no visualization, if n, every n-th item will be visualized.
      """
    pcd_dir = osp.join(extracted_data_dir, 'pcd')

    os.makedirs(pcd_output_dir, exist_ok=True)
    os.makedirs(rgb_output_dir, exist_ok=True)

    print(f"Starting batch segmentation...")
    print(f"PCDs will be saved to: {pcd_output_dir}")
    print(f"Segmentation images will be saved to: {rgb_output_dir}")
    if visualize_every_n > 0:
        print(f"Visualization is enabled for every {visualize_every_n} item(s).")

    pcd_filenames = sorted(os.listdir(pcd_dir))
    for i, pcd_filename in enumerate(pcd_filenames):
        if not pcd_filename.endswith(".pcd"): continue

        logger.info(f"\n--- Processing item {i + 1}/{len(pcd_filenames)}: {pcd_filename} ---")

        timestamp_str = pcd_filename.replace("pointcloud_", "").replace(".pcd", "")
        rgb_filename_base = f"rgb_image_{timestamp_str}"
        pcd_filepath = osp.join(pcd_dir, pcd_filename)
        rgb_filepath = osp.join(extracted_data_dir, 'rgb', f"{rgb_filename_base}.png")

        if not osp.exists(rgb_filepath): continue

        try:
            # Temporarily set the processor's visualize flag for spot-checking
            vis_segment = False
            if visualize_every_n > 0 and (i + 1) % visualize_every_n == 0:
                vis_segment = True

            pcd_result, mask_img, annotated_dino, annotated_sam = processor.segment_single_pcd_rgb(rgb_filepath,
                                                                                                   pcd_filepath,
                                                                                                   vis_segment)
            # Always save results in batch mode
            if pcd_result and pcd_result.has_points():
                pcd_save_path = osp.join(pcd_output_dir, f"{osp.splitext(pcd_filename)[0]}_segmented.pcd")
                o3d.io.write_point_cloud(pcd_save_path, pcd_result)
                print(f"--> Segmented PCD saved: {pcd_save_path}")

            if mask_img is not None:
                mask_save_path = osp.join(rgb_output_dir, f"{rgb_filename_base}_mask.png")
                cv2.imwrite(mask_save_path, mask_img)

            dino_save_path = osp.join(rgb_output_dir, f"{rgb_filename_base}_dino_annotated.jpg")
            sam_save_path = osp.join(rgb_output_dir, f"{rgb_filename_base}_sam_annotated.jpg")
            cv2.imwrite(dino_save_path, annotated_dino)
            cv2.imwrite(sam_save_path, annotated_sam)
            print(f"--> Segmentation images saved for {rgb_filename_base}")

        except Exception as e:
            print(f"ERROR processing {pcd_filename}: {e}")



# Example
if __name__ == "__main__":
    script_start_time = time.time()

    # --- User Configuration ---
    # Mode: "single" or "batch"
    MODE = "single"
    SINGLE_SAVE = False #  save flag in single mode
    OVERWRITE_IN_SINGLE = False # !!!!! this is dangerous, can overwrite data, use with caution !!!!!

    # --- Batch Mode Specific Settings ---
    # Visualize every Nth item during a batch run. Set to 0 to disable all visualization in batch mode.
    VISUALIZE_EVERY_N_IN_BATCH = 10

    # --- Path Configuration ---
    # Project root directory (adjust if your script is elsewhere)
    try:
        PROJECT_ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), ".."))
    except NameError:
        # __file__ is not defined in some interactive environments (like Jupyter)
        PROJECT_ROOT_DIR = os.getcwd()

    # Directory containing the output from the first script (e.g., realsense_data_0611)
    EXTRACTED_DATA_INPUT_DIR = os.environ.get(
        "RGBENCH_DATA_ROOT",
        osp.join(PROJECT_ROOT_DIR, "data", "sample"),
    )
    # Path to the PointCloudProcessor's configuration YAML file
    PROCESSOR_CONFIG_YAML_PATH = osp.join(PROJECT_ROOT_DIR, "config",
                                          "segment_realsense_pcd.yaml")  # <<--- !!! CHECK AND MODIFY THIS !!!

    # --- Paths for Single File Test ---
    # These paths are only used when MODE = "single"
    SINGLE_TEST_PCD_PATH = os.path.join(EXTRACTED_DATA_INPUT_DIR, "pcd", "pointcloud_1753696464.817768.pcd")  # <<--- Modify to the point cloud file you want to test
    SINGLE_TEST_RGB_PATH =  os.path.join(EXTRACTED_DATA_INPUT_DIR, "rgb", "rgb_image_1753696464.817768.png")  # <<--- Modify to the image file you want to test


    # --- Path and Configuration Validation ---
    if not osp.isdir(EXTRACTED_DATA_INPUT_DIR): sys.exit(
        f"FATAL ERROR: Input directory not found: {EXTRACTED_DATA_INPUT_DIR}")
    if not osp.exists(PROCESSOR_CONFIG_YAML_PATH): PROCESSOR_CONFIG_YAML_PATH = None

    # Define output structure based on the input directory
    pcd_output_dir = osp.join(EXTRACTED_DATA_INPUT_DIR, "segment_pcds")
    rgb_output_dir = osp.join(EXTRACTED_DATA_INPUT_DIR, "segment_rgb")
    os.makedirs(pcd_output_dir, exist_ok=True)
    os.makedirs(rgb_output_dir, exist_ok=True)

    print("-" * 50)
    print(f"Mode: {MODE}")
    print(f"Input Data Directory: {EXTRACTED_DATA_INPUT_DIR}")
    print(f"PCD Output Directory: {pcd_output_dir}")
    print(f"RGB Output Directory: {rgb_output_dir}")
    print(f"Processor Config Path: {PROCESSOR_CONFIG_YAML_PATH or 'Using defaults'}")
    print("-" * 50)

    # --- Processor Initialization ---
    intrinsics_json_path = osp.join(EXTRACTED_DATA_INPUT_DIR, "intrinsics", "camera_intrinsics.json")
    if not osp.exists(intrinsics_json_path): sys.exit(f"FATAL ERROR: 'camera_intrinsics.json' not found")
    camera = RealSenseCamera()
    if not camera.load_calibration(intrinsics_json_path): sys.exit("FATAL ERROR: Failed to load intrinsics.")
    try:
        config = OmegaConf.load(PROCESSOR_CONFIG_YAML_PATH) if PROCESSOR_CONFIG_YAML_PATH else None
        processor = PointCloudProcessor(camera, config)
    except Exception as e:
        sys.exit(f"FATAL ERROR: Could not initialize PointCloudProcessor: {e}")

    # --- Mode Execution ---
    if MODE == 'batch':
        # Batch mode is for production: always saves, visualization is for spot-checking.
        run_batch_segmentation(
            processor,
            EXTRACTED_DATA_INPUT_DIR,
            pcd_output_dir,
            rgb_output_dir,
            visualize_every_n=VISUALIZE_EVERY_N_IN_BATCH
        )
    elif MODE == 'single':
        # Single mode is for debugging: always visualizes (if enabled in config), only saves PCD.
        if not (osp.exists(SINGLE_TEST_PCD_PATH) and osp.exists(SINGLE_TEST_RGB_PATH)):
            sys.exit(f"FATAL ERROR: Single test files not found.")

        processor.visualize_postprocess = True

        # In single mode, always show the annotated images for debugging if visualization is on.
        pcd_result, mask_img, annotated_dino, annotated_sam = processor.segment_single_pcd_rgb(SINGLE_TEST_RGB_PATH,
                                                                                               SINGLE_TEST_PCD_PATH,
                                                                                               vis=True)
        #  save the PCD in single mode to check the result
        if SINGLE_SAVE and pcd_result and pcd_result.has_points():
            save_path = osp.join(pcd_output_dir,
                                 f"{osp.splitext(osp.basename(SINGLE_TEST_PCD_PATH))[0]}_SINGLE_TEST_RESULT.pcd")
            o3d.io.write_point_cloud(save_path, pcd_result)
            print(f"--> Single test PCD result saved to: {save_path}")

        if OVERWRITE_IN_SINGLE:
            print("\n'OVERWRITE_IN_SINGLE' is True. Saving results to original file locations...")

            # 1. Parse timestamp from the input filename to construct output filenames
            pcd_filename = osp.basename(SINGLE_TEST_PCD_PATH)
            timestamp_str = pcd_filename.replace("pointcloud_", "").replace(".pcd", "")
            rgb_filename_base = f"rgb_image_{timestamp_str}"

            # 2. Construct the full paths for all 4 output files
            pcd_save_path = osp.join(pcd_output_dir, f"pointcloud_{timestamp_str}_segmented.pcd")
            mask_save_path = osp.join(rgb_output_dir, f"{rgb_filename_base}_mask.png")
            dino_save_path = osp.join(rgb_output_dir, f"{rgb_filename_base}_dino_annotated.jpg")
            sam_save_path = osp.join(rgb_output_dir, f"{rgb_filename_base}_sam_annotated.jpg")

            # 3. Save all results, overwriting existing files
            if pcd_result and pcd_result.has_points():
                o3d.io.write_point_cloud(pcd_save_path, pcd_result)
                print(f"--> Overwritten PCD: {pcd_save_path}")

            if mask_img is not None:
                cv2.imwrite(mask_save_path, mask_img)
                print(f"--> Overwritten Mask: {mask_save_path}")

            if annotated_dino is not None:
                cv2.imwrite(dino_save_path, annotated_dino)
                print(f"--> Overwritten DINO Annotation: {dino_save_path}")

            if annotated_sam is not None:
                cv2.imwrite(sam_save_path, annotated_sam)
                print(f"--> Overwritten SAM Annotation: {sam_save_path}")

    print(f"\nTotal execution time: {time.time() - script_start_time:.2f} seconds.")


