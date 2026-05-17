#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script re-processes a SINGLE pair of RGB image and Point Cloud files.

It is designed to fix or re-run segmentation on a specific file that may have
failed or produced a poor result during a batch process.

The script automatically determines the location of the corresponding RGB image
and intrinsics file based on the input PCD file path. It saves the new results
to the default 'segment_pcds' and 'segment_rgb' directories, overwriting any
previous output for that specific file.
"""
import os
import os.path as osp
import sys
import time
import argparse
import cv2
import open3d as o3d

# Correctly set up the system path to find the 'common' and 'third_party' modules.
# This assumes this script is in a 'tools' directory, and the project root is one level up.
try:
    PROJECT_ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), ".."))
    if PROJECT_ROOT_DIR not in sys.path:
        sys.path.append(PROJECT_ROOT_DIR)
except NameError:
    # Handle cases where __file__ is not defined (e.g., interactive environments)
    PROJECT_ROOT_DIR = os.getcwd()
    if PROJECT_ROOT_DIR not in sys.path:
        sys.path.append(PROJECT_ROOT_DIR)

from omegaconf import OmegaConf
# It is assumed that the user's library contains these correctly defined classes/functions.
from rgbench.point_cloud import PointCloudProcessor
from third_party.realsense.realsense import RealSenseCamera


def main():
    """
    Main function to parse arguments and run single-file reprocessing.
    """
    default_config_path = osp.join(PROJECT_ROOT_DIR, "config", "segment_realsense_pcd.yaml") # ！！！need to revise ! ! !
    parser = argparse.ArgumentParser(description="Reprocess a single Point Cloud file.")
    parser.add_argument(
        "--pcd_path",
        required=True,
        help="Full path to the specific .pcd file you want to reprocess."
    )
    parser.add_argument(
        "--overwrite",
        required=True,
        help="True for overwrite, false for check."
    )
    parser.add_argument(
        "--config_path",
        default=default_config_path,
        help="Path to the PointCloudProcessor's configuration YAML file."
    )
    parser.add_argument(
        "--output_pcd_dir",
        default=None,
        help="(Optional) Override the default output directory for segmented PCDs."
    )
    parser.add_argument(
        "--output_rgb_dir",
        default=None,
        help="(Optional) Override the default output directory for segmented 2D images."
    )
    parser.add_argument(
        "--visualize",
        action="store_false",
        help="If set, enables visualization for debugging."
    )
    args = parser.parse_args()
    script_start_time = time.time()

    # --- 1. Validate and derive paths ---
    if not osp.exists(args.pcd_path):
        sys.exit(f"FATAL ERROR: Input PCD file not found: {args.pcd_path}")
    if not osp.exists(args.config_path):
        sys.exit(f"FATAL ERROR: Processor configuration file not found: {args.config_path}")

    # Auto-detect related file and directory paths
    pcd_filename = osp.basename(args.pcd_path)
    pcd_parent_dir = osp.dirname(args.pcd_path)  # e.g., .../robot_data.../pcd
    data_root_dir = osp.dirname(pcd_parent_dir)  # e.g., .../robot_data.../

    timestamp_str = pcd_filename.replace("pointcloud_", "").replace(".pcd", "")
    rgb_filename_base = f"rgb_image_{timestamp_str}"

    rgb_path = osp.join(data_root_dir, 'rgb', f"{rgb_filename_base}.png")
    intrinsics_path = osp.join(data_root_dir, 'intrinsics', 'camera_intrinsics.json')

    if not osp.exists(rgb_path):
        sys.exit(f"FATAL ERROR: Corresponding RGB file not found at: {rgb_path}")
    if not osp.exists(intrinsics_path):
        sys.exit(f"FATAL ERROR: Intrinsics file not found at: {intrinsics_path}")

    # Determine output directories
    pcd_output_dir = args.output_pcd_dir if args.output_pcd_dir else osp.join(data_root_dir, "segment_pcds")
    rgb_output_dir = args.output_rgb_dir if args.output_rgb_dir else osp.join(data_root_dir, "segment_rgb")
    os.makedirs(pcd_output_dir, exist_ok=True)
    os.makedirs(rgb_output_dir, exist_ok=True)

    print("-" * 50)
    print("Reprocessing Single File")
    print(f"  - Input PCD: {args.pcd_path}")
    print(f"  - Input RGB: {rgb_path}")
    print(f"  - Output PCD Dir: {pcd_output_dir}")
    print(f"  - Output RGB Dir: {rgb_output_dir}")
    print(f"  - Config: {args.config_path}")
    print(f"  - Visualize: {args.visualize}")
    print("-" * 50)

    # --- 2. Initialize Processor ---
    try:
        config = OmegaConf.load(args.config_path)
        camera = RealSenseCamera()
        if not camera.load_calibration(intrinsics_path):
            sys.exit("FATAL ERROR: Failed to load intrinsics.")
        processor = PointCloudProcessor(camera, config)


    except Exception as e:
        sys.exit(f"FATAL ERROR: Could not initialize PointCloudProcessor: {e}")

    # --- 3. Run Segmentation and Save ---
    try:
        pcd_result, mask_img, annotated_dino, annotated_sam = processor.segment_single_pcd_rgb(
            rgb_path, args.pcd_path, vis=args.visualize
        )

        # save all results, overwriting previous ones
        if args.overwrite == "True":
            print("Overwriting previous results...")
            if pcd_result and pcd_result.has_points():
                pcd_save_path = osp.join(pcd_output_dir, f"{osp.splitext(pcd_filename)[0]}_segmented.pcd")
                o3d.io.write_point_cloud(pcd_save_path, pcd_result)
                print(f"--> Overwrote segmented PCD: {pcd_save_path}")

            if mask_img is not None:
                mask_save_path = osp.join(rgb_output_dir, f"{rgb_filename_base}_mask.png")
                cv2.imwrite(mask_save_path, mask_img)
                dino_save_path = osp.join(rgb_output_dir, f"{rgb_filename_base}_dino_annotated.jpg")
                sam_save_path = osp.join(rgb_output_dir, f"{rgb_filename_base}_sam_annotated.jpg")
                cv2.imwrite(dino_save_path, annotated_dino)
                cv2.imwrite(sam_save_path, annotated_sam)
                print(f"--> Overwrote segmentation images in: {rgb_output_dir}")


    except Exception as e:
        print(f"ERROR during processing: {e}")

    print(f"\nReprocessing finished in {time.time() - script_start_time:.2f} seconds.")


if __name__ == '__main__':
    main()
