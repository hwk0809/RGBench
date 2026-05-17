# -*- coding: utf-8 -*-
"""
This script provides a command-line interface to run batch segmentation on one
or multiple realsense data directories.

It can operate in two ways:
1.  If the input path is a specific data directory (containing rgb/, pcd/),
    it processes only that directory.
2.  If the input path is a parent directory, it automatically discovers and
    processes all valid data subdirectories within it.
3.  If the input path is a parent directory Data_path like : Data_path/blue_dress/data1, data2, data3
    includes a '--multi_item_color_mode' to automatically handle different items with color-based configurations.
"""
import os
import os.path as osp
import sys
import time
import argparse
from loguru import logger

# Correctly set up the system path to find the 'common' and 'third_party' modules.
# This assumes the script is run from the project's root directory.
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
from rgbench.point_cloud import PointCloudProcessor, run_batch_segmentation
from third_party.realsense.realsense import RealSenseCamera


def find_target_dirs(root_path: str) -> list:
    """
    Finds all valid data directories to process based on the root path.

    Args:
        root_path (str): The initial path provided by the user.

    Returns:
        list: A list of absolute paths to the directories that need processing.
    """
    target_dirs = []
    # Check if the root_path itself is a valid data directory
    if osp.isdir(osp.join(root_path, "pcd")) and osp.isdir(osp.join(root_path, "rgb")):
        target_dirs.append(root_path)
        print(f"Input path is a valid data directory. Processing only: {root_path}")
    else:
        # If not, scan its subdirectories
        print(f"Input path is a parent directory. Scanning for data folders in: {root_path}")
        for sub_dir in os.listdir(root_path):
            potential_dir = osp.join(root_path, sub_dir)
            if osp.isdir(potential_dir) and \
                    osp.isdir(osp.join(potential_dir, "pcd")) and \
                    osp.isdir(osp.join(potential_dir, "rgb")):
                target_dirs.append(potential_dir)
    return target_dirs


def main():
    """
    Main function to parse arguments and orchestrate the batch processing.
    """
    # command
    # 1. for single case
    # python batch_segment.py --input_path /path/to/project_root/data/parent_folder --config_path /path/to/project_root/configs/my_processor_config.yaml
    #
    # 2. for batch processing (parent folder containing multiple data sets)
    # python batch_segment.py \
    # --input_path /path/to/project_root/data/parent_folder \
    # --config_path /path/to/project_root/configs/my_processor_config.yam
    # --visualize_every_n 10

    # configure default paths !!!need to be updated if the script is moved!!!
    default_config_path = osp.join(PROJECT_ROOT_DIR, "config", "segment_realsense_pcd.yaml")

    parser = argparse.ArgumentParser(description="Batch Segmentation for Realsense Data.")
    parser.add_argument(
        "--input_path",
        required=True,
        help="Path to the data directory. Can be a parent folder containing multiple "
             "data sets or a single data set folder."
    )

    parser.add_argument(
        "--config_path",
        default=default_config_path,
        help="Path to the PointCloudProcessor's configuration YAML file."
    )
    parser.add_argument(
        "--visualize_every_n",
        type=int,
        default=0,
        help="In batch mode, visualize the result for spot-checking every N items. "
             "Set to 0 (default) to disable all visualization."
    )
    parser.add_argument(
        "--multi_item_color_mode",
        action="store_true",  # This makes it a flag.
        help="Enable multi-item color mode. Expects --input_path to be a directory "
             "containing item subfolders (e.g., 'blue_dress')."
    )
    args = parser.parse_args()

    # --- 1. Validate Paths ---
    if not osp.isdir(args.input_path):
        print(f"FATAL ERROR: Input path does not exist or is not a directory: {args.input_path}")
        sys.exit(1)
    if not osp.exists(args.config_path):
        print(f"FATAL ERROR: Processor configuration file not found: {args.config_path}")
        sys.exit(1)

    # --- 2. Initialize Logger ---
    logger.remove()
    log_dir = osp.join(PROJECT_ROOT_DIR, "log", "segment")
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"segment_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log"
    log_filepath = osp.join(log_dir, log_filename)

    # add a sink to display logs in the console
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )

    # add a sink to write logs to a file
    logger.add(
        log_filepath,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        enqueue=True,
        rotation="20 MB",
        retention="10 days"
    )

    # This will capture all print statements and redirect them to the logger
    logger.patch(lambda record: record.update(name="print_capture"))

    logger.info("Logger initialized. All output will be captured.")
    logger.info(f"Log file saved to: {log_filepath}")

    # --- 3. Initialize Processor (once) ---
    try:
        config = OmegaConf.load(args.config_path)
        # We initialize a dummy camera here. The actual intrinsics will be loaded per directory.
        camera = RealSenseCamera()
        processor = PointCloudProcessor(camera, config)
    except Exception as e:
        sys.exit(f"FATAL ERROR: Could not initialize PointCloudProcessor: {e}")

    # --- 4. Loop through directories and process ---
    total_start_time = time.time()
    # =========================================================================
    # NEW: Multi-Item Color Mode Logic
    # =========================================================================
    if args.multi_item_color_mode:
        print("✅ INFO: Multi-Item Color Mode ENABLED.")
        base_config = OmegaConf.load(args.config_path)

        item_folders = [d for d in os.listdir(args.input_path) if
                        osp.isdir(os.path.join(args.input_path, d)) and '_' in d]

        if not item_folders:
            sys.exit(f"ERROR: No item subfolders (containing '_') found in '{args.input_path}' for multi-item mode.")

        print(f"Found {len(item_folders)} item folders to process.")

        for item_name in item_folders:
            item_path = osp.join(args.input_path, item_name)
            print("\n" + "#" * 80)
            print(f"## Processing Item: {item_name}")
            print("#" * 80)

            color = item_name.split('_')[0]
            class_prompt = f"{color} cloth"
            item_config = base_config.copy()
            item_config.segmentation.classes = [class_prompt]
            print(f"Using generated class: ['{class_prompt}']")

            try:
                camera = RealSenseCamera()
                processor = PointCloudProcessor(camera, item_config)
            except Exception as e:
                print(f"ERROR: Could not initialize processor for '{item_name}': {e}. Skipping.")
                continue

            target_data_dirs = find_target_dirs(item_path)
            if not target_data_dirs:
                print(f"WARNING: No data captures (with pcd/rgb) found inside '{item_name}'. Skipping.")
                continue

            print(f"Found {len(target_data_dirs)} data capture(s) for '{item_name}'.")

            for i, data_dir in enumerate(target_data_dirs):
                print(f"\n--- Processing data capture {i + 1}/{len(target_data_dirs)}: {osp.basename(data_dir)} ---")
                dir_start_time = time.time()
                pcd_output_dir = osp.join(data_dir, "segment_pcds")
                rgb_output_dir = osp.join(data_dir, "segment_rgb")
                intrinsics_path = osp.join(data_dir, "intrinsics", "camera_intrinsics.json")
                if not osp.exists(intrinsics_path):
                    intrinsics_path = osp.join(item_path, "intrinsics", "camera_intrinsics.json")
                    if not osp.exists(intrinsics_path):
                        print(f"WARNING: No 'camera_intrinsics.json' found. Skipping capture.")
                        continue
                if not processor.camera.load_calibration(intrinsics_path):
                    print(f"WARNING: Failed to load intrinsics. Skipping capture.")
                    continue
                run_batch_segmentation(
                    processor=processor, extracted_data_dir=data_dir,
                    pcd_output_dir=pcd_output_dir, rgb_output_dir=rgb_output_dir,
                    visualize_every_n=args.visualize_every_n
                )
                print(f"--- Finished data capture in {time.time() - dir_start_time:.2f} seconds. ---")

    else:
        print("ℹ️ INFO: Running in Standard Mode (single config for all data).")

        target_data_dirs = find_target_dirs(args.input_path)
        if not target_data_dirs:
            print(
                f"No valid data directories found in {args.input_path}. A valid directory must contain 'pcd' and 'rgb' subfolders.")
            sys.exit(0)

        print(f"\nFound {len(target_data_dirs)} data director(y/ies) to process.")

        try:
            config = OmegaConf.load(args.config_path)
            camera = RealSenseCamera()
            processor = PointCloudProcessor(camera, config)
        except Exception as e:
            sys.exit(f"FATAL ERROR: Could not initialize PointCloudProcessor: {e}")

        for i, data_dir in enumerate(target_data_dirs):
            print("\n" + "=" * 70)
            print(f"Processing directory {i + 1}/{len(target_data_dirs)}: {data_dir}")
            print("=" * 70)
            dir_start_time = time.time()
            pcd_output_dir = osp.join(data_dir, "segment_pcds")
            rgb_output_dir = osp.join(data_dir, "segment_rgb")
            intrinsics_path = osp.join(data_dir, "intrinsics", "camera_intrinsics.json")
            if not osp.exists(intrinsics_path):
                print(f"WARNING: No 'camera_intrinsics.json' found. Skipping.")
                continue
            if not processor.camera.load_calibration(intrinsics_path):
                print(f"WARNING: Failed to load intrinsics. Skipping.")
                continue
            run_batch_segmentation(
                processor=processor, extracted_data_dir=data_dir,
                pcd_output_dir=pcd_output_dir, rgb_output_dir=rgb_output_dir,
                visualize_every_n=args.visualize_every_n
            )
            print(f"Finished processing directory in {time.time() - dir_start_time:.2f} seconds.")

    total_end_time = time.time()
    print("\n" + "=" * 80)
    print("Batch processing complete.")
    print(f"Total execution time: {total_end_time - total_start_time:.2f} seconds.")
    print("=" * 80)


if __name__ == '__main__':
    main()