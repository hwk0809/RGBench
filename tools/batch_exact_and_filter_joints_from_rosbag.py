#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --- Standard Library Imports ---
import rosbag
import sys
import csv
import os
import glob
import argparse

# --- Add Project Root to Python Path ---
# This allows us to import from the 'common' module.
try:
    # Assumes this script is in 'tools/' and the project root is one level up.
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)
except NameError:
    # Fallback for environments where __file__ is not defined.
    PROJECT_ROOT = os.getcwd()

# --- Custom Module Imports ---
# from rgbench.csv_data import process_and_save_joint_data, process_and_save_end_pose_data
from rgbench.csv_data import process_and_save_combined_data
from loguru import logger


# --- STAGE 1: ROSBAG EXTRACTION LOGIC ---

def flatten_message(msg, parent_key='', sep='.'):
    """Recursively flattens a ROS message into a single-level dictionary."""
    items = {}
    try:
        slots = msg.__slots__
    except AttributeError:
        return {parent_key: msg} if parent_key else {}

    for slot in slots:
        new_key = f"{parent_key}{sep}{slot}" if parent_key else slot
        value = getattr(msg, slot)
        if hasattr(value, '__slots__'):
            items.update(flatten_message(value, new_key, sep=sep))
        elif isinstance(value, (list, tuple)):
            items[new_key] = str(value)
        else:
            items[new_key] = value
    return items


def extract_topics_to_csv(bag_path: str, joints_output_dir: str, topics_to_extract: list) -> dict:
    """
    Extracts specified topics from a bag file into CSV files.
    Returns a mapping from topic name to the path of the created CSV file.
    """
    logger.info(f"Stage 1: Starting extraction from '{os.path.basename(bag_path)}'")
    output_csv_paths = {}

    try:
        bag = rosbag.Bag(bag_path, 'r')
    except Exception as e:
        logger.error(f"Could not open bag file {bag_path}: {e}")
        return {}

    with bag:
        available_topics = bag.get_type_and_topic_info().topics.keys()
        for topic_name in topics_to_extract:
            if topic_name not in available_topics:
                logger.warning(f"Topic '{topic_name}' not found in bag. Skipping.")
                continue

            # MODIFIED: Removed the '_raw' suffix from the initial extracted file.
            safe_topic_name = topic_name.strip('/').replace('/', '_')
            output_filename = os.path.join(joints_output_dir, f"{safe_topic_name}.csv")
            output_csv_paths[topic_name] = output_filename

            try:
                with open(output_filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',')
                    is_first_message = True
                    message_count = 0
                    for _, msg, t in bag.read_messages(topics=[topic_name]):
                        message_count += 1
                        flat_msg = flatten_message(msg)
                        if is_first_message:
                            headers = ["rosbagTimestamp", "header.stamp.secs", "header.stamp.nsecs"] + sorted(
                                [k for k in flat_msg if 'header' not in k])
                            writer.writerow(headers)
                            is_first_message = False

                        flat_msg['header.stamp.secs'] = msg.header.stamp.secs
                        flat_msg['header.stamp.nsecs'] = msg.header.stamp.nsecs

                        values = [str(t)] + [str(flat_msg.get(key, '')) for key in headers[1:]]
                        writer.writerow(values)

                    if message_count > 0:
                        logger.success(f"Extracted {message_count} messages to '{output_filename}'")
            except Exception as e:
                logger.error(f"Failed to write CSV file {output_filename}: {e}")
                del output_csv_paths[topic_name]

    return output_csv_paths


# --- STAGE 2: PIPELINE ORCHESTRATION ---

def run_processing_pipeline(bag_file_path: str, output_base_dir: str, args: argparse.Namespace):
    """
    Manages the full pipeline for a single bag file: extraction and processing.
    """
    logger.info("=" * 70)
    logger.info(f"▶️  Checking Bag File: {os.path.basename(bag_file_path)}")

    joints_output_dir = os.path.join(output_base_dir, 'joints')

    # --- MODIFIED: Updated the logic to check for the correct final file names ---
    if not args.overwrite:
        final_files_to_check = []
        for topic_name in [args.left_arm_topic, args.right_arm_topic]:
            base_name = topic_name.strip('/').replace('/', '_')  # e.g., 'left_arm_joint_states'

            # Expected final processed joint file, e.g., '.../left_arm_joint_states_processed.csv'
            combined_file = os.path.join(joints_output_dir, f"{base_name}_and_end_pose.csv")
            final_files_to_check.append(combined_file)
            # processed_joint_file = os.path.join(joints_output_dir, f"{base_name}_processed.csv")
            # final_files_to_check.append(processed_joint_file)

            # Expected final end pose file, e.g., '.../left_arm_end_pose_piper.csv'
            # This logic mimics the replace() call in your csv_data_utils.py
            # end_pose_base_name = base_name.replace('_joint_states', '')
            # end_pose_file = os.path.join(joints_output_dir, f"{end_pose_base_name}_end_pose_{args.robot_name}.csv")
            # final_files_to_check.append(end_pose_file)

        if all(os.path.exists(f) for f in final_files_to_check):
            logger.success(
                f"All final processed files already exist for '{os.path.basename(bag_file_path)}'. Skipping.")
            return

    os.makedirs(joints_output_dir, exist_ok=True)
    logger.info(f"   Output will be saved in: {joints_output_dir}")

    topics = {'left': args.left_arm_topic, 'right': args.right_arm_topic}

    # MODIFIED: Look for initial CSVs without the '_raw' suffix.
    initial_csv_paths = {
        arm: os.path.join(joints_output_dir, f"{name.strip('/').replace('/', '_')}.csv")
        for arm, name in topics.items()
    }

    if not args.skip_extraction:
        extract_topics_to_csv(bag_file_path, joints_output_dir, list(topics.values()))
    else:
        logger.info("Stage 1: --skip-extraction is enabled. Bypassing rosbag reading.")

    logger.info("Stage 2: Starting processing and filtering of initial CSV data.")
    for arm_key, initial_csv_path in initial_csv_paths.items():
        if not os.path.exists(initial_csv_path):
            logger.error(
                f"Initial CSV file not found: '{initial_csv_path}'. Cannot proceed with filtering. Please run without --skip-extraction first.")
            continue

        logger.info(f"--- Processing for {arm_key} ({os.path.basename(initial_csv_path)}) ---")
        try:
            process_and_save_combined_data(
                csv_path=initial_csv_path,
                robot_name=args.robot_name,
                arm_key=arm_key,
                output_dir=joints_output_dir,
                save=True
            )
        except Exception as e:
            logger.error(f"An error occurred during the filtering stage for {initial_csv_path}: {e}")


def main(args: argparse.Namespace):
    """Finds bag files and orchestrates the processing pipeline."""
    input_path = os.path.expanduser(args.input_path)
    bag_files = []

    if os.path.isdir(input_path):
        bag_files = sorted(glob.glob(os.path.join(input_path, '*.bag')))
        if args.output_dir is None:
            args.output_dir = input_path
    elif os.path.isfile(input_path):
        if input_path.endswith('.bag'):
            bag_files.append(input_path)
            if args.output_dir is None:
                args.output_dir = os.path.dirname(input_path)
        else:
            logger.error(f"Specified input file is not a .bag file: '{input_path}'")
            return
    else:
        logger.error(f"Specified input path does not exist: '{input_path}'")
        return

    if not bag_files:
        logger.error("No .bag files found to process.")
        return

    logger.info(f"Found {len(bag_files)} .bag file(s). Root output directory: '{os.path.expanduser(args.output_dir)}'")

    for bag_file_path in bag_files:
        bag_basename = os.path.splitext(os.path.basename(bag_file_path))[0]
        specific_output_dir = os.path.join(os.path.expanduser(args.output_dir), bag_basename)
        run_processing_pipeline(bag_file_path, specific_output_dir, args)

    logger.info("=" * 70)
    logger.info("🎉 All tasks have been completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="A complete pipeline to extract and process joint state data from ROS bags.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('input_path', type=str,
                        help='Path to a folder of .bag files OR a single .bag file.')
    parser.add_argument('-o', '--output_dir', type=str, default=None,
                        help='Root output folder. Defaults to the input directory.')

    parser.add_argument('--skip-extraction', action='store_true',
                        help='If set, skips reading the rosbag and only runs filtering on existing initial CSVs.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Force reprocessing and overwrite all existing final files.')

    parser.add_argument('--robot-name', type=str, default='piper', choices=['piper', 'k1'],
                        help='Name of the robot model to use for Forward Kinematics.')
    parser.add_argument('--left-arm-topic', type=str, default='/left_arm/joint_states',
                        help='Topic name for the left arm.')
    parser.add_argument('--right-arm-topic', type=str, default='/right_arm/joint_states',
                        help='Topic name for the right arm.')

    args = parser.parse_args()
    main(args)