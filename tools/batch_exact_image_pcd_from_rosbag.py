#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rosbag
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import os
import json
import math
import struct
import csv
import collections
import argparse
import glob


# --- Helper Function: Save PCD file (no changes needed) ---
def save_pcd_file(filepath, points_list, is_xyzrgb=False):
    # (This function remains unchanged)
    valid_points_lines = []
    final_num_points = 0
    for point_data in points_list:
        x, y, z = point_data[0], point_data[1], point_data[2]
        if math.isnan(x) or math.isinf(x) or \
                math.isnan(y) or math.isinf(y) or \
                math.isnan(z) or math.isinf(z):
            continue
        if is_xyzrgb:
            rgb_float = point_data[3]
            rgb_int_to_write = 0
            if not (math.isnan(rgb_float) or math.isinf(rgb_float)):
                try:
                    rgb_int_to_write = struct.unpack('I', struct.pack('f', rgb_float))[0]
                except struct.error:
                    print(f"Warning: Could not convert RGB float {rgb_float}. Defaulting to black.")
                    rgb_int_to_write = 0
            valid_points_lines.append(f"{x:.6f} {y:.6f} {z:.6f} {rgb_int_to_write}\n")
        else:
            valid_points_lines.append(f"{x:.6f} {y:.6f} {z:.6f}\n")
        final_num_points += 1

    header_fields = "FIELDS x y z"
    header_type = "TYPE F F F"
    header_size = "SIZE 4 4 4"
    header_count = "COUNT 1 1 1"
    if is_xyzrgb:
        header_fields += " rgb"
        header_type += " U"
        header_size += " 4"
        header_count += " 1"

    header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
{header_fields}
{header_size}
{header_type}
{header_count}
WIDTH {final_num_points}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {final_num_points}
DATA ascii
"""
    with open(filepath, 'w') as f:
        f.write(header)
        if final_num_points > 0:
            f.writelines(valid_points_lines)

    if final_num_points < len(points_list):
        print(
            f"Info: Filtered out {len(points_list) - final_num_points} invalid points from {os.path.basename(filepath)}.")


# --- Core Processing Function: Process a single bag file ---
def process_bag_file(bag_file_path, output_base_dir, config):
    # (This function remains unchanged)
    print("=" * 80)
    print(f"▶️  Processing Bag File: {os.path.basename(bag_file_path)}")
    print(f"   Output will be saved to: {output_base_dir}")

    # 1. Create output directories
    intrinsics_dir = os.path.join(output_base_dir, 'intrinsics')
    pcd_dir = os.path.join(output_base_dir, 'pcd')
    rgb_dir = os.path.join(output_base_dir, 'rgb')
    os.makedirs(intrinsics_dir, exist_ok=True)
    os.makedirs(pcd_dir, exist_ok=True)
    os.makedirs(rgb_dir, exist_ok=True)

    # 2. Prepare data buffers and CSV files
    pcd_queue = collections.deque()
    rgb_queue = collections.deque()
    pcd_timestamps_file = os.path.join(output_base_dir, 'pcd_timestamps.csv')
    rgb_timestamps_file = os.path.join(output_base_dir, 'rgb_timestamps.csv')

    try:
        bag = rosbag.Bag(bag_file_path, 'r')
    except Exception as e:
        print(f"❌ Error: An error occurred while opening the rosbag file: {e}")
        return

    # --- Phase 1: Reading messages into buffers ---
    print("   Phase 1: Reading all point cloud and image messages into buffers...")
    intrinsics_saved = False
    topics_to_read = [config.info_topic, config.pcd_topic, config.rgb_topic]
    with bag:
        for topic, msg, t_ros in bag.read_messages(topics=topics_to_read):
            if not hasattr(msg, 'header') or not hasattr(msg.header, 'stamp'):
                continue

            current_msg_time_sec = msg.header.stamp.to_sec()

            if topic == config.info_topic and not intrinsics_saved:
                intrinsics_data = {'timestamp_sec': current_msg_time_sec, 'height': msg.height, 'width': msg.width,
                                   'distortion_model': msg.distortion_model, 'D': list(msg.D), 'K': list(msg.K),
                                   'R': list(msg.R), 'P': list(msg.P)}
                intrinsics_file = os.path.join(intrinsics_dir, f"camera_intrinsics.json")
                with open(intrinsics_file, 'w') as f:
                    json.dump(intrinsics_data, f, indent=4)
                print(f"   Camera intrinsics saved to: {intrinsics_file}")
                intrinsics_saved = True
            elif topic == config.pcd_topic:
                pcd_queue.append(msg)
            elif topic == config.rgb_topic:
                rgb_queue.append(msg)

    print(f"   Reading complete. Buffers contain {len(pcd_queue)} point clouds and {len(rgb_queue)} RGB images.")
    print("   Phase 2: Synchronizing and saving paired data...")

    # --- Phase 2: Synchronize and save data ---
    bridge = CvBridge()
    last_save_time = -float('inf')
    saved_pairs_count = 0
    with open(pcd_timestamps_file, 'w', newline='') as pcd_csv, open(rgb_timestamps_file, 'w', newline='') as rgb_csv:
        pcd_writer = csv.writer(pcd_csv)
        rgb_writer = csv.writer(rgb_csv)
        pcd_writer.writerow(['timestamp_sec', 'filename'])
        rgb_writer.writerow(['timestamp_sec', 'filename'])

        while pcd_queue and rgb_queue:
            pcd_time = pcd_queue[0].header.stamp.to_sec()
            rgb_time = rgb_queue[0].header.stamp.to_sec()
            time_diff = abs(pcd_time - rgb_time)

            if time_diff <= config.tolerance:
                pair_time = pcd_time
                if pair_time >= last_save_time + config.interval:
                    pcd_msg_to_save = pcd_queue.popleft()
                    rgb_msg_to_save = rgb_queue.popleft()
                    timestamp_str = f"{pair_time:.6f}"

                    try:
                        has_rgb = any(field.name == 'rgb' for field in pcd_msg_to_save.fields)
                        field_names = ("x", "y", "z", "rgb") if has_rgb else ("x", "y", "z")
                        points_list = list(pc2.read_points(pcd_msg_to_save, field_names=field_names, skip_nans=True))
                        if points_list:
                            pcd_filename = f"pointcloud_{timestamp_str}.pcd"
                            pcd_filepath = os.path.join(pcd_dir, pcd_filename)
                            save_pcd_file(pcd_filepath, points_list, is_xyzrgb=has_rgb)
                            pcd_writer.writerow([timestamp_str, pcd_filename])
                        else:
                            continue
                    except Exception as e:
                        print(f"   ❌ Error: Error processing point cloud at timestamp {pair_time:.6f}: {e}")
                        continue

                    try:
                        cv_image = bridge.imgmsg_to_cv2(rgb_msg_to_save, desired_encoding='bgr8')
                        rgb_filename = f"rgb_image_{timestamp_str}.png"
                        rgb_filepath = os.path.join(rgb_dir, rgb_filename)
                        cv2.imwrite(rgb_filepath, cv_image)
                        rgb_writer.writerow([timestamp_str, rgb_filename])
                    except Exception as e:
                        print(f"   ❌ Error: Error processing RGB image at timestamp {pair_time:.6f}: {e}")
                        continue
                    last_save_time = pair_time
                    saved_pairs_count += 1
                    if saved_pairs_count % 20 == 0:
                        print(f"   ...Successfully saved {saved_pairs_count} pairs...")
                else:
                    pcd_queue.popleft()
                    rgb_queue.popleft()
            elif pcd_time < rgb_time:
                pcd_queue.popleft()
            else:
                rgb_queue.popleft()

    print("-" * 50)
    print(f"✅ Processing complete! A total of {saved_pairs_count} synchronized pairs were saved.")


# --- Main Function: Find files and start batch processing ---
def main(args):
    """
    Main function, finds all bag files and starts the processing workflow for each.
    Now supports both a single file and a directory as input.
    """
    input_path = args.input_path
    output_dir = args.output_dir

    # --- MODIFICATION START: Set default output directory to be same as input ---
    if output_dir is None:
        if os.path.isdir(input_path):
            output_dir = input_path
        elif os.path.isfile(input_path):
            output_dir = os.path.dirname(os.path.abspath(input_path))
        # Handle edge case where dirname is empty
        if not output_dir:
            output_dir = '.'
    # --- MODIFICATION END ---

    bag_files = []
    # --- Check if input is a directory or a file (original logic) ---
    if os.path.isdir(input_path):
        print(f"Input is a directory. Searching for .bag files in '{input_path}'...")
        bag_files = glob.glob(os.path.join(input_path, '*.bag'))
        bag_files.sort()
    elif os.path.isfile(input_path):
        if input_path.endswith('.bag'):
            print(f"Input is a single file: '{input_path}'")
            bag_files.append(input_path)
        else:
            print(f"Error: The specified input file is not a .bag file: '{input_path}'")
            return
    else:
        print(f"Error: The specified input path does not exist: '{input_path}'")
        return

    if not bag_files:
        print(f"Error: No .bag files to process.")
        return

    print(f"Found {len(bag_files)} .bag file(s) to process. Starting...")

    for bag_file_path in bag_files:
        bag_basename = os.path.splitext(os.path.basename(bag_file_path))[0]
        # Use the determined output_dir as the base for each specific output
        specific_output_dir = os.path.join(output_dir, bag_basename)
        process_bag_file(bag_file_path, specific_output_dir, args)

    print("=" * 80)
    print("🎉 All tasks have been completed!")


if __name__ == '__main__':
    # --- Configure Command-Line Arguments ---
    parser = argparse.ArgumentParser(
        description="Batch process ROS bag files from a folder or a single file to extract synchronized point clouds and RGB images.")

    parser.add_argument('input_path', type=str,
                        help='Path to the input folder containing .bag files, OR path to a single .bag file.')

    # --- MODIFICATION START: Change default to None and update help text ---
    parser.add_argument('-o', '--output_dir', type=str, default=None,
                        help='Root output folder. Defaults to the input directory if not specified.')
    # --- MODIFICATION END ---

    parser.add_argument('--interval', type=float, default=0.2,
                        help='Time interval in seconds to save data pairs. Default: 0.2')
    parser.add_argument('--tolerance', type=float, default=0.05,
                        help='Timestamp synchronization tolerance in seconds between point cloud and image. Default: 0.05')

    # ROS Topic Name Arguments
    parser.add_argument('--info_topic', type=str, default='/camera/color/camera_info',
                        help='Camera info topic name.')
    parser.add_argument('--pcd_topic', type=str, default='/camera/depth/color/points',
                        help='Point cloud topic name.')
    parser.add_argument('--rgb_topic', type=str, default='/camera/color/image_rect_color',
                        help='RGB image topic name.')

    args = parser.parse_args()
    main(args)