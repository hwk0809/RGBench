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
import collections  # Import collections for the efficient deque

# --- Configuration Parameters ---
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DEFAULT_SAMPLE_ROOT = os.path.join(REPO_ROOT, "data", "sample")
BAG_FILE_PATH = os.environ.get(
    "RGBENCH_ROSBAG_PATH",
    os.path.join(DEFAULT_SAMPLE_ROOT, "rosbag", "robot_data.bag"),
)  # <--- Path to your rosbag file
OUTPUT_BASE_DIR = os.environ.get(
    "RGBENCH_ROSBAG_OUTPUT_DIR",
    os.path.join(DEFAULT_SAMPLE_ROOT, "realsense_data"),
)  # <--- Name of the output directory

# Interval in seconds to save the paired point cloud and RGB image
SAVE_INTERVAL_SECONDS = 0.2

# Timestamp synchronization tolerance (in seconds). A pair of point cloud and image messages
# is considered a match if their timestamp difference is within this value.
# Typical values range from 0.02 to 0.05, depending on the sensor's data publishing synchronization.
TIME_SYNC_TOLERANCE = 0.05

# ROS Topic Names (adjust according to your rosbag)
CAMERA_INFO_TOPIC = '/camera/color/camera_info'
POINT_CLOUD_TOPIC = '/camera/depth_registered/points'
RGB_IMAGE_TOPIC = '/camera/color/image_rect_color'


# --- Helper Function: Save PCD file (same as your original code) ---
def save_pcd_file(filepath, points_list, is_xyzrgb=False):
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
            f"Info: Filtered out {len(points_list) - final_num_points} invalid points from {os.path.basename(filepath)}")


def main():
    # 1. Create output directories
    intrinsics_dir = os.path.join(OUTPUT_BASE_DIR, 'intrinsics')
    pcd_dir = os.path.join(OUTPUT_BASE_DIR, 'pcd')
    rgb_dir = os.path.join(OUTPUT_BASE_DIR, 'rgb')

    os.makedirs(intrinsics_dir, exist_ok=True)
    os.makedirs(pcd_dir, exist_ok=True)
    os.makedirs(rgb_dir, exist_ok=True)

    # 2. Prepare data buffers and CSV files
    pcd_queue = collections.deque()
    rgb_queue = collections.deque()

    pcd_timestamps_file = os.path.join(OUTPUT_BASE_DIR, 'pcd_timestamps.csv')
    rgb_timestamps_file = os.path.join(OUTPUT_BASE_DIR, 'rgb_timestamps.csv')

    try:
        bag = rosbag.Bag(BAG_FILE_PATH, 'r')
    except FileNotFoundError:
        print(f"Error: Bag file not found at {BAG_FILE_PATH}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while opening the rosbag: {e}")
        return

    # --- Phase 1: Iterate through the Rosbag, read messages into queues, and save camera intrinsics ---
    print(f"Opening rosbag: {BAG_FILE_PATH}")
    print("Phase 1: Reading all point cloud and image messages into buffers...")

    intrinsics_saved = False
    topics_to_read = [CAMERA_INFO_TOPIC, POINT_CLOUD_TOPIC, RGB_IMAGE_TOPIC]

    with bag:
        total_messages = bag.get_message_count(topic_filters=topics_to_read)
        print(f"Total messages to process on specified topics: {total_messages}.")

        for topic, msg, t_ros in bag.read_messages(topics=topics_to_read):
            # Ensure the message has a header and a stamp
            if not hasattr(msg, 'header') or not hasattr(msg.header, 'stamp'):
                print(
                    f"Warning: Message on topic '{topic}' at rosbag time {t_ros.to_sec():.6f} is missing header.stamp. Skipping.")
                continue

            current_msg_time_sec = msg.header.stamp.to_sec()

            # Save camera intrinsics (only once)
            if topic == CAMERA_INFO_TOPIC and not intrinsics_saved:
                intrinsics_data = {
                    'timestamp_sec': current_msg_time_sec,
                    'height': msg.height, 'width': msg.width,
                    'distortion_model': msg.distortion_model,
                    'D': list(msg.D), 'K': list(msg.K),
                    'R': list(msg.R), 'P': list(msg.P)
                }
                intrinsics_file = os.path.join(intrinsics_dir, f"camera_intrinsics.json")
                with open(intrinsics_file, 'w') as f:
                    json.dump(intrinsics_data, f, indent=4)
                print(f"Camera intrinsics saved to: {intrinsics_file}")
                intrinsics_saved = True

            # Add point cloud and image messages to their respective queues
            elif topic == POINT_CLOUD_TOPIC:
                pcd_queue.append(msg)
            elif topic == RGB_IMAGE_TOPIC:
                rgb_queue.append(msg)

    print(f"Reading complete. Buffers contain {len(pcd_queue)} point clouds and {len(rgb_queue)} RGB images.")
    print("-" * 50)
    print("Phase 2: Synchronizing and saving paired data...")

    # --- Phase 2: Synchronize queues and save at the specified interval ---
    bridge = CvBridge()
    last_save_time = -float('inf')
    saved_pairs_count = 0

    with open(pcd_timestamps_file, 'w', newline='') as pcd_csv, \
            open(rgb_timestamps_file, 'w', newline='') as rgb_csv:

        pcd_writer = csv.writer(pcd_csv)
        rgb_writer = csv.writer(rgb_csv)
        pcd_writer.writerow(['timestamp_sec', 'filename'])
        rgb_writer.writerow(['timestamp_sec', 'filename'])

        # Process while both queues still have data
        while pcd_queue and rgb_queue:
            pcd_time = pcd_queue[0].header.stamp.to_sec()
            rgb_time = rgb_queue[0].header.stamp.to_sec()

            time_diff = abs(pcd_time - rgb_time)

            if time_diff <= TIME_SYNC_TOLERANCE:
                # Found a synchronized pair, now check if it meets the save interval
                pair_time = pcd_time  # Use the point cloud time as the representative time for this pair

                if pair_time >= last_save_time + SAVE_INTERVAL_SECONDS:
                    # Get the messages to save
                    pcd_msg_to_save = pcd_queue.popleft()
                    rgb_msg_to_save = rgb_queue.popleft()

                    # Create filenames using a consistent timestamp format
                    timestamp_str = f"{pair_time:.6f}"

                    # 1. Save Point Cloud
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
                            print(f"Skipping empty point cloud at time {pair_time:.6f}")
                            continue  # If the point cloud is empty, don't save this pair
                    except Exception as e:
                        print(f"Error processing point cloud at time {pair_time:.6f}: {e}")
                        continue

                    # 2. Save RGB Image
                    try:
                        cv_image = bridge.imgmsg_to_cv2(rgb_msg_to_save, desired_encoding='bgr8')
                        rgb_filename = f"rgb_image_{timestamp_str}.png"
                        rgb_filepath = os.path.join(rgb_dir, rgb_filename)
                        cv2.imwrite(rgb_filepath, cv_image)
                        rgb_writer.writerow([timestamp_str, rgb_filename])
                    except Exception as e:
                        print(f"Error processing RGB image at time {pair_time:.6f}: {e}")
                        # Note: The point cloud might have been saved, but the image saving failed
                        continue

                    # Update status
                    last_save_time = pair_time
                    saved_pairs_count += 1
                    if saved_pairs_count % 10 == 0:
                        print(f"Successfully saved {saved_pairs_count} pairs...")

                else:
                    # This pair is too close to the last saved pair, skip it
                    # Discard both and look for the next pair
                    pcd_queue.popleft()
                    rgb_queue.popleft()

            elif pcd_time < rgb_time:
                # Point cloud message is too old, no matching image, discard it
                pcd_queue.popleft()
            else:  # rgb_time < pcd_time
                # Image message is too old, no matching point cloud, discard it
                rgb_queue.popleft()

    print("-" * 50)
    print(
        f"Processing complete! Total of {saved_pairs_count} synchronized pairs of point clouds and images were saved.")
    print(f"Data saved to directory: '{OUTPUT_BASE_DIR}'")
    print(f"Point cloud timestamp index: {pcd_timestamps_file}")
    print(f"RGB image timestamp index: {rgb_timestamps_file}")


if __name__ == '__main__':
    main()
