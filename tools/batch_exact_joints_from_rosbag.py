#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rosbag
import sys
import csv
import time
import os
import glob
import argparse


def flatten_message(msg, parent_key='', sep='.'):
    """
    Recursively flattens a ROS message into a single-level dictionary.

    Args:
        msg: The ROS message object.
        parent_key (str): The base key for nested fields.
        sep (str): The separator between nested field names.

    Returns:
        dict: A flattened dictionary representation of the message.
    """
    items = {}
    try:
        # Get all attribute "slots" of the message.
        slots = msg.__slots__
    except AttributeError:
        # Return if the object is not a standard ROS message.
        return {parent_key: msg} if parent_key else {}

    for slot in slots:
        new_key = f"{parent_key}{sep}{slot}" if parent_key else slot
        value = getattr(msg, slot)

        if hasattr(value, '__slots__'):
            # Recursively flatten nested messages.
            items.update(flatten_message(value, new_key, sep=sep))
        elif isinstance(value, (list, tuple)):
            # Convert lists/tuples (like in sensor_msgs/JointState) to a string representation.
            items[new_key] = str(value)
        else:
            # Handle primitive data types.
            items[new_key] = value
    return items


def process_bag_file(bag_file_path, output_base_dir, args):
    """
    Processes a single ROS bag file to extract specified joint_state topics into CSV files.

    Args:
        bag_file_path (str): The full path to the .bag file.
        output_base_dir (str): The primary output directory for this bag file's data.
        args: The command-line arguments containing topic names.
    """
    print("-" * 60)
    print(f"▶️  Processing Bag File: {os.path.basename(bag_file_path)}")

    # 1. Define target topics from arguments.
    target_topics = [args.left_arm_topic, args.right_arm_topic]

    # 2. Create the 'joints' subdirectory for the output CSVs.
    joints_output_dir = os.path.join(output_base_dir, 'joints')
    try:
        # The 'exist_ok=True' flag prevents an error if the directory already exists.
        os.makedirs(joints_output_dir, exist_ok=True)
        print(f"   Output will be saved to: {joints_output_dir}")
    except OSError as e:
        print(f"❌ Error: Could not create directory {joints_output_dir}: {e}")
        return

    # 3. Open the bag file.
    try:
        bag = rosbag.Bag(bag_file_path, 'r')
    except Exception as e:
        print(f"❌ Error: Could not open bag file {bag_file_path}: {e}")
        return

    with bag:
        # Get a list of all available topics in the bag for validation.
        available_topics = bag.get_type_and_topic_info().topics.keys()

        for topic_name in target_topics:
            # Check if the target topic exists in this bag file.
            if topic_name not in available_topics:
                print(f"   🟡 Info: Topic '{topic_name}' not found in {os.path.basename(bag_file_path)}. Skipping.")
                continue

            print(f"   Extracting Topic: {topic_name}")

            # Create a filesystem-safe name for the CSV file.
            safe_topic_name = topic_name.strip('/').replace('/', '_')
            output_filename = os.path.join(joints_output_dir, f"{safe_topic_name}.csv")

            try:
                with open(output_filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',')
                    is_first_message = True
                    message_count = 0

                    # Read all messages from the specified topic.
                    for _, msg, t in bag.read_messages(topics=[topic_name]):
                        message_count += 1

                        # Flatten the complex ROS message into a simple dictionary.
                        flat_msg_dict = flatten_message(msg)

                        # On the first message, write the CSV header row.
                        if is_first_message:
                            # Sort keys for consistent column order.
                            headers = ["rosbagTimestamp"] + sorted(flat_msg_dict.keys())
                            writer.writerow(headers)
                            is_first_message = False

                        # Write the message data as a new row.
                        values = [str(t)] + [str(flat_msg_dict.get(key, '')) for key in headers[1:]]
                        writer.writerow(values)

                    if message_count > 0:
                        print(f"   ✅ Success: Wrote {message_count} messages to {output_filename}")

            except Exception as e:
                print(f"   ❌ Error: Failed to write CSV file {output_filename}: {e}")


def main(args):
    """
    Finds all .bag files based on input path and initiates processing for each.
    """
    input_path = args.input_path
    bag_files = []

    # Check if the input path is a directory or a single file.
    if os.path.isdir(input_path):
        print(f"Input is a directory. Searching for .bag files in '{input_path}'...")
        # Use glob to find all files ending with .bag.
        bag_files = sorted(glob.glob(os.path.join(input_path, '*.bag')))
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
        print("Error: No .bag files found to process.")
        return

    print(f"Found {len(bag_files)} .bag file(s) to process. Starting...")
    print("=" * 60)

    # Process each found bag file.
    for bag_file_path in bag_files:
        # Create a specific output directory named after the bag file (without extension).
        bag_basename = os.path.splitext(os.path.basename(bag_file_path))[0]
        specific_output_dir = os.path.join(args.output_dir, bag_basename)

        process_bag_file(bag_file_path, specific_output_dir, args)

    print("=" * 60)
    print("🎉 All tasks have been completed!")


if __name__ == '__main__':
    # --- Configure Command-Line Arguments ---
    parser = argparse.ArgumentParser(
        description="Extracts joint_state topics from ROS bag files into CSVs.",
        formatter_class=argparse.RawTextHelpFormatter  # For better help text formatting.
    )

    parser.add_argument('input_path', type=str,
                        help='Path to the input folder containing .bag files, OR path to a single .bag file.')

    parser.add_argument('-o', '--output_dir', type=str, default='.',
                        help='Root output folder to store all extracted data. Defaults to the current directory.')

    # --- ROS Topic Name Arguments ---
    parser.add_argument('--left-arm-topic', type=str, default='/left_arm/joint_states',
                        help='The topic name for the left arm joint states.')

    parser.add_argument('--right-arm-topic', type=str, default='/right_arm/joint_states',
                        help='The topic name for the right arm joint states.')

    args = parser.parse_args()
    main(args)