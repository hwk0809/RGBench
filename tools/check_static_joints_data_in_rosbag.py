#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import rosbag
import rospy  # use rospy's Time and Duration for precise time handling
import argparse
import numpy as np
import multiprocessing
from tqdm import tqdm

# --- Configuration ---
TOPICS_TO_CHECK = ["/left_arm/joint_states", "/right_arm/joint_states"]
FLOAT_TOLERANCE = 1e-6
# Limit the search for the "last message" to the trailing seconds of the bag; this dramatically improves throughput
LAST_SECONDS_WINDOW = 2.0
# --------------------

# Output colors
COLOR_RED = '\033[91m'
COLOR_GREEN = '\033[92m'
COLOR_YELLOW = '\033[93m'
COLOR_NC = '\033[0m'


def check_bag_file_fast(bag_path):
    """
    Fast core function for checking a single bag file.
    Reads only the first message per topic and the last message within the trailing window.
    """
    try:
        with rosbag.Bag(bag_path, 'r') as bag:
            # Retrieve the bag start/end time (cheap metadata)
            try:
                end_time_float = bag.get_end_time()
                start_search_time = rospy.Time.from_sec(end_time_float - LAST_SECONDS_WINDOW)
            except Exception:
                # Skip if the bag is too short/corrupt to expose timestamps
                return (False, bag_path, "unable to read timestamps")

            for topic in TOPICS_TO_CHECK:
                if bag.get_message_count(topic_filters=[topic]) < 2:
                    continue

                # 1. Fast retrieval of the first message
                try:
                    _, first_msg, _ = next(bag.read_messages(topics=[topic]))
                except StopIteration:
                    continue

                # 2. Fast retrieval of the last message
                last_msg = None
                # Search only within the trailing window
                for _, msg, _ in bag.read_messages(topics=[topic], start_time=start_search_time):
                    last_msg = msg

                # If no message arrived in the trailing window (e.g., topic stopped publishing early),
                # we cannot judge motion; treat as not "fully static" to be safe.
                if last_msg is None:
                    continue

                # 3. Core comparison
                if np.allclose(first_msg.position, last_msg.position, atol=FLOAT_TOLERANCE):
                    return (True, bag_path, topic)

    except Exception as e:
        return (False, bag_path, f"processing error: {e}")

    return (False, bag_path, None)


def main():
    parser = argparse.ArgumentParser(description="Fast parallel batch check for static joint data inside ROS bag files.")
    parser.add_argument("directory", help="Root directory to scan.")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"{COLOR_RED}Error: directory '{args.directory}' does not exist.{COLOR_NC}")
        return

    print(f"Scanning directory: {args.directory}")

    bag_files = []
    for root, _, files in os.walk(args.directory):
        for file in files:
            if file.endswith(".bag"):
                bag_files.append(os.path.join(root, file))

    if not bag_files:
        print(f"{COLOR_YELLOW}No .bag files found in this directory.{COLOR_NC}")
        return

    num_processes = multiprocessing.cpu_count()
    print(f"Found {len(bag_files)} .bag files; running on {num_processes} CPU cores...")

    static_bags = []
    error_bags = []

    with multiprocessing.Pool(processes=num_processes) as pool:
        with tqdm(total=len(bag_files), desc="Checking", unit="bag") as pbar:
            for result in pool.imap_unordered(check_bag_file_fast, bag_files):
                is_static, path, detail = result
                if is_static:
                    static_bags.append((path, detail))
                elif detail is not None and "Error" in detail:
                    error_bags.append((path, detail))
                pbar.update(1)

    print("\n" + "=" * 50)
    print("Check complete.")
    print("=" * 50)

    if not static_bags and not error_bags:
        print(f"{COLOR_GREEN}All {len(bag_files)} bag files contain moving joint data.{COLOR_NC}")
    else:
        if static_bags:
            print(f"{COLOR_RED}Found {len(static_bags)} bag file(s) with apparently static joint data:{COLOR_NC}")
            for path, topic in static_bags:
                relative_path = os.path.relpath(path, start=args.directory)
                print(f"  - {COLOR_YELLOW}file:{COLOR_NC} ./{relative_path} ({COLOR_RED}reason: {topic} static{COLOR_NC})")
        if error_bags:
            print(f"\n{COLOR_RED}The following {len(error_bags)} file(s) errored during processing:{COLOR_NC}")
            for path, error_msg in error_bags:
                relative_path = os.path.relpath(path, start=args.directory)
                print(f"  - {COLOR_YELLOW}file:{COLOR_NC} ./{relative_path} ({COLOR_RED}{error_msg}{COLOR_NC})")


if __name__ == "__main__":
    # Run main only when executed directly
    main()
