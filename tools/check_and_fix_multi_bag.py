#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm  # Import the tqdm library


# Define terminal output colors
class bcolors:
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    HEADER = '\033[95m'


def check_bag(file_path):
    """
    Function to check a single bag file.
    Returns a tuple: (file_path, error_message_or_None).
    """
    try:
        # We use Popen to have more control, e.g., for timeouts.
        process = subprocess.Popen(
            ['rosbag', 'check', file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # Add a timeout (e.g., 600 seconds = 10 minutes) to prevent getting stuck on one file.
        _, stderr = process.communicate(timeout=600)

        if process.returncode != 0:
            return (file_path, stderr.strip())
        return (file_path, None)  # None indicates the file is OK.
    except subprocess.TimeoutExpired:
        return (file_path, "Check timed out after 10 minutes.")
    except Exception as e:
        return (file_path, str(e))


def main():
    parser = argparse.ArgumentParser(
        description="Quickly check all ROS bag files in a directory using parallel processing and a progress bar.",
    )
    parser.add_argument("directory", type=str, help="The root directory to scan.")
    parser.add_argument(
        "-j", "--jobs", type=int, default=cpu_count(),
        help=f"Number of parallel jobs to run. Defaults to your number of CPU cores ({cpu_count()})."
    )
    args = parser.parse_args()

    # Check for ROS environment
    if "ROS_VERSION" not in os.environ:
        print(
            f"{bcolors.FAIL}Error: ROS environment not detected. Please source your ROS setup file first.{bcolors.ENDC}")
        exit(1)

    if not os.path.isdir(args.directory):
        print(f"{bcolors.FAIL}Error: Directory '{args.directory}' not found.{bcolors.ENDC}")
        exit(1)

    print(f"Scanning for .bag files in {args.directory}...")
    bag_files = []
    for root, _, files in os.walk(args.directory):
        for filename in files:
            if filename.endswith('.bag'):
                bag_files.append(os.path.join(root, filename))

    if not bag_files:
        print("No .bag files found.")
        return

    print(f"Found {len(bag_files)} bag files. Starting check with {args.jobs} parallel jobs...")

    corrupted_files = []

    # Use tqdm to wrap the pool's map call to display a progress bar
    with Pool(processes=args.jobs) as pool:
        # Using imap_unordered updates the progress bar more frequently, as it yields results as they become available.
        results = list(tqdm(pool.imap_unordered(check_bag, bag_files), total=len(bag_files), desc="Checking bags"))

    # Process the results after the pool is finished
    for file_path, error_msg in results:
        if error_msg:
            corrupted_files.append((file_path, error_msg))

    print(f"\n{bcolors.HEADER}--- Check Complete ---{bcolors.ENDC}")
    if not corrupted_files:
        print(f"{bcolors.OKGREEN}All {len(bag_files)} bag files are OK!{bcolors.ENDC}")
    else:
        print(f"{bcolors.FAIL}Found {len(corrupted_files)} corrupted bag files:{bcolors.ENDC}")
        for f, err in corrupted_files:
            print(f"  - File: {f}")
            print(f"    Reason: {err}")
        print("\nYou can try to fix them using the command: `rosbag fix <corrupted_file.bag> <fixed_file.bag>`")


if __name__ == "__main__":
    main()