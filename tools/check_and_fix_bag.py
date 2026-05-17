#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import argparse
import shutil

# Define terminal output colors
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def check_and_fix_rosbags(directory, replace_on_fix=False):
    """
    Recursively checks and attempts to fix all ROS bag files in a given directory.

    :param directory: The root directory to scan.
    :param replace_on_fix: If True, replaces the original file with the fixed one upon success.
    """
    print(f"{bcolors.HEADER}--- Starting scan in directory: {directory} ---\n{bcolors.ENDC}")
    found_bags = []
    corrupted_bags = 0
    fixed_bags = 0
    failed_to_fix_bags = 0

    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.bag'):
                file_path = os.path.join(root, filename)
                found_bags.append(file_path)

                print(f"{bcolors.OKBLUE}Checking: {file_path}{bcolors.ENDC}")

                # 1. Check the bag file using `rosbag check`
                try:
                    check_command = ['rosbag', 'check', file_path]
                    check_result = subprocess.run(
                        check_command,
                        capture_output=True,
                        text=True,
                        check=True  # Throws CalledProcessError if command returns a non-zero exit code
                    )
                    print(f"{bcolors.OKGREEN}  -> Status: OK{bcolors.ENDC}")

                except subprocess.CalledProcessError as e:
                    corrupted_bags += 1
                    print(f"{bcolors.FAIL}  -> Status: Corrupted!{bcolors.ENDC}")
                    print(f"{bcolors.WARNING}  -> Error message: {e.stderr.strip()}{bcolors.ENDC}")
                    print(f"{bcolors.OKCYAN}  -> Attempting to fix...{bcolors.ENDC}")

                    # 2. Attempt to fix using `rosbag fix`
                    fixed_bag_path = file_path.replace('.bag', '.fixed.bag')

                    # Prevent trying to fix a file that is already marked as .fixed
                    if fixed_bag_path == file_path:
                        print(f"{bcolors.FAIL}  -> Cannot create a new name for an already fixed file, skipping.{bcolors.ENDC}")
                        failed_to_fix_bags += 1
                        continue

                    try:
                        fix_command = ['rosbag', 'fix', file_path, fixed_bag_path]
                        fix_result = subprocess.run(
                            fix_command,
                            capture_output=True,
                            text=True,
                            check=True
                        )
                        print(f"{bcolors.OKGREEN}  -> Fix successful! New file saved to: {fixed_bag_path}{bcolors.ENDC}")
                        fixed_bags += 1

                        # 3. (Optional) Replace the original file
                        if replace_on_fix:
                            print(f"{bcolors.OKCYAN}  -> Replacing original file with the fixed version...{bcolors.ENDC}")
                            try:
                                shutil.move(fixed_bag_path, file_path)
                                print(f"{bcolors.OKGREEN}  -> Replacement successful!{bcolors.ENDC}")
                            except Exception as move_err:
                                print(f"{bcolors.FAIL}  -> Replacement failed: {move_err}{bcolors.ENDC}")

                    except subprocess.CalledProcessError as fix_e:
                        failed_to_fix_bags += 1
                        print(f"{bcolors.FAIL}  -> Fix failed!{bcolors.ENDC}")
                        print(f"{bcolors.WARNING}  -> Error message: {fix_e.stderr.strip()}{bcolors.ENDC}")

                print("-" * 30)

    print(f"\n{bcolors.HEADER}--- Scan Complete ---{bcolors.ENDC}")
    print(f"Found a total of {len(found_bags)} bag files.")
    print(f"{bcolors.FAIL}Found {corrupted_bags} corrupted files.{bcolors.ENDC}")
    print(f"{bcolors.OKGREEN}Successfully fixed {fixed_bags} files.{bcolors.ENDC}")
    if failed_to_fix_bags > 0:
        print(f"{bcolors.FAIL}Failed to fix {failed_to_fix_bags} files.{bcolors.ENDC}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quickly check and fix all ROS bag files within a directory and its subdirectories.",
        formatter_class=argparse.RawTextHelpFormatter # Preserve help text formatting
    )
    parser.add_argument(
        "directory",
        type=str,
        help="The root directory containing the bag files to check."
    )
    parser.add_argument(
        "--replace",
        action="store_true", # This makes it a flag; if present, its value is True
        help="If set, automatically replaces the original corrupted file with the new fixed file upon successful repair.\n"
             "WARNING: This is a destructive operation. Use with caution."
    )

    args = parser.parse_args()

    # Check if the ROS environment is sourced
    if "ROS_VERSION" not in os.environ:
        print(f"{bcolors.FAIL}Error: ROS environment not detected. Please source your ROS setup file first (e.g., `source /opt/ros/noetic/setup.bash`){bcolors.ENDC}")
        exit(1)

    if not os.path.isdir(args.directory):
        print(f"{bcolors.FAIL}Error: Directory '{args.directory}' not found.{bcolors.ENDC}")
        exit(1)

    check_and_fix_rosbags(args.directory, args.replace)