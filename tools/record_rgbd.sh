#!/bin/bash

# ==============================================================================
# --- Configure here ---
# ==============================================================================

# 1. List of topics to record
TOPICS_TO_RECORD=(
    "/camera/color/camera_info"
    "/camera/color/image_rect_color"
    "/camera/depth/color/points"
    "/left_arm/end_pose"
    "/left_arm/end_rotation_pose"
    "/left_arm/joint_states"
    "/right_arm/end_pose"
    "/right_arm/end_rotation_pose"
    "/right_arm/joint_states"
)

# 2. Retry settings for the pre-flight check
#    The script will retry until every topic is present
MAX_RETRIES=5       # maximum number of retries
RETRY_DELAY=2       # seconds to wait between retries

# 3. Default Bag file prefix
DEFAULT_BAG_PREFIX="robot_data"

# ==============================================================================
# --- End of configuration --- the script body below does not need to be modified
# ==============================================================================

# --- Argument parsing (optional -f flag to set filename) ---
BAG_PREFIX="$DEFAULT_BAG_PREFIX"
while getopts ":f:" opt; do
  case ${opt} in
    f) BAG_PREFIX=$OPTARG ;;
    \?) echo "Invalid option: -$OPTARG" 1>&2; exit 1 ;;
    :) echo "Option -$OPTARG requires an argument" 1>&2; exit 1 ;;
  esac
done

# --- Pre-flight check ---
echo "Running pre-flight check..."
echo "--------------------------------------------------"

retry_count=0
all_topics_found=false

while [ $retry_count -lt $MAX_RETRIES ]; do
    # Read the active topic list, redirecting stderr so rostopic noise does not pollute output
    active_topics=$(rostopic list 2>/dev/null)

    # Verify that the ROS Master is online
    if [ -z "$active_topics" ]; then
        echo "Error: cannot connect to the ROS Master (roscore)."
        echo "   Make sure roscore and your nodes are running."
        exit 1
    fi

    missing_topics=()
    # Iterate every topic to record
    for topic in "${TOPICS_TO_RECORD[@]}"; do
        # Use grep -qx for exact whole-line matching
        if ! echo "$active_topics" | grep -qx "$topic"; then
            # Not in the active list; mark as missing
            missing_topics+=("$topic")
        fi
    done

    # If nothing is missing, every topic was found
    if [ ${#missing_topics[@]} -eq 0 ]; then
        echo "All ${#TOPICS_TO_RECORD[@]} topics found. Check passed."
        all_topics_found=true
        break # success, exit loop
    else
        # Report missing topics
        echo "Attempt $((retry_count + 1))/$MAX_RETRIES: the following topics are not yet published:"
        printf "   - %s\n" "${missing_topics[@]}"
        retry_count=$((retry_count + 1))

        # Retry if attempts remain
        if [ $retry_count -lt $MAX_RETRIES ]; then
            echo "   Retrying in ${RETRY_DELAY} seconds..."
            sleep $RETRY_DELAY
        fi
    fi
done

echo "--------------------------------------------------"

# If the flag is still false the final check failed
if [ "$all_topics_found" = false ]; then
    echo "Error: after $MAX_RETRIES attempts some topics are still missing."
    echo "Recording cancelled. Please check that your ROS nodes are running."
    exit 1
fi

# --- Start recording ---
echo "Starting ROS data recording..."
echo "Press Ctrl+C to stop."

rosbag record -o "$BAG_PREFIX" "${TOPICS_TO_RECORD[@]}"

# --- Post-recording verification (kept as a final safety net) ---
BAG_FILE=$(find . -maxdepth 1 -name "${BAG_PREFIX}*.bag" -print0 | xargs -0 ls -t | head -1)

if [ -z "$BAG_FILE" ]; then
    echo "Error: no recorded bag file found."
    exit 1
fi

echo ""
echo "Recording finished. Data saved to: $BAG_FILE"
echo "Performing final content verification..."

RECORDED_TOPICS=$(rosbag info -y "$BAG_FILE" | grep 'topic:' | sed 's/.*topic: //')
final_check_ok=true
for topic in "${TOPICS_TO_RECORD[@]}"; do
    if ! echo "$RECORDED_TOPICS" | grep -qx "$topic"; then
        echo "[final check failed] $topic"
        final_check_ok=false
    fi
done

if [ "$final_check_ok" = true ]; then
    echo "Final check passed. Data complete."
else
    echo "Warning: some topics may have been interrupted during recording."
fi
