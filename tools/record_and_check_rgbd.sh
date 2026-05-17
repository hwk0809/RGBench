#!/bin/bash

# Enable job control; this is key to robust background process handling
set -m

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

# 2. Pre-flight check retry settings
MAX_RETRIES=5
RETRY_DELAY=2

# 3. Default Bag file prefix
DEFAULT_BAG_PREFIX="robot_data"

# 4. Continuous dynamic check settings (*** tune sensitivity here ***)
TOPICS_TO_CHECK_MOVEMENT=(
"/left_arm/joint_states"
 "/right_arm/joint_states"
  )

CONTINUOUS_CHECK_INTERVAL=0.2   # seconds between checks (recommend 1 or 0.5)
STATIC_THRESHOLD_COUNT=2    # number of consecutive identical samples that trigger a warning (recommend 2)
MESSAGE_TIMEOUT=1           # message timeout

# ==============================================================================
# --- End of configuration ---
# ==============================================================================

# --- Globals and color definitions ---
BAG_RECORDER_PID=0
ROSBAG_LOG_FILE="rosbag_recorder.log"
declare -A last_positions
declare -A static_counters
declare -A has_warned

COLOR_YELLOW='\033[1;33m'
COLOR_GREEN='\033[1;32m'
COLOR_NC='\033[0m'

# --- Graceful exit handler ---
function cleanup {
    echo ""
    echo -e "${COLOR_YELLOW}Stop signal received (Ctrl+C)...${COLOR_NC}"
    if [ $BAG_RECORDER_PID -eq 0 ]; then
        echo "   - Recording has not started yet; exiting."
        exit 1
    fi
    if ps -p $BAG_RECORDER_PID > /dev/null; then
        echo "   - Sending graceful stop signal to rosbag record (PID: $BAG_RECORDER_PID)..."
        kill -SIGINT $BAG_RECORDER_PID
        echo "   - Waiting for rosbag record to finish writing and exit..."
        wait $BAG_RECORDER_PID
        echo "   - rosbag record has exited."
        echo "   - Waiting 1 second for the filesystem to sync..."
        sleep 1
    else
        echo -e "   - ${COLOR_YELLOW}Warning: when cleanup was called, the rosbag record process (PID: $BAG_RECORDER_PID) was already gone.${COLOR_NC}"
    fi
}

# --- Register trap ---
trap cleanup SIGINT SIGTERM

# --- Argument parsing ---
BAG_PREFIX="$DEFAULT_BAG_PREFIX"
while getopts ":f:" opt; do
  case ${opt} in
    f) BAG_PREFIX=$OPTARG ;;
    \?) echo "Invalid option: -$OPTARG" 1>&2; exit 1 ;;
    :) echo "Option -$OPTARG requires an argument" 1>&2; exit 1 ;;
  esac
done

# --- Pre-flight check (do the topics exist) ---
echo "Running pre-flight check (verifying topics exist)..."
echo "--------------------------------------------------"
retry_count=0
all_topics_found=false
while [ $retry_count -lt $MAX_RETRIES ]; do
    active_topics=$(rostopic list 2>/dev/null)
    if [ -z "$active_topics" ]; then
        echo "Error: cannot connect to the ROS Master (roscore)."
        exit 1
    fi
    missing_topics=()
    for topic in "${TOPICS_TO_RECORD[@]}"; do
        if ! echo "$active_topics" | grep -qx "$topic"; then
            missing_topics+=("$topic")
        fi
    done
    if [ ${#missing_topics[@]} -eq 0 ]; then
        echo "All ${#TOPICS_TO_RECORD[@]} topics found. Check passed."
        all_topics_found=true
        break
    else
        echo "Attempt $((retry_count + 1))/$MAX_RETRIES: the following topics are not yet published:"
        printf "   - %s\n" "${missing_topics[@]}"
        retry_count=$((retry_count + 1))
        if [ $retry_count -lt $MAX_RETRIES ]; then
            echo "   Retrying in ${RETRY_DELAY} seconds..."
            sleep $RETRY_DELAY
        fi
    fi
done
echo "--------------------------------------------------"
if [ "$all_topics_found" = false ]; then
    echo "Error: after $MAX_RETRIES attempts some topics are still missing. Recording cancelled."
    exit 1
fi

# --- Initialize check state ---
for topic in "${TOPICS_TO_CHECK_MOVEMENT[@]}"; do
    static_counters["$topic"]=0
    last_positions["$topic"]=""
    has_warned["$topic"]=0
done

# --- Start recording (background) ---
echo "Starting ROS data recording..."
rosbag record -o "$BAG_PREFIX" "${TOPICS_TO_RECORD[@]}" > "$ROSBAG_LOG_FILE" 2>&1 &
BAG_RECORDER_PID=$!
echo "rosbag record started in background, PID: $BAG_RECORDER_PID"
echo "   rosbag logs redirected to: $ROSBAG_LOG_FILE"
echo "   (tail -f $ROSBAG_LOG_FILE in another terminal to watch live)"
echo "--------------------------------------------------"
echo "Now monitoring data dynamics... (Ctrl+C to stop everything)"
echo ""

# --- Continuous dynamic check loop ---
while ps -p $BAG_RECORDER_PID > /dev/null; do
    for topic in "${TOPICS_TO_CHECK_MOVEMENT[@]}"; do
        current_pos=$(timeout $MESSAGE_TIMEOUT rostopic echo -n 1 "$topic" 2>/dev/null | grep "position:")

        if [ -z "$current_pos" ]; then
            continue
        fi

        last_pos="${last_positions[$topic]}"

        if [ -n "$last_pos" ] && [ "$last_pos" == "$current_pos" ]; then
            static_counters["$topic"]=$((static_counters["$topic"] + 1))
        else
            static_counters["$topic"]=0
            if [ ${has_warned["$topic"]} -eq 1 ]; then
                echo -e "${COLOR_GREEN}INFO: topic '$topic' has resumed motion.${COLOR_NC}"
                has_warned["$topic"]=0
            fi
        fi

        last_positions["$topic"]="$current_pos"

        if [ ${static_counters[$topic]} -ge $STATIC_THRESHOLD_COUNT ] && [ ${has_warned["$topic"]} -eq 0 ]; then
            echo -e "${COLOR_YELLOW}Warning: topic '$topic' appears static (${static_counters[$topic]} consecutive identical samples)${COLOR_NC}"
            has_warned["$topic"]=1
        fi
    done
    sleep $CONTINUOUS_CHECK_INTERVAL
done

# --- Final confirmation after the script ends ---
echo ""
echo "Recording flow finished."
BAG_FILE=$(find . -maxdepth 1 -name "${BAG_PREFIX}*.bag" -print0 | xargs -0 ls -t | head -1)
if [ -z "$BAG_FILE" ]; then
    echo "Error: no recorded bag file found."
    exit 1
fi
echo "Data saved to: $BAG_FILE"
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
