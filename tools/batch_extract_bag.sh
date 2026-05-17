#!/bin/bash

# ==============================================================================
# --- 1. Configure the topics to extract ---
#
# In the parentheses below, list every topic you want to keep in the new bag files.
# ==============================================================================
TOPICS_TO_KEEP=(
    "/left_arm/joint_states"
    "/right_arm/joint_states"
)
# ==============================================================================
# --- 2. Configure the default output name ---
#
# This default suffix is used when no argument is passed at runtime.
# Example: input "a.bag" -> output "a_filtered.bag" inside the "filtered/" directory.
# ==============================================================================
DEFAULT_OUTPUT_NAME="_filtered"
# ==============================================================================
# --- End of configuration --- the script body below usually does not need modifying
# ==============================================================================

# --- Usage hint ---
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    echo "Usage: $0 [optional suffix]"
    echo ""
    echo "Function: batch-process every .bag file in the current directory. Creates a new"
    echo "          subdirectory and writes the filtered bag files with the chosen suffix."
    echo ""
    echo "Examples:"
    echo "  # Default suffix ('${DEFAULT_OUTPUT_NAME}')"
    echo "  # (a.bag -> filtered/a_filtered.bag)"
    echo "  $0"
    echo ""
    echo "  # Custom suffix '_joints'"
    echo "  # (a.bag -> joints/a_joints.bag)"
    echo "  $0 _joints"
    exit 0
fi

# --- Argument handling ---
# Use the user-supplied argument if any; otherwise fall back to the default.
SUFFIX=${1:-$DEFAULT_OUTPUT_NAME}
# Strip the leading underscore (if present) to get a clean directory name.
DIR_NAME=${SUFFIX#_}

# --- Pre-checks ---
# Make sure the topic list has been populated
if [ ${#TOPICS_TO_KEEP[@]} -eq 0 ]; then
    echo "Error: please edit the script and add topics to TOPICS_TO_KEEP."
    exit 1
fi

# Make sure the current directory contains .bag files
shopt -s nullglob
BAG_FILES=(*.bag)
if [ ${#BAG_FILES[@]} -eq 0 ]; then
    echo "Info: no .bag files found in the current directory."
    exit 0
fi
shopt -u nullglob # restore default behavior

# --- Create the output directory ---
echo "Preparing output directory..."
mkdir -p "$DIR_NAME"
if [ $? -ne 0 ]; then
    echo "Error: cannot create directory '$DIR_NAME'. Check permissions."
    exit 1
fi

# --- Build the expression rosbag filter needs (only once) ---
filter_expression=""
first=true
for topic in "${TOPICS_TO_KEEP[@]}"; do
    if [ "$first" = true ]; then
        filter_expression="topic == '$topic'"
        first=false
    else
        filter_expression="$filter_expression or topic == '$topic'"
    fi
done

# --- Start batch processing ---
echo "Starting batch extraction..."
echo "--------------------------------------------------"
echo "Topics to keep:"
printf "  - %s\n" "${TOPICS_TO_KEEP[@]}"
echo "Output directory: '${DIR_NAME}'"
echo "Output filename suffix: '${SUFFIX}'"
echo "--------------------------------------------------"

processed_count=0
skipped_count=0

# Iterate every .bag file
for INPUT_BAG in "${BAG_FILES[@]}"; do
    # Build output filename and full path
    BASENAME=$(basename "$INPUT_BAG" .bag)
    OUTPUT_FILENAME="${BASENAME}${SUFFIX}.bag"
    OUTPUT_BAG="$DIR_NAME/$OUTPUT_FILENAME"

    # Skip if output already exists
    if [ -f "$OUTPUT_BAG" ]; then
        echo "Skipping: output '$OUTPUT_BAG' already exists."
        skipped_count=$((skipped_count + 1))
        continue
    fi

    # --- Extract a single file ---
    echo "Processing: '$INPUT_BAG' -> '$OUTPUT_BAG'"
    rosbag filter "$INPUT_BAG" "$OUTPUT_BAG" "$filter_expression"

    if [ $? -eq 0 ]; then
        echo "   Success"
        processed_count=$((processed_count + 1))
    else
        echo "   Failed: error while processing '$INPUT_BAG'."
        # Remove any partial output produced before the failure
        rm -f "$OUTPUT_BAG"
    fi
done

echo "--------------------------------------------------"
echo "Batch processing complete."
echo "   Successfully processed: $processed_count files"
echo "   Skipped: $skipped_count files"
