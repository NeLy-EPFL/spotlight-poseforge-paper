#!/bin/bash

# Batch generate muscle activity videos for multiple experiments
# Usage: ./batch_make_videos.sh "/path/to/data/fly*pattern*" [segments...]
#
# Example:
#   ./batch_make_videos.sh "/mnt/upramdya_data/VAS/spotlight_data/260305_G213ximmortGCaMP8m_vib/fly001*ice*6.*V*" femur
#   ./batch_make_videos.sh "/mnt/upramdya_data/VAS/spotlight_data/260305_G213ximmortGCaMP8m_vib/fly*ice*6.*V*" femur tibia

set -e  # Exit on error

# Check if pattern is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <folder_pattern> [segments...]"
    echo "Example: $0 '/path/to/fly*pattern*' femur"
    exit 1
fi

PATTERN="$1"
shift  # Remove first argument, remaining are segments

# Default segments if none provided
if [ $# -eq 0 ]; then
    SEGMENTS="femur"
else
    SEGMENTS="$@"
fi

echo "==================================================================="
echo "Batch Video Generation"
echo "==================================================================="
echo "Pattern: $PATTERN"
echo "Segments: $SEGMENTS"
echo "==================================================================="
echo ""

# Find all matching folders that have a "processed" subfolder
FOLDERS=()
for folder in $PATTERN; do
    if [ -d "$folder/processed" ]; then
        FOLDERS+=("$folder")
    fi
done

if [ ${#FOLDERS[@]} -eq 0 ]; then
    echo "ERROR: No experiment folders found matching pattern: $PATTERN"
    echo "Looking for folders with 'processed' subfolder"
    exit 1
fi

echo "Found ${#FOLDERS[@]} experiment folders:"
for folder in "${FOLDERS[@]}"; do
    echo "  - $(basename "$folder")"
done
echo ""

# Process each folder
SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_FOLDERS=()

for folder in "${FOLDERS[@]}"; do
    exp_name=$(basename "$folder")
    output_video="$folder/muscle_activity_video.mp4"
    
    echo "==================================================================="
    echo "Processing: $exp_name"
    echo "==================================================================="
    
    # Run the video generation
    if python fig4_vid8_muscle/make_muscle_activity_video.py \
        --exp_folder "$folder" \
        --segments $SEGMENTS \
        --output "$output_video"; then
        echo "SUCCESS: Video saved to $output_video"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "ERROR: Failed to generate video for $exp_name"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_FOLDERS+=("$exp_name")
    fi
    
    echo ""
done

# Summary
echo "==================================================================="
echo "BATCH PROCESSING COMPLETE"
echo "==================================================================="
echo "Total experiments: ${#FOLDERS[@]}"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"

if [ $FAIL_COUNT -gt 0 ]; then
    echo ""
    echo "Failed experiments:"
    for failed in "${FAILED_FOLDERS[@]}"; do
        echo "  - $failed"
    done
fi

echo "==================================================================="
