#!/bin/bash
# Remove videos not found in video_list.txt
# Usage:
#   bash cleanup_videos.sh --dry-run    # show what would be removed
#   bash cleanup_videos.sh              # actually remove

VIDEO_DIR=~/aggressive_behavior_project/videos
KEEP_LIST=~/aggressive_behavior_project/video_list.txt

DRY_RUN=false
if [ "$1" = "--dry-run" ]; then
    DRY_RUN=true
fi

to_remove=0
to_keep=0

for f in "$VIDEO_DIR"/*.mp4; do
    basename=$(basename "$f")
    if ! grep -qxF "$basename" "$KEEP_LIST"; then
        to_remove=$((to_remove + 1))
        if [ "$DRY_RUN" = true ]; then
            echo "WOULD REMOVE: $basename"
        else
            rm "$f"
        fi
    else
        to_keep=$((to_keep + 1))
        if [ "$DRY_RUN" = true ]; then
            echo "WOULD KEEP: $basename"
        fi
    fi
done

echo ""
echo "Keep: $to_keep"
echo "Remove: $to_remove"
if [ "$DRY_RUN" = true ]; then
    echo "(dry run -- nothing deleted)"
fi
