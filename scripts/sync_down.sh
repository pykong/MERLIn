#!/bin/bash

# Script to upload sources to a cloud destination for training
# Edit sync.cfg accordingly.

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $SCRIPT_DIR/../sync.cfg

# Configure list of files to be transferred
include_file=$(mktemp)
echo "$SYNC_DOWN_PATTERN" | tr ',' '\n' > "$include_file"

# Create dest dir
timestamp=$(date +"%Y%m%d_%H%M%S")
dest_dir=$LOCAL_SAVE_DIR$timestamp

# Command
# Use --dry-run for debugging purposes
echo "Saving to: $dest_dir"
rsync -avzr --files-from="$include_file" "$REMOTE_USER"@"$REMOTE_IP":"$REMOTE_DIR" $dest_dir
