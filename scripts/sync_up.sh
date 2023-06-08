#!/bin/bash

# Script to upload sources to a cloud destination for training
# Edit sync.cfg accordingly.


# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change the working directory to the script directory
cd "$SCRIPT_DIR"

# Load configurations
source sync.cfg

# Configure list of files to be transferred
include_file=$(mktemp)
echo "$SYNC_UP_PATTERN" | tr ',' '\n' > "$include_file"

# Command
# Use --dry-run for debugging purposes
rsync -avz --delete --files-from="$include_file" "$LOCAL_DIR" "$REMOTE_USER"@"$REMOTE_IP":"$REMOTE_DIR"
