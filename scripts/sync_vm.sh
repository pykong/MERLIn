#!/bin/bash

# Script to upload sources to a cloud destination for training
# Edit sync.cfg accordingly.
# You can also overwrite the destination.ip via command line arg.
# ./sync_vm.sh new.destination.ip

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change the working directory to the script directory
cd "$SCRIPT_DIR"

# Load configurations
source sync.cfg

# Overwrite DESTINATION_IP if command line argument is provided
if [ ! -z "$1" ]; then
  DESTINATION_IP="$1"
fi

# Split EXCLUDE_PATTERNS into an array and prepare rsync exclude options
IFS=',' read -r -a EXCLUDE_PATTERNS_ARRAY <<< "$EXCLUDE_PATTERNS"
EXCLUDE_OPTIONS=()
for pattern in "${EXCLUDE_PATTERNS_ARRAY[@]}"; do
  EXCLUDE_OPTIONS+=("--exclude=$pattern")
done

# Command
rsync -avz "${EXCLUDE_OPTIONS[@]}" --include="$INCLUDE_PATTERN" "$SOURCE_DIR" "$DESTINATION_USER"@"$DESTINATION_IP":"$DESTINATION_DIR"
