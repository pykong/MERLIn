#!/bin/bash

# Read configuration file
source sync.cfg

# Open a new terminal and run the ssh command in the background/disowned
gnome-terminal -- bash -c "ssh -t $REMOTE_USER@$REMOTE_IP 'cd $REMOTE_DIR; bash'" &
