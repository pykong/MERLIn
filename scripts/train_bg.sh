#!/bin/sh

# This scripts starts training in the background.
# The output will be directed to a log file in results/.
# That log file is going to be watched until the training ends.

# ensure the 'results' directory exists
mkdir -p results

# start the long running process
# replace 'long-running-command' with your actual command
nohup long-running-command > results/nohup.log 2>&1 &

# get the process id of the long-running command
pid=$!

echo "Started long running process with pid $pid"

# start a loop in the background that waits for the long running process
# to finish, and then kills the tail command
(tail -f results/nohup.log & echo $! > tail.pid) &

# wait for the long-running process to finish
wait $pid

# once it finishes, kill the tail process
kill $(cat tail.pid)

# cleanup
rm tail.pid
