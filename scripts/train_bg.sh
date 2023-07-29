#!/bin/sh

shutdown=false

# Check if the first argument is --shutdown
if [ "$1" = "--shutdown" ]
then
  shutdown=true
fi

# This scripts starts training in the background.
# The output will be directed to a log file in results/.
# That log file is going to be watched until the training ends.

# ensure the 'results' directory exists
mkdir -p results

# get the directory of the current script
dir=$(dirname "$0")

# start the monitoring script in the background
# replace 'monitor.sh' with your actual script name
nohup "$dir/monitor.sh" > results/monitor.log 2>&1 &
monitor_pid=$!
echo "Started monitoring with pid $monitor_pid"

# start the long running process
# replace 'long-running-command' with your actual command
nohup poetry run train > results/nohup.log 2>&1 &
pid=$!
echo "Started long running process with pid $pid"

# start a loop in the background that waits for the long running process
# to finish, and then kills the tail command
(tail -f results/nohup.log & echo $! > tail.pid) &

# wait for the long-running process to finish
wait $pid

# once it finishes, kill the tail process and the monitoring script
kill $(cat tail.pid)
kill $monitor_pid

# cleanup
rm tail.pid

# If the shutdown argument was passed, shutdown the system
if $shutdown
then
  echo "Shutting down..."
  sudo shutdown -h now
fi
