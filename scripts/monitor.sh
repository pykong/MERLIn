#!/bin/bash

while true; do
    timestamp=$(date "+%Y.%m.%d-%H.%M.%S")
    cpu_util=$(top -b -n2 -d0.01 | grep "Cpu(s)" | tail -n 1 | awk '{print $2 + $4}')
    mem_util=$(free | grep Mem | awk '{print $3/$2 * 100.0}')
    gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    echo "$timestamp CPU: $cpu_util%, Mem: $mem_util%, GPU: $gpu_util%"
    sleep 10
done
