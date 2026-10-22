#!/bin/bash
# Samples GPU utilization (%) and memory used (MiB) every 30 s for time_left and the GPU status hook.
# Started by record_start.sh; the GPU is shared with the eval server, so samples include eval load.
while true; do
    echo "$(date +%s) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')" \
        >> /var/tmp/ao_gpu.log
    sleep 30
done
