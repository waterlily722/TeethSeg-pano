#!/bin/bash
cd /oral_llm/xiweidai/dentalagent2/Instance_seg_teeth
echo "==========================================="
echo "Starting 50-epoch training at $(date)"
echo "==========================================="
echo ""

python3 -u train_50epoch.py 2>&1

echo ""
echo "==========================================="
echo "Training finished at $(date)"
echo "==========================================="
