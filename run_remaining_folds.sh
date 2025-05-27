#!/bin/bash

# Script to run training for folds 2-8 sequentially
# When one fold completes, the next one automatically starts

for fold in {0..8}
do
    echo "Starting training for fold $fold"
    python -m launch_scripts.train_comp --logger wandb --save-logs --fold $fold --name cqt_v1_fold_$fold
    
    echo "Completed training for fold $fold"
done

echo "All folds completed!"