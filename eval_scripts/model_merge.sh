#!/bin/bash
# Set CHECKPOINT_DIR to your experiment checkpoint dir (e.g. .../golf/exp_name)
LOCAL_DIR=${CHECKPOINT_DIR:-/path/to/your/GOLF/checkpoints/your_exp_dir}
TARGET_DIR=$LOCAL_DIR/merged_hf_model

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")/golf/verl"

for step in 200; do
    echo "开始合并 global_step_$step..."
    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir $LOCAL_DIR/global_step_$step/actor \
        --target_dir $TARGET_DIR/global_step_$step
    echo "global_step_$step 合并完成"
    echo "---"
done

echo "所有步骤合并完成!"
