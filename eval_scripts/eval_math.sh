#!/bin/bash
# Set PROJECT_ROOT to your repo parent path (e.g. /path/to/your/projects)
ROOT=${PROJECT_ROOT:-/path/to/your/projects}

cd $ROOT/GOLF/eval_scripts

DATA=${EVAL_DATA:-$ROOT/GOLF/data/valid.math.parquet}
OUTPUT_DIR=${EVAL_OUTPUT_DIR:-$ROOT/GOLF/results/in_math}

# Set MODEL_PATHS to your checkpoint dirs (merged_hf_model/global_step_XXX)
MODEL_PATHS=(
    "$ROOT/GOLF/checkpoints/qwen3-8b-no-think/golf/your_exp_name/merged_hf_model/global_step_xxx"
)
MODEL_NAMES=(
    "your_model_name_global_step_xxx"
)
TEMPLATES=(
    "refine"
)

for i in "${!MODEL_PATHS[@]}"; do
    MODEL_PATH=${MODEL_PATHS[$i]}
    MODEL_NAME=${MODEL_NAMES[$i]}
    TEMPLATE=${TEMPLATES[$i]}

    echo "Running inference for $MODEL_NAME ..."

    python generate_vllm.py \
      --model_path "$MODEL_PATH" \
      --input_file "$DATA" \
      --remove_system True \
      --output_file "$OUTPUT_DIR/$MODEL_NAME.jsonl" \
      --template "$TEMPLATE" \
      --n 8
done

