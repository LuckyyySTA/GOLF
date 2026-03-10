#!/bin/bash -l
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"  # vllm 不需要真实的 key 时可设为 EMPTY
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:80/v1}"

# Fix CUDA multiprocessing issue
export VLLM_WORKER_MULTIPROC_METHOD=spawn

ROOT=${PROJECT_ROOT:-/path/to/your/projects}
cd ${FUZZY_EVAL_WORKDIR:-$ROOT/GOLF/eval_scripts}
# Optional: cp -r ${NLTK_DATA_SRC} /root/nltk_data

MODELS=(
    "$ROOT/GOLF/checkpoints/your_model_dir/merged_hf_model/global_step_xxx"
)
MODEL_NAMES=(
    "your_model_name_global_step_xxx"
)
OUTPUT_ROOT="${FUZZY_OUTPUT_ROOT:-$ROOT/GOLF/results/fuzzy}"

BENCHMARKS=(
    # "creativewritingv3"
    # "alpacaeval2"
    # "wildbench"
    # "arena_hard_v1"
    # "arena_hard_v2"
    # "ifeval"
)

# Sampling parameters
TEMPERATURE=0.7
TOP_P=0.95

for BENCHMARK in "${BENCHMARKS[@]}"; do
for i in "${!MODELS[@]}"; do

# Get model path and corresponding name
MODEL="${MODELS[$i]}"
MODEL_NAME="${MODEL_NAMES[$i]}"

# Check if already completed (both prediction and evaluation)
OUTPUT_DIR="${OUTPUT_ROOT}/${BENCHMARK}-compare/${MODEL_NAME}"
if ls "${OUTPUT_DIR}"/*${BENCHMARK}*.json > /dev/null 2>&1; then
    if ls "${OUTPUT_DIR}"/*${BENCHMARK}*.json.score > /dev/null 2>&1; then
        echo "⏭  Skipping ${BENCHMARK} - already completed (prediction + evaluation)"
        continue
    else
        echo "⚠️  Found predictions but no evaluation scores, will re-run"
    fi
fi
mkdir -p ${OUTPUT_DIR}

echo "=================================================="
echo "🚀 Running: ${BENCHMARK}"
echo "   Model: ${MODEL_NAME}"
echo "=================================================="


# Check if we have a longcot config file
# First place to look is inside the model directory
if [ -f "${MODEL}/longcot_config.json" ]; then
    LONGCOT_CONFIG="${MODEL}/longcot_config.json"
    EXTRA_ARGS="--longcot_config ${LONGCOT_CONFIG}"
    MAX_TOKENS=8192 # allow for longcot
    echo "🔍 Using longcot config: ${LONGCOT_CONFIG}"
elif [ -f "outputs/model_configs/${MODEL_NAME}.json" ]; then
    LONGCOT_CONFIG="outputs/model_configs/${MODEL_NAME}.json"
    EXTRA_ARGS="--longcot_config ${LONGCOT_CONFIG}"
    MAX_TOKENS=8192 # allow for longcot
    echo "🔍 Using longcot config: ${LONGCOT_CONFIG}"
else
    EXTRA_ARGS=""
    MAX_TOKENS=4096
fi

 
if [ "$BENCHMARK" == "arena_hard_v1" ] || [ "$BENCHMARK" == "arena_hard_v2" ] ; then
    MAX_TOKENS=8192
fi

# Set max model length
MAX_MODEL_LEN=$((4096 + MAX_TOKENS))

# Enable parallel evaluation for benchmarks that use external APIs (GPT-4 as judge)
if [ "$BENCHMARK" == "creativewritingv3" ] || [ "$BENCHMARK" == "alpacaeval2" ] || [ "$BENCHMARK" == "wildbench" ] || [ "$BENCHMARK" == "wildbench_newref" ] || [ "$BENCHMARK" == "arena_hard_v2" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --parallel_eval"
    echo "📡 Parallel evaluation enabled (requires internet access for API calls)"
fi

# Run the benchmark
CUDA_VISIBLE_DEVICES=0 python run_benchmarks_sampling.py \
    --benchmark ${BENCHMARK} \
    --model vllm/$MODEL \
    --output_dir ${OUTPUT_DIR} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --max_tokens ${MAX_TOKENS} \
    --max_model_len ${MAX_MODEL_LEN} \
    ${EXTRA_ARGS} \
    --force_overwrite

# Check result
if [ $? -eq 0 ]; then
    echo "✅ Successfully completed: ${BENCHMARK}"
else
    echo "❌ Failed: ${BENCHMARK}"
    exit 1
fi

echo ""

done
done

echo "🎉 All benchmarks completed!"

