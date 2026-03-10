#!/usr/bin/env bash
set -xeuo pipefail

export WANDB_MODE=online
# export WANDB_API_KEY="your_key"  # or set WANDB_MODE=disabled

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export RAY_LOGGING_LEVEL=DEBUG
export HYDRA_FULL_ERROR=1

project_name='golf'
exp_name='qwen3-8b-code-critique-grpo-rollout8-hybrid-env-feedback-mix-clip-0.28-refine-final'

# Paths (set PROJECT_ROOT and MODEL_PATH for your environment)
ROOT=${PROJECT_ROOT:-/path/to/your/projects}
HOME="${ROOT}/GOLF"

SCRIPT_PATH="${BASH_SOURCE[0]}"
if [[ "$SCRIPT_PATH" != /* ]]; then
    SCRIPT_PATH="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)/$(basename "$SCRIPT_PATH")"
else
    SCRIPT_DIR=$(cd "$(dirname "$SCRIPT_PATH")" && pwd)
    SCRIPT_PATH="${SCRIPT_DIR}/$(basename "$SCRIPT_PATH")"
fi
SCRIPT_NAME=$(basename "$SCRIPT_PATH")

cd $HOME/golf/verl

MODEL_PATH=${MODEL_PATH:-/path/to/pretrained_models/Qwen3-8B}
CKPTS_DIR="${HOME}/checkpoints/qwen3-8b-no-think/${project_name}/${exp_name}"

# Code dataset (LiveCodeBench or HumanEval+)
TRAIN_FILE=${TRAIN_FILE:-$ROOT/GOLF/data/lcb_v6_train.parquet}
TEST_FILE=${TEST_FILE:-$ROOT/GOLF/data/lcb_v6_test.parquet}

ROLLOUT_DATA_DIR=${CKPTS_DIR}/rollout_data
VALIDATION_DATA_DIR=${CKPTS_DIR}/validation_data

mkdir -p ${ROLLOUT_DATA_DIR}
mkdir -p ${VALIDATION_DATA_DIR}

# Logging
LOG_DIR=${CKPTS_DIR}/logs
mkdir -p ${LOG_DIR}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=${LOG_DIR}/train_${exp_name}_${TIMESTAMP}.log

# Save script
SCRIPT_BACKUP_DIR=${CKPTS_DIR}/scripts
mkdir -p ${SCRIPT_BACKUP_DIR}
SCRIPT_BACKUP_PATH=${SCRIPT_BACKUP_DIR}/${SCRIPT_NAME%.sh}_${TIMESTAMP}.sh
if [[ -f "$SCRIPT_PATH" ]]; then
    cp "$SCRIPT_PATH" "${SCRIPT_BACKUP_PATH}"
    echo "Script saved to: ${SCRIPT_BACKUP_PATH}" | tee -a ${LOG_FILE}
else
    echo "Warning: Could not find script file at $SCRIPT_PATH to backup" | tee -a ${LOG_FILE}
fi

# Off-policy injection configuration
num_off_policy_targets=1
max_available_targets=1
use_sft_multitask_loss=False
use_off_policy_loss=True
off_policy_normalize=False
off_policy_reshape="p_div_p_0.1"
off_policy_loss_impl='seq'
loss_remove_token_mean=True
loss_remove_clip=False
injection_strategy='adaptive'
target_selection_strategy='best_k'
prefix_strategy='random'
success_threshold=0.125  # At least 1/8 correct

# ============ CODE-SPECIFIC Critique Refiner Configuration ============
use_critique_refiner=True      # Enable critique refiner
train_refiner_mode=True        # Hybrid training mode
num_targets_from_critique=1
critique_key='critique'        # Use 'feedback' field from code execution
critique_type='environment'    # ⭐ KEY: Use environment feedback (test failures, errors, etc.)
refine_type='aggregated'       # Aggregate multiple failed attempts
refine_temperature=1.0
refine_top_p=0.9
aggregate_only_wrong=True      # Only aggregate failed code attempts

# Reward configuration
reward_impl_version=6
val_reward_impl_version=6
off_policy_reward_manager='naive'
reward_manager='naive'
val_reward_manager='naive'

adv_estimator=grpo

clip_ratio_low=0.2
clip_ratio_high=0.28

# ============ ALIGNED WITH SDPO HYPERPARAMETERS ============
max_prompt_length=10240      # SDPO: 2048
max_response_length=8192    # SDPO: 8192
enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 2))
overlong_penalty_factor=1.0

loss_agg_mode="seq-mean-token-sum-norm"

train_prompt_bsz=32         # SDPO: 32
val_prompt_bsz=32
n_resp_per_prompt=8         # SDPO: 8
train_prompt_mini_bsz=8     # SDPO: 8 (ppo_mini_batch_size)

NNODES=1
NGPUS_PER_NODE=8

# Algorithm (training generation)
temperature=1.0
top_p=1.0
top_k=-1

# ============ ALIGNED WITH SDPO VALIDATION SETTINGS ============
val_temperature=0.6   # SDPO: 0.6 (not greedy, use sampling for diversity)
val_top_p=0.95        # SDPO: 0.95
val_do_sample=True    # SDPO: True
val_n=4               # SDPO: 4 (generate 4 samples for validation)

filter_groups_enabled=False
filter_groups_metric='seq_reward'
max_num_gen_batches=10

# Performance
sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))

offload=True
gen_tp=4
fsdp_size=-1

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

echo "Training started at $(date)" | tee -a ${LOG_FILE}
echo "Log file: ${LOG_FILE}" | tee -a ${LOG_FILE}
echo "========================================" | tee -a ${LOG_FILE}
echo "Code Task with Environment Feedback" | tee -a ${LOG_FILE}
echo "Using Critique Refiner: ${use_critique_refiner}" | tee -a ${LOG_FILE}
echo "Train Refiner Mode: ${train_refiner_mode}" | tee -a ${LOG_FILE}
echo "Critique Type: ${critique_type} (environment feedback)" | tee -a ${LOG_FILE}
echo "========================================" | tee -a ${LOG_FILE}

python3 -m verl.adaptive_mix_src.main_mix_dapo \
    algorithm.adv_estimator=${adv_estimator} \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='right' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.val_batch_size=${val_prompt_bsz} \
    data.shuffle=True \
    data.num_off_policy_targets=${num_off_policy_targets} \
    data.max_available_targets=${max_available_targets} \
    data.reward_impl_version=${reward_impl_version} \
    data.val_reward_impl_version=${val_reward_impl_version} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.filter_groups.enable=${filter_groups_enabled} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.override_config.max_position_embeddings=32768 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0.000 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=${val_do_sample} \
    actor_rollout_ref.rollout.val_kwargs.n=${val_n} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.rollout.max_prefix_len=${max_response_length} \
    actor_rollout_ref.rollout.prefix_share_across_samples=False \
    actor_rollout_ref.rollout.prefix_strategy=random \
    actor_rollout_ref.rollout.min_prefix_ratio=1.0 \
    actor_rollout_ref.rollout.max_prefix_ratio=1.0 \
    actor_rollout_ref.rollout.prefix_reward_weight_alpha=1.0 \
    actor_rollout_ref.ref.use_ref=False \
    actor_rollout_ref.actor.use_sft_multitask_loss=${use_sft_multitask_loss} \
    actor_rollout_ref.actor.use_off_policy_loss=${use_off_policy_loss} \
    actor_rollout_ref.actor.off_policy_normalize=${off_policy_normalize} \
    actor_rollout_ref.actor.off_policy_reshape=${off_policy_reshape} \
    actor_rollout_ref.actor.off_policy_loss_impl=${off_policy_loss_impl} \
    actor_rollout_ref.actor.loss_remove_token_mean=${loss_remove_token_mean} \
    actor_rollout_ref.actor.loss_remove_clip=${loss_remove_clip} \
    actor_rollout_ref.rollout.injection_strategy=${injection_strategy} \
    actor_rollout_ref.rollout.target_selection_strategy=${target_selection_strategy} \
    actor_rollout_ref.rollout.success_threshold=${success_threshold} \
    actor_rollout_ref.rollout.use_critique_refiner=${use_critique_refiner} \
    actor_rollout_ref.rollout.train_refiner_mode=${train_refiner_mode} \
    actor_rollout_ref.rollout.num_targets_from_critique=${num_targets_from_critique} \
    actor_rollout_ref.rollout.critique_refiner.critique_key=${critique_key} \
    actor_rollout_ref.rollout.critique_refiner.critique_type=${critique_type} \
    actor_rollout_ref.rollout.critique_refiner.refine_type=${refine_type} \
    actor_rollout_ref.rollout.critique_refiner.refine_temperature=${refine_temperature} \
    actor_rollout_ref.rollout.critique_refiner.refine_top_p=${refine_top_p} \
    actor_rollout_ref.rollout.critique_refiner.aggregate_only_wrong=${aggregate_only_wrong} \
    reward_model.reward_manager=${reward_manager} \
    reward_model.reward_manager_shaping_function_name=threshold_0 \
    reward_model.compute_score_name=mean_exp_log_softmax \
    reward_model.repetition_penalty=False \
    reward_model.off_policy_reward_manager=${off_policy_reward_manager} \
    reward_model.val_reward_manager=${val_reward_manager} \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=5 \
    trainer.save_freq=10 \
    trainer.total_epochs=30 \
    trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
    trainer.validation_data_dir="${VALIDATION_DATA_DIR}" \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.default_hdfs_dir=null \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 \
    2>&1 | tee -a ${LOG_FILE}

echo "========================================" | tee -a ${LOG_FILE}
echo "Training finished at $(date)" | tee -a ${LOG_FILE}
