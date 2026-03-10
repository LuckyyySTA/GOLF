#!/usr/bin/env bash
set -xeuo pipefail

export WANDB_MODE=online
# export WANDB_API_KEY="your_key"  # or set WANDB_MODE=disabled

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export RAY_LOGGING_LEVEL=DEBUG
export HYDRA_FULL_ERROR=1

project_name='golf'
# Hybrid training: 同时训练 generate 和 refine 能力
exp_name='llama-3.1-8b-instruct-critique-grpo-rollout8-prompt-8k-resp-4k-hybrid-training-mix-incorrect-only-threshold-0.375-wildchat-prompt-enhanced'

# Paths (set PROJECT_ROOT and MODEL_PATH for your environment)
ROOT=${PROJECT_ROOT:-/path/to/your/projects}
HOME="${ROOT}/GOLF"

# Get script path and name BEFORE cd command
SCRIPT_PATH="${BASH_SOURCE[0]}"
# Resolve to absolute path if relative
if [[ "$SCRIPT_PATH" != /* ]]; then
    SCRIPT_PATH="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)/$(basename "$SCRIPT_PATH")"
else
    SCRIPT_DIR=$(cd "$(dirname "$SCRIPT_PATH")" && pwd)
    SCRIPT_PATH="${SCRIPT_DIR}/$(basename "$SCRIPT_PATH")"
fi
SCRIPT_NAME=$(basename "$SCRIPT_PATH")

cd $HOME/golf/verl
MODEL_PATH=${MODEL_PATH:-/path/to/pretrained_models/Llama-3.1-8B-Instruct}
CKPTS_DIR="${HOME}/checkpoints/llama-3.1-8b-instruct/${project_name}/${exp_name}"

TRAIN_FILE=${TRAIN_FILE:-$ROOT/GOLF/data/wildchat-if_train_with_ref_filtered_with_target.parquet}
TEST_FILE=${TEST_FILE:-$ROOT/GOLF/data/wildchat-if_val_with_target.parquet}

ROLLOUT_DATA_DIR=${CKPTS_DIR}/rollout_data
VALIDATION_DATA_DIR=${CKPTS_DIR}/validation_data

mkdir -p ${ROLLOUT_DATA_DIR}
mkdir -p ${VALIDATION_DATA_DIR}

# Logging
LOG_DIR=${CKPTS_DIR}/logs
mkdir -p ${LOG_DIR}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=${LOG_DIR}/train_${exp_name}_${TIMESTAMP}.log

# Save current script to checkpoint directory with timestamp
SCRIPT_BACKUP_DIR=${CKPTS_DIR}/scripts
mkdir -p ${SCRIPT_BACKUP_DIR}
SCRIPT_BACKUP_PATH=${SCRIPT_BACKUP_DIR}/${SCRIPT_NAME%.sh}_${TIMESTAMP}.sh
if [[ -f "$SCRIPT_PATH" ]]; then
    cp "$SCRIPT_PATH" "${SCRIPT_BACKUP_PATH}"
    echo "Script saved to: ${SCRIPT_BACKUP_PATH}" | tee -a ${LOG_FILE}
else
    echo "Warning: Could not find script file at $SCRIPT_PATH to backup" | tee -a ${LOG_FILE}
fi

# MIX - Off-policy injection 配置
# 重要：hybrid mode 会从 refiner 样本中自动选择高分的替换初始样本
num_off_policy_targets=1  # 从 refiner 样本中选择最多 1 个高分样本替换初始样本
max_available_targets=1
use_sft_multitask_loss=False
use_off_policy_loss=True
off_policy_normalize=False
off_policy_reshape="p_div_p_0.1"
off_policy_loss_impl='seq' # token or seq
loss_remove_token_mean=True
loss_remove_clip=False
injection_strategy='adaptive' # 当初始 rollouts 全错时才从 refiner 中选择
target_selection_strategy='best_k'
prefix_strategy='random'
success_threshold=0.375  # 成功阈值 (比例 0-1): 0.125 表示至少需要 12.5% 的正确样本 (n=8 时至少 1 个), 1.0 表示需要全部正确

# Off-Policy KL Regularization (防止 refine 样本偏离当前策略太远)

# Critique Refiner Configuration - HYBRID TRAINING MODE (Batch Expansion 2n)
use_critique_refiner=True  # 启用 critique refiner
train_refiner_mode=True    # 启用 hybrid training: batch 扩充到 2n
num_targets_from_critique=1
critique_key='critique'
critique_type='simple'
refine_type='aggregated' # 聚合多个 rollouts
refine_temperature=1.0  # 增加多样性
refine_top_p=0.9
aggregate_only_wrong=True  # 是否只聚合错误的样本 (True: 只聚合错误样本, False: 聚合所有样本)

# reward
reward_impl_version=6
val_reward_impl_version=6
off_policy_reward_manager='batch' #prob/random/naive
reward_manager='batch'
val_reward_manager='batch'

adv_estimator=grpo

clip_ratio_low=0.2
clip_ratio_high=0.2

max_prompt_length=$((1024 * 8))
max_response_length=$((1024 * 4))
enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="seq-mean-token-sum-norm"

train_prompt_bsz=64
val_prompt_bsz=512
n_resp_per_prompt=8
train_prompt_mini_bsz=64

NNODES=4
NGPUS_PER_NODE=8

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_temperature=0.6
val_top_p=1.0

filter_groups_enabled=False
filter_groups_metric='seq_reward'
max_num_gen_batches=10

# Performance Related Parameter
sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))

offload=True
gen_tp=2
fsdp_size=-1

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0



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
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
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
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    reward_model.reward_manager_shaping_function_name=threshold_0 \
    reward_model.compute_score_name=mean_exp_log_softmax \
    reward_model.repetition_penalty=False \
    reward_model.off_policy_reward_manager=${off_policy_reward_manager} \
    reward_model.val_reward_manager=${val_reward_manager} \
    reward_model.format_mode=R1_nothink \
    reward_model.format_coefficient=0.0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=20000 \
    trainer.save_freq=20 \
    trainer.total_epochs=2 \
    trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
    trainer.validation_data_dir="${VALIDATION_DATA_DIR}" \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.default_hdfs_dir=null \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 \
    2>&1 | tee -a ${LOG_FILE}

# Log completion
echo "========================================" | tee -a ${LOG_FILE}
echo "Training finished at $(date)" | tee -a ${LOG_FILE}

