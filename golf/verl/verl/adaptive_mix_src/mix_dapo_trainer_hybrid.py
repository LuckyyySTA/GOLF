# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
Supports hybrid training mode: on-policy GRPO + critique-based off-policy refinement.
"""
import os
import uuid
import math
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Dict, List, Optional
import random
import json

import numpy as np
import torch
from tqdm import tqdm

from omegaconf import OmegaConf, open_dict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.utils.torch_functional import get_response_mask
from verl.utils.reward_score.feedback.code import extract_code
from verl.adaptive_mix_src.reward_manager import _CODE_SOURCES
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    apply_kl_penalty,
    Role,
    ResourcePoolManager,
    WorkerType,
    compute_response_mask,
    StatefulDataLoader,
    ValidationGenerationsLogger,
)
from verl.utils.profiler import marked_timer

import logging
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'INFO'))


def compute_advantage(
    data: DataProto,
    adv_estimator,
    gamma=1.0,
    lam=1.0,
    norm_adv_by_std_in_grpo=True,
    config: Optional[AlgoConfig] = None,
):
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)

    if adv_estimator == 'grpo':
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise ValueError(f"Unsupported adv_estimator: '{adv_estimator}'. Only 'grpo' is supported.")

    return data



class MIXRayDAPOTrainer(RayPPOTrainer):
    """
    Hybrid GRPO trainer with critique-based off-policy refinement.
    Runs on the driver process; all compute is dispatched to Ray workers via RPC.
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        reward_fn=None,
        val_reward_fn=None,
        device_name=None,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only hybrid engine is supported'
        assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device

        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        # Critic is only used with GAE; GRPO does not need it
        if config.critic.enable is not None:
            self.use_critic = bool(config.critic.enable)
        elif self.config.algorithm.adv_estimator == 'gae':
            self.use_critic = True
        else:
            self.use_critic = False

        self.reward_manager = self.config.reward_model.get("reward_manager", "naive")

        self.use_critique_refiner = self.config.actor_rollout_ref.rollout.get('use_critique_refiner', False)
        self.critique_refiner = None

        self._create_dataloader()

    # ─────────────────────────────────────────────────────────────────────────
    # Worker initialization
    # ─────────────────────────────────────────────────────────────────────────

    def init_workers(self):
        """Create Ray resource pools and spawn worker groups."""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # Actor + Rollout (hybrid engine)
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRollout],
            config=self.config.actor_rollout_ref,
            role="actor_rollout",
        )
        self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls

        # Optional: critic
        if self.use_critic:
            from verl.trainer.ppo.ray_trainer import omega_conf_to_dataclass
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic],
                config=omega_conf_to_dataclass(self.config.critic),
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # Optional: reference policy
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # Spawn all worker groups
        all_wg = {}
        wg_kwargs = {"device_name": self.device_name}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            assert OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.trainer, "worker_nsight_options")
            )

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            all_wg.update(wg_dict.spawn(prefix_set=class_dict.keys()))

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        # Actor/rollout last so vLLM can estimate KV-cache memory accurately
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

        # Initialize CritiqueRefiner now that the actor worker group is ready
        if self.use_critique_refiner:
            from .critique_refiner import CritiqueRefiner
            critique_config = self.config.actor_rollout_ref.rollout.get('critique_refiner', {})
            self.critique_refiner = CritiqueRefiner(
                actor_rollout_wg=self.actor_rollout_wg,
                tokenizer=self.tokenizer,
                config=self.config,
                critique_key=critique_config.get('critique_key', 'critique'),
                max_refine_length=critique_config.get('max_refine_length', None),
                refine_temperature=critique_config.get('refine_temperature', 0.7),
                refine_top_p=critique_config.get('refine_top_p', 0.9),
                refine_type=critique_config.get('refine_type', 'aggregated'),
                reward_fn=self.reward_fn,
            )
            logger.info(
                f"CritiqueRefiner initialized: refine_type={critique_config.get('refine_type', 'aggregated')}"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Data
    # ─────────────────────────────────────────────────────────────────────────

    def _create_dataloader(self):
        from torch.utils.data import SequentialSampler
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        from .rl_dataset_with_target import RLHFDatasetWithTarget

        self.train_dataset = RLHFDatasetWithTarget(
            data_files=self.config.data.train_files,
            tokenizer=self.tokenizer,
            config=self.config.data,
            max_target_length=self.config.actor_rollout_ref.rollout.max_prefix_len,
        )

        if self.config.data.shuffle:
            from verl.adaptive_mix_src.rl_dataset_with_target import ResumableRandomSampler
            sampler = ResumableRandomSampler(data_source=self.train_dataset)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=sampler,
        )

        self.val_dataset = RLHFDataset(
            data_files=self.config.data.val_files,
            tokenizer=self.tokenizer,
            config=self.config.data,
        )
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=len(self.val_dataset),
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"
        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, "
            f"Size of val dataloader: {len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Logging helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _dump_generations(self, inputs, outputs, gts, scores, data_source, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
            "data_source": data_source,
        }
        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = [json.dumps({k: v[i] for k, v in base_data.items()}, ensure_ascii=False) for i in range(n)]
        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Dumped generations to {filename}")

    # ─────────────────────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────────────────────

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        sample_inputs, sample_outputs, sample_gts, sample_scores = [], [], [], []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            input_ids = test_batch.batch["input_ids"]
            sample_inputs.extend(self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids)
            sample_gts.extend(
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            )

            non_tensor_keys_to_pop = ["raw_prompt_ids"]
            for key in ["multi_modal_data", "raw_prompt", "tools_kwargs", "interaction_kwargs", "agent_name"]:
                if key in test_batch.non_tensor_batch:
                    non_tensor_keys_to_pop.append(key)

            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=non_tensor_keys_to_pop,
            )
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                test_gen_batch, self.actor_rollout_wg.world_size
            )
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print("validation generation end")

            output_ids = test_output_gen_batch.batch["responses"]
            sample_outputs.extend(self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)

            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)
            reward_tensor_lst.append(reward_tensor)
            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(
                test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0])
            )

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                data_source=data_source_lst[0].tolist(),
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()
        data_sources = np.concatenate(data_source_lst, axis=0)

        data_source_reward: dict = {}
        for i in range(reward_tensor.shape[0]):
            src = data_sources[i]
            data_source_reward.setdefault(src, []).append(reward_tensor[i].item())

        return {f'val/test_score/{src}': np.mean(rewards) for src, rewards in data_source_reward.items()}

    # ─────────────────────────────────────────────────────────────────────────
    # Main training loop
    # ─────────────────────────────────────────────────────────────────────────

    def fit(self):
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0

        self._load_checkpoint()

        # Initial validation
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.trainer.profile_steps
            if self.config.trainer.profile_steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None
        num_gen_batches = 0
        self.teacher_source_tracker = defaultdict(int)

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.trainer.profile_continuous_steps
                        else curr_step_profile
                    )

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1

                # Separate generation inputs from metadata
                pop_batch_keys = ["input_ids", "attention_mask", "position_ids", "tgt_input_ids"]
                pop_non_tensor_keys = ["raw_prompt_ids", "extra_info"]
                if "multi_modal_data" in new_batch.non_tensor_batch:
                    pop_non_tensor_keys.append("multi_modal_data")
                gen_batch = new_batch.pop(
                    batch_keys=pop_batch_keys,
                    non_tensor_batch_keys=pop_non_tensor_keys,
                )
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.gen_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):

                    # ── Step 1: On-policy generation ──────────────────────────
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    new_batch = new_batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                    )
                    new_batch = new_batch.union(gen_batch_output)

                    # ── Step 2: Reward computation & Feedback collection────────────────────────────
                    with marked_timer("reward", timing_raw, "yellow"):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        reward_result = self.reward_fn(new_batch, return_dict=True)
                        reward_tensor = reward_result["reward_tensor"]
                        reward_extra_infos_dict = reward_result.get("reward_extra_info", {})

                        new_batch.batch["token_level_scores"] = reward_tensor
                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # ── Step 3 (Hybrid): Expand batch to 2n ──────────────
                        train_refiner_mode = self.config.actor_rollout_ref.rollout.get('train_refiner_mode', False)
                        if train_refiner_mode and self.use_critique_refiner and self.critique_refiner is not None:
                            print("Hybrid training mode: expanding batch to 2n (initial + refiner rollouts)")

                            with marked_timer("refiner_rollout", timing_raw, "purple"):
                                refiner_result = self._rollout_with_refiner_prompts(
                                    gen_batch=gen_batch,
                                    new_batch=new_batch,
                                )
                                refiner_gen_batch_output = refiner_result['refiner_gen_batch_output']
                                all_correct_mask = refiner_result['all_correct_mask']

                            if refiner_gen_batch_output is not None:
                                # Refiner rollouts get a "_refiner" uid suffix so advantages are computed
                                # independently from the original rollouts
                                refiner_uids = np.array(
                                    [uid + "_refiner" for uid in new_batch.non_tensor_batch['uid']],
                                    dtype=object,
                                )
                                refiner_new_batch = deepcopy(new_batch)

                                for key in ['prompts', 'responses', 'input_ids', 'attention_mask', 'position_ids']:
                                    if key in refiner_new_batch.batch.keys():
                                        refiner_new_batch.batch.pop(key)

                                refiner_new_batch = refiner_new_batch.union(refiner_gen_batch_output)
                                refiner_new_batch.non_tensor_batch['uid'] = refiner_uids

                                with marked_timer("refiner_reward", timing_raw, "purple"):
                                    if self.use_rm:
                                        refiner_rm_out = self.rm_wg.compute_rm_score(refiner_new_batch)
                                        refiner_new_batch = refiner_new_batch.union(refiner_rm_out)

                                    refiner_reward_result = self.reward_fn(refiner_new_batch, return_dict=True)
                                    refiner_reward_tensor = refiner_reward_result["reward_tensor"]

                                    raw_refiner_score = refiner_reward_tensor.sum(-1).mean().item()
                                    print(f"Refiner raw score (before zeroing): {raw_refiner_score:.4f}")

                                    # Zero rewards for all-correct prompts (no learning signal)
                                    # print(self.tokenizer.decode(refiner_new_batch.batch['responses'][0], skip_special_tokens=True))
                                    # print(refiner_reward_result['reward_extra_info']['critique'][0])

                                    need_refine_score = []
                                    if all_correct_mask is not None:
                                        n = self.config.actor_rollout_ref.rollout.n
                                        attn = refiner_new_batch.batch['attention_mask']
                                        prompt_len = refiner_new_batch.batch['prompts'].shape[-1]
                                        for prompt_idx, is_all_correct in enumerate(all_correct_mask):
                                            start_idx = prompt_idx * n
                                            end_idx = (prompt_idx + 1) * n
                                            if is_all_correct:
                                                for s_idx in range(start_idx, end_idx):
                                                    valid_resp_len = attn[s_idx, prompt_len:].sum().item()
                                                    refiner_reward_tensor[s_idx, valid_resp_len - 1] = torch.tensor(0.0)
                                            else:
                                                need_refine_score.extend(
                                                    refiner_reward_tensor[start_idx:end_idx].sum(-1).tolist()
                                                )
                                        all_correct_count = sum(all_correct_mask)
                                        if all_correct_count > 0:
                                            print(
                                                f"Zeroed rewards for {all_correct_count} all-correct prompts "
                                                f"({all_correct_count * n} samples)"
                                            )
                                    metrics['hybrid/actual_refiner_score'] = np.mean(need_refine_score) if need_refine_score else 0.0
                                    refiner_new_batch.batch["token_level_scores"] = refiner_reward_tensor

                                # Concatenate original (n) + refiner (n) → 2n
                                original_batch_size = len(new_batch)
                                new_batch.meta_info['hybrid_mode'] = True
                                new_batch.meta_info['original_batch_size'] = original_batch_size

                                concatenated_batch = {
                                    key: torch.cat([new_batch.batch[key], refiner_new_batch.batch[key]], dim=0)
                                    for key in new_batch.batch.keys()
                                }
                                new_batch.batch = TensorDict(
                                    concatenated_batch,
                                    batch_size=torch.Size([next(iter(concatenated_batch.values())).shape[0]]),
                                )
                                for key, value in new_batch.non_tensor_batch.items():
                                    if isinstance(value, np.ndarray) and key in refiner_new_batch.non_tensor_batch:
                                        new_batch.non_tensor_batch[key] = np.concatenate(
                                            [value, refiner_new_batch.non_tensor_batch[key]]
                                        )

                                reward_tensor = new_batch.batch["token_level_scores"]
                                print(
                                    f"Hybrid training: batch {original_batch_size} → {len(new_batch)}, "
                                    f"initial mean score: {reward_tensor[:original_batch_size].sum(-1).mean():.3f}, "
                                    f"refiner mean score: {reward_tensor[original_batch_size:].sum(-1).mean():.3f}"
                                )
                            else:
                                logger.warning("Refiner rollout failed, falling back to original batch only")
                                train_refiner_mode = False

                        # ── Step 4: Off-policy injection ──────────────────────
                        n_prefix = self.config.actor_rollout_ref.rollout.n_prefix
                        if n_prefix <= 0:
                            # No injection: add all-zero prefix_mask in-place, skip pop/union.
                            new_batch.batch['prefix_mask'] = torch.zeros_like(
                                new_batch.batch['responses'], dtype=torch.bool
                            )
                        else:
                            new_gen_batch_output = new_batch.pop(
                                batch_keys=['prompts', 'responses', 'input_ids', 'attention_mask', 'position_ids',
                                            'token_level_scores']
                            )
                            new_gen_batch_output.non_tensor_batch = new_batch.non_tensor_batch
                            new_gen_batch_output.meta_info = new_batch.meta_info
                            original_token_level_scores = new_gen_batch_output.batch['token_level_scores'].clone()

                            hybrid_mode = new_batch.meta_info.get('hybrid_mode', False)
                            if hybrid_mode:
                                orig_bs = new_batch.meta_info.get('original_batch_size', len(original_token_level_scores))
                                original_sequence_scores = original_token_level_scores[:orig_bs].sum(-1)
                            else:
                                original_sequence_scores = original_token_level_scores.sum(-1)
                            metrics['critic/original_score/mean'] = torch.mean(original_sequence_scores).item()

                            new_gen_batch_output = self.mix_off_policy_rollout(new_gen_batch_output)
                            injected_sequence_scores = new_gen_batch_output.batch['token_level_scores'].sum(-1)

                            if hybrid_mode:
                                score_delta = injected_sequence_scores[:orig_bs] - original_sequence_scores
                            else:
                                score_delta = injected_sequence_scores - original_sequence_scores
                            metrics['critic/score_delta/mean'] = torch.mean(score_delta).item()

                            new_batch = new_batch.union(new_gen_batch_output)

                        if 'prefix_ratios' in new_batch.meta_info:
                            metrics['batch/avg_prefix_ratio'] = float(np.mean(new_batch.meta_info['prefix_ratios']))

                        if self.teacher_source_tracker:
                            total_injections = sum(self.teacher_source_tracker.values())
                            metrics['teacher_usage/total_inject_num'] = total_injections
                            for src, cnt in self.teacher_source_tracker.items():
                                metrics[f'teacher_usage/{src}'] = cnt / total_injections

                        # Hybrid-specific metrics
                        if train_refiner_mode:
                            uids = new_batch.non_tensor_batch['uid']
                            is_refiner = np.array(['_refiner' in str(uid) for uid in uids])
                            is_initial = ~is_refiner
                            if is_initial.any():
                                initial_scores = reward_tensor[is_initial].sum(-1)
                                metrics['hybrid/initial_score_mean'] = torch.mean(initial_scores).item()
                                metrics['hybrid/initial_score_max'] = torch.max(initial_scores).item()
                            if is_refiner.any():
                                refiner_scores = reward_tensor[is_refiner].sum(-1)
                                metrics['hybrid/refiner_score_mean'] = torch.mean(refiner_scores).item()
                                metrics['hybrid/refiner_score_max'] = torch.max(refiner_scores).item()
                            metrics['hybrid/batch_size'] = len(new_batch)
                            metrics['hybrid/initial_count'] = int(is_initial.sum())
                            metrics['hybrid/refiner_count'] = int(is_refiner.sum())

                        if 'injection_stats' in new_batch.meta_info:
                            stats = new_batch.meta_info['injection_stats']
                            metrics.update({
                                'injection/total_samples': stats['total_samples'],
                                'injection/injected_samples': stats['injected_samples'],
                                'injection/injection_rate': stats['injection_rate'],
                                'injection/actual_injected_count': stats['actual_injected_count'],
                                'injection/actual_injection_rate': stats['actual_injection_rate'],
                                'injection/success_rate': 1.0 - stats['injection_rate'],
                            })
                            for key in ['avg_effective_targets', 'max_effective_targets', 'min_effective_targets',
                                        'critique_reward']:
                                if key in stats:
                                    metrics[f'batch/{key}' if 'targets' in key else f'injection/{key}'] = stats[key]

                        # ── Rejection sampling ────────────────────────────────
                        uids = new_batch.non_tensor_batch['uid']
                        unique_uids = np.unique(uids)
                        valid_mask = torch.ones(len(uids), dtype=torch.bool)
                        fail_value = 0
                        success_value = 1

                        solve_none = solve_all = 0
                        for uid in unique_uids:
                            uid_mask = uids == uid
                            uid_rewards = reward_tensor[uid_mask].sum(-1)
                            if (uid_rewards == fail_value).all():
                                valid_mask[uid_mask] = False
                                solve_none += 1
                            elif (uid_rewards == success_value).all():
                                valid_mask[uid_mask] = False
                                solve_all += 1

                        if self.config.trainer.skip_valid_mask:
                            valid_mask[:] = True

                        metrics['batch/solve_none'] = solve_none
                        metrics['batch/solve_all'] = solve_all
                        metrics['batch/solved'] = (reward_tensor.sum(-1) == success_value).sum().item() / len(uids)
                        metrics['batch/failed'] = (reward_tensor.sum(-1) == fail_value).sum().item() / len(uids)

                        prefix_mask = new_batch.batch['prefix_mask']
                        off_policy_mask = prefix_mask.any(-1)
                        on_policy_mask = ~off_policy_mask
                        metrics['batch/on_solved'] = (
                            (reward_tensor[on_policy_mask].sum(-1) == success_value).sum().item()
                            / (on_policy_mask.sum().item() + 1e-6)
                        )
                        metrics['batch/off_solved'] = (
                            (reward_tensor[off_policy_mask].sum(-1) == success_value).sum().item()
                            / (off_policy_mask.sum().item() + 1e-6)
                        )

                        # KL penalty (optional)
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    batch = new_batch

                    # ── Step 5: Update ────────────────────────────────────────
                    batch.batch["response_mask"] = compute_response_mask(batch)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        entropy_agg = agg_loss(
                            loss_mat=entropys,
                            loss_mask=batch.batch["response_mask"],
                            loss_agg_mode=self.config.actor_rollout_ref.actor.loss_agg_mode,
                        )
                        metrics["actor/entropy"] = entropy_agg.item()
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw, "olive"):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, "brown"):
                        norm_adv = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            norm_adv_by_std_in_grpo=norm_adv,
                            config=self.config.algorithm,
                        )

                        # Scale advantages by prefix weight (on-policy vs off-policy tokens)
                        prefix_mask = batch.batch['prefix_mask']
                        advantages = batch.batch['advantages']
                        alpha = self.config.actor_rollout_ref.rollout.prefix_reward_weight_alpha
                        beta = self.config.actor_rollout_ref.rollout.prefix_reward_weight_beta
                        prefix_weight = prefix_mask.float() * alpha + (~prefix_mask).float() * beta
                        batch.batch['advantages'] = prefix_weight * advantages

                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, "green"):
                            val_metrics = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        if 'avg_score' not in val_metrics:
                            val_metrics['avg_score'] = np.mean(
                                [v for k, v in val_metrics.items() if k.startswith('val/test_score/')]
                            )
                        metrics.update(val_metrics)
                        self.maybe_save_best_hf(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw, "green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.trainer.profile_steps
                        if self.config.trainer.profile_steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.trainer.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(compute_throughout_metrics(
                    batch=batch, timing_raw=timing_raw, n_gpus=self.resource_pool_manager.get_n_gpus()
                ))
                timing_raw = defaultdict(float)
                metrics["train/num_gen_batches"] = num_gen_batches

                batch = None
                num_gen_batches = 0

                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1

    # ─────────────────────────────────────────────────────────────────────────
    # Checkpointing
    # ─────────────────────────────────────────────────────────────────────────

    def maybe_save_best_hf(self, val_metrics: dict):
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'best', 'actor')
        os.makedirs(actor_local_path, exist_ok=True)

        best_score = -float('inf')
        metrics_file = f'{actor_local_path}/metrics.json'
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                best_score = json.load(f)['best_avg_score']
        else:
            print('No current best checkpoint found. Best score set to -inf.')

        cur_score = val_metrics['avg_score']
        if cur_score > best_score:
            print(f'Saving best checkpoint with score {cur_score} at {actor_local_path}')
            self.actor_rollout_wg.save_checkpoint_hf(actor_local_path)
            with open(metrics_file, 'w') as f:
                f.write(json.dumps({'best_avg_score': cur_score, 'global_step': self.global_steps}) + '\n')

    # ─────────────────────────────────────────────────────────────────────────
    # Hybrid training: refiner rollout generation
    # ─────────────────────────────────────────────────────────────────────────

    def _rollout_with_refiner_prompts(
        self,
        gen_batch: DataProto,
        new_batch: DataProto,
    ) -> Optional[dict]:
        """
        Generate n refiner rollouts per prompt using aggregated critique prompts.

        Steps:
          1. Build one aggregated prompt per original prompt (all n rollouts + critiques).
          2. Repeat each prompt n times and generate n refined responses.
          3. Return refiner outputs + all_correct_mask for reward zeroing.

        Args:
            gen_batch:  prompt-only batch (input_ids / extra_info / meta_info).
            new_batch:  full batch after on-policy generation + reward computation
                        (responses, token_level_scores, critique, uid, …).
        """
        if not self.use_critique_refiner or self.critique_refiner is None:
            logger.error("Refiner training mode requires critique_refiner to be enabled")
            return None

        n = self.config.actor_rollout_ref.rollout.n
        original_batch_size = new_batch.batch['responses'].size(0) // n
        logger.info(f"Using initial rollouts: {original_batch_size} prompts × {n} rollouts = {original_batch_size * n}")

        critique_type = self.config.actor_rollout_ref.rollout.critique_refiner.get('critique_type', 'simple')
        refiner_prompt_data = self.critique_refiner._extract_aggregated_refiner_prompts(
            gen_batch=gen_batch,
            gen_batch_out=new_batch,
            critique_type=critique_type,
        )

        if not refiner_prompt_data or len(refiner_prompt_data['prompts']) == 0:
            logger.warning("No refiner prompts generated, cannot proceed with refiner rollout")
            return None

        aggregated_prompts = refiner_prompt_data['prompts']
        all_correct_mask = refiner_prompt_data.get('all_correct_mask', [False] * len(aggregated_prompts))
        logger.info(f"Generated {len(aggregated_prompts)} aggregated refiner prompts")

        messages_list = [
            [{"role": "user", "content": p}]
            for p in aggregated_prompts
        ]

        refiner_prompts_with_template = [
            self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, enable_thinking=True
            )
            for messages in messages_list
        ]

        refiner_prompt_inputs, attention_masks, position_ids = self.critique_refiner._tokenize_texts(
            refiner_prompts_with_template,
            max_length=self.config.data.max_prompt_length,
        )

        refiner_gen_batch = DataProto.from_single_dict({
            'input_ids': refiner_prompt_inputs,
            'attention_mask': attention_masks,
            'position_ids': position_ids,
        })
        refiner_gen_batch.meta_info = gen_batch.meta_info.copy()
        refiner_gen_batch = refiner_gen_batch.repeat(repeat_times=n, interleave=True)

        logger.info(f"Generating {original_batch_size * n} refiner rollouts (n={n} per prompt)...")
        refiner_gen_batch_output = self.actor_rollout_wg.generate_sequences(refiner_gen_batch)
        logger.info(f"Refiner rollout complete: {original_batch_size} × {n} = {original_batch_size * n} samples")

        return {
            'refiner_gen_batch_output': refiner_gen_batch_output,
            'all_correct_mask': all_correct_mask,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Off-policy injection
    # ─────────────────────────────────────────────────────────────────────────

    def _build_code_only_response(self, response_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """
        For a code task: decode the refiner response, extract the longest ```python``` block,
        re-tokenize it as "```python\\n{code}\\n```" and return a new response tensor of the
        same fixed length.  Returns None if no code block is found.
        """
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id
        max_len = response_tensor.shape[-1]

        valid_ids = response_tensor[response_tensor != pad_id]
        text = self.tokenizer.decode(valid_ids, skip_special_tokens=True)

        code = extract_code(text)
        if code is None:
            return None

        new_text = f"```python\n{code.strip()}\n```"
        new_ids = self.tokenizer.encode(new_text, add_special_tokens=False)
        if not new_ids or new_ids[-1] != eos_id:
            new_ids.append(eos_id)
        new_ids = new_ids[:max_len]

        pad_len = max_len - len(new_ids)
        return torch.tensor(
            new_ids + [pad_id] * pad_len,
            dtype=response_tensor.dtype,
            device=response_tensor.device,
        )

    def mix_off_policy_rollout(self, gen_batch_out: DataProto):
        """
        Replace failing on-policy rollouts with successful refiner (off-policy) responses.

        The off-policy targets are the refiner rollouts in the second half of the 2n
        batch.  Only refiner samples with score == 1.0 are eligible as injection
        targets.  The original prompt is preserved; only the response is replaced.
        """
        prompts = gen_batch_out.batch['prompts']
        responses = gen_batch_out.batch['responses']
        input_ids = gen_batch_out.batch['input_ids']
        attention_mask = gen_batch_out.batch['attention_mask']
        position_ids = gen_batch_out.batch['position_ids']
        reward_tensor = gen_batch_out.batch['token_level_scores']
        data_source = gen_batch_out.non_tensor_batch['data_source']

        batch_size = responses.size(0)
        prefix_mask = torch.zeros_like(responses, dtype=torch.bool)

        n = self.config.actor_rollout_ref.rollout.n
        n_prefix = self.config.actor_rollout_ref.rollout.n_prefix
        hybrid_mode = gen_batch_out.meta_info.get('hybrid_mode', False)

        if n_prefix <= 0 or not hybrid_mode:
            gen_batch_out.batch['prefix_mask'] = prefix_mask
            return gen_batch_out

        num_prompts = gen_batch_out.meta_info.get('original_batch_size') // n
        refiner_start = num_prompts * n  # second half of 2n batch

        injection_strategy = self.config.actor_rollout_ref.rollout.get('injection_strategy', 'adaptive')
        success_threshold = self.config.actor_rollout_ref.rollout.get('success_threshold', 1.0 / n)
        success_threshold_count = math.ceil(success_threshold * n)

        needs_injection_count = 0
        actual_injected_count = 0

        for i in range(num_prompts):
            init_start = i * n
            ref_start = refiner_start + i * n

            # Classify initial rollout positions for this prompt
            init_scores = reward_tensor[init_start:init_start + n].sum(-1)
            incorrect_positions = (init_scores != 1.0).nonzero(as_tuple=True)[0].tolist()
            if not incorrect_positions:
                continue

            # Decide whether to inject based on strategy
            correct_count = n - len(incorrect_positions)
            sample_success = correct_count >= success_threshold_count
            if injection_strategy == 'adaptive':
                needs_injection = not sample_success
            elif injection_strategy == 'hybrid':
                needs_injection = (not sample_success) or (random.random() < 0.3)
            else:  # 'always'
                needs_injection = True

            if not needs_injection:
                continue
            needs_injection_count += 1

            # Eligible targets: refiner samples with score == 1.0
            ref_scores = reward_tensor[ref_start:ref_start + n].sum(-1)
            correct_refiner = (ref_scores == 1.0).nonzero(as_tuple=True)[0].tolist()
            if not correct_refiner:
                logger.info(f"Prompt {i}: no correct refiner outputs, skipping injection")
                continue
            
            # Pair wrong positions with correct refiner responses (up to n_prefix)
            n_inject = min(n_prefix, len(incorrect_positions), len(correct_refiner))
            selected_wrong = random.sample(incorrect_positions, n_inject)
            selected_refiner = random.sample(correct_refiner, n_inject)

            orig_prompt_len = prompts[init_start].shape[-1]
            for wrong_pos, ref_local in zip(selected_wrong, selected_refiner):
                global_idx = init_start + wrong_pos
                ref_global = ref_start + ref_local
                refined_response = responses[ref_global]

                # For code tasks: strip thinking, keep only the ```python``` block.
                # This reduces distribution shift when the refiner produces reasoning
                # text before the code, but the original prompt expects bare code output.
                if data_source[global_idx] in _CODE_SOURCES:
                    code_only = self._build_code_only_response(refined_response)
                    if code_only is None:
                        logger.info(
                            f"Prompt {i}, pos {wrong_pos}: refiner response has no code block, skipping injection"
                        )
                        continue
                    refined_response = code_only

                # Replace response; keep original prompt, recompute masks / position ids
                prompt_attn = attention_mask[global_idx, :orig_prompt_len]
                response_attn = get_response_mask(
                    response_id=refined_response.unsqueeze(0),
                    eos_token=self.tokenizer.eos_token_id,
                    dtype=prompt_attn.dtype,
                ).squeeze(0)
                prompt_pos = position_ids[global_idx, :orig_prompt_len]
                delta = torch.arange(1, refined_response.shape[-1] + 1, device=refined_response.device)

                responses[global_idx] = refined_response
                input_ids[global_idx] = torch.cat([prompts[global_idx], refined_response], dim=-1)
                attention_mask[global_idx] = torch.cat([prompt_attn, response_attn], dim=-1)
                position_ids[global_idx] = torch.cat([prompt_pos, prompt_pos[-1] + delta], dim=-1)
                reward_tensor[global_idx] = reward_tensor[ref_global]
                data_source[global_idx] = 'critique_refined'

                prefix_mask[global_idx, :response_attn.sum().item()] = True
                self.teacher_source_tracker['critique_refined'] += 1
                actual_injected_count += 1

        logger.info(
            f"Off-policy injection: {needs_injection_count}/{num_prompts} prompts needed injection, "
            f"actual injected={actual_injected_count} samples"
        )

        injection_stats = {
            'total_samples': num_prompts,
            'injected_samples': needs_injection_count,
            'injection_rate': needs_injection_count / max(num_prompts, 1),
            'actual_injected_count': actual_injected_count,
            'actual_injection_rate': actual_injected_count / max(needs_injection_count, 1),
            'critique_reward': 1.0 if actual_injected_count > 0 else 0.0,
        }

        mixed_batch = TensorDict(
            {
                'prompts': prompts,
                'responses': responses,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'prefix_mask': prefix_mask,
                'token_level_scores': reward_tensor,
            },
            batch_size=batch_size,
        )
        return DataProto(batch=mixed_batch, meta_info={'injection_stats': injection_stats})
