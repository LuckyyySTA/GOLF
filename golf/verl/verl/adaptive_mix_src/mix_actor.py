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
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple

import torch
import numpy as np
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import masked_mean
from verl.utils.seqlen_balancing import rearrange_micro_batches
import verl.utils.torch_functional as verl_F
from verl.workers.config import ActorConfig

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

__all__ = ['MIXDataParallelPPOActor']

from verl.workers.actor.dp_actor import DataParallelPPOActor

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class MIXDataParallelPPOActor(DataParallelPPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config: Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """
    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        super().__init__(config, actor_module, actor_optimizer)
        self.use_adaptive_temperature = self.config.get('use_adaptive_temperature', False)
        self.adaptive_temperature_target_entropy = self.config.get('adaptive_temperature_target_entropy', 1.0)
        if self.use_adaptive_temperature:
            self.log_alpha = torch.tensor(np.log(self.config.entropy_coeff), dtype=torch.float)
            self.log_alpha.requires_grad = True
            from torch import optim
            self.alpha_optimizer = optim.AdamW([self.log_alpha],
                                          lr=self.config.alpha_lr,
                                          betas=(0.9, 0.999),
                                          weight_decay=1e-2)
        else:
            self.alpha_optimizer = None
            
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
       
        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
            'prefix_mask'
        ]
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        # if self.config.use_off_policy_loss and self.config.off_policy_loss_impl == 'seq':
        #     select_keys.append('on_logprobs_mean')
        #     select_keys.append('on_logprobs_std')
        # if self.config.use_off_policy_loss and self.config.use_off_policy_probs or self.config.get('use_off_policy_kl_loss', False):
        if self.config.use_off_policy_loss and "refiner_old_log_probs" in data.batch.keys():
            select_keys.append('refiner_old_log_probs')
            
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        print(data.batch.keys())
        print(f"before resize len(mini_batches): {len(data) // self.config.ppo_mini_batch_size}")
        # ppo_mini_batch_size = len(data)
        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details.
        mini_batches = data.split(self.config.ppo_mini_batch_size)
        # mini_batches = data.split(ppo_mini_batch_size)
        print(f"after resize len(mini_batches): {len(mini_batches)}")

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(mini_batches):
                # split batch into micro_batches
                mini_batch = data
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()
                if self.alpha_optimizer is not None:
                    self.alpha_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    print("MICROBATCH STEP")
                    micro_batch = micro_batch.to(get_device_id())  # actor device is cpu when using offload
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]
                    
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode
                    
                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation
                        
                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )
                    
                    clip_ratio = self.config.clip_ratio

                    if self.config.use_sft_multitask_loss:
                        assert self.config.use_off_policy_loss is False, 'Either use off-policy loss or sft multitask loss. You cannot set both to be True.'
                        from .mix_core_alg import compute_sft_pure_loss
                        off_policy_mask = model_inputs['prefix_mask'].any(-1) # [No]
                        off_policy_logprob = log_prob[off_policy_mask]
                        off_policy_eos_mask = response_mask[off_policy_mask]
                        
                        sft_loss = compute_sft_pure_loss(log_prob=off_policy_logprob,
                                                        eos_mask=off_policy_eos_mask)
                        
                        on_policy_mask = ~off_policy_mask
                        on_policy_logprob = log_prob[on_policy_mask]
                        on_policy_old_logprob = old_log_prob[on_policy_mask]
                        
                        # assert self.config.algorithm.adv_estimator == 'grpo_split'
                        # The on-policy advantages should not be computed together with the off-policy rewards
                        on_policy_advantages = advantages[on_policy_mask]
                        on_policy_eos_mask = response_mask[on_policy_mask]
                        
                        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                        # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla
                        # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                        # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                        policy_loss_fn = get_policy_loss_fn(loss_mode)
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                            old_log_prob=on_policy_old_logprob,
                            log_prob=on_policy_logprob,
                            advantages=on_policy_advantages,
                            response_mask=on_policy_eos_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                        )
                        
                        pg_loss = sft_loss * self.config.sft_loss_coef + pg_loss

                    elif self.config.use_off_policy_loss:
                        from .mix_core_alg import compute_token_on_off_policy_loss, compute_token_on_seq_off_policy_loss
                        
                        if self.config.off_policy_loss_impl == 'seq':
                            loss_fn = compute_token_on_seq_off_policy_loss
                        elif self.config.off_policy_loss_impl == 'token':
                            loss_fn = compute_token_on_off_policy_loss
                        else:
                            raise ValueError(f"Invalid off-policy loss impl: {self.config.off_policy_loss_impl} should be one of ['seq', 'token']")

                        ret_dict = loss_fn(old_log_prob=old_log_prob, 
                            log_prob=log_prob,
                            advantages=advantages,
                            eos_mask=response_mask,
                            prefix_mask=model_inputs['prefix_mask'],
                            target_log_probs=model_inputs['refiner_old_log_probs'] if "refiner_old_log_probs" in model_inputs else None,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                        )
                        pg_loss = ret_dict['pg_loss']
                        off_pg_loss = ret_dict['off_pg_loss']
                        on_pg_loss = ret_dict['on_pg_loss']
                        off_pg_clipfrac = ret_dict['off_pg_clipfrac']
                        pg_clipfrac = ret_dict['on_pg_clipfrac']
                        pg_clipfrac_lower = ret_dict['on_pg_clipfrac_lower']
                        ppo_kl = ret_dict['ppo_kl']
                        
                        # === Off-Policy KL Regularization ===
                        # Prevent refine samples from deviating too far from their original generation distribution
                        # KL( p_θ_rollout(refine_y | aggregated_x) || p_θ_current(refine_y | original_x) )
                        off_policy_kl_loss = torch.tensor(0.0, device=log_prob.device)
                        if self.config.get('use_off_policy_kl_loss', False):
                            off_policy_kl_coef = self.config.get('off_policy_kl_coef', 0.1)
                            off_policy_mask = model_inputs['prefix_mask'].any(-1)  # [bsz]
                            
                            if off_policy_mask.any() and 'refiner_old_log_probs' in model_inputs:
                                refiner_old_log_probs = model_inputs['refiner_old_log_probs']  # p(y|aggregated_x)
                                # current log_prob is p(y|original_x)
                                
                                # Reverse KL: KL(p_rollout || p_current) = E_p_rollout[log(p_rollout/p_current)]
                                # = E_p_rollout[log p_rollout - log p_current]
                                # Approximated as: exp(refiner_old_log_probs) * (refiner_old_log_probs - log_prob)
                                kl_div = torch.exp(refiner_old_log_probs) * (refiner_old_log_probs - log_prob)  # [bsz, seq_len]
                                
                                # Only apply to off-policy samples (where refiner_old_log_probs != 0)
                                off_kl_div = kl_div[off_policy_mask]
                                off_response_mask = response_mask[off_policy_mask]
                                
                                # Only compute KL on positions where refiner_old_log_probs is non-zero
                                # (i.e., positions that were actually replaced by refiner samples)
                                refiner_mask = (refiner_old_log_probs[off_policy_mask] != 0)
                                combined_mask = off_response_mask & refiner_mask
                                
                                if combined_mask.any():
                                    off_policy_kl_loss = agg_loss(
                                        loss_mat=off_kl_div, 
                                        loss_mask=combined_mask, 
                                        loss_agg_mode=loss_agg_mode
                                    )
                                    
                                    # Add to pg_loss
                                    pg_loss = pg_loss + off_policy_kl_loss * off_policy_kl_coef
                        
                        data = {
                            'actor/on_pg_loss': on_pg_loss.detach().item(),
                            'actor/off_pg_clipfrac': off_pg_clipfrac.detach().item(),
                        }
                        if 'off_pg_loss' in ret_dict:
                            data['actor/off_pg_loss'] = off_pg_loss.detach().item()
                        for k in range(self.config.num_off_policy_targets):
                            target_key = f'off_policy_target_{k}_loss'
                            if target_key in ret_dict:
                                data[f'actor/{target_key}'] = ret_dict[target_key].detach().item()

                        if 'off_policy_prob' in ret_dict:
                            data['actor/off_policy_prob'] = ret_dict['off_policy_prob'].detach().item()
                        for k in range(self.config.num_off_policy_targets):
                            target_key = f'off_policy_target_{k}_prob'
                            if target_key in ret_dict:
                                data[f'actor/{target_key}'] = ret_dict[target_key].detach().item()
                                
                        if 'on_policy_prob' in ret_dict:
                            data['actor/on_policy_prob'] = ret_dict['on_policy_prob'].detach().item()
                        if 'off_ratio_mean' in ret_dict:
                            data['actor/off_ratio_mean'] = ret_dict['off_ratio_mean'].detach().item()
                        if 'off_ratio_max_clip_frac' in ret_dict:
                            data['actor/off_ratio_max_clip_frac'] = ret_dict['off_ratio_max_clip_frac'].detach().item()
                        if 'off_ratio_min_clip_frac' in ret_dict:
                            data['actor/off_ratio_min_clip_frac'] = ret_dict['off_ratio_min_clip_frac'].detach().item()
                        
                        # Log off-policy KL metrics
                        if self.config.get('use_off_policy_kl_loss', False):
                            data['actor/off_policy_kl_loss'] = off_policy_kl_loss.detach().item()
                            data['actor/off_policy_kl_coef'] = self.config.get('off_policy_kl_coef', 0.1)
                    
                        append_to_dict(metrics, data)
   
                    else:
                        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                        # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla
                        # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                        # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                        policy_loss_fn = get_policy_loss_fn(loss_mode)
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                        )
                        
                    # compute entropy loss from entropy
                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        entropy_loss = torch.tensor(0.0)
                        policy_loss = pg_loss

                    # compute policy loss
                    if self.config.use_adaptive_temperature:
                        if self.config.use_adaptive_temperature_fixed is False:
                            target_entropy = self.config.adaptive_temperature_target_entropy
                            entropy_coeff = self.log_alpha.exp()
                            if self.config.adaptive_temperature_clip > 0:
                                entropy_coeff = torch.clamp(entropy_coeff, max=self.config.adaptive_temperature_clip)
                            alpha_loss = verl_F.masked_mean(entropy - target_entropy, response_mask).detach() * entropy_coeff
                            alpha_loss = alpha_loss / self.gradient_accumulation
                            alpha_loss.backward()
                            
                            policy_loss = pg_loss - entropy_loss * entropy_coeff.detach().item()
                            micro_batch_metrics['actor/alpha_loss'] = alpha_loss.detach().item()
                            micro_batch_metrics['actor/entropy_coeff'] = entropy_coeff.detach().item()
                            micro_batch_metrics['actor/log_alpha'] = self.log_alpha.detach().item()
                        else: # fixed strategy for entropy coeff
                            target_entropy = self.config.adaptive_temperature_target_entropy
                            # cur_entropy = verl_F.masked_mean(entropy, response_mask)
                            entropy_coeff = (target_entropy / entropy_loss).detach().item() * self.config.entropy_coeff
                            policy_loss = pg_loss - entropy_loss * entropy_coeff
                            micro_batch_metrics['actor/entropy_coeff'] = entropy_coeff
                    else:
                        policy_loss = pg_loss - entropy_loss * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs['ref_log_prob']
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics['actor/kl_loss'] = kl_loss.detach().item()
                        micro_batch_metrics['actor/kl_coef'] = self.config.kl_loss_coef
                        
                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    loss.backward()
                    
                    micro_batch_metrics.update(
                        {
                            'actor/entropy_loss': entropy_loss.detach().item(),
                            "actor/pg_loss": pg_loss.detach().item() * loss_scale_factor,
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {'actor/grad_norm': grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        if self.alpha_optimizer is not None:
            self.alpha_optimizer.zero_grad()
        return metrics
    
    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        if self.alpha_optimizer is not None:
            self.alpha_optimizer.step()
        return grad_norm
