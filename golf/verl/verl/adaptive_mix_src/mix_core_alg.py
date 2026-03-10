import numpy as np
import torch
from collections import defaultdict
from omegaconf import DictConfig
from typing import Any, Callable, Optional

import verl.utils.torch_functional as verl_F
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.core_algos import register_policy_loss, register_adv_est, agg_loss
@register_policy_loss("sft_pure")
def compute_sft_pure_loss(log_prob, eos_mask):
    sft_losses = -log_prob
    sft_loss = verl_F.masked_mean(sft_losses, eos_mask)
    return sft_loss

@register_adv_est("grpo_split")
def compute_grpo_outcome_advantage_split(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   on_policy_mask: torch.Tensor,
                                   epsilon: float = 1e-6,
                                   norm_adv_by_std_in_grpo: bool = True):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    
    # assert 检查
    # off_idx = (~on_policy_mask).nonzero(as_tuple=True)[0]
    # for i in off_idx.tolist():
    #     # 1) off policy 的 reward 行和必须为 1
    #     assert token_level_rewards[i].sum() == 1.0
    #     pid = index[i]
    #     same = [j for j, v in enumerate(index) if v == pid]
    #     assert len(same) == 8
    #     other = [j for j in same if j != i]
    #     assert token_level_rewards[other].sum() == 0

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            # only include on-policy samples for mean and std calculation
            if on_policy_mask[i].item() is True:
                id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        # process std
        for idx in id2std:
            if id2std[idx].item() == 0:
                id2std[idx] = torch.tensor(1.0)
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = (scores[i] - id2mean[index[i]])
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores

@register_policy_loss("mixed_token_off_policy_loss")
def compute_token_on_off_policy_loss(
    old_log_prob, 
    log_prob, 
    advantages, 
    eos_mask, 
    prefix_mask, 
    target_log_probs=None,  # Renamed from target_probs for clarity (stores log probabilities)
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
):
    """
    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO.
        prefix_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    device = log_prob.device
    
    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter ε for standard PPO.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get(  # Lower bound of the ratio for dual-clip PPO.
        "clip_ratio_c", 3.0
    )

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )
    
    # off_cliprange = config.off_policy_cliprange
    off_normalize = config.off_policy_normalize
    off_max_clip = config.off_policy_max_clip if config.off_policy_max_clip != -1 else None
    off_min_clip = config.off_policy_min_clip if config.off_policy_min_clip != -1 else None
    all_max_clip = config.all_max_clip if config.all_max_clip != -1 else None
    off_policy_reshape = config.get("off_policy_reshape", "no_reshape")
    off_policy_reshape_weight = config.get("off_policy_reshape_weight", 1.0)
    off_policy_reshape_pow_exp = config.get("off_policy_reshape_pow_exp", 0.5)
    off_policy_reshape_epsilon = config.get("off_policy_reshape_epsilon", 0.01)
    on_policy_reshape = config.get("on_policy_reshape", "no_reshape")
    on_policy_reshape_weight = config.get("on_policy_reshape_weight", 1.0)
    on_policy_reshape_pow_exp = config.get("on_policy_reshape_pow_exp", 0.5)
    loss_remove_token_mean = config.get("loss_remove_token_mean", False)
    loss_remove_clip = config.get("loss_remove_clip", False)
    
    # on-policy loss
    
    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    if on_policy_reshape == "no_reshape":
        ratio = torch.exp(negative_approx_kl) # [bsz, l]
    elif on_policy_reshape == "logp":
        ratio = log_prob - old_log_prob
    elif on_policy_reshape == "p_logp":
        ratio = torch.exp(negative_approx_kl) + on_policy_reshape_weight * negative_approx_kl
    elif on_policy_reshape == "square_root":
        ratio = torch.exp(negative_approx_kl) # [bsz, l]
        ratio = torch.sqrt(ratio)
    elif on_policy_reshape == "pow":
        ratio = torch.exp(negative_approx_kl) # [bsz, l]
        ratio = torch.pow(ratio, on_policy_reshape_pow_exp)
    elif on_policy_reshape == "p_div_p_0.1":
        prob = torch.exp(log_prob)
        old_prob = torch.exp(old_log_prob)
        f_prob = prob / (prob + 0.1)
        f_old_prob = old_prob / (old_prob + 0.1)
        ratio = f_prob / f_old_prob
    elif on_policy_reshape == "p_div_p_0.5":
        prob = torch.exp(log_prob)
        old_prob = torch.exp(old_log_prob)
        f_prob = prob / (prob + 0.5)
        f_old_prob = old_prob / (old_prob + 0.5)
        ratio = f_prob / f_old_prob
    else:
        raise ValueError(f"Invalid on_policy_reshape: {on_policy_reshape}")

    on_pg_losses = -advantages * ratio
    
    if loss_remove_clip is False:
        if cliprange_low is None:
            cliprange_low = cliprange
        if cliprange_high is None:
            cliprange_high = cliprange
        
        on_pg_losses2 = -advantages * torch.clamp(
            ratio, 1 - cliprange_low, 1 + cliprange_high
        )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
        clip_on_pg_losses1 = torch.maximum(
            on_pg_losses, on_pg_losses2
        )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
        on_pg_clipfrac = verl_F.masked_mean(torch.gt(on_pg_losses2, on_pg_losses).float(), eos_mask)

        on_pg_losses3 = -advantages * clip_ratio_c
        clip_pg_losses2 = torch.min(on_pg_losses3, clip_on_pg_losses1)
        on_pg_clipfrac_lower = verl_F.masked_mean(
            torch.gt(clip_on_pg_losses1, on_pg_losses3) * (advantages < 0).float(), eos_mask
        )

        on_pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_on_pg_losses1)
        loss_mask = (~prefix_mask) * eos_mask
        on_pg_loss = agg_loss(loss_mat=on_pg_losses, loss_mask=loss_mask, loss_agg_mode=loss_agg_mode)

    else:
        loss_mask = (~prefix_mask) * eos_mask
        on_pg_loss = agg_loss(loss_mat=on_pg_losses, loss_mask=loss_mask, loss_agg_mode=loss_agg_mode)
        # on_pg_loss = verl_F.masked_mean(on_pg_losses, (~prefix_mask) * eos_mask) #token-mean
        on_pg_clipfrac = torch.tensor(0.0)
        on_pg_clipfrac_lower = torch.tensor(0.0)
    
    if on_pg_loss.isnan().item() is True:
        on_pg_loss = torch.tensor(0.0, device=device)
    
    # compute off-policy loss
    if target_log_probs is None:
        # No target provided: use current policy probability (or its transformation)
        print(f"reshape function: {off_policy_reshape}")
        off_ratio = torch.exp(log_prob) # [bsz, l]
        if off_policy_reshape == "no_reshape":
            pass
        elif off_policy_reshape == "logp":
            off_ratio = log_prob * off_policy_reshape_weight
        elif off_policy_reshape == "p_logp":
            off_ratio = log_prob * off_policy_reshape_weight + off_ratio
        elif off_policy_reshape == "square_root":
            off_ratio = torch.sqrt(off_ratio)
        elif off_policy_reshape == "p_div_p_0.1":
            off_ratio = off_ratio / (off_ratio + 0.1)
        elif off_policy_reshape == "p_div_p_0.5":
            off_ratio = off_ratio / (off_ratio + 0.5)
        elif off_policy_reshape == "p_div_p_0.3":
            off_ratio = off_ratio / (off_ratio + 0.3)
        elif off_policy_reshape == "pow":
            off_ratio = torch.pow(off_ratio, off_policy_reshape_pow_exp)
        elif off_policy_reshape == "f_function":
            # f(t) = t - t log t  (t < 1)
            # f(t) = t - log t    (t >= 1)
            epsilon = off_policy_reshape_epsilon
            t = (off_ratio + epsilon) / (1.0 + epsilon)
            logt = torch.log(t)
            f1 = t - t * logt      # t < 1
            f2 = t - logt          # t >= 1
            off_ratio = torch.where(t < 1, f1, f2)
        else:
            raise ValueError(f"Invalid off_policy_reshape: {off_policy_reshape}")
    else:
        # Target provided: compute importance sampling ratio in log space (more stable)
        # off_ratio = p_θ_current(y|x) / p_θ_target(y|aggregated_x)
        assert target_log_probs.shape == log_prob.shape
        log_ratio = log_prob - target_log_probs  # log(p_current / p_target)
        log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)  # Clamp for stability
        off_ratio = torch.exp(log_ratio)  # p_current / p_target
        off_ratio = off_ratio * prefix_mask
        
    # clip off-policy ratio
    if off_max_clip is not None:
        off_ratio = torch.clamp(off_ratio, max=off_max_clip)
        off_ratio_max_clip_frac = verl_F.masked_mean((off_ratio == off_max_clip).float(), prefix_mask * eos_mask)
    else:
        off_ratio_max_clip_frac = torch.tensor(0.0)
        
    if off_min_clip is not None:
        off_ratio = torch.clamp(off_ratio, min=off_min_clip)
        off_ratio_min_clip_frac = verl_F.masked_mean((off_ratio == off_min_clip).float(), prefix_mask * eos_mask)
    else:
        off_ratio_min_clip_frac = torch.tensor(0.0)

    off_ratio_mean = verl_F.masked_mean(off_ratio, prefix_mask * eos_mask)
    if off_ratio_mean.isnan().any().item():
        off_ratio_mean = torch.tensor(0.0)

    off_pg_losses = -advantages * off_ratio
    
    off_pg_loss = verl_F.masked_mean(off_pg_losses, prefix_mask * eos_mask)
    if off_pg_loss.isnan().item() is True:
        off_pg_loss = torch.tensor(0.0)
    off_pg_clipfrac = torch.tensor(0.0)
    
    prefix_mask = prefix_mask.float()
    pg_losses = off_pg_losses * prefix_mask + on_pg_losses * (1 - prefix_mask)
    
    # log on/off probs
    off_policy_probs = torch.exp(log_prob)
    off_policy_prob = verl_F.masked_mean(off_policy_probs, prefix_mask * eos_mask)
    if off_policy_prob.isnan().item() is True:
        off_policy_prob = torch.tensor(0.0)
    on_policy_probs = torch.exp(old_log_prob)
    on_policy_prob = verl_F.masked_mean(on_policy_probs, (1.0-prefix_mask) * eos_mask)
    if on_policy_prob.isnan().item() is True:
        on_policy_prob = torch.tensor(0.0)
            
    if all_max_clip is not None:
        p_on = torch.exp(log_prob)
        p_on_mask = (p_on <= all_max_clip).float()
        eos_mask = eos_mask * p_on_mask
        pg_losses = pg_losses * p_on_mask
        
    if loss_remove_token_mean is True:
        pg_loss = (pg_losses * eos_mask).sum() / eos_mask.shape[-1]
        print(f'no token mean: mean normalization {eos_mask.shape[-1]}, using loss_agg_mode: seq-mean-token-sum-norm')
    else:
        # pg_loss = verl_F.masked_mean(pg_losses, eos_mask)
        pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=eos_mask, loss_agg_mode=loss_agg_mode)
        print(f'using loss_agg_mode: {loss_agg_mode}')

    return {
        "pg_loss": pg_loss,
        "off_pg_loss": off_pg_loss,
        "on_pg_loss": on_pg_loss,
        "off_pg_clipfrac": off_pg_clipfrac,
        "on_pg_clipfrac": on_pg_clipfrac,
        "on_pg_clipfrac_lower": on_pg_clipfrac_lower,
        "ppo_kl": ppo_kl,
        "off_policy_prob": off_policy_prob,
        "on_policy_prob": on_policy_prob,
        "off_ratio_mean": off_ratio_mean,
        "off_ratio_max_clip_frac": off_ratio_max_clip_frac,
        "off_ratio_min_clip_frac": off_ratio_min_clip_frac,
    }
@register_policy_loss("mixed_seq_off_policy_loss")
def compute_token_on_seq_off_policy_loss(
    old_log_prob, 
    log_prob, 
    advantages, 
    eos_mask, 
    prefix_mask, 
    target_log_probs=None,  # Renamed from target_probs for clarity (stores log probabilities)
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
):
    """
    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO.
        prefix_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    device = log_prob.device
    
    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio 
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get(
        "clip_ratio_c", 3.0
    )

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )
    
    # off_cliprange = config.off_policy_cliprange
    off_normalize = config.off_policy_normalize
    off_max_clip = config.off_policy_max_clip if config.off_policy_max_clip != -1 else None
    off_min_clip = config.off_policy_min_clip if config.off_policy_min_clip != -1 else None
    all_max_clip = config.all_max_clip if config.all_max_clip != -1 else None
    off_policy_reshape = config.get("off_policy_reshape", "no_reshape")
    off_policy_reshape_weight = config.get("off_policy_reshape_weight", 1.0)
    off_policy_reshape_pow_exp = config.get("off_policy_reshape_pow_exp", 0.5)
    off_policy_reshape_epsilon = config.get("off_policy_reshape_epsilon", 0.01)
    on_policy_reshape = config.get("on_policy_reshape", "no_reshape")
    on_policy_reshape_weight = config.get("on_policy_reshape_weight", 1.0)
    on_policy_reshape_pow_exp = config.get("on_policy_reshape_pow_exp", 0.5)
    loss_remove_token_mean = config.get("loss_remove_token_mean", False)
    loss_remove_clip = config.get("loss_remove_clip", False)
    
    # on-policy loss
    
    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    if on_policy_reshape == "no_reshape":
        ratio = torch.exp(negative_approx_kl) # [bsz, l]
    elif on_policy_reshape == "logp":
        ratio = log_prob - old_log_prob
    elif on_policy_reshape == "p_logp":
        ratio = torch.exp(negative_approx_kl) + on_policy_reshape_weight * negative_approx_kl
    elif on_policy_reshape == "square_root":
        ratio = torch.exp(negative_approx_kl) # [bsz, l]
        ratio = torch.sqrt(ratio)
    elif on_policy_reshape == "pow":
        ratio = torch.exp(negative_approx_kl) # [bsz, l]
        ratio = torch.pow(ratio, on_policy_reshape_pow_exp)
    elif on_policy_reshape == "p_div_p_0.1":
        prob = torch.exp(log_prob)
        old_prob = torch.exp(old_log_prob)
        f_prob = prob / (prob + 0.1)
        f_old_prob = old_prob / (old_prob + 0.1)
        ratio = f_prob / f_old_prob
    elif on_policy_reshape == "p_div_p_0.5":
        prob = torch.exp(log_prob)
        old_prob = torch.exp(old_log_prob)
        f_prob = prob / (prob + 0.5)
        f_old_prob = old_prob / (old_prob + 0.5)
        ratio = f_prob / f_old_prob
    else:
        raise ValueError(f"Invalid on_policy_reshape: {on_policy_reshape}")

    on_pg_losses = -advantages * ratio
    
    if loss_remove_clip is False:
        if cliprange_low is None:
            cliprange_low = cliprange
        if cliprange_high is None:
            cliprange_high = cliprange
        
        on_pg_losses2 = -advantages * torch.clamp(
            ratio, 1 - cliprange_low, 1 + cliprange_high
        )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
        clip_on_pg_losses1 = torch.maximum(
            on_pg_losses, on_pg_losses2
        )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
        on_pg_clipfrac = verl_F.masked_mean(torch.gt(on_pg_losses2, on_pg_losses).float(), eos_mask)

        on_pg_losses3 = -advantages * clip_ratio_c
        clip_pg_losses2 = torch.min(on_pg_losses3, clip_on_pg_losses1)
        on_pg_clipfrac_lower = verl_F.masked_mean(
            torch.gt(clip_on_pg_losses1, on_pg_losses3) * (advantages < 0).float(), eos_mask
        )

        on_pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_on_pg_losses1)
        loss_mask = (~prefix_mask) * eos_mask
        on_pg_loss = agg_loss(loss_mat=on_pg_losses, loss_mask=loss_mask, loss_agg_mode=loss_agg_mode)

    else:
        loss_mask = (~prefix_mask) * eos_mask
        on_pg_loss = agg_loss(loss_mat=on_pg_losses, loss_mask=loss_mask, loss_agg_mode=loss_agg_mode)
        # on_pg_loss = verl_F.masked_mean(on_pg_losses, (~prefix_mask) * eos_mask) #token-mean
        on_pg_clipfrac = torch.tensor(0.0)
        on_pg_clipfrac_lower = torch.tensor(0.0)
    
    if on_pg_loss.isnan().item() is True:
        on_pg_loss = torch.tensor(0.0, device=device)
    
    # compute off-policy loss
    off_target_losses = []
    off_target_probs = []
    all_off_ratio_stats = {
        'off_ratio_mean': [],
        'off_ratio_max_clip_frac': [],
        'off_ratio_min_clip_frac': []
    }

    combined_off_pg_losses = torch.zeros_like(on_pg_losses, device=device)
    off_policy_indices = torch.where(prefix_mask.any(dim=-1))[0].tolist()
    # print(f"off_policy_indices: {off_policy_indices}")
    effective_k = len(off_policy_indices)
    print(f"reshape function: {off_policy_reshape}")
    print(f"effective_k: {effective_k}")
    for k in range(effective_k):
        target_mask = torch.zeros_like(prefix_mask, dtype=torch.bool, device=device)
        target_indices = [idx for i, idx in enumerate(off_policy_indices) if i % effective_k == k]
        
        for sample_idx in target_indices:
            target_mask[sample_idx] = prefix_mask[sample_idx]
    
        if target_mask.any():
            # Check if target_log_probs is all zeros (dataset samples)
            if target_log_probs is not None and target_log_probs.abs().sum().item() == 0:
                print(f"⚠️ WARNING: target_log_probs is all zeros!")
                print(f"target_log_probs shape: {target_log_probs.shape}")
                print(f"target_mask sum: {target_mask.sum().item()}")
            
            if target_log_probs is None:
                # No target provided: use current policy probability (or its transformation)
                off_ratio = torch.exp(log_prob) # [bsz, l]
                if off_policy_reshape == "no_reshape":
                    pass
                elif off_policy_reshape == "logp":
                    off_ratio = log_prob * off_policy_reshape_weight
                elif off_policy_reshape == "p_logp":
                    off_ratio = log_prob * off_policy_reshape_weight + off_ratio
                elif off_policy_reshape == "square_root":
                    off_ratio = torch.sqrt(off_ratio)
                elif off_policy_reshape == "p_div_p_0.1":
                    off_ratio = off_ratio / (off_ratio + 0.1)
                elif off_policy_reshape == "p_div_p_0.5":
                    off_ratio = off_ratio / (off_ratio + 0.5)
                elif off_policy_reshape == "p_div_p_0.3":
                    off_ratio = off_ratio / (off_ratio + 0.3)
                elif off_policy_reshape == "pow":
                    off_ratio = torch.pow(off_ratio, off_policy_reshape_pow_exp)
                elif off_policy_reshape == "f_function":
                    # f(t) = t - t log t  (t < 1)
                    # f(t) = t - log t    (t >= 1)
                    epsilon = off_policy_reshape_epsilon
                    t = (off_ratio + epsilon) / (1.0 + epsilon)
                    logt = torch.log(t)
                    f1 = t - t * logt      # t < 1
                    f2 = t - logt          # t >= 1
                    off_ratio = torch.where(t < 1, f1, f2)
                else:
                    raise ValueError(f"Invalid off_policy_reshape: {off_policy_reshape}")
            else:
                # Target provided: compute importance sampling ratio in log space (more stable)
                # off_ratio = p_θ_current(y|x) / p_θ_target(y|aggregated_x)
                assert target_log_probs.shape == log_prob.shape
                log_ratio = log_prob - target_log_probs  # log(p_current / p_target)
                log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)  # Clamp for stability
                off_ratio = torch.exp(log_ratio)  # p_current / p_target
                off_ratio = off_ratio * target_mask.float()
            
            # clip off-policy ratio
            if off_max_clip is not None:
                off_ratio = torch.clamp(off_ratio, max=off_max_clip)
                target_max_clip_frac = verl_F.masked_mean((off_ratio == off_max_clip).float(), target_mask * eos_mask)
                all_off_ratio_stats['off_ratio_max_clip_frac'].append(target_max_clip_frac)
            else:
                all_off_ratio_stats['off_ratio_max_clip_frac'].append(torch.tensor(0.0, device=device))
                
            if off_min_clip is not None:
                off_ratio = torch.clamp(off_ratio, min=off_min_clip)
                target_min_clip_frac = verl_F.masked_mean((off_ratio == off_min_clip).float(), target_mask * eos_mask)
                all_off_ratio_stats['off_ratio_min_clip_frac'].append(target_min_clip_frac)
            else:
                all_off_ratio_stats['off_ratio_min_clip_frac'].append(torch.tensor(0.0, device=device))

            target_off_ratio_mean = verl_F.masked_mean(off_ratio, target_mask * eos_mask)
            if target_off_ratio_mean.isnan().any().item():
                target_off_ratio_mean = torch.tensor(0.0, device=device)
            all_off_ratio_stats['off_ratio_mean'].append(target_off_ratio_mean)

            target_off_pg_losses = -advantages * off_ratio
            target_off_pg_loss = verl_F.masked_mean(target_off_pg_losses, target_mask * eos_mask)
            
            if target_off_pg_loss.isnan().item() is True:
                target_off_pg_loss = torch.tensor(0.0, device=device)
            
            off_target_losses.append(target_off_pg_loss)

            combined_off_pg_losses = combined_off_pg_losses + target_off_pg_losses * target_mask.float()

            target_off_policy_probs = torch.exp(log_prob)
            target_off_policy_prob = verl_F.masked_mean(target_off_policy_probs, target_mask.float() * eos_mask)
            if target_off_policy_prob.isnan().item() is True:
                target_off_policy_prob = torch.tensor(0.0, device=device)
            off_target_probs.append(target_off_policy_prob)
        else:
            off_target_losses.append(torch.tensor(0.0, device=device))
            off_target_probs.append(torch.tensor(0.0, device=device))
            all_off_ratio_stats['off_ratio_mean'].append(torch.tensor(0.0, device=device))
            all_off_ratio_stats['off_ratio_max_clip_frac'].append(torch.tensor(0.0, device=device))
            all_off_ratio_stats['off_ratio_min_clip_frac'].append(torch.tensor(0.0, device=device))

    if len(off_target_losses) > 0:
        off_pg_loss = torch.stack(off_target_losses).mean()
    else:
        off_pg_loss = torch.tensor(0.0, device=device)
    
    off_ratio_mean = torch.stack(all_off_ratio_stats['off_ratio_mean']).mean() if all_off_ratio_stats['off_ratio_mean'] else torch.tensor(0.0, device=device)
    off_ratio_max_clip_frac = torch.stack(all_off_ratio_stats['off_ratio_max_clip_frac']).mean() if all_off_ratio_stats['off_ratio_max_clip_frac'] else torch.tensor(0.0, device=device)
    off_ratio_min_clip_frac = torch.stack(all_off_ratio_stats['off_ratio_min_clip_frac']).mean() if all_off_ratio_stats['off_ratio_min_clip_frac'] else torch.tensor(0.0, device=device)

    off_pg_clipfrac = torch.tensor(0.0, device=device)
    
    prefix_mask = prefix_mask.float()
    pg_losses = combined_off_pg_losses + on_pg_losses * (1 - prefix_mask)

    if len(off_target_probs) > 0:
        off_policy_prob = torch.stack(off_target_probs).mean()
    else:
        off_policy_probs = torch.exp(log_prob)
        off_policy_prob = verl_F.masked_mean(off_policy_probs, prefix_mask * eos_mask)
        if off_policy_prob.isnan().item():
            off_policy_prob = torch.tensor(0.0, device=device)
        
    on_policy_probs = torch.exp(old_log_prob)
    on_policy_prob = verl_F.masked_mean(on_policy_probs, (1.0-prefix_mask) * eos_mask)
    if on_policy_prob.isnan().item() is True:
        on_policy_prob = torch.tensor(0.0, device=device)
            
    if all_max_clip is not None:
        p_on = torch.exp(log_prob)
        p_on_mask = (p_on <= all_max_clip).float()
        eos_mask = eos_mask * p_on_mask
        pg_losses = pg_losses * p_on_mask
        
    if loss_remove_token_mean is True:
        pg_loss = (pg_losses * eos_mask).sum() / eos_mask.shape[-1]
        print(f'no token mean: mean normalization {eos_mask.shape[-1]}, using loss_agg_mode: seq-mean-token-sum-norm')
    else:
        # pg_loss = verl_F.masked_mean(pg_losses, eos_mask)
        pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=eos_mask, loss_agg_mode=loss_agg_mode)
        print(f'using loss_agg_mode: {loss_agg_mode}')
    
    result = {
        "pg_loss": pg_loss,
        "off_pg_loss": off_pg_loss,
        "on_pg_loss": on_pg_loss,
        "off_pg_clipfrac": off_pg_clipfrac,
        "on_pg_clipfrac": on_pg_clipfrac,
        "on_pg_clipfrac_lower": on_pg_clipfrac_lower,
        "ppo_kl": ppo_kl,
        "off_policy_prob": off_policy_prob,
        "on_policy_prob": on_policy_prob,
        "off_ratio_mean": off_ratio_mean,
        "off_ratio_max_clip_frac": off_ratio_max_clip_frac,
        "off_ratio_min_clip_frac": off_ratio_min_clip_frac,
    }

    for k, target_loss in enumerate(off_target_losses):
        result[f'off_policy_target_{k}_loss'] = target_loss
    for k, target_prob in enumerate(off_target_probs):  
        result[f'off_policy_target_{k}_prob'] = target_prob
    
    
    return result
