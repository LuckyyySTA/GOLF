"""
Reward manager for GRPO training with critique-based refinement.
Supports ifeval, ifbench, livecodebench, and wildchat-if data sources.
"""
from collections import defaultdict

from verl import DataProto
import torch
from verl.utils.reward_score import ifeval, ifbench, feedback


_CODE_SOURCES = {"code", "livecodebench", "humanevalplus"}
_IFEVAL_SOURCES = {"ifeval", "ifbench"}


def _select_rm_score_fn(data_source):
    if data_source in _IFEVAL_SOURCES:
        return ifeval.compute_score if data_source == "ifeval" else ifbench.compute_score
    elif data_source in _CODE_SOURCES:
        return feedback.compute_score
    else:
        raise NotImplementedError(f"Unsupported data_source: '{data_source}'")


class RewardManager:
    """Compute task-specific scalar rewards from model responses."""

    def __init__(self, tokenizer, num_examine, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict: bool = False):
        # Fast path: pre-computed scores (e.g., from hybrid refiner)
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            extra_info["num_turns"] = data_item.non_tensor_batch.get("__num_turns__", None)

            compute_score_fn = _select_rm_score_fn(data_source)

            if data_source in _IFEVAL_SOURCES:
                score = compute_score_fn(solution_str=response_str, ground_truth=ground_truth, extra_info=extra_info)
            elif data_source in _CODE_SOURCES:
                score = compute_score_fn(
                    data_source=data_source, solution_str=response_str, ground_truth=ground_truth, extra_info=extra_info)
                if isinstance(score, dict) and "feedback" in score:
                    score["critique"] = score["feedback"]
            else:
                raise NotImplementedError(f"Unsupported data_source: '{data_source}'")
                
            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if already_print_data_sources.get(data_source, 0) < self.num_examine:
                already_print_data_sources[data_source] = already_print_data_sources.get(data_source, 0) + 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        return reward_tensor
