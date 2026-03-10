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

import re
from verl import DataProto
import random
# from verl.utils.reward_score import _default_compute_score
import torch
import numpy as np
from functools import partial
from verl.utils.reward_score import prime_math
from collections import defaultdict
from verl.workers.reward_manager.format import format_reward

def sigmoid_k(x, k=6):
    """
    Maps a value x in [0, 1] to [0, 1] using a sigmoid-like S-shaped curve.

    Parameters:
        x (float): Input value in the range [0, 1].
        k (float): Steepness parameter for the sigmoid curve. Higher values make the curve steeper.

    Returns:
        float: Mapped value in the range [0, 1].
    """
    if k == 0: # vanilla sigmoid
        return 1 / (1 + np.exp(-x))

    # Shift and scale x to the range [-k, k]
    x_scaled = k * (2 * x - 1)
    
    # Apply the sigmoid function
    sigmoid = 1 / (1 + np.exp(-x_scaled))
    
    # Scale the output to [0, 1]
    return sigmoid

def threshold_t_sigmoid_k(x, t, k=6):
    """
    Maps a value x in [0, 1] to [0, 1] using a sigmoid-like S-shaped curve.

    Parameters:
        x (float): Input value in the range [0, 1].
        k (float): Steepness parameter for the sigmoid curve. Higher values make the curve steeper.

    Returns:
        float: Mapped value in the range [0, 1].
    """
    # Shift and scale x to the range [-k, k]
    x_scaled = k * (2 * x - 1)
    
    # Apply the sigmoid function
    sigmoid = 1 / (1 + np.exp(-x_scaled))

    result = 0 if sigmoid < t else sigmoid
    
    return result


def threshold_t_sigmoidv2_k(x, t, k=6):
    # concave curve
    if x < t:
        result = 0
    else:
        x = x - t
        x = x * k
        result = 1 / (1 + np.exp(-x))
    return result


def threshold_t_sigmoidv2fixed_k(x, t, k=6):
    if x < t:
        result = 0
    else:
        x = (x- t) * k
        result = 1 / (1 + np.exp(-x))  * ((1 - t) / 0.5) - ((1-t) - t) 
    
    return result


def threshold_t_sigmoidv3_k(x, t, k=6):
    # convex curve
    if x < t:
        result = 0
    else:
        x = (x - 1) * k
        result = 1 / (1 + np.exp(-x)) + 0.5
    return result


def leaky_relu_like(score, threshold, alpha=0.01):
    """
    Maps a score from [0, 1] to [0, 1] using a Leaky ReLU-like function.

    Parameters:
    - score: The input score in the range [0, 1].
    - threshold: The threshold below which the score is scaled.
    - alpha: The slope for scores below the threshold (default is 0.01).

    Returns:
    - The transformed score in the range [0, 1].
    """
    if score < threshold:
        return alpha * score
    else:
        return score


def threshold_t_tanh_k(score, t, k=6):
    # Apply tanh transformation with a configurable scaling factor
    transformed_score = (np.tanh(score * k - k / 2) + 1) / 2
    
    # Threshold values smaller than 0.05 to 0
    if transformed_score < t:
        transformed_score = 0
    
    return transformed_score


class ProbRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score_name=None, 
                 shaping_function_name=None, discrete_function_name=None, 
                 format_coefficient=0, save_results_dir=None, reward_type='pr',
                 gt_tokens_one_more=False, gt_tokens_one_more_adjusted=False,
                 format_mode='R1_nothink') -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        # assert compute_score is None
        self.compute_score_name = compute_score_name
        print(f"{shaping_function_name=}")
        if shaping_function_name == 'identity':
            self.shaping_function = lambda x:x
        elif shaping_function_name == 'one_minus':
            self.shaping_function = lambda x: 1 - x
        elif shaping_function_name == 'random':
            self.shaping_function = lambda x: random.random()
        elif shaping_function_name.startswith('threshold'):
            threshold = float(shaping_function_name.split('_')[-1])
            self.shaping_function = lambda x: 0 if x < threshold else x
        elif shaping_function_name.startswith('sigmoid_'):
            print(f"Selecting sigmoid_k function.")
            k = float(shaping_function_name.split('_')[-1])
            self.shaping_function = partial(sigmoid_k, k=k)
        elif shaping_function_name.startswith('leaky_'):
            # e.g., leaky_0.05
            print(f"Using leaky-relu like function")
            threshold = float(shaping_function_name.split('_')[1])
            self.shaping_function = partial(leaky_relu_like, threshold=threshold)
        elif shaping_function_name.startswith('comp'): # compound
            # comp_threshold_0.3_sigmoid_6
            threshold = float(shaping_function_name.split('_')[2])
            k = float(shaping_function_name.split('_')[4])
            if 'sigmoidv2fixed' in shaping_function_name:
                print(f"Using sigmoid v2fixed")
                self.shaping_function = partial(threshold_t_sigmoidv2fixed_k, t=threshold, k=k)
            elif 'sigmoidv3' in shaping_function_name:
                print(f"Using sigmoid v3")
                self.shaping_function = partial(threshold_t_sigmoidv3_k, t=threshold, k=k)
            elif 'sigmoidv2' in shaping_function_name:
                print(f"Using sigmoid v2")
                self.shaping_function = partial(threshold_t_sigmoidv2_k, t=threshold, k=k)
            elif 'sigmoid' in shaping_function_name:
                print(f"Using sigmoid v1")
                self.shaping_function = partial(threshold_t_sigmoid_k, t=threshold, k=k)
            elif 'tanh' in shaping_function_name:
                self.shaping_function = partial(threshold_t_tanh_k, t=threshold, k=k)
            else:
                raise ValueError
        else:
            print(f"{shaping_function_name=}")
            raise NotImplementedError(f"{shaping_function_name=}")
        self.discrete_function_name = discrete_function_name
        self.format_coefficient = format_coefficient
        assert reward_type in ['pr', 'pr+vr']
        self.reward_type = reward_type
        self.format_mode = format_mode
        self.gt_tokens_one_more = gt_tokens_one_more
        self.gt_tokens_one_more_adjusted = gt_tokens_one_more_adjusted

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]
        
        if 'tgt_input_ids' in data.batch.keys():
            print(f"{data.batch['tgt_input_ids'].shape=}")
            print(f"{data.batch['tgt_input_ids'][0]=}")
            print(f"{data.batch['tgt_input_ids'][0].shape=}")
            print(f"{data.batch['tgt_input_ids'][0].dtype=}")
            print()

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        extracted_answer_list = []
        format_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        scoreA_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        scoreB_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses'] # len(response_ids): 1024
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum() # 329
            valid_response_ids = response_ids[:valid_response_length] 

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences) # <|im_start|>system\n..the answer is: \boxed{10, 30, 40}<|im_end|>

            prompt_str = self.tokenizer.decode(valid_prompt_ids) # '<|im_start|>system\nA conversation between ... answer here </answer>.<|im_end|>\n<|im_start|>user\nLet the parabola ... and $X_{M}$.<|im_end|>\n<|im_start|>assistant\n'
            predict_str = self.tokenizer.decode(valid_response_ids) # To determine the relationship ... relative to each other and the parabola.<|im_end|>
            
            format_score = format_reward(predict_str=predict_str, format_mode=self.format_mode) 

            scoreA, scoreB, extracted_answer = self.compute_scoreA_scoreB_and_extracted_answer(data_item, valid_response_ids=valid_response_ids)
            score_delta = scoreB - scoreA
            score_delta = self.shaping_function(score_delta)
            if self.discrete_function_name is not None and self.discrete_function_name != 'identity':
                if self.discrete_function_name.startswith('bin_'):
                    num_bins = int(self.discrete_function_name.split('_')[1])
                    score_delta = self.map_to_bins(score_delta, num_bins)

            if self.format_coefficient == -1:
                score = score_delta if format_score == 1 else 0
            else:
                score = (1 - self.format_coefficient) * (score_delta) + self.format_coefficient * format_score
            
            if isinstance(score, dict):
                score = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                score = score
            
            reward_tensor[i, valid_response_length - 1] = score
            format_reward_tensor[i, valid_response_length - 1] = format_score
            scoreB_tensor[i, valid_response_length - 1] = scoreB
            scoreA_tensor[i, valid_response_length - 1] = scoreA

            data_source = data_item.non_tensor_batch['data_source']
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"{data_source=} {sequences_str=}", flush=True)

            extracted_answer_list.append(extracted_answer)

        # return reward_tensor, scoreB_tensor, scoreA_tensor, format_reward_tensor, extracted_answer_list
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "scoreB_tensor": scoreB_tensor,
                "scoreA_tensor": scoreA_tensor,
                "format_reward_tensor": format_reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor, scoreB_tensor, scoreA_tensor, format_reward_tensor


    @staticmethod
    def map_to_bins(score: float, num_bins: int) -> float:
        """
        Maps a score in [0,1] to the nearest discrete value based on num_bins.
        
        :param score: A float between 0 and 1.
        :param num_bins: The number of discrete bins.
        :return: The mapped discrete value.
        """
        if num_bins < 1:
            raise ValueError("num_bins must be at least 1")
        
        step = 1 / num_bins
        return round(score / step) * step

    def compute_scoreB(self, old_log_probs, ground_truth_mask):
        if ground_truth_mask.sum() == 0:
            scoreB = 0
        else:
            old_log_probs_in_gt = old_log_probs[ground_truth_mask.bool()] # 
            if self.gt_tokens_one_more and self.gt_tokens_one_more_adjusted:
                old_log_probs_in_gt[-1] = min(old_log_probs_in_gt[-1], np.log(0.5))
            # Convert to BF32 for higher precision calculations
            old_log_probs_in_gt = old_log_probs_in_gt.to(torch.float32)
            # mean of probs
            if self.compute_score_name == 'mean_exp_log_softmax':
                scoreB = torch.mean(torch.exp(old_log_probs_in_gt)).item()
            # mean log probs
            elif self.compute_score_name == 'mean_log_softmax':
                scoreB = torch.mean(old_log_probs_in_gt).item()
            # product of probs
            elif self.compute_score_name == 'exp_sum_log_softmax':
                scoreB = torch.exp(torch.sum(old_log_probs_in_gt)).item()
            # root of the product of probs
            elif self.compute_score_name == 'exp_mean_log_softmax':
                scoreB = torch.exp(torch.mean(old_log_probs_in_gt)).item() 
            else:
                raise ValueError
            # Convert the final score back to BF16 if necessary
            scoreB = torch.tensor(scoreB, dtype=torch.bfloat16).item()

        return scoreB

    def compute_scoreA_scoreB_and_extracted_answer(self, data_item, valid_response_ids):
        data_source = data_item.non_tensor_batch['data_source']
        if self.reward_type == 'pr+vr' and (any(dataset_name in data_source for dataset_name in ["numina_cn_k12", "numina_synthetic_math", "numina_olympiads", 
                                                                "numina_synthetic_amc", "numina_aops_forum", "numina_amc_aime",
                                                                "Math-500", "AIME2024", "AMC2023", "DAPO-Math-17k",
                                                                "OlympiadBench", "Minerva", "simplelr_deepscaler"])):

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            solution_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True),
            res = prime_math.compute_score(solution_str, ground_truth)
            scoreB = float(res[0])
            scoreA = 0
            extracted_answer = res[1]
        else:
            scoreB = self.compute_scoreB(data_item.batch['old_log_probs_pr'], data_item.batch['ground_truth_mask_pr']) # shape: [1024]
            scoreA = data_item.non_tensor_batch['reward_model'].get('scoreA', 0.0)

            predict_str = self.tokenizer.decode(valid_response_ids) # To determine the relationship ... relative to each other and the parabola.<|im_end|>
            match = re.search(r'<answer>(.*?)</answer>', predict_str, re.DOTALL)
            extracted_answer = match.group(1).strip() if match else ""
        return scoreA, scoreB, extracted_answer