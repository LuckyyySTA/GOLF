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

from omegaconf import DictConfig
import os
from typing import List, Union

import pandas as pd
import copy 
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from verl.utils.fs import copy_local_path_from_hdfs

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl.utils.torch_functional import pad_sequence_to_length


import logging
import os
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'INFO'))


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

    return {**tensors, **non_tensors}

from verl.utils.dataset.rl_dataset import RLHFDataset

class RLHFDatasetWithTarget(RLHFDataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 data_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 config: DictConfig,
                 target_key='target',
                 max_target_length=8192,
                 target_list_key='target_lst',
                 target_probs_key='target_ds_qwen_7b_probs',
        ):
        super().__init__(data_files, tokenizer, config)
        
        self.max_target_length = max_target_length
        self.filter_targets = config.get('filter_targets', False)
        self.target_key = target_key
        self.sample_target_ratio = config.get('sample_target_ratio', 1.0)
        self.target_list_key = target_list_key
        self.target_probs_key = target_probs_key
        self.max_num_targets = config.get('max_available_targets', 10)

        self.num_off_policy_targets = config.get('num_off_policy_targets', 2)
        self.length_strategy = config.get('length_strategy', 'short')  # 'long' or 'short'

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)

        # chat = row_dict.pop(self.prompt_key)

        prompt_with_chat_template = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        
        raw_prompt = prompt_with_chat_template # for text-only data
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")
        
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]
        
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        
        # process target
        if getattr(self, 'target_key', "target_key") in row_dict:
            tgt = None
            if self.target_key in row_dict:
                tgt = row_dict.pop(self.target_key)
            sample = np.random.rand() < self.sample_target_ratio
            
            if tgt is not None and sample is True:
                tgt = tgt[0]
            
                if prompt_with_chat_template.endswith('<think>\n') and tgt['content'].startswith('<think>\n'):
                    tgt['content'] = tgt['content'][len('<think>\n'):]
                tgt_input_ids = self.tokenizer(tgt['content'], add_special_tokens=False, return_tensors='pt')['input_ids'].reshape(-1) # [1, l]

                # NOTE: we don't need to do this because we add eos token id at mix_dapo_trainer.py
                
                # if tgt_input_ids[-1].item() != self.tokenizer.eos_token_id:
                #     eos_tensor = torch.tensor([self.tokenizer.eos_token_id], device=tgt_input_ids.device, dtype=tgt_input_ids.dtype).reshape(-1)
                #     tgt_input_ids = torch.cat([tgt_input_ids, eos_tensor], dim=-1)
                
                tgt_input_ids = tgt_input_ids.reshape(1, -1)
            else:
                tgt_input_ids = torch.tensor([], dtype=torch.long).reshape(1, 0) # empty target, will be pad to max_target_length

            # padding or truncate
            sequence_length = tgt_input_ids.shape[-1]
            if sequence_length < self.max_target_length:
                # right pad for tgt_input_ids
                tgt_input_ids = pad_sequence_to_length(tgt_input_ids,
                                                max_seq_len=self.max_target_length,
                                                pad_token_id=self.tokenizer.pad_token_id,
                                                left_pad=False)
            else:
                assert self.truncation in ('right', 'error')
                tgt_input_ids = tgt_input_ids[:, :self.max_target_length]
            
            tgt_input_ids = tgt_input_ids.squeeze(0)

            row_dict['tgt_input_ids'] = tgt_input_ids
        
        # process target_list
        if getattr(self, 'target_list_key', "target_list_key") in row_dict:
            target_list = row_dict.pop(self.target_list_key)
            tgt_input_ids = torch.tensor([], dtype=torch.long).reshape(1, 0)
                    
            if target_list is None:
                tgt_input_ids_lst = [torch.zeros_like(tgt_input_ids).fill_(self.tokenizer.pad_token_id)] * self.max_num_targets
                
            else:
                if self.length_strategy == 'short':
                    sorted_target_list = sorted(target_list, key=lambda x: x['quality_score'], reverse=True) # sort by quality_score
                elif self.length_strategy == 'long':
                    sorted_target_list = sorted(target_list, key=lambda x: x['quality_score'], reverse=False) # sort by quality_score
                target_source_list = [tgt['source'] for tgt in sorted_target_list]
                tgt_input_ids_lst = [self._process_target(tgt['content'], prompt_with_chat_template, add_eos=True) for tgt in sorted_target_list]
                if len(tgt_input_ids_lst) <= self.max_num_targets:
                    tgt_input_ids_lst.extend([torch.zeros_like(tgt_input_ids_lst[0]).fill_(self.tokenizer.pad_token_id)] * (self.max_num_targets - len(tgt_input_ids_lst)))
                else:
                    tgt_input_ids_lst = tgt_input_ids_lst[:self.max_num_targets]
                
                    
            row_dict['tgt_input_ids'] = torch.stack(tgt_input_ids_lst, dim=0) # [max_num_targets, max_target_length]
            row_dict['target_source_list'] = target_source_list
        
        
        if getattr(self, 'target_probs_key', "target_probs_key") in row_dict:
            target_probs = row_dict.pop(self.target_probs_key)
            if target_probs is not None:
                target_probs_pt = torch.tensor(target_probs, dtype=torch.float32, device=tgt_input_ids.device)
                target_probs_pt = target_probs_pt.reshape(1, -1)
                # truncation
                # prompt_len = (input_ids[0] != self.tokenizer.pad_token_id).sum()
                tgt_len = (tgt_input_ids != self.tokenizer.pad_token_id).sum()
                try:
                    assert target_probs_pt.shape[-1] == tgt_len+1
                except Exception as e:
                    breakpoint()
                
                # same padding as tgt_input_ids
                if target_probs_pt.shape[-1] < self.max_target_length:
                    target_probs_pt = pad_sequence_to_length(target_probs_pt,
                                                max_seq_len=self.max_target_length,
                                                pad_token_id=-1,
                                                left_pad=False)
                else:
                    assert self.truncation in ('right', 'error')
                    target_probs_pt = target_probs_pt[:, :self.max_target_length]
                row_dict['target_probs'] = target_probs_pt.squeeze(0) # [max_target_length]
            else:
                row_dict['target_probs'] = torch.zeros_like(tgt_input_ids, dtype=torch.float32, device=tgt_input_ids.device).fill_(-1)

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict
    def _process_target(self, tgt: str, prompt: str, add_eos=False) -> torch.Tensor:
        if prompt.endswith('<think>\n') and tgt.startswith('<think>\n'):
            tgt = tgt[len('<think>\n'):]
        tgt_input_ids = self.tokenizer(tgt, add_special_tokens=False, return_tensors='pt')['input_ids'].reshape(-1) # [1, l]
        if add_eos:
            tgt_input_ids = torch.cat([tgt_input_ids, torch.tensor([self.tokenizer.eos_token_id], device=tgt_input_ids.device, dtype=tgt_input_ids.dtype).reshape(-1)])

        tgt_input_ids = tgt_input_ids.reshape(1, -1)
        # padding or truncate
        sequence_length = tgt_input_ids.shape[-1]
        if sequence_length < self.max_target_length:
            # right pad for tgt_input_ids
            tgt_input_ids = pad_sequence_to_length(tgt_input_ids,
                                            max_seq_len=self.max_target_length,
                                            pad_token_id=self.tokenizer.pad_token_id,
                                            left_pad=False)
        else:
            assert self.truncation in ('right', 'error')
            tgt_input_ids = tgt_input_ids[:, :self.max_target_length]
        
        tgt_input_ids = tgt_input_ids.squeeze(0)

        return tgt_input_ids

from verl import DataProto
class BufferedDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.batch_size = dataloader.batch_size
        self.buffer = []
        self.dataloader_iter = None

    def start_new_epoch(self):
        """Reset for new epoch"""
        self.dataloader_iter = iter(self.dataloader)

    def get_next_batch(self):
        try:
            return next(self.dataloader_iter)
        except StopIteration:
            raise StopIteration

    def __len__(self):
        return len(self.dataloader)

    def add_to_buffer(self, samples):
        if len(self.buffer) == 0:
            self.buffer = samples
        else:
            self.buffer = DataProto.concat([self.buffer, samples])

    def get_from_buffer(self, count, dp_size):
        if count > self.buffer_size():
            count = (self.buffer_size() // dp_size) * dp_size
        samples = self.buffer.slice(range(0, count))
        self.buffer = self.buffer.slice(range(count, self.buffer_size()))
        return samples

    def buffer_size(self):
        return len(self.buffer)

import torch

class ResumableRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.
    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
        generator (Generator): Generator used in sampling.
    """
    #data_source: Sized
    #replacement: bool

    def __init__(self, data_source):
        self.data_source = data_source
        self.generator = torch.Generator()
        self.generator.manual_seed(47)
        
        self.perm_index = 0
        self.perm = torch.randperm(self.num_samples, generator=self.generator)
        
    @property
    def num_samples(self) -> int:
        return len(self.data_source)

    def __iter__(self):
        if self.perm_index >= len(self.perm):
            self.perm_index = 0
            self.perm = torch.randperm(self.num_samples, generator=self.generator)
            
        while self.perm_index < len(self.perm):
            self.perm_index += 1
            yield self.perm[self.perm_index-1].item() # the output index should be int

    def __len__(self):
        return self.num_samples
    
    def get_state(self):
        return {"perm": self.perm, "perm_index": self.perm_index, "generator_state": self.generator.get_state()}
    
    def set_state(self, state):
        self.perm = state["perm"]
        self.perm_index = state["perm_index"]
        self.generator.set_state(state["generator_state"])
