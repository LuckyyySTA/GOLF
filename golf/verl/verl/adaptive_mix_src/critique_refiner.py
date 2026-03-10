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
Critique-based Off-Policy Sample Refiner (hybrid mode).

Constructs aggregated refiner prompts from multiple on-policy rollouts + their
environment/task feedback, then drives a second generation round to produce
improved off-policy targets for hybrid training.
"""
import logging
from typing import Optional, List, Dict
import torch

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import tokenize_and_postprocess_data

logger = logging.getLogger(__file__)


class CritiqueRefiner:
    """
    Constructs aggregated critique prompts and generates refined off-policy
    rollouts in hybrid training mode.
    """

    def __init__(
        self,
        actor_rollout_wg,
        tokenizer,
        config,
        critique_key: str = 'critique',
        max_refine_length: Optional[int] = None,
        refine_temperature: float = 0.7,
        refine_top_p: float = 0.9,
        refine_type: str = 'aggregated',
        reward_fn=None,
    ):
        self.actor_rollout_wg = actor_rollout_wg
        self.tokenizer = tokenizer
        self.config = config
        self.critique_key = critique_key
        self.max_refine_length = max_refine_length or config.data.max_response_length
        self.refine_temperature = refine_temperature
        self.refine_top_p = refine_top_p

    # ─────────────────────────────────────────────────────────────────────────
    # Task-specific aggregated prompt builders
    # ─────────────────────────────────────────────────────────────────────────

    def _construct_aggregated_refine_prompt(
        self,
        original_prompt: str,
        all_responses: List[str],
        all_critiques: List[str],
        all_scores: List[float],
        data_source: str,
    ) -> str:
        if data_source == 'ifeval':
            return self._construct_ifeval_aggregated_prompt(
                original_prompt, all_responses, all_critiques, all_scores
            )
        elif data_source in ['openr1-math-46k', 'amc_aime', 'aops_forum', 'cn_contest',
                             'number_theory', 'olympiads_ref', 'olympiads', 'inequalities', 'math_dapo']:
            return self._construct_math_aggregated_prompt(
                original_prompt, all_responses, all_critiques, all_scores
            )
        elif data_source == 'wildchat-if':
            return self._construct_wildchat_aggregated_prompt(
                original_prompt, all_responses, all_critiques, all_scores
            )
        elif data_source in ['code', 'livecodebench', 'humanevalplus']:
            return self._construct_code_aggregated_prompt(
                original_prompt, all_responses, all_critiques, all_scores
            )
        else:
            raise ValueError(f"Unsupported data source for aggregated prompt: '{data_source}'")

    def _build_aggregated_prompt(
        self,
        header: str,
        candidates_header: str,
        final_instructions: str,
        all_responses: List[str],
        all_critiques: List[str],
        all_scores: List[float],
        segment_fn,
    ) -> str:
        """
        Shared scaffold for all aggregated prompt builders.
        Greedily adds candidate segments (shortest first) until the token budget runs out.
        """
        max_prompt_length = self.config.data.max_prompt_length

        def token_count(text: str) -> int:
            return len(self.tokenizer.encode(text, add_special_tokens=False))

        header_tokens = token_count(header + candidates_header)
        final_tokens = token_count(final_instructions)
        current_tokens = header_tokens

        items = sorted(zip(all_responses, all_critiques, all_scores), key=lambda x: token_count(x[0]))

        candidates_text = ""
        for idx, (response, critique, score) in enumerate(items, 1):
            segment = segment_fn(idx, response, critique, score)
            seg_tokens = token_count(segment)
            if current_tokens + seg_tokens + final_tokens <= max_prompt_length:
                candidates_text += segment
                current_tokens += seg_tokens
            else:
                break

        return header + candidates_header + candidates_text + final_instructions

    def _construct_wildchat_aggregated_prompt(
        self,
        original_prompt: str,
        all_responses: List[str],
        all_critiques: List[str],
        all_scores: List[float],
    ) -> str:
        header = f"Given the following inputs:\n\n**Problem**: {original_prompt}\n\n"
        candidates_header = "**Candidate Responses with Feedback**:"
        final_instructions = (
            "\n\nPlease synthesize an improved response by:\n\n"
            "- Learning from the mistakes identified in the critiques - avoid repeating the same errors.\n"
            "- Incorporating the strengths and good aspects mentioned in the critiques.\n"
            "- Synthesizing the best parts from all candidates while addressing their individual weaknesses.\n"
            "- Fully satisfying the user instruction and meeting all requirements.\n\n"
            "**CRITICAL OUTPUT REQUIREMENTS:**\n"
            "- Provide ONLY the synthesized response itself, nothing more.\n"
            "- DO NOT start with any meta phrases like \"Improved Response:\", \"Here is the synthesized response:\", or similar introductory text.\n"
            "- DO NOT end with any meta commentary, notes, or explanations such as \"Note: This response meets all requirements...\", \"This addresses the user's needs...\", or any other additional remarks.\n"
            "- Your output should be the pure, direct response to the user's original instruction - as if you were directly answering them without any wrapper text or self-commentary."
        )

        def segment_fn(idx, response, critique, score):
            seg = f"\n\n--- Candidate Response {idx} (Score: {score:.3f}) ---\n"
            seg += f"Response:\n{response}\n\nFeedback:\n{critique}"
            return seg

        return self._build_aggregated_prompt(
            header, candidates_header, final_instructions,
            all_responses, all_critiques, all_scores, segment_fn
        )

    def _construct_ifeval_aggregated_prompt(
        self,
        original_prompt: str,
        all_responses: List[str],
        all_critiques: List[str],
        all_scores: List[float],
    ) -> str:
        header = f"Given the following inputs:\n\n**User Instruction**: {original_prompt}\n\n"
        candidates_header = "**Candidate Responses with Feedback**:"
        final_instructions = (
            "\n\nPlease synthesize an improved response by:\n\n"
            "- Learning from the mistakes identified in the critiques - avoid repeating the same errors.\n"
            "- Incorporating the strengths and good aspects mentioned in the critiques.\n"
            "- Synthesizing the best parts from all candidates while addressing their individual weaknesses.\n"
            "- Fully satisfying the user instruction and meeting all requirements.\n\n"
            "**CRITICAL OUTPUT REQUIREMENTS:**\n"
            "- Provide ONLY the synthesized response itself, nothing more.\n"
            "- DO NOT start with any meta phrases like \"Improved Response:\", \"Here is the synthesized response:\", or similar introductory text.\n"
            "- DO NOT end with any meta commentary, notes, or explanations such as \"Note: This response meets all requirements...\", \"This addresses the user's needs...\", or any other additional remarks.\n"
            "- Your output should be the pure, direct response to the user's original instruction - as if you were directly answering them without any wrapper text or self-commentary."
        )

        def segment_fn(idx, response, critique, score):
            seg = f"\n\n--- Candidate Response {idx} (Score: {score:.3f}) ---\n"
            seg += f"Response:\n{response}\n\nFeedback:\n{critique}"
            return seg

        return self._build_aggregated_prompt(
            header, candidates_header, final_instructions,
            all_responses, all_critiques, all_scores, segment_fn
        )

    def _construct_math_aggregated_prompt(
        self,
        original_prompt: str,
        all_responses: List[str],
        all_critiques: List[str],
        all_scores: List[float],
    ) -> str:
        header = f"Given the following inputs:\n\n**Problem**: {original_prompt}\n\n"
        candidates_header = "**Solution Attempts with Feedback**:\n"
        final_instructions = (
            "\n\nPlease synthesize an improved solution by:\n\n"
            "- Carefully analyze the feedback to identify which steps in each attempt were incorrect or problematic.\n"
            "- Understand WHY those steps were wrong by examining the feedback.\n"
            "- Keep the correct reasoning steps and valid calculations from the candidates.\n"
            "- Fix or replace the incorrect steps with your own correct reasoning - do NOT simply copy answers.\n"
            "- Build a complete, coherent solution with genuine step-by-step reasoning.\n\n"
            "**CRITICAL Requirements:**\n"
            "- You MUST derive the solution through authentic mathematical reasoning.\n"
            "- DO NOT hack by writing arbitrary steps and then forcing a specific final answer.\n"
            "- Every step must logically follow from the previous one with valid mathematical operations.\n"
            "- Use the feedback to understand errors, but work through the solution yourself.\n"
            "- If the feedback mentions a correct answer, you should arrive at it naturally through proper reasoning, not by working backwards from it.\n\n"
            "**Output Format:**\n"
            "- Answer the original question directly and self-contained.\n"
            "- Do NOT mention or allude to candidates/attempts/feedback/critiques "
            "(e.g., avoid phrases like \"after reviewing\", \"based on feedback\", \"one attempt suggests\").\n"
            "- Start solving immediately (no preface or meta commentary).\n"
            "- Provide clear step-by-step reasoning with necessary calculations.\n"
            "- End with the final answer formatted as `\\boxed{}`.\n"
            "- DO NOT start with any meta phrases like \"Improved solution:\", \"Here is the improved solution:\", or similar introductory text.\n"
            "- DO NOT add any concluding notes, remarks, or meta commentary after the boxed answer.\n"
            "- Your output should be the pure, direct response to the user's original instruction - as if you were directly answering them without any wrapper text or self-commentary.\n"
        )

        def segment_fn(idx, response, critique, score):
            seg = f"\n\n--- Solution Attempt {idx} (Score: {score:.3f}) ---\n{response}\n"
            if critique:
                seg += f"Feedback: {critique}"
            return seg

        return self._build_aggregated_prompt(
            header, candidates_header, final_instructions,
            all_responses, all_critiques, all_scores, segment_fn
        )

    def _construct_code_aggregated_prompt(
        self,
        original_prompt: str,
        all_responses: List[str],
        all_critiques: List[str],
        all_scores: List[float],
    ) -> str:
        header = (
            f"You are given a coding problem and several failed solution attempts. "
            f"Each attempt is accompanied by execution feedback from the environment "
            f"(e.g., failed test cases, runtime errors, or compilation errors).\n\n"
            f"**Problem**:\n{original_prompt}\n\n"
        )
        candidates_header = "**Failed Attempts with Execution Feedback**:\n"
        final_instructions = (
            "\n\nYour task is to synthesize a correct solution by learning from their mistakes:\n\n"
            "1. **Analyse the failures**: For each attempt, read the execution feedback carefully. "
            "Identify the exact root cause – wrong algorithm, incorrect edge-case handling, "
            "off-by-one error, wrong data type, etc.\n\n"
            "2. **Extract what is useful**: Even in wrong attempts there may be partial insight "
            "(e.g., a correct parsing strategy, a useful helper function). "
            "Note what is worth keeping and what must be discarded.\n\n"
            "3. **Derive the correct approach**: Based on your analysis, think through the problem "
            "from scratch and construct a solution that avoids all identified errors. "
            "Trace through the failing test cases mentally to validate your logic before coding.\n\n"
            "4. **Output**: Write your final solution in a single ```python ... ``` code block. "
            "The block must contain complete, self-contained, executable code.\n"
        )

        def segment_fn(idx, response, critique, score):
            seg = f"\n--- Attempt {idx} (Score: {score:.2f}) ---\n{response}\n\n"
            if critique:
                seg += f"**Execution Feedback**:\n{critique}\n"
            return seg

        return self._build_aggregated_prompt(
            header, candidates_header, final_instructions,
            all_responses, all_critiques, all_scores, segment_fn
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Critique extraction
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_critiques(self, gen_batch_out: DataProto, critique_type: str) -> list:
        """
        Extract per-rollout critique strings from the batch.

        For code tasks ('environment'): uses execution feedback from the reward model.
        For math tasks ('simple' / 'answer'): generates correctness labels or GT hints.
        """
        critiques = []
        reward_tensor = gen_batch_out.batch['token_level_scores']
        reward_model_list = gen_batch_out.non_tensor_batch['reward_model'].tolist()

        for rollout_idx, reward_model_item in enumerate(reward_model_list):
            score = reward_tensor[rollout_idx].sum().item()
            is_correct = score == 1.0

            if critique_type == 'environment':
                feedback = reward_model_item.get('critique', '').strip()
                critique = feedback if feedback else (
                    "All tests passed!" if is_correct else "Some tests failed. Please review your solution."
                )
            elif critique_type == 'simple':
                critique = "The answer is correct." if is_correct else "The answer is incorrect. Please try again."
            elif critique_type == 'answer':
                ground_truth = reward_model_item.get('ground_truth', None)
                critique = (
                    f"The answer is correct, the ground_truth is {ground_truth}"
                    if is_correct else
                    f"The answer is incorrect, the ground_truth is {ground_truth}"
                )
            else:
                raise ValueError(f"Unsupported critique type: '{critique_type}'")

            critiques.append(critique)

        return critiques

    # ─────────────────────────────────────────────────────────────────────────
    # Tokenization helper
    # ─────────────────────────────────────────────────────────────────────────

    def _tokenize_texts(self, texts: list, max_length: int, left_pad: bool = True) -> tuple:
        tokenize_config = {
            "tokenizer": self.tokenizer,
            "max_length": max_length,
            "pad_token_id": self.tokenizer.pad_token_id,
            "left_pad": left_pad,
            "truncation": "right",
        }
        tokenized_data = [tokenize_and_postprocess_data(prompt=text, **tokenize_config) for text in texts]
        input_ids = torch.stack([data[0][0] for data in tokenized_data])
        attention_masks = torch.stack([data[1][0] for data in tokenized_data])
        position_ids = compute_position_id_with_mask(attention_masks)
        return input_ids, attention_masks, position_ids

    # ─────────────────────────────────────────────────────────────────────────
    # Correctness check
    # ─────────────────────────────────────────────────────────────────────────

    def _is_sample_correct(self, score: float, critique: str, data_source: str) -> bool:
        """Return True if the sample is considered correct for the given data source."""
        if data_source in ['openr1-math-46k', 'amc_aime', 'aops_forum', 'cn_contest',
                           'number_theory', 'olympiads_ref', 'olympiads', 'inequalities', 'math_dapo']:
            return score == 1.0
        # Binary correctness for code, ifeval, wildchat
        return score == 1.0

    # ─────────────────────────────────────────────────────────────────────────
    # Aggregated refiner prompt extraction (entry point for hybrid training)
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_aggregated_refiner_prompts(
        self,
        gen_batch: DataProto,
        gen_batch_out: DataProto,
        critique_type: str,
    ) -> Optional[Dict]:
        """
        Build one aggregated refiner prompt per original prompt.

        Each prompt concatenates all n on-policy rollouts (optionally filtered to
        wrong-only) with their critiques and scores, forming the input for the
        refiner's generation round.

        Returns:
            {
              'prompts':          List[str]  – one per original prompt
              'prompt_indices':   List[int]
              'all_correct_mask': List[bool] – True → all rollouts correct or all wrong
                                              (refiner reward will be zeroed)
            }
        """
        n = self.config.actor_rollout_ref.rollout.n
        original_batch_size = gen_batch_out.batch['responses'].size(0) // n

        aggregate_only_wrong = getattr(
            self.config.actor_rollout_ref.rollout.critique_refiner, 'aggregate_only_wrong', False
        )

        data_source = gen_batch_out.non_tensor_batch['data_source'][0]

        # Select critique source
        if data_source in ['openr1-math-46k', 'amc_aime', 'aops_forum', 'cn_contest',
                           'number_theory', 'olympiads_ref', 'olympiads', 'inequalities', 'math_dapo']:
            critiques = self._extract_critiques(gen_batch_out, critique_type=critique_type)
        elif data_source in ['code', 'livecodebench', 'humanevalplus', 'ifeval', 'wildchat-if']:
            critiques = gen_batch_out.non_tensor_batch['critique']
        else:
            logger.warning(f"Unsupported data source: '{data_source}', skipping refiner prompt extraction")
            return None

        reward_tensor = gen_batch_out.batch['token_level_scores']
        aggregated_prompts, prompt_indices, all_correct_mask = [], [], []

        for prompt_idx in range(original_batch_size):
            start_idx = prompt_idx * n
            end_idx = (prompt_idx + 1) * n

            prompt_responses, prompt_critiques, prompt_scores = [], [], []
            for rollout_idx in range(start_idx, end_idx):
                score = reward_tensor[rollout_idx].sum().item()
                critique = critiques[rollout_idx]
                response_text = self.tokenizer.decode(
                    gen_batch_out.batch['responses'][rollout_idx], skip_special_tokens=True
                )
                if aggregate_only_wrong and self._is_sample_correct(score, critique, data_source):
                    continue
                prompt_responses.append(response_text)
                prompt_critiques.append(critique)
                prompt_scores.append(score)

            # All-correct → refiner gets no learning signal (reward zeroed later).
            # is_all_correct = aggregate_only_wrong and (
            #     len(prompt_responses) == 0 or len(prompt_responses) == n
            # )
            # All-wrong is kept: the refiner can still learn by aggregating all failed attempts.
            is_all_correct = aggregate_only_wrong and len(prompt_responses) == 0

            # Extract original question text
            extra_info = gen_batch.non_tensor_batch['extra_info']
            if "question" in extra_info[0]:
                question = extra_info[start_idx]['question']
            elif "problem" in extra_info[0] and data_source == "livecodebench":
                problem = extra_info[start_idx]['problem']
                prompt_prefix = (
                    "You are a coding expert. You will be given a coding problem, and you need to write "
                    "a correct Python program that matches the specification and passes all tests. "
                    "The time limit is 1 second. You may start by outlining your thought process. "
                    "In the end, please provide the complete code in a code block enclosed with ``` ```.\n\n"
                )
                question = problem.split(prompt_prefix)[1]
            else:
                raise KeyError(f"Cannot find question in extra_info keys: {list(extra_info[0].keys())}")

            aggregated_prompt = self._construct_aggregated_refine_prompt(
                original_prompt=question,
                all_responses=prompt_responses,
                all_critiques=prompt_critiques,
                all_scores=prompt_scores,
                data_source=data_source,
            )
            aggregated_prompts.append(aggregated_prompt)
            prompt_indices.append(prompt_idx)
            all_correct_mask.append(is_all_correct)

        all_correct_count = sum(all_correct_mask)
        if aggregate_only_wrong:
            msg = f"Extracted {len(aggregated_prompts)} aggregated refiner prompts (wrong-only)"
            if all_correct_count:
                msg += f", {all_correct_count} prompts all-correct → reward will be zeroed"
            print(msg)
        else:
            print(f"Extracted {len(aggregated_prompts)} aggregated refiner prompts")

        return {
            'prompts': aggregated_prompts,
            'prompt_indices': prompt_indices,
            'all_correct_mask': all_correct_mask,
        }
