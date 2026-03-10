# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
Wildchat GRM compute_score function for prime.py integration.
Extracted from grm.py to provide a simple synchronous interface for ProcessPoolExecutor.
"""

import random
import time
from typing import Optional

from openai import OpenAI

# GRM configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30

# Default GRM nodes (can be overridden via extra_info)
DEFAULT_GRM_URLS = [
    "http://10.146.236.29:8000", "http://10.146.229.49:8000", "http://10.146.232.186:8000",
    "http://10.146.225.237:8000", "http://10.146.229.181:8000", "http://10.146.230.39:8000",
    "http://10.146.225.215:8000", "http://10.146.225.188:8000", "http://10.146.231.15:8000",
    "http://10.146.232.194:8000", "http://10.144.160.156:8000", "http://10.146.234.57:8000"
]
DEFAULT_MODEL_NAME = "qwen3-30b-a3b-instruct-2507"

PAIRWISE_TEMPLATE = """You are given a user question and two responses.
- Response A: a model-generated answer that may need improvement.
- Response B: a high-quality reference answer (for evaluation only).

Your task is to act as an impartial judge and decide which response is better for the user.

First, think step by step and put your analysis in <reasoning> and </reasoning> tags. In your reasoning:
- Identify the key requirements of the user question (instruction following, relevance, completeness, factuality/safety, style/format).
- Compare Response A and Response B on these requirements.
- Response A may be better than Response B if it follows the user more closely, is safer, or better matches the requested style.
- Avoid position bias and length bias.

Then, provide an actionable critique in <critique> and </critique> tags **for the model-generated answer (Response A) only**. This critique will be shown to another model that only sees that answer. Therefore:
- Write the critique as if there is ONLY ONE answer.
- Do NOT mention or hint that there was another response, a reference answer, "Assistant A/B", "the other answer", "the second answer", or "the reference".
- Point out the current answer's strengths.
- Point out missing/incorrect/unsafe/irrelevant parts.
- Give concrete suggestions on what to add/remove/fix.
- Do NOT copy or paraphrase the content of Response B.

Finally, output your verdict in <answer> and </answer> tags:
- <answer> [[A]] </answer> if the model-generated answer (Response A) is better
- <answer> [[B]] </answer> if the reference answer (Response B) is better

Format your output EXACTLY like this:
<reasoning> your step-by-step comparison of A vs B </reasoning>
<critique>
your self-contained, neutral feedback for improving the model-generated answer
</critique>
<answer> [[A]] or [[B]] </answer>

Below are the user's question and the two responses:

[User Question]
{instruction}

[The Start of Response A]
{response_a}
[The End of Response A]

[The Start of Response B]
{response_b}
[The End of Response B]
"""


def _extract_answer(text: str) -> float:
    """Extract answer from GRM response and convert to score (A=1.0, B=0.0)"""
    import re
    
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_match = re.search(answer_pattern, text, re.DOTALL | re.IGNORECASE)
    
    if answer_match:
        answer_content = answer_match.group(1).strip()
        
        # Check various formats
        if re.search(r'\[\[A\]\]', answer_content, re.IGNORECASE):
            return 1.0
        elif re.search(r'\[\[B\]\]', answer_content, re.IGNORECASE):
            return 0.0
        elif re.search(r'<A>|\[A\]|\{A\}|\(A\)', answer_content, re.IGNORECASE):
            return 1.0
        elif re.search(r'<B>|\[B\]|\{B\}|\(B\)', answer_content, re.IGNORECASE):
            return 0.0
        elif 'A' in answer_content:
            return 1.0
        elif 'B' in answer_content:
            return 0.0
    
    return 0.0  # Default to B (reference is better)


def _call_grm_api(client: OpenAI, messages: list, model_name: str, max_retries: int = DEFAULT_MAX_RETRIES) -> Optional[str]:
    """Call GRM API with retry logic"""
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
                timeout=DEFAULT_TIMEOUT
            )
            
            if completion.choices and len(completion.choices) > 0:
                content = completion.choices[0].message.content
                if isinstance(content, str) and content.strip():
                    return content.strip()
            
            time.sleep(0.1)
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.1)
            continue
    
    return None


def compute_score(
    data_source: str,
    model_output: str,
    ground_truth: str,
    extra_info: dict = None
) -> float:
    """
    Compute GRM score for model output compared to ground truth (reference answer).
    
    This is a synchronous function designed to be called by prime.py's ProcessPoolExecutor.
    Each process will create its own OpenAI client to avoid serialization issues.
    
    Args:
        data_source: Data source name (e.g., "wildchat")
        model_output: Model-generated response to evaluate
        ground_truth: Reference/ground truth answer
        extra_info: Dict containing:
            - "question": The original user question (required)
            - "grm_urls": List of GRM URLs (optional, uses default if not provided)
            - "grm_model_name": GRM model name (optional, uses default if not provided)
    
    Returns:
        float: Score between 0.0 and 1.0
               - 1.0: Model output is better than or equal to reference
               - 0.0: Reference is better than model output
    """
    if extra_info is None:
        extra_info = {}
    
    # Get question from extra_info
    question = extra_info.get("question", "")
    if not question:
        print(f"Warning: No question provided in extra_info, using empty question")
        return 0.0
    
    # Get GRM configuration
    grm_urls = DEFAULT_GRM_URLS # extra_info.get("grm_urls", DEFAULT_GRM_URLS)
    model_name = DEFAULT_MODEL_NAME # extra_info.get("grm_model_name", DEFAULT_MODEL_NAME)
    
    # Create OpenAI client (each process creates its own)
    base_url = random.choice(grm_urls) if isinstance(grm_urls, list) else grm_urls
    client = OpenAI(api_key="EMPTY", base_url=f"{base_url}/v1")
    
    # Build prompt
    prompt = PAIRWISE_TEMPLATE.replace(
        "{instruction}", question
    ).replace(
        "{response_a}", model_output
    ).replace(
        "{response_b}", ground_truth
    )
    
    messages = [{"role": "user", "content": prompt}]
    
    # Call GRM API
    try:
        response = _call_grm_api(client, messages, model_name)
        if response is not None:
            score = _extract_answer(response)
            return score
    
    except Exception as e:
        print(f"GRM API error: {e}")
    
    return 0.0  # Default: reference is better