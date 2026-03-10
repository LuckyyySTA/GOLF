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

import asyncio
import re
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Union

from openai import OpenAI

# GRM service configuration parameters
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30

GENRM_PROMPT_TEMPLATE = "You are a skilled little expert at scoring responses. You should evaluate given responses based on the given judging criteria.\nGiven the context of the conversation (the last round is the User's query) and multiple responses from the Assistant, you need to refer to the [General Evaluation Criteria] to score the responses. Based on the general evaluation criteria, state potential other specific criteria to the query, the weights of different criteria, and then provide an overall comprehensive score upon them. The score is 0 or 1, with 1 indicating that the response is correct.\nBefore scoring, please analyze step by step. Your scoring needs to be as strict as possible.\n\n#### Evaluation Criteria ####\n1. Instruction Adherence:\n   - Fully Adhered: The response fully complies with all instructions and requirements of the question.\n   - Partially Adhered: The response meets most of the instructions but has some omissions or misunderstandings.\n   - Basically Adhered: The response meets some instructions, but the main requirements are not fulfilled.\n   - Not Adhered: The response does not meet any instructions.\n   Example: If the question requires three examples and the response provides only one, it falls under \"Partially Adhered.\"\n2. Clarity:\n   - Very Clear: The response is fluent, well-structured, and logically clear.\n   - Clear but Minor Issues: The response is mostly clear but has some minor language or structural issues.\n   - Basically Clear: The response has noticeable language or logic issues but is still understandable.\n   - Not Clear: The response is disjointed, illogical, and hard to understand.\n   Example: If the response has complex sentence structures and lacks punctuation, it falls under \"Basically Clear\" or \"Not Clear.\"\n3. Accuracy:\n   - Completely Accurate: All information and data are completely accurate.\n   - Mostly Accurate: Most information is accurate, with minor errors.\n   - Some Errors: There are some noticeable errors affecting comprehension.\n   - Mostly Incorrect: There are numerous errors seriously affecting the credibility of the information.\n   Example: If a specific data point is incorrectly cited but doesn't affect the overall conclusion, it falls under \"Mostly Accurate.\"\n\n#### Conversation Context ####\n{}\n#### Responses to be Scored ####\n{}\n\n#### Output Format Requirements ####\n\nOutput with three lines\nSpecific Criteria: <Other potential criteria specific to the query and the context, and the weights of each criteria>.\nAnalysis: <Compare different responses based on given Criteria>.\nScores: <the overall comprehensive score of the response, e.g., \\boxed{{x}}>."

GENRM_PAIRWISE_PROMPT_TEMPLATE = "You are a skilled little expert at scoring responses. You should evaluate given responses based on the given judging criteria.\nGiven the context of the conversation (the last round is the User's query) and multiple responses from the Assistant, you need to refer to the [General Evaluation Criteria] to score the responses. Based on the general evaluation criteria, state potential other specific criteria to the query, the weights of different criteria, and then provide an overall comprehensive score upon them.\nEach score is an integer between 1 and 10, with a higher score indicating that the response meets the relevant criteria more closely. For example, a score of 1 means the response does not meet the criteria at all, a score of 6 means the response meets only some parts, and a score of 10 means the response perfectly meets the evaluation criteria.\nBefore scoring, please analyze step by step. Your scoring needs to be as strict as possible.\n\n#### Evaluation Criteria ####\n1. Instruction Adherence:\n   - Fully Adhered (9-10 points): The response fully complies with all instructions and requirements of the question.\n   - Partially Adhered (6-8 points): The response meets most of the instructions but has some omissions or misunderstandings.\n   - Basically Adhered (3-5 points): The response meets some instructions, but the main requirements are not fulfilled.\n   - Not Adhered (1-2 points): The response does not meet any instructions.\n   Example: If the question requires three examples and the response provides only one, it falls under \"Partially Adhered.\"\n2. Usefulness:\n   - Highly Useful (9-10 points): The response provides comprehensive and accurate information, fully addressing the issue.\n   - Useful but Incomplete (6-8 points): The response provides some useful information, but lacks details or accuracy.\n   - Limited Usefulness (3-5 points): The response offers little useful information, with most content being irrelevant or incorrect.\n   - Useless or Incorrect (1-2 points): The response is completely irrelevant or incorrect.\n   Example: If there are factual errors in the response but the overall direction is correct, it falls under \"Useful but Incomplete.\"\n3. Level of Detail:\n   - Very Detailed (9-10 points): The response includes ample details covering all aspects of the issue.\n   - Detailed but Slightly Lacking (6-8 points): The response is fairly detailed but misses some important details.\n   - Basically Detailed (3-5 points): The response provides some details but is not thorough enough overall.\n   - Not Detailed (1-2 points): The response is very brief and lacks necessary details.\n   Example: If the response provides only a simple conclusion without an explanation, it falls under \"Not Detailed.\"\n4. Relevance:\n   - Highly Relevant (9-10 points): The response is highly relevant to the question, with information closely aligned with the topic.\n   - Generally Relevant (6-8 points): The response is generally relevant but includes some unnecessary information.\n   - Partially Relevant (3-5 points): The response has a lot of content that deviates from the topic.\n   - Not Relevant (1-2 points): The response is completely irrelevant.\n   Example: If the response strays from the topic but still provides some relevant information, it falls under \"Partially Relevant.\"\n\n#### Conversation Context ####\n{}\n#### Responses to be Scored ####\n{}\n\n#### Output Format Requirements ####\n\nOutput with three lines\nSpecific Criteria: <Other potential criteria specific to the query and the context, and the weights of each criteria>.\nAnalysis: <Compare different responses based on given Criteria>.\nScores: <the overall comprehensive score of all responses in order, separate by comma in the boxed, e.g., \\boxed{{x, x}} if there exists 2 responses>."

POINTWISE_PROMPT_TEMPLATE = """You are given a user question and a single response from an AI assistant. Your task is to act as an impartial judge and evaluate how well the response fulfills the user's instructions.

Think carefully about how to assess the quality of the response, and enclose your reasoning within <reasoning> and </reasoning> tags. Your reasoning should include your evaluation criteria, a clear understanding of what an ideal response would look like for this particular question, and a concrete example of such an ideal or reference answer if possible. Then compare the assistant's response to your ideal or reference answer, explaining how it aligns with or deviates from your expectations. Be specific and avoid vague or overly general judgments. Remain as objective as possible. **Be critical and rigorous in your evaluation—do not be lenient.**

In addition to your reasoning, provide a concise, actionable critique of the assistant's response for improvement. The critique should (a) highlight key strengths and weaknesses, (b) point out concrete errors, omissions, safety or factuality issues, and (c) give clear, targeted suggestions for fixing them. Enclose this critique within <critique> and </critique> tags. Important: in the <critique> section, only give analysis and modification suggestions (what to change and how to change it). Do NOT rewrite the full answer, do NOT output a "revised" or "improved" version of the response, and do NOT copy large spans of the original answer. 

Finally, assign the assistant's response a score from 1 to 10. **Use strict standards and fully utilize the entire 1-10 scale.** Use integers only (no decimals). The score distribution should be:
- 1-2: Fundamentally flawed, mostly irrelevant, or severely harmful
- 3-4: Significant issues that prevent it from being useful; major gaps or errors
- 5-6: Partially helpful but with substantial room for improvement; meets only basic requirements
- 7-8: Good quality with some noticeable issues or missing elements
- 9: Very good, minor issues only
- 10: Exceptional, comprehensive, and nearly perfect

**Important calibration notes:**
- **Fully utilize the 1-10 range**: Do not cluster scores in the 7-9 range. Spread your scores across the entire scale based on actual quality.
- Scores of 9-10 should be rare and reserved for truly exceptional responses
- Scores of 1-4 should be given when responses have fundamental problems
- Be especially critical of: factual errors, incomplete answers, poor reasoning, ignoring parts of the question, verbosity without substance, lack of specificity
- Avoid grade inflation: if a response has clear deficiencies, assign a correspondingly lower score

Choose the number that best matches your judgment after applying these strict standards. Enclose the score within <score> and </score> tags.

Format your output like this:
<reasoning> your_thinking_process </reasoning>
<critique> your_critique (only what to fix and how to fix, no rewritten answer) </critique>
<score> your_integer_score_from_1_to_10 </score>

Below are the user's question and the assistant's response:

[User Question]
{instruction}

[The Start of Assistant's Answer]
{response}
[The End of Assistant's Answer]
"""

RUBRIC_BASED_PROMPT_TEMPLATE = """You are an expert evaluator. Given a user prompt, a generated response, and a list of quality rubrics, please rate the overall quality of the response on a scale of 1 to 10 based on how well it satisfies the rubrics.
Consider all rubrics holistically when determining your score. A response that violates multiple rubrics should receive a lower score, while a response that satisfies all rubrics should receive a higher score.
Start your response with <score> and ends with </score>. The value should be an integer between 1 and 10.
Example response:
<score> your_integer_score_from_1_to_10 </score>

Given the following prompt, response, and rubrics, please rate the overall quality of the response on a scale of 1 to 10 based on how well it satisfies the rubrics.

<prompt>
{instruction}
</prompt>

<response>
{response}
</response>

<rubrics>
{rubric_list_string}
</rubrics>

Your evaluation:"""

PAIRWISE_WITH_HIDDEN_REF_TEMPLATE = """You are given a user question and two responses.
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
- Do NOT mention or hint that there was another response, a reference answer, “Assistant A/B”, “the other answer”, “the second answer”, or “the reference”.
- Point out the current answer’s strengths.
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

Below are the user’s question and the two responses:

[User Question]
{instruction}

[The Start of Response A]
{response_a}
[The End of Response A]

[The Start of Response B]
{response_b}
[The End of Response B]
"""

def extract_rubric_score(text: str) -> tuple[str, float]:
    """
    Extract score from GRM response that follows RUBRIC_BASED_PROMPT_TEMPLATE format.
    
    Expected format:
    <critique> your_critique </critique>
    <score> your_score </score>
    
    Args:
        text: The GRM response text
        
    Returns:
        A tuple of (critique: str, score: float). Returns empty string and default score 1.0 if extraction fails.
    """
    text = text.strip()
    critique = ""
    score = 1.0

    # Extract score content between <score> and </score> tags
    score_pattern = r'<score>(.*?)</score>'
    score_match = re.search(score_pattern, text, re.DOTALL | re.IGNORECASE)
    if score_match:
        score_content = score_match.group(1).strip()
        # Extract the first float number from the score content
        score_number_match = re.search(r'(\d+(?:\.\d+)?)', score_content)
        if score_number_match:
            try:
                score = float(score_number_match.group(1))
                # Ensure score is in valid range [1, 10]
                score = max(1.0, min(10.0, score))
            except ValueError:
                score = 1.0
    
    return critique, score

def extract_critique_and_score(text: str) -> tuple[str, float]:
    """
    Extract critique and score from GRM response that follows POINTWISE_PROMPT_TEMPLATE format.
    
    Expected format:
    <critique> your_critique </critique>
    <score> your_score </score>
    
    Args:
        text: The GRM response text
        
    Returns:
        A tuple of (critique: str, score: float). Returns empty string and default score 1.0 if extraction fails.
    """
    text = text.strip()
    critique = ""
    score = 1.0
    
    # Extract critique content between <critique> and </critique> tags
    critique_pattern = r'<critique>(.*?)</critique>'
    critique_match = re.search(critique_pattern, text, re.DOTALL | re.IGNORECASE)
    if critique_match:
        critique = critique_match.group(1).strip()
    
    # Extract score content between <score> and </score> tags
    score_pattern = r'<score>(.*?)</score>'
    score_match = re.search(score_pattern, text, re.DOTALL | re.IGNORECASE)
    if score_match:
        score_content = score_match.group(1).strip()
        # Extract the first float number from the score content
        score_number_match = re.search(r'(\d+(?:\.\d+)?)', score_content)
        if score_number_match:
            try:
                score = float(score_number_match.group(1))
                # Ensure score is in valid range [1, 10]
                score = max(1.0, min(10.0, score))
            except ValueError:
                score = 1.0
    
    return critique, score

def extract_critique_and_answer(text: str) -> tuple[str, float]:
    """
    Extract critique and answer from GRM response that follows PAIRWISE_WITH_HIDDEN_REF_TEMPLATE format.
    
    Expected format:
    <critique> your_critique </critique>
    <answer> [[A]] or [[B]] </answer>
    
    Args:
        text: The GRM response text
        
    Returns:
        A tuple of (critique: str, score: float). 
        - If answer is A, score is 1.0
        - If answer is B, score is 0.0
        - Returns empty string and default score 0.0 if extraction fails.
    """
    text = text.strip()
    critique = ""
    score = 0.0
    
    # Extract critique content between <critique> and </critique> tags
    critique_pattern = r'<critique>(.*?)</critique>'
    critique_match = re.search(critique_pattern, text, re.DOTALL | re.IGNORECASE)
    if critique_match:
        critique = critique_match.group(1).strip()
    
    # Extract answer content between <answer> and </answer> tags
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_match = re.search(answer_pattern, text, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer_content = answer_match.group(1).strip()
        
        # Check various formats in order of priority for robustness
        # 1. Standard format: [[A]] or [[B]]
        if re.search(r'\[\[A\]\]', answer_content, re.IGNORECASE):
            score = 1.0
        elif re.search(r'\[\[B\]\]', answer_content, re.IGNORECASE):
            score = 0.0
        # 2. Other bracket/marker formats: <A>, [A], {A}, (A)
        elif re.search(r'<A>|\[A\]|\{A\}|\(A\)', answer_content, re.IGNORECASE):
            score = 1.0
        elif re.search(r'<B>|\[B\]|\{B\}|\(B\)', answer_content, re.IGNORECASE):
            score = 0.0
        # 3. Final fallback: simple string search (if both A and B exist, use the first one)
        else:
            # Fallback: try to find A or B in the answer content
            if 'A' in answer_content:
                score = 1.0
            elif 'B' in answer_content:
                score = 0.0

    return critique, score

def _init_grm_clients(grm_urls: Union[str, List[str]]) -> List[OpenAI]:
    """Initialize OpenAI clients for GRM nodes with connection pooling"""
    if isinstance(grm_urls, str):
        # Handle comma-separated string
        if "," in grm_urls:
            grm_urls = [url.strip() for url in grm_urls.split(",") if url.strip()]
        else:
            grm_urls = [grm_urls]
    
    if not grm_urls:
        raise ValueError("grm_urls cannot be empty")
    
    return [OpenAI(api_key="EMPTY", base_url=f"{url}/v1") for url in grm_urls]


def call_grm_api(
    messages: List[dict],
    client: OpenAI,
    model_name: str,
    temperature: float = 0.7,
    max_retries: int = DEFAULT_MAX_RETRIES
) -> Optional[str]:
    """
    Call GRM API using OpenAI SDK with retry logic.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        client: OpenAI client instance
        model_name: Model name for inference
        temperature: Sampling temperature
        max_retries: Maximum retry attempts
        
    Returns:
        Response content string, or None if all attempts fail
    """
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
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

async def compute_score_async(data_source, solution_str, ground_truth, extra_info, executor=None, **grm_kwargs):
    """异步计算单个样本的分数（使用线程池执行同步调用）"""
    problem = extra_info["question"]
    
    # Get GRM configuration with defaults
    grm_urls = [
        "http://10.144.168.29:80"
    ]
    model_name = "qwen3-30b-a3b-instruct-2507-tir-cold-start"
    
    # Initialize clients if not provided
    if "grm_clients" not in grm_kwargs:
        grm_kwargs["grm_clients"] = _init_grm_clients(grm_urls)
    
    clients = grm_kwargs["grm_clients"]
    client = random.choice(clients)  # Load balancing
    
    # Build prompt
    prompt = PAIRWISE_WITH_HIDDEN_REF_TEMPLATE.replace(
        "{instruction}", problem
    ).replace(
        "{response_a}", solution_str
    ).replace(
        "{response_b}", ground_truth
    )
    # prompt = RUBRIC_BASED_PROMPT_TEMPLATE.replace(
    #     "{instruction}", problem
    # ).replace(
    #     "{response}", solution_str
    # ).replace(
    #     "{rubric_list_string}", extra_info["rubric"]
    # )
    # prompt = POINTWISE_PROMPT_TEMPLATE.replace(
    #     "{instruction}", problem
    # ).replace(
    #     "{response}", solution_str
    # )
    messages = [{"role": "user", "content": prompt}]
    try:
        # Run sync call in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            executor,  # Use custom executor if provided
            call_grm_api,
            messages,
            client,
            model_name,
            0.7,
            DEFAULT_MAX_RETRIES
        )

        import pdb; pdb.set_trace()
        if response is not None:
            # critique, reward_score = extract_critique_and_score(response)
            critique, reward_score = extract_critique_and_answer(response)
            # critique, reward_score = extract_rubric_score(response)
            return {"score": reward_score, "critique": critique}
    
    except Exception as e:
        print(f"GRM API error: {e}")
    
    return {"score": 0.0, "critique": ""}


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos, max_workers: int = 256, **grm_kwargs):
    """
    批量计算分数
    
    Args:
        data_sources: List of data sources
        solution_strs: List of solution strings to evaluate
        ground_truths: List of ground truth/reference answers
        extra_infos: List of extra info dicts containing 'question' field
        max_workers: Maximum number of concurrent workers (default: 128, recommend: num_grm_nodes * 8-16)
        **grm_kwargs: GRM configuration (grm_url, grm_model_name)
    
    Returns:
        List of score dicts with 'score' and 'critique' fields
    """
    # Pre-initialize clients for efficiency (shared across all tasks)
    if "grm_clients" not in grm_kwargs:
        grm_urls = grm_kwargs.get("grm_url", [
            "http://10.144.168.29:80"
        ])
        grm_kwargs["grm_clients"] = _init_grm_clients(grm_urls)
    
    async def _async_batch_score():
        # Create custom thread pool executor for high concurrency
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = [
                compute_score_async(data_source, solution_str, ground_truth, extra_info, executor, **grm_kwargs)
                for data_source, solution_str, ground_truth, extra_info in zip(
                    data_sources, solution_strs, ground_truths, extra_infos
                )
            ]
            return await asyncio.gather(*tasks)
    
    return asyncio.run(_async_batch_score())

if __name__ == "__main__":
    data_sources = ["wildchat"] * 256
    solution_strs = ["Stoicism: A philosophy that emphasizes reason, self-control, and acceptance of life's circumstances."] * 256
    ground_truths = ["Stoicism: A philosophy valuing reason, virtue, and accepting what's beyond one's control."] * 256
    extra_infos = [{"question": "In 15 words or less, explain Stoicism?", "rubric": "The response is very very detail, and the answer is correct."}] * 256
    
    # 支持单个URL或URL列表
    grm_kwargs = {
        "grm_url": [
            "http://10.144.168.29:80"
        ], 
        "grm_model_name": "qwen3-30b-a3b-instruct-2507-tir-cold-start"
    }

    # 高并发调用：12个节点 * 10 = 120 并发（默认128已足够）
    while True:
        # print(f"Starting computation...")
        time_start = time.time()
        scores = compute_score_batch(
            data_sources, solution_strs, ground_truths, extra_infos,
            max_workers=32,
            **grm_kwargs
        )
        time_end = time.time()
        import pdb; pdb.set_trace()
        # print(f"Time taken: {time_end - time_start} seconds")
        critiques = [score["critique"] for score in scores]
        reward = [score["score"] for score in scores]
        time.sleep(10)

        print(f"Time taken: {time_end - time_start} seconds")
