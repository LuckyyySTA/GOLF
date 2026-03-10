#export HF_ENDPOINT=https://hf-mirror.com  
import os
import json
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch
import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import psutil

try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")
    raise

import re

THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"

SYSTEM_PROMPT = "You are a helpful assistant. Please reason step by step to solve the problem and put the final answer within the <answer> </answer> tags."


def single_verify_response(model_output: str, golden_answer: str, timeout_score: float = 0.0):
    """
    Verify a single response using math_verify.
    This function will be executed in a separate process.
    
    Args:
        model_output: Model's output text
        golden_answer: Ground truth answer
        timeout_score: Score to return when timeout occurs
    
    Returns:
        float: Score (1.0 for correct, 0.0 for incorrect)
    """
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    
    ret_score = 0.0
    
    # Handle empty or None output
    if model_output == '' or model_output is None:
        return 0.0
    
    # Add \boxed{} if not already present
    ground_truth_boxed = "\\boxed{" + golden_answer + "}"
    if 'boxed' in model_output:
        formatted_output = model_output
    else:
        formatted_output = "\\boxed{" + model_output + "}"
    
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [formatted_output])
    except TimeoutException:
        ret_score = timeout_score
    except Exception:
        # Catch all other exceptions and treat as incorrect
        ret_score = 0.0
    
    return float(ret_score)


async def async_single_verify(model_output: str, golden_answer: str, executor, timeout: float = 60.0):
    """
    Async wrapper for single verification with timeout.
    
    Args:
        model_output: Model's output text
        golden_answer: Ground truth answer
        executor: ProcessPoolExecutor instance
        timeout: Timeout in seconds
    
    Returns:
        float: Score or None if timeout/error
    """
    loop = asyncio.get_running_loop()
    try:
        future = loop.run_in_executor(
            executor, 
            partial(single_verify_response, model_output, golden_answer, 0.0)
        )
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        print(f"[Timeout] Verification timeout for output: {model_output[:50]}...")
        return 0.0
    except Exception as e:
        print(f"[Error] Verification failed: {e}, output: {model_output[:50]}...")
        return 0.0


async def parallel_verify_async(responses: list[str], golden_answer: str, num_processes: int = 8, timeout: float = 60.0):
    """
    Verify multiple responses in parallel using async processing.
    
    Args:
        responses: List of model responses to verify
        golden_answer: Ground truth answer (single answer for all responses)
        num_processes: Number of parallel processes
        timeout: Timeout per verification task in seconds
    
    Returns:
        List of float scores
    """
    scores = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        try:
            # Create async tasks for all responses
            tasks = [
                async_single_verify(response, golden_answer, executor, timeout=timeout)
                for response in responses
            ]
            results = await asyncio.gather(*tasks, return_exceptions=False)
        except Exception as e:
            print(f"[Exception] Async gather failed: {e}")
            # Return all zeros if gather fails
            return [0.0] * len(responses)
        finally:
            # Ensure all processes are terminated
            terminated_count = 0
            for pid, proc in executor._processes.items():
                try:
                    p = psutil.Process(pid)
                    p.terminate()
                    try:
                        p.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        p.kill()
                    terminated_count += 1
                except Exception:
                    pass
            if terminated_count > 0:
                print(f"[Shutdown] {terminated_count} subprocess(es) terminated.")
    
    # Process results
    for result in results:
        if isinstance(result, Exception) or result is None:
            scores.append(0.0)
        else:
            scores.append(float(result))
    
    return scores


async def parallel_verify_batch_async(responses: list[str], golden_answers: list[str], num_processes: int = 8, timeout: float = 60.0):
    """
    Verify multiple responses with different answers in parallel using async processing.
    
    Args:
        responses: List of model responses to verify
        golden_answers: List of ground truth answers (one per response)
        num_processes: Number of parallel processes
        timeout: Timeout per verification task in seconds
    
    Returns:
        List of float scores
    """
    assert len(responses) == len(golden_answers), "responses and golden_answers must have the same length"
    
    scores = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        try:
            # Create async tasks for all response-answer pairs
            tasks = [
                async_single_verify(response, answer, executor, timeout=timeout)
                for response, answer in zip(responses, golden_answers)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=False)
        except Exception as e:
            print(f"[Exception] Async gather failed: {e}")
            # Return all zeros if gather fails
            return [0.0] * len(responses)
        finally:
            # Ensure all processes are terminated
            terminated_count = 0
            for pid, proc in executor._processes.items():
                try:
                    p = psutil.Process(pid)
                    p.terminate()
                    try:
                        p.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        p.kill()
                    terminated_count += 1
                except Exception:
                    pass
            if terminated_count > 0:
                print(f"[Shutdown] {terminated_count} subprocess(es) terminated.")
    
    # Process results
    for result in results:
        if isinstance(result, Exception) or result is None:
            scores.append(0.0)
        else:
            scores.append(float(result))
    
    return scores


def labeling_responses_batch(responses: list[str], golden_answers: list[str], num_processes: int = 8, timeout: float = 60.0):
    """
    Label responses using async parallel verification with different answers for each response.
    
    Args:
        responses: List of model responses to verify
        golden_answers: List of ground truth answers (one per response)
        num_processes: Number of parallel processes (default 8)
        timeout: Timeout per verification in seconds (default 60)
    
    Returns:
        List of boolean labels indicating correctness
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        scores = loop.run_until_complete(
            parallel_verify_batch_async(responses, golden_answers, num_processes, timeout)
        )
    finally:
        loop.close()
    
    # Convert scores to boolean labels
    return [bool(score) for score in scores]


def labeling_responses(responses: list[str], golden_answer: str, num_processes: int = 8, timeout: float = 60.0):
    """
    Label responses using async parallel verification with timeout handling.
    
    Args:
        responses: List of model responses to verify
        golden_answer: Ground truth answer
        num_processes: Number of parallel processes (default 8)
        timeout: Timeout per verification in seconds (default 60)
    
    Returns:
        List of boolean labels indicating correctness
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        scores = loop.run_until_complete(
            parallel_verify_async(responses, golden_answer, num_processes, timeout)
        )
    finally:
        loop.close()
    
    # Convert scores to boolean labels
    return [bool(score) for score in scores]

def make_conv_zero(question):
    question = question + "\n\nPresent the answer in LaTex format: \\boxed{Your answer}"
    content = f"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {question}. Assistant:"
    return content

def make_conv_zero_code(question):
    question = question + "\n\nWrite Python code to solve the problem. Present the code in \n```python\nYour code\n```\nat the end."
    content = f"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {question}. Assistant:"
    return content

def make_conv_prime_sft(question, tokenizer):
    # for math problem
    content = question + "\n\nPresent the answer in LaTex format: \\boxed{Your answer}"
    # for code problem
    # content = question + "\n\nWrite Python code to solve the problem. Present the code in \n```python\nYour code\n```\nat the end." 
    msg = [
        {"role": "user", "content": content}
    ]
    chat = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    return chat

def apply_qwen_math_template(question: str):
    return (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )

def apply_llama_math_template(question: str):
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "Please reason step by step, and put your final answer within \\boxed{}.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        + question + "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

def simplerl_template(question: str):
    return (
        '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
        + question
        + '\nPlease reason step by step, and put your final answer within\\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n'
    )

def extract_last_answer(text):
    # Find all occurrences of <answer>...</answer>
    answers = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    
    if answers:
        return answers[-1]
    else:
        return ""
    
def score_generated_file(input_file, dec_output_path, output_file, n=1, add_think_before_answer=False, no_split_think=False, model_path=None, num_processes=8, verify_timeout=60.0):
    """
    Score a previously generated file without requiring GPU.
    
    Args:
        input_file: Original input parquet file (for data_source info)
        dec_output_path: Path to the .decoded.jsonl file with generated responses
        output_file: Path to save the scored results
        n: Number of samples per prompt
        add_think_before_answer: Whether to add think delimiter before answer
        no_split_think: Whether to skip splitting think tags
        model_path: Optional model path for tokenizer (for accurate token counting)
        num_processes: Number of parallel processes for verification (default 8)
        verify_timeout: Timeout per verification in seconds (default 60.0)
    """
    # Read original data for data_source info
    df = pd.read_parquet(input_file)
    data_sources = df['data_source'].tolist()
    
    # Load generated outputs
    print(f'Loading generated outputs from: {dec_output_path}')
    jss = []
    with open(dec_output_path, 'r') as f:
        for line in f:
            jss.append(json.loads(line))
    
    outputs = [item['generated_text'] for item in jss]
    prompts = [item['prompt'] for item in jss]
    answers = [item['answer'] for item in jss]
    
    # Initialize tokenizer for response length calculation if model_path is provided
    tokenizer = None
    if model_path:
        print(f'Loading tokenizer from: {model_path}')
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        print('No model_path provided, using word count for token approximation')
    
    print(f'Loaded {len(outputs)} generated responses')
    
    from collections import defaultdict
    from tqdm import tqdm
    
    print('Preprocessing responses...')
    
    # Step 1: Preprocess all responses (handle think/answer format)
    processed_texts = []
    response_lengths = []
    metadata = []  # Store original info for each response
    
    for i in tqdm(range(len(outputs)), total=len(outputs), desc="Preprocessing"):
        generated_text = outputs[i]
        prompt = prompts[i]
        answer = answers[i]
        
        # Calculate which original prompt this response belongs to
        original_prompt_idx = i // n if n > 1 else i
        data_source = data_sources[original_prompt_idx]
        
        # Calculate response length
        if tokenizer:
            response_tokens = tokenizer.encode(generated_text, add_special_tokens=False)
            response_length = len(response_tokens)
        else:
            # Simple token count (approximation without tokenizer)
            response_length = len(generated_text.split())
        
        response_lengths.append(response_length)
        
        # Handle think and answer format
        processed_text = generated_text
        skip_verify = False
        
        think_format = False
        answer_format = False
        if prompt.endswith(THOUGHT_DELIMITER_START+'\n') or add_think_before_answer is True:
            processed_text = THOUGHT_DELIMITER_START + '\n' + processed_text
            think_format = True
        if no_split_think:
            think_format = False

        if '<answer>' in processed_text and '</answer>' in processed_text:
            answer_format = True
            pattern_answer = extract_last_answer(processed_text)
        
        if think_format:
            try:
                processed_text = processed_text.split(THOUGHT_DELIMITER_END)[1]
            except Exception as e:
                skip_verify = True
        if answer_format:
            try:
                processed_text = processed_text + f'The final answer is \\boxed{{{pattern_answer}}}.'
            except Exception as e:
                skip_verify = True
        
        processed_texts.append(processed_text)
        metadata.append({
            'index': i,
            'prompt': prompt,
            'data_source': data_source,
            'original_prompt_idx': original_prompt_idx,
            'answer': answer,
            'skip_verify': skip_verify,
            'response_length': response_length
        })
    
    # Step 2: Batch verify all responses
    print(f'Batch verifying {len(outputs)} responses with {num_processes} parallel processes...')
    
    # Separate responses that need verification
    verify_indices = []
    verify_texts = []
    verify_answers = []
    
    for i, meta in enumerate(metadata):
        if not meta['skip_verify']:
            verify_indices.append(i)
            verify_texts.append(processed_texts[i])
            verify_answers.append(meta['answer'])
    
    print(f'  - {len(verify_texts)} responses to verify')
    print(f'  - {len(outputs) - len(verify_texts)} responses skipped (format errors)')
    
    # Initialize all labels as False
    all_labels = [False] * len(outputs)
    
    if len(verify_texts) > 0:
        try:
            # Batch verify all responses at once!
            verify_labels = labeling_responses_batch(
                verify_texts, 
                verify_answers, 
                num_processes=num_processes, 
                timeout=verify_timeout
            )
            
            # Assign results back to their original positions
            for idx, label in zip(verify_indices, verify_labels):
                all_labels[idx] = label
                
        except Exception as e:
            print(f"[Error] Batch verification failed: {e}")
            # Labels remain False
    
    # Step 3: Organize results
    print('Organizing results...')
    rets = defaultdict(list)
    response_lengths_by_source = defaultdict(list)
    save_data = []
    avg = 0
    prompt_correctness = defaultdict(list)
    
    for i, meta in enumerate(metadata):
        correctness = all_labels[i]
        data_source = meta['data_source']
        original_prompt_idx = meta['original_prompt_idx']
        
        rets[data_source].append(correctness)
        response_lengths_by_source[data_source].append(meta['response_length'])
        prompt_correctness[original_prompt_idx].append(correctness)
        
        save_data.append({
            'prompt': meta['prompt'],
            'data_source': data_source,
            'generated_text': processed_texts[i],
            'answer': meta['answer'],
            'correctness': correctness,
            'response_tokens': meta['response_length']
        })
        if correctness:
            avg += 1
    
    diff_cnt = 0

    print('accuracy: ', avg / len(outputs))
    print('diff_cnt: ', diff_cnt)
    all_response_lengths = [length for lengths in response_lengths_by_source.values() for length in lengths]
    overall_avg_length = np.array(all_response_lengths).mean() if all_response_lengths else 0
    print(f'overall avg response tokens: {overall_avg_length:.2f}')
    
    # Calculate avg@n and pass@n: for each prompt, calculate metrics based on its n responses
    if n > 1:
        # For each prompt, calculate the average correctness of its n responses
        prompt_avg_n = [np.mean(correctness_list) for correctness_list in prompt_correctness.values() if len(correctness_list) > 0]
        overall_avg_n = np.mean(prompt_avg_n) if prompt_avg_n else 0.0
        print(f'avg@{n}: {overall_avg_n:.4f} (average correctness across {len(prompt_avg_n)} prompts)')
        
        # Calculate pass@n: for each prompt, check if at least one response is correct
        prompt_pass_n = [int(any(correctness_list)) for correctness_list in prompt_correctness.values() if len(correctness_list) > 0]
        overall_pass_n = np.mean(prompt_pass_n) if prompt_pass_n else 0.0
        print(f'pass@{n}: {overall_pass_n:.4f} (prompts with at least one correct response out of {len(prompt_pass_n)} prompts)')
        
        # Calculate avg@n and pass@n by data source
        avg_n_by_source = defaultdict(list)
        pass_n_by_source = defaultdict(list)
        for prompt_idx, correctness_list in prompt_correctness.items():
            if len(correctness_list) > 0 and prompt_idx < len(data_sources):
                data_source = data_sources[prompt_idx]
                avg_n_by_source[data_source].append(np.mean(correctness_list))
                pass_n_by_source[data_source].append(int(any(correctness_list)))
    
    accs = []
    avg_lengths = []
    evaluation_summary = {
        'overall_accuracy': avg / len(outputs),
        'overall_avg_response_tokens': overall_avg_length,
        'data_source_results': {}
    }
    
    # Add avg@n and pass@n to summary if n > 1
    if n > 1:
        evaluation_summary[f'avg@{n}'] = float(overall_avg_n)
        evaluation_summary[f'pass@{n}'] = float(overall_pass_n)
    
    for data_source, labels in rets.items():
        acc = np.array(labels).mean()
        avg_length = np.array(response_lengths_by_source[data_source]).mean()
        print(f'{data_source}: accuracy={acc:.4f}, avg_response_tokens={avg_length:.2f}', end='')
        
        # Add avg@n and pass@n for this data source if n > 1
        if n > 1 and data_source in avg_n_by_source:
            avg_n_source = np.array(avg_n_by_source[data_source]).mean()
            pass_n_source = np.array(pass_n_by_source[data_source]).mean()
            print(f', avg@{n}={avg_n_source:.4f}, pass@{n}={pass_n_source:.4f}', end='')
        print()
        
        accs.append(acc)
        avg_lengths.append(avg_length)
        
        evaluation_summary['data_source_results'][data_source] = {
            'accuracy': float(acc),
            'avg_response_tokens': float(avg_length),
            'sample_count': len(labels)
        }
        
        # Add avg@n and pass@n to data source results if n > 1
        if n > 1 and data_source in avg_n_by_source:
            evaluation_summary['data_source_results'][data_source][f'avg@{n}'] = float(np.array(avg_n_by_source[data_source]).mean())
            evaluation_summary['data_source_results'][data_source][f'pass@{n}'] = float(np.array(pass_n_by_source[data_source]).mean())
    
    evaluation_summary['overall_avg_accuracy_by_source'] = float(np.array(accs).mean())
    evaluation_summary['overall_avg_tokens_by_source'] = float(np.array(avg_lengths).mean())
    
    print('avg acc by source: ', np.array(accs).mean())
    print('avg response tokens by source: ', np.array(avg_lengths).mean())
    
    try:
        with open(output_file, 'w') as f:
            for item in save_data:
                f.write(json.dumps(item) + '\n')
        summary_file = output_file.replace('.jsonl', '_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(evaluation_summary, f, indent=2, ensure_ascii=False)
        
        print(f'Results saved to: {output_file}')
        print(f'Summary saved to: {summary_file}')
        
    except Exception as e:
        print(f'Error: {e}')
        print(f'Output file: {output_file}')


def main(input_file, output_file, model_path=None, debug=False, remove_system=True, template='own', temperature=0.6, top_p=1.0, max_tokens=8192, n=1, force_generate=True, add_think_before_answer=False, any_true=False, skip_scoring=False, skip_generation=False, output_eval=None, no_split_think=False, num_processes=8, verify_timeout=60.0):
    """
    Main function for generation and/or scoring.
    
    Args:
        skip_generation: If True, skip generation and only score existing decoded file
        skip_scoring: If True, skip scoring after generation
        model_path: Required for generation, optional for scoring only
        num_processes: Number of parallel processes for verification (default 8)
        verify_timeout: Timeout per verification in seconds (default 60.0)
    """
    df = pd.read_parquet(input_file)
    dec_output_path = output_file.replace('.jsonl', '') + '.decoded.jsonl'
    
    # If skip_generation is True, directly score the existing file
    if skip_generation:
        print('Skipping generation, scoring existing file...')
        if not os.path.exists(dec_output_path):
            raise FileNotFoundError(f'Decoded file not found: {dec_output_path}. Please generate first.')
        score_generated_file(input_file, dec_output_path, output_file, n=n, 
                           add_think_before_answer=add_think_before_answer,
                           no_split_think=no_split_think,
                           model_path=model_path,
                           num_processes=num_processes,
                           verify_timeout=verify_timeout)
        return
    
    # Generation requires model_path
    if model_path is None:
        raise ValueError('model_path is required for generation')
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # print(torch.cuda.device_count())
    llm = LLM(model=model_path, tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.9) 

    if force_generate or (not os.path.exists(dec_output_path)):
        messages = df['prompt'].tolist()
        
        assert remove_system is True
        if remove_system:
            print('remove system')
            assert messages[0][0]['role'] == 'system'
            messages = [message[1:] for message in messages]
            
        else:
            assert remove_system is False
            print('not remove system')
            
        answers = df['reward_model'].tolist()
        answers = [answer['ground_truth'] for answer in answers]
        # if debug:
            # answers = answers[:10]
        assert len(messages) == len(answers)
                
        print(messages[0])
        print(f"temperature: {temperature}, top_p: {top_p}, max_tokens: {max_tokens}, n: {n}, template: {template}")
        outputs, gen_prompts = generate_vllm(messages, llm, tokenizer, template=template, temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=n)
        print('Example input: ', gen_prompts[0])
        # rets = {}
        # save the outputs first
        with open(dec_output_path, 'w') as fo:
            for i, output in enumerate(outputs):
                prompt = output.prompt
                for j in range(n):
                    generated_text = output.outputs[j].text
                    item = {
                        'prompt': prompt,
                        'generated_text': generated_text,
                        'answer': answers[i]
                    }
                    fo.write(json.dumps(item) + '\n')
        
        print(f'Generation completed. Results saved to: {dec_output_path}')
    
    # If skip_scoring is True, stop here
    if skip_scoring:
        print('Skipping scoring as requested.')
        return
    
    # Otherwise, proceed with scoring
    print('Starting scoring...')
    score_generated_file(input_file, dec_output_path, output_file, n=n,
                        add_think_before_answer=add_think_before_answer,
                        no_split_think=no_split_think,
                        model_path=model_path,
                        num_processes=num_processes,
                        verify_timeout=verify_timeout)

def generate_vllm(messages, llm, tokenizer, template='own', temperature=0.6, top_p=0.95, max_tokens=8192, n=1):
    gen_prompts = []
    # max_tokens is for the maximum length for generation.
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=8192, n=n)
    for i in range(len(messages)):
        cur_message = messages[i]
        if template == 'own': 
            message = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": cur_message[0]['content']}
            ]
            gen_prompt = tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True
            )
        elif template == 'refine':
            message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": cur_message[0]['content'] + "\nPlease reason step by step, and put your final answer within \\boxed{}."}
            ]
            gen_prompt = tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        elif template == 'luffy':
            message = [
            {"role": "user", "content": cur_message[0]['content']}
            ]
            gen_prompt = tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True
            )
        elif template == 'simplerl':
            gen_prompt = simplerl_template(cur_message[0]['content'])
        elif template == 'qwen':
            gen_prompt = apply_qwen_math_template(cur_message[0]['content'])
        elif template == 'llama':
            message = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": cur_message[0]['content']}
            ]
            gen_prompt = tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True
            )
        elif template == 'prime':
            gen_prompt = make_conv_zero(cur_message[0]['content'])
        elif template == 'prime_sft':
            gen_prompt = make_conv_prime_sft(cur_message[0]['content'], tokenizer)
        elif template == 'prime_code':
            gen_prompt = make_conv_zero_code(cur_message[0]['content'])
        elif template == 'no':
            gen_prompt = cur_message[0]['content']
        else: raise ValueError(f'Invalid template: {template}')
        gen_prompts.append(gen_prompt)

    outputs = llm.generate(gen_prompts, sampling_params)
    return outputs, gen_prompts

if __name__ == "__main__":
    import fire
    fire.Fire(main)
