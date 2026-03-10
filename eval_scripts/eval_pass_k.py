import json
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch
import time
from tqdm import tqdm

THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"

from generate_vllm import generate_vllm, extract_last_answer, labeling_responses, labeling_responses_batch

def calculate_pass_at_k(c: int, n: int = 20, k: int = 1) -> float:
    import math
    if c == 0:
        return 0.0   
    if k > n:
        return 0.0 
    def n_choose_k(n, k):
        return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
    if n-c < k:
        return 1.0
    else:
        numerator = n_choose_k(n - c, k)
        denominator = n_choose_k(n, k)
        return 1.0 - (numerator / denominator)

def main(input_file, output_file, model_path, remove_system=True, template='own', temperature=0.6, top_p=1.0, max_tokens=8192, n=1, m=1, add_think_before_answer=False, add_oat_evaluate=False, any_true=False, skip_scoring=False, output_eval=None, no_split_think=False, num_processes=32, verify_timeout=60.0):
    df = pd.read_parquet(input_file)
    total_samples = len(df)
    
    print(f"Loading dataset: {input_file}")
    print(f"Total samples: {total_samples}")
    print(f"Model: {model_path}")
    print(f"Template: {template}")
    print(f"Generation parameters: temperature={temperature}, top_p={top_p}, max_tokens={max_tokens}, n={n}")
    print(f"Verification parameters: num_processes={num_processes}, verify_timeout={verify_timeout}")

    from collections import defaultdict
    rets = defaultdict(lambda: defaultdict(list))
    save_data = []
    avg = 0
    
    start_time = time.time()
    
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(torch.cuda.device_count())
    llm = LLM(model=model_path, tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.9)
    
    model_load_time = time.time() - start_time
    print(f"Model loaded successfully in {model_load_time:.2f} seconds")
    
    # Step 1: Generate all responses
    print("\n=== Step 1: Generating responses ===")
    all_generations = []
    generation_start_time = time.time()
    
    progress_bar = tqdm(total=total_samples, desc="Generating", unit="sample")
    
    for idx, (_, item) in enumerate(df.iterrows()):
        message = item['prompt']
        assert remove_system is True
        if remove_system:
            if idx == 0:
                print('remove system')
            assert message[0]['role'] == 'system'
            message = message[1:]
        else:
            assert remove_system is False
            if idx == 0:
                print('not remove system')
                
        answer = item['reward_model']['ground_truth']
        data_source = item['data_source']
        prompt = message[0]['content']

        if idx == 0:
            print(message[0])
            print(f"temperature: {temperature}, top_p: {top_p}, max_tokens: {max_tokens}, n: {n}")

        outputs, gen_prompts = generate_vllm([message], llm, tokenizer, template=template, temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=n)
        
        if idx == 0:
            print('Example input: ', gen_prompts[0])
        
        outputs = [out.outputs[j].text for out in outputs for j in range(n)]
        
        all_generations.append({
            'idx': idx,
            'prompt': prompt,
            'data_source': data_source,
            'answer': answer,
            'outputs': outputs,
        })
        
        progress_bar.update(1)
    
    progress_bar.close()
    generation_time_total = time.time() - generation_start_time
    print(f"Generation completed in {generation_time_total:.2f}s ({total_samples/generation_time_total:.2f} samples/s)")
    
    # Step 2: Preprocess all responses
    print("\n=== Step 2: Preprocessing responses ===")
    all_texts_to_verify = []
    all_answers_to_verify = []
    response_metadata = []  # Track which sample and output each verification belongs to
    
    for gen_data in tqdm(all_generations, desc="Preprocessing"):
        for output_idx, generated_text in enumerate(gen_data['outputs']):
            think_format = False
            answer_format = False
            skip_verify = False
            
            processed_text = generated_text
            
            if THOUGHT_DELIMITER_END in processed_text:
                if THOUGHT_DELIMITER_START not in processed_text or add_think_before_answer is True:
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
            
            all_texts_to_verify.append(processed_text)
            all_answers_to_verify.append(gen_data['answer'])
            response_metadata.append({
                'sample_idx': gen_data['idx'],
                'output_idx': output_idx,
                'skip_verify': skip_verify
            })
    
    # Step 3: Batch verify all responses
    print(f"\n=== Step 3: Batch verifying {len(all_texts_to_verify)} responses ===")
    verification_start_time = time.time()
    
    try:
        all_labels = labeling_responses_batch(
            all_texts_to_verify,
            all_answers_to_verify,
            num_processes=num_processes,
            timeout=verify_timeout
        )
        # Set skipped responses to False
        for i, meta in enumerate(response_metadata):
            if meta['skip_verify']:
                all_labels[i] = False
    except Exception as e:
        print(f"[Error] Batch verification failed: {e}")
        all_labels = [False] * len(all_texts_to_verify)
    
    verification_time_total = time.time() - verification_start_time
    print(f"Verification completed in {verification_time_total:.2f}s ({len(all_texts_to_verify)/verification_time_total:.2f} responses/s)")
    
    # Step 4: Organize results
    print("\n=== Step 4: Computing pass@k metrics ===")
    
    # Group labels by sample
    sample_labels = {}
    for i, meta in enumerate(response_metadata):
        sample_idx = meta['sample_idx']
        if sample_idx not in sample_labels:
            sample_labels[sample_idx] = []
        sample_labels[sample_idx].append(all_labels[i])
    
    # Compute pass@k for each sample
    for gen_data in all_generations:
        idx = gen_data['idx']
        data_source = gen_data['data_source']
        prompt = gen_data['prompt']
        answer = gen_data['answer']
        
        labels = sample_labels[idx]
        pass_count = sum(labels)
        
        label = False
        if pass_count > 0:
            avg += 1 
            label = True
            
        rets[data_source]['correctness'].append(label)
        save_data.append({
            'prompt': prompt,
            'data_source': data_source,
            'answer': answer,
            'correctness': label,
        })
        
        for i in range(n):
            k = i*m
            if k == 0 or k == 1:
                pass_at_1 = calculate_pass_at_k(pass_count, n, 1)
                rets[data_source]['pass_at_1'].append(pass_at_1)
                save_data.append({'pass_at_1': pass_at_1})
            elif k <= n:
                pass_at_k = calculate_pass_at_k(pass_count, n, k)
                rets[data_source][f'pass_at_{k}'].append(pass_at_k)
                save_data.append({f'pass_at_{k}': pass_at_k})
            else:
                break

    total_processing_time = time.time() - start_time
    actual_processing_time = total_processing_time - model_load_time
    
    print(f"\n=== Timing Summary ===")
    print(f"Model loading time: {model_load_time:.2f}s")
    print(f"Generation time: {generation_time_total:.2f}s ({total_samples/generation_time_total:.2f} samples/s)")
    print(f"Verification time: {verification_time_total:.2f}s ({len(all_texts_to_verify)/verification_time_total:.2f} responses/s)")
    print(f"Total processing time: {actual_processing_time:.2f}s")
    print(f"Total time (including model loading): {total_processing_time:.2f}s")
    print(f"Average time per sample: {actual_processing_time/total_samples:.2f}s")
    print(f"Overall speed: {total_samples/actual_processing_time:.2f} samples/s")
    
    accs = []
    evaluation_summary = {}
    
    print(f"\n=== Evaluation Results ===")
    for data_source, result in rets.items():
        # print(data_source, len(labels))
        labels = result['correctness']
        pass_k_results = []
        acc = np.array(labels).mean()
        
        for i in range(n):
            k = i*m
            if k == 0 or k ==1:
                pass_1 = np.array(result['pass_at_1']).mean()
                pass_k_results.append(round(float(pass_1), 4))
            elif k <= n:
                pass_k = np.array(result[f'pass_at_{k}']).mean()
                pass_k_results.append(round(float(pass_k), 4))
            else:
                break
        print(f'{data_source}:pass_at_1={pass_1:.4f} pass_at_{n}={acc:.4f} ({np.sum(labels)}/{len(labels)})')
        accs.append(acc)
        
        evaluation_summary[data_source] = {
            'pass_at_1' : round(float(pass_1), 4),
            f'pass_at_{n}': round(float(acc), 4),
            'pass_at_k_results': pass_k_results,
            'sample_count': len(labels),
            'correct_count': int(np.sum(labels))
        }
    
    overall_acc = np.array(accs).mean()
    evaluation_summary[f'overall_pass_at_{n}_by_source'] = float(overall_acc)
    evaluation_summary['timing_info'] = {
        'model_load_time_seconds': float(model_load_time),
        'generation_time_seconds': float(generation_time_total),
        'verification_time_seconds': float(verification_time_total),
        'total_processing_time_seconds': float(actual_processing_time),
        'average_time_per_sample_seconds': float(actual_processing_time/total_samples),
        'processing_speed_samples_per_second': float(total_samples/actual_processing_time),
        'generation_speed_samples_per_second': float(total_samples/generation_time_total),
        'verification_speed_responses_per_second': float(len(all_texts_to_verify)/verification_time_total)
    }
    evaluation_summary['verification_config'] = {
        'num_processes': num_processes,
        'verify_timeout': verify_timeout,
        'total_responses_verified': len(all_texts_to_verify)
    }
    
    print(f'\nOverall average accuracy by source: {overall_acc:.4f}')
    print(f'Total samples processed: {total_samples}')
    
    try:
        summary_file = output_file.replace('.jsonl', f'_pass_{n}.json')
        with open(summary_file, 'w') as f:
            json.dump(evaluation_summary, f, indent=2, ensure_ascii=False)
        
        # print(f'Results saved to: {output_file}')
        print(f'Results saved to: {summary_file}')
        
    except Exception as e:
        print(f'Error: {e}')
        print(f'Output file: {output_file}')
        

if __name__ == "__main__":
    import fire
    fire.Fire(main)
