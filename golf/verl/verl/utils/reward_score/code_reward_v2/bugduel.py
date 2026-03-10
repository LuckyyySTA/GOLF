import ast
import base64
import json
import logging
import os
import pickle
import random
import re
import string
import time
from datetime import datetime
from enum import Enum
from typing import Dict, Literal, Optional, Union

import requests
# from verifiers.rules.code_reward.apis import POOLS_10, POOLS_6, POOLS_95
from pydantic import BaseModel
# import fire
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

random.seed(42)


DEBUG_MODEL_APIS=os.getenv('DEBUG_MODEL_APIS', '').split(',') if os.getenv('DEBUG_MODEL_APIS') else []
DEBUG_MODEL_NAMES=os.getenv('DEBUG_MODEL_NAMES', '').split(',') if os.getenv('DEBUG_MODEL_NAMES') else []


template="""Observe the following code task and correct implementation. Your task is to first add bug(s) to the code and then to explain the bug you add in 50 words. You have to write the implementation again. You should put <bug> </bug>, <exp> </exp> in the beginning and end of the code and explanation.
Make sure the bugs incurs unexpected behaviors and are hard to find. Do not write anything else in your response. You must not add any comment about the bug in code.
<task>
{question}
</task>
<implementation>
{solution}
</implementation>
"""

debug_template="""Analyze the given code task and the faulty code below. Your objective is to identify and fix the issues in the code. 

Requirements:
- Provide your analysis in <think></think> tags (maximum 100 words)
- Provide the corrected complete solution in <code></code> tags
- Do not include any other content in your response

Format your response exactly as:
<think>
[Your brief analysis of the problem and solution approach]
</think>

<code>
[Complete corrected code]
</code>

Here is the code task and the faulty implementation:
<task>
{question}
</task>
<faulty_implementation>
{solution}
</faulty_implementation>
"""

def send_chat_completion(
    user_message="你是谁",
    model="Seed-Coder-8B-Instruct-debug-bs32",
    max_tokens=128,
    base_url="http://10.39.18.251:80/v1",  # 假设使用第二个URL
    authorization_token=""
):
    """
    发送聊天完成请求到指定的API端点
    
    Args:
        user_message (str): 用户消息
        model (str): 模型名称
        max_tokens (int): 最大令牌数
        base_url (str): API基础URL
        authorization_token (str): 授权令牌
    
    Returns:
        dict: API响应的JSON数据
    """
    
    url = f"{base_url}/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {authorization_token}"
    }
    
    data = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user", 
                "content": user_message
            }
        ]
    }
    while True:
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()  # 抛出HTTP错误异常
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            print(f"请求错误: {e}")
            time.sleep(2)
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            time.sleep(2)
            
            
            
def compute_score_debug(solution_str, ground_truth, extra_info,
                       thinking, zero, negative_format_score,
                       format_score, correct_score,
                       extract_all,fix_language, run_timeout,
                       connection_timeout):
    def extract_debug_code(item):
        if '<code>' in item and '</code>' in item:
            code = item.split('<code>')[-1].split('</code>')[0].strip()
        else:
            code = ''
        return code
        
    code = extract_debug_code(solution_str)
    if code == '':
        print('no valid code')
        return 0
    else:
        from verl.utils.reward_score.code_reward_v2.__init__ import \
            compute_score
        return compute_score(
            ground_truth['real_style'], f'```python\n{code}\n```', 
            {**ground_truth, 'style': ground_truth['real_style']}, extra_info=extra_info, 
            thinking=thinking, zero=zero, 
            negative_format_score=negative_format_score,
            format_score=format_score, correct_score=correct_score, 
            extract_all=extract_all,
            fix_language=fix_language,
            run_timeout = run_timeout,
            connection_timeout=connection_timeout
        )['score']


def compute_score_hack(solution_str, ground_truth, extra_info,
                       thinking, zero, negative_format_score,
                       format_score, correct_score,
                       extract_all,fix_language, run_timeout,
                       connection_timeout,
                       debug_n=8):
    def call_debug_model_api(prompt):
        random.shuffle(DEBUG_MODEL_APIS)
        api_url = DEBUG_MODEL_APIS[0]
        response = send_chat_completion(
                    user_message=prompt,
                    model=DEBUG_MODEL_NAMES[0],
                    max_tokens=4096,
                    base_url=api_url,
                )
        return response
    
    def extract_buggy_code(item):
        if '<bug>' in item and '</bug>' in item and '<exp>' in item and '</exp>' in item:
            code = item.split('<bug>')[-1].split('</bug>')[0].strip()
        elif '<bug>' in item and '</bug>' in item and '<think>' in item and '</think>' in item:
            code = item.split('<bug>')[-1].split('</bug>')[0].strip()
        else:
            code = ''
        return code
        
    def extract_debug_code(item):
        if '<think>' in item and '</think>' in item and '<code>' in item and '</code>' in item:
            code = item.split('<code>')[-1].split('</code>')[0].strip()
        else:
            code = ''
        return code

    def is_valid_python_code(text):
        """检查是否为有效的Python代码"""
        try:
            ast.parse(text)
            return True
        except SyntaxError:
            return False
        
    def dump_bugduel(data):
        if int(os.environ.get("RANK")) == 0:
            # 写到json文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            EXPERIMENT_NAME = os.getenv('EXPERIMENT_NAME', 'default_experiment')
            id = extra_info.get('id', 'unknown_id').replace('/','-')
            save_root = f'/cpfs/user/mjren/projects/RLLoggingBoard/bugduel/{EXPERIMENT_NAME}'
            if not os.path.exists(save_root):
                os.makedirs(save_root)
            # 根据 data['solution_str'] 计算一个简短的hash作为文件名的一部分
            hash_name = abs(hash(data['solution_str'])) % (10 ** 8)
            save_path = os.path.join(save_root, f'{id}_{timestamp}_{hash_name}.json')
            print(f'saving to {save_path}')
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=4)
        
    if DEBUG_MODEL_APIS == []:
        raise ValueError("DEBUG_MODEL_APIS is not set")
    assert len(DEBUG_MODEL_APIS) == len(DEBUG_MODEL_NAMES), "DEBUG_MODEL_APIS and DEBUG_MODEL_NAMES must have the same length"

    problem = extra_info['problem']
    code = extract_buggy_code(solution_str)
    # 这里只能放置有效的python代码，否则debug模型无法正常工作
    code = code if is_valid_python_code(code) else ''
    
    if code == '':
        print('no valid code')
        dump_bugduel({'problem': problem, 'solution_str': solution_str, 'DEBUG_MODEL_APIS': DEBUG_MODEL_APIS, 'DEBUG_MODEL_NAMES': DEBUG_MODEL_NAMES, 'info': 'no valid code extracted'})
        return 0
    else:
        if 'assert ' in code or 'test_' in code or 'raise ' in code:
            dump_bugduel({'problem': problem, 'solution_str': solution_str, 'DEBUG_MODEL_APIS': DEBUG_MODEL_APIS, 'DEBUG_MODEL_NAMES': DEBUG_MODEL_NAMES, 'info': 'hack occur, give some error tests'})
            return 0
        # 构建prompt
        prompt = debug_template.format(question=problem, solution=code)
        # 调用 DEBUG_MODEL_APIS 得到n个response
        responses = [call_debug_model_api(prompt) for _ in range(debug_n)]
        # dump 下 responses 以便分析
        
        # 抽取每个response的修正代码
        fixed_codes = [extract_debug_code(item) for item in responses if extract_debug_code(item) != '']
        if len(fixed_codes) == 0:
            # buggy code导致debug模型无法表现出正常行为
            dump_bugduel({'problem': problem, 'solution_str': solution_str, 'prompt': prompt, 'DEBUG_MODEL_APIS': DEBUG_MODEL_APIS, 'DEBUG_MODEL_NAMES': DEBUG_MODEL_NAMES, 'responses': responses, 'fixed_codes': fixed_codes})
            return 0
        else:
            from verl.utils.reward_score.code_reward_v2.__init__ import \
                compute_score

            # 计算每个code的正确性
            scores = []
            for fixed_code in fixed_codes:
                scores.append(compute_score(
                    ground_truth['real_style'], f'```python\n{fixed_code}\n```', 
                    {**ground_truth, 'style': ground_truth['real_style']}, extra_info=extra_info, 
                    thinking=thinking, zero=zero, 
                    negative_format_score=negative_format_score,
                    format_score=format_score, correct_score=correct_score, 
                    extract_all=extract_all,
                    fix_language=fix_language,
                    run_timeout=run_timeout,
                    connection_timeout=connection_timeout
                ))
            # 计算平均分
            avg_score = sum(scores) / len(scores)
            dump_bugduel({'problem': problem, 'solution_str': solution_str, 'DEBUG_MODEL_APIS': DEBUG_MODEL_APIS, 'DEBUG_MODEL_NAMES': DEBUG_MODEL_NAMES, 'responses': responses, 'fixed_codes': fixed_codes, 'scores': scores, 'avg_score': avg_score})
            return 1-avg_score


