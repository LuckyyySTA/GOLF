import ast
import base64
import json
import logging
import os
import pickle
import random
import re
import string
from datetime import datetime
from enum import Enum
from typing import Dict, Literal, Optional, Union

import requests
# from verifiers.rules.code_reward.apis import POOLS_10, POOLS_6, POOLS_95
from pydantic import BaseModel
# import fire
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

DEFAULT_RUNTIMEOUT = 30
DEFAULT_CONNECTION_TIMEOUT = 60


MAX_RETRY=int(os.getenv('SANDBOX_MAX_RETRY', 5))
SUPPORT_LANGUAGES=['python', 'pytest', 'cpp', 'nodejs', 'go', 'go_test', 'java', 'csharp', 'typescript', 'rust']

# API_POOLS=POOLS_95
# OJ_API_POOLS=POOLS_6+POOLS_10
os.environ['API_POOLS'] = 'https://agi-mcp.devops.xiaohongshu.com/api/code/code_sandbox'
os.environ['OJ_API_POOLS'] = 'https://agi-mcp.devops.xiaohongshu.com/api/code/code_sandbox'

API_POOLS=os.getenv('API_POOLS', '').split(',') if os.getenv('API_POOLS') else None
OJ_API_POOLS=os.getenv('OJ_API_POOLS', '').split(',') if os.getenv('OJ_API_POOLS') else None

assert not(API_POOLS is None or len(API_POOLS) == 0), 'Please set environment variable API_POOLS'
assert not(OJ_API_POOLS is None or len(OJ_API_POOLS) == 0), 'Please set environment variable OJ_API_POOLS'

print(f'API_POOLS: {API_POOLS}')
print(f'OJ_API_POOLS: {OJ_API_POOLS}')

cnt=0

def get_endpoint():
    API = random.choice(API_POOLS)
    print(API)
    return f'{API}'

def get_oj_endpoint():
    API = random.choice(OJ_API_POOLS)
    print(API)
    return f'{API}'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def normalize_language(language):
    language_map = {
        # JavaScript
        'javascript': 'nodejs',
        'js': 'nodejs',
        
        # Python
        'python3': 'python',
        'py': 'python',
        'py3': 'python',
        
        # C++
        'c++': 'cpp',
        'cxx': 'cpp',
        'cc': 'cpp',
        
        # Java
        'java8': 'java',
        'java11': 'java',
        'java17': 'java',
        'java21': 'java',
        
        # C#
        'c#': 'csharp',
        'cs': 'csharp',
        'dotnet': 'csharp',
        '.net': 'csharp',
        
        # TypeScript
        'ts': 'typescript',
        
        # Go
        'golang': 'go',
        
        # Rust
        'rs': 'rust',
    }
    
    normalized = language.lower()
    return language_map.get(normalized, normalized)


class CommandRunStatus(str, Enum):
    Finished = 'Finished'
    TimeLimitExceeded = 'TimeLimitExceeded'
    # ignore this in logic as this state cause run_code to throw
    Error = 'Error'


class CommandRunResult(BaseModel):
    status: CommandRunStatus
    execution_time: Optional[float] = None
    return_code: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None


class RunCodeRequest(BaseModel):
    code: str
    language: Literal['python', 'pytest', 'cpp', 'nodejs', 'go', 'go_test', 'java', 'csharp', 'typescript', 'rust']
    compile_timeout: float = 30
    run_timeout: float = DEFAULT_RUNTIMEOUT
    files: Dict[str, str] = {}

class OJConfig(BaseModel):
    language: Literal['python', 'pytest', 'cpp', 'nodejs', 'go', 'go_test', 'java', 'csharp', 'typescript', 'rust']
    compile_timeout: float = 30
    run_timeout: float = DEFAULT_RUNTIMEOUT
    locale: str="en"
    dataset_type: str="CommonOJDataset"
    provided_data: dict = {}
    extra: dict = {}

class OJRequest(BaseModel):
    completion: str
    dataset: str='code_contests_train'
    id: str=''
    config: OJConfig

class RunStatus(str, Enum):
    # all command finished successfully
    Success = 'Success'
    # one of the process has non-zero return code
    Failed = 'Failed'
    # error on sandbox side, ignore this in logic as this state cause run_code to throw
    SandboxError = 'SandboxError'

class RunCodeResponse(BaseModel):
    status: RunStatus
    message: str
    compile_result: Optional[CommandRunResult] = None
    run_result: Optional[CommandRunResult] = None


class OJResponse(BaseModel):
    extracted_code: str
    accepted: bool = None
    tests: list = None

def on_retry_error(s):
    e = s.outcome.exception()
    logger.error(f'give up requesting sandbox. error: {e}')
    raise e


def before_retry_sleep(s):
    logger.warning(f'error requesting sandbox for {s.attempt_number} time(s), will retry... error: {s.outcome.exception()}')


def is_malicuous_code(code: str) -> bool:
    if 'os.killpg' in code:
        return True
    else:
        return False

def save_failed_request(request_data: Union[RunCodeRequest, OJRequest]):
    # 生成文件名：时间戳 + 随机5字符（字母数字）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
    filename = f"/cpfs/user/mjren/logs/sandboxfusion/aborted_{timestamp}_{random_str}.pkl"
    
    # 使用 pickle 存储对象
    with open(filename, 'wb') as f:  # 注意是二进制写入模式 'wb'
        pickle.dump(request_data, f)
    
    print(f"请求失败，已使用 pickle 存储到文件: {filename}")
    
    
class AutoTestRequest(BaseModel):
    language: Literal['python-test', 'cpp-test', 'go-test', 'java-test', 'javascript-test', 'rust-test']
    compile_timeout: float = 30
    run_timeout: float = DEFAULT_RUNTIMEOUT
    files: Dict[str, str] = {}
    
@retry(wait=wait_exponential_jitter(max=1),
       stop=stop_after_attempt(MAX_RETRY),
       before_sleep=before_retry_sleep,
       retry_error_callback=on_retry_error)
def autotest_in_sandbox(request: AutoTestRequest, connection_timeout, run_timeout) -> RunCodeResponse:
    # 计算post使用时间
    start_time = datetime.now()
    result = requests.post(get_endpoint()+'/auto_test', 
                            json=request.model_dump(), 
                            timeout=connection_timeout)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    if duration > connection_timeout or duration > run_timeout:
        logger.warning(f'autotest_in_sandbox request took too long: {duration} seconds (connection timeout {connection_timeout})')
        return None
    # print(result.json())
    # print(request.model_dump())
    if result.status_code == 500 and ('error_code' in json.loads(result.text) and json.loads(result.text)['error_code'] == 'function_proxy_error'):
        return None
    if result.status_code != 200:
        if duration > run_timeout:
            return None
        raise Exception(f'Faas api responded with code {result.status_code}: {result.text}')
    resp = RunCodeResponse(**result.json())
    if resp.status == RunStatus.SandboxError:
        if duration > run_timeout:
            return None
        raise Exception(f'Sanbox responded with error: {resp.message}')
    logger.debug(f'sandbox request success. request = {request.model_dump_json(indent=2)}. response = {resp.model_dump_json(indent=2)}')
    return resp


# wait=wait_exponential_jitter(): 这是定义等待重试的时间策略。wait_exponential_jitter() 结合了指数递增和随机抖动的等待时间。每次重试之间的等待时间会成指数增长，但为了避免高峰或拥塞，它还会加上随机抖动。
# stop=stop_after_attempt(12): 这是定义重试的终止条件。这里表示最多会进行 12 次重试。
# before_sleep=before_retry_sleep: 这个参数指定了在每次重试前执行的操作，比如记录日志或处理其他逻辑。before_retry_sleep 可能是一个回调函数，用来在重试之前记录或处理某些数据。
# retry_error_callback=on_retry_error: 当所有的重试都失败后，会执行 on_retry_error 这个回调函数，用于处理最终的错误，可能记录日志或采取其他补救措施。
@retry(wait=wait_exponential_jitter(max=1),
       stop=stop_after_attempt(MAX_RETRY),
       before_sleep=before_retry_sleep,
       retry_error_callback=on_retry_error)
def run_code_in_sandbox(request: RunCodeRequest, connection_timeout, run_timeout) -> RunCodeResponse:
    start_time = datetime.now()
    result = requests.post(get_endpoint()+'/run_code', 
                            json=request.model_dump(), 
                            timeout=connection_timeout)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    if duration > connection_timeout or duration > run_timeout:
        logger.warning(f'run_code_in_sandbox request took too long: {duration} seconds (connection timeout {connection_timeout})')
        return None

    if result.status_code == 500 and ('error_code' in json.loads(result.text) and json.loads(result.text)['error_code'] == 'function_proxy_error'):
        return None
    if result.status_code != 200:
        if duration > run_timeout:
            return None
        raise Exception(f'Faas api responded with code {result.status_code}: {result.text}')
    resp = RunCodeResponse(**result.json())
    if resp.status == RunStatus.SandboxError:
        if duration > run_timeout:
            return None
        raise Exception(f'Sanbox responded with error: {resp.message}')
    logger.debug(f'sandbox request success. request = {request.model_dump_json(indent=2)}. response = {resp.model_dump_json(indent=2)}')
    return resp


@retry(wait=wait_exponential_jitter(max=1),
       stop=stop_after_attempt(MAX_RETRY),
       before_sleep=before_retry_sleep,
       retry_error_callback=on_retry_error)
def oj_in_sandbox(request: OJRequest, connection_timeout, run_timeout) -> RunCodeResponse:
    # print(request.model_dump())
    start_time = datetime.now()
    result = requests.post(get_oj_endpoint()+'/submit', 
                            json=request.model_dump(), 
                            timeout=connection_timeout)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    if duration > connection_timeout or duration > run_timeout:
        logger.warning(f'oj_in_sandbox request took too long: {duration} seconds (connection timeout {connection_timeout})')
        return None
    
    if result.status_code == 500 and ('error_code' in json.loads(result.text) and json.loads(result.text)['error_code'] == 'function_proxy_error'):
        return None
    if result.status_code != 200:
        if duration > run_timeout:
            return None
        raise Exception(f'Faas api responded with code {result.status_code}: {result.text}')
    resp = OJResponse(**result.json())
    logger.debug(f'sandbox request success. request = {request.model_dump_json(indent=2)}. response = {resp.model_dump_json(indent=2)}')
    return resp


class SummaryMapping(BaseModel):
    Success: str = RunStatus.Success
    Failed: str = RunStatus.Failed
    CompileFailed: Optional[str] = None
    CompileTimeout: Optional[str] = None
    RunFailed: Optional[str] = None
    RunTimeout: Optional[str] = None


def summary_result(result: RunCodeResponse, mapping: SummaryMapping) -> str:
    if result.compile_result is None and result.run_result is None:
        # note: this should not happen
        if result.status == RunStatus.Success:
            return mapping.Success
        if result.status == RunStatus.Failed:
            return mapping.Failed
        raise Exception(f'unexpected result status {result.status}')
    if result.run_result is None:
        # compile error
        if result.compile_result.status == CommandRunStatus.TimeLimitExceeded:
            return mapping.CompileTimeout or mapping.Failed
        return_code = result.compile_result.return_code
        if return_code is None:
            raise Exception(f'invalid sandbox result: no return code with status {result.compile_result.status}')
        if return_code != 0:
            return mapping.CompileFailed or mapping.Failed
        raise Exception(f'invalid sandbox result: compiled succesfully with no run result')
    if result.run_result.status == CommandRunStatus.TimeLimitExceeded:
        return mapping.RunTimeout or mapping.Failed
    return_code = result.run_result.return_code
    # print(f'return_code {return_code}')


def extract_solutions(solution_str, thinking, zero, extract_all=False, default_language='python'):
    """Extract the equation from the solution string."""
    # 修改逻辑，将所有代码块都抽取出来
    # Remove everything before the first "Assistant:"
    # TODO support extract from <answer> tag
    # print(f'zero : {zero}, thinking: {thinking}')
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    elif "<|start_header_id|>assistant" in solution_str:
        solution_str = solution_str.split("<|start_header_id|>assistant", 1)[1]
    
    # 正则匹配 <answer> 标签中可能存在的代码块 (包括语言和代码)
    # answer_pattern = r'<answer>.*?```(\w+)?\n(.*?)```.*?</answer>'
    # TODO为了支持多格式（部分模型的answer不会wrap），直接使用```提取
    if not thinking:
        # answer_pattern = r'```(\w+)?\n(.*?)```'
        answer_pattern = r'```([a-zA-Z0-9+#-]*)?\n(.*?)```'
        matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))  # re.DOTALL 允许匹配多行内容
    elif thinking and not zero:
        # if '</think>' not in solution_str:
        if solution_str.count('<think>')!=1 or solution_str.count('</think>')!=1:
            return [(None, None)]
        else:
            solution_str = solution_str.split('</think>')[-1]
            # answer_pattern = r'```(\w+)?\n(.*?)```'
            answer_pattern = r'```([a-zA-Z0-9+#-]*)?\n(.*?)```'
            matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    elif thinking and zero:
        if '</think>' not in solution_str:
            return [(None, None)]
        else:
            answer_pattern = r'<answer>.*?```(\w+)?\n(.*?)```.*?</answer>'
            answer_pattern = r'<answer>.*?```([a-zA-Z0-9+#-]*)?\n(.*?)```.*?</answer>'
            matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))

    # 如果找到匹配项，提取最后一个 <answer> 中的语言和代码
    if matches:
        if not extract_all:
            last_match = matches[-1]
            print('extract last code block only')
            # last_match = matches[0]
            # print('extract first code block only')
            language = last_match.group(1) or default_language  # 如果没有指定语言，默认为 python
            code = last_match.group(2).strip()
            language = normalize_language(language)
            return [(language, code)]
        else:
            return [(normalize_language((match.group(1) or default_language)), match.group(2).strip()) for match in matches]
        # return language, code
    else:
        return [(None, None)]
    


def compute_score_stdio(solution_str, ground_truth, 
                        thinking, zero, import_str, 
                        run_timeout,
                        format_score, 
                        negative_format_score,
                        correct_score,
                        extract_all,
                        fix_language,
                        connection_timeout):
    """The scoring function for countdown task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        correct_score: the score for the correct answer
    """
    exec_times = []
    scores = []
    
    global cnt
    solutions = extract_solutions(solution_str=solution_str, thinking=thinking, zero=zero, extract_all=extract_all, default_language='python' if not fix_language else fix_language)
    
    for language, code in solutions:
        exec_time = None
        
        if language is None or language not in SUPPORT_LANGUAGES or code is None:
            print(f'unsupported language {language}')
            scores.append(negative_format_score)
            exec_times.append(None)
            continue
        if fix_language and normalize_language(language) != normalize_language(fix_language):
            scores.append(negative_format_score)
            exec_times.append(None)
            continue
        if is_malicuous_code(code):
            scores.append(format_score)
            exec_times.append(None)
            continue
        
        cnt += 1
        tests = ground_truth
        reward = 0
        oj_data = {
            "id": 1,                          # Unique identifier
            "content": '',                     # Problem statement
            "test": tests
        }
        try:
            result = oj_in_sandbox(OJRequest(completion=f"```{language}\n{import_str}\n\n{code}\n```", 
                                                config=OJConfig(language=language, provided_data=oj_data, 
                                                                extra={'run_all_cases': False},
                                                                run_timeout=run_timeout if run_timeout else DEFAULT_RUNTIMEOUT,)), 
                                    connection_timeout=connection_timeout if connection_timeout else DEFAULT_CONNECTION_TIMEOUT,
                                    run_timeout=run_timeout if run_timeout else DEFAULT_RUNTIMEOUT)
            exec_time = [(test['exec_info']['run_result']['execution_time'], test['test_info']) for test in result.tests]
            
            if result is None:
                reward = format_score
            else:
                # assert code == result.extracted_code, f'code not equal: {code} vs {result.extracted_code}'
                pass_rate = int(result.accepted)
                reward = pass_rate*correct_score + format_score
        except requests.exceptions.ConnectionError as e:
            reward = format_score
        except Exception as e:
            logger.warning(f'oj_in_sandbox failed: {e}')
            reward = format_score
            
        scores.append(reward)
        exec_times.append(exec_time)
        
    valid_exec_times = [sum([i[0] for i in exec_time]) for exec_time, score in zip(exec_times, scores) if score==correct_score + format_score]
    return max(scores), min(valid_exec_times) if valid_exec_times else None
    
    
def compute_score_pytest(solution_str, pytest_code, 
                         thinking, zero, import_str, 
                         run_timeout,
                         format_score, 
                         negative_format_score,
                         correct_score,
                         extract_all,
                         connection_timeout):
    exec_times = []
    scores = []
    
    global cnt
    solutions = extract_solutions(solution_str=solution_str, thinking=thinking, zero=zero, extract_all=extract_all)

    for language, code in solutions:
        exec_time = None
        
        if language is None or language not in SUPPORT_LANGUAGES or code is None:
            # return negative_format_score, exec_time
            print(f'unsupported language {language}')
            scores.append(negative_format_score)
            exec_times.append(None)
            continue
        
        if is_malicuous_code(code):
            # return format_score, exec_time
            scores.append(format_score)
            exec_times.append(None)
            continue
        
        cnt += 1
        reward = 0
        try:
            result = run_code_in_sandbox(RunCodeRequest(code=f'{import_str}\n\n{code}\n\n{pytest_code}', 
                                                        language='pytest',
                                                        run_timeout=run_timeout if run_timeout else DEFAULT_RUNTIMEOUT),            
                                            connection_timeout=connection_timeout if connection_timeout else DEFAULT_CONNECTION_TIMEOUT,
                                            run_timeout=run_timeout if run_timeout else DEFAULT_RUNTIMEOUT)
            if result is None:
                reward = format_score
            else:
                exec_time = result.run_result.execution_time
                pass_rate = 1 if result.status == RunStatus.Success else 0
                reward = pass_rate*correct_score+format_score
        except requests.exceptions.ConnectionError as e:
            reward = format_score
        except Exception as e:
            logger.warning(f'oj_in_sandbox failed: {e}')
            reward = format_score
        
        scores.append(reward)
        exec_times.append(exec_time)
    # return reward, exec_time
    valid_exec_times = [exec_time for exec_time, score in zip(exec_times, scores) if score==correct_score + format_score]
    return max(scores), min(valid_exec_times) if valid_exec_times else None


def compute_score_aider_format(response,
                        tests: dict,
                        already_solutions: dict,
                        test_language: str,
                        run_timeout,
                        correct_score,
                        connection_timeout):
    
    def extract_editing_solutions(response):
        # 匹配代码块的正则表达式
        # 匹配 filename.ext 后面跟着 ``` 开始的代码块
        def format_filename(filename):
            # - java 需要加上路径src/main/java
            # - rust 需要加上路径src
            if filename.endswith('.java'):
                return f'src/main/java/{filename}'
            elif filename.endswith('.rs'):
                return f'src/{filename}'
            else:
                return filename
        
        pattern = r'(\S+\.\w+)\s*\n```(?:\w+)?\s*\n(.*?)\n```'
        matches = re.findall(pattern, response, re.DOTALL)
        results = {format_filename(filename): content for filename, content in matches}
        
        return results
    
    test_language2command = {
        'python': 'python-test',
        'java': 'java-test',
        'cpp': 'cpp-test',
        'javascript': 'javascript-test',
        'go': 'go-test',
        'rust': 'rust-test',
    }
    test_command = test_language2command.get(test_language)
    
    global cnt
    exec_time = None
    
    reward = 0
    solutions = extract_editing_solutions(response)
    encodeed_solution = {solution: base64.b64encode(solutions[solution].encode('utf-8')).decode('utf-8') for solution in solutions}
    encodeed_already_solutions = {solution: base64.b64encode(already_solutions[solution].encode('utf-8')).decode('utf-8') for solution in already_solutions}
    encodeed_test = {test: base64.b64encode(tests[test].encode('utf-8')).decode('utf-8') for test in tests}
    
    try:
        result = autotest_in_sandbox(AutoTestRequest(language=test_command,
                                                    files={**encodeed_already_solutions, **encodeed_solution, **encodeed_test},   # test要放在后面, 防止被模型生成的solution hack
                                                    run_timeout=run_timeout if run_timeout else DEFAULT_RUNTIMEOUT), 
                                        run_timeout=run_timeout if run_timeout else DEFAULT_RUNTIMEOUT,
                                        connection_timeout=connection_timeout if connection_timeout else DEFAULT_CONNECTION_TIMEOUT)
        
        exec_time = result.run_result.execution_time
        if result is None:
            reward = 0
        else:
            pass_rate = 1 if result.status == RunStatus.Success else 0
            reward = pass_rate*correct_score
    except requests.exceptions.ConnectionError as e:
        reward = 0
    except Exception as e:
        logger.warning(f'oj_in_sandbox failed: {e}')
        reward = 0
    
    return reward, exec_time


def compute_score_assertions(solution_str, assertions, entry_point_str,
                             thinking, zero, import_str, 
                             run_timeout,
                             format_score, 
                             negative_format_score,
                             correct_score,
                             extract_all,
                             connection_timeout):
    exec_times = []
    scores = []
    
    exec_time = None
    global cnt
    solutions = extract_solutions(solution_str=solution_str, thinking=thinking, zero=zero, extract_all=extract_all)
    
    for language, code in solutions:
        
        exec_time = None
        
        if language is None or language not in SUPPORT_LANGUAGES or code is None:
            print(f'unsupported language {language}')
            scores.append(negative_format_score)
            exec_times.append(None)
            continue
        if is_malicuous_code(code):
            scores.append(format_score)
            exec_times.append(None)
            continue
            
        cnt += 1
        reward = 0
        solution=f"{import_str}\n\n{code}"
        encodeed_solution = base64.b64encode(solution.encode('utf-8')).decode('utf-8')
        try:
            assertions_code = "\n".join(assertions)
            
            if 'class Solution' not in solution:
                final_code = f"""
{import_str}
from solution import {entry_point_str}

{assertions_code}
                """
            else:
                final_code = f"""
{import_str}
from solution import Solution

{entry_point_str} = Solution().{entry_point_str}

{assertions_code}
                """
            
            result = run_code_in_sandbox(RunCodeRequest(code=final_code, 
                                                        language=language,
                                                        files={'solution.py': encodeed_solution},
                                                        run_timeout=run_timeout if run_timeout else DEFAULT_RUNTIMEOUT), 
                                            connection_timeout=connection_timeout if connection_timeout else DEFAULT_CONNECTION_TIMEOUT,
                                            run_timeout=run_timeout if run_timeout else DEFAULT_RUNTIMEOUT)
            exec_time = result.run_result.execution_time
            if result is None:
                reward = format_score
            else:
                pass_rate = 1 if result.status == RunStatus.Success else 0
                reward = pass_rate*correct_score+format_score
        except requests.exceptions.ConnectionError as e:
            reward = format_score
        except Exception as e:
            logger.warning(f'oj_in_sandbox failed: {e}')
            reward = format_score
        
        scores.append(reward)
        exec_times.append(exec_time)
            
    # return reward, exec_time
    valid_exec_times = [exec_time for exec_time, score in zip(exec_times, scores) if score==correct_score + format_score]
    return max(scores), min(valid_exec_times) if valid_exec_times else None


def compute_score_call(solution_str, test_functions, entry_point_str,
                       thinking, zero, import_str, 
                       run_timeout,
                       format_score, 
                       negative_format_score,
                       correct_score,
                       extract_all,
                       connection_timeout):
    exec_times = []
    scores = []
    
    exec_time = None
    global cnt
    solutions = extract_solutions(solution_str=solution_str, thinking=thinking, zero=zero, extract_all=extract_all)
    
    for language, code  in solutions:
        
        if language is None or language not in SUPPORT_LANGUAGES or code is None:
            print(f'unsupported language {language}')
            scores.append(negative_format_score)
            exec_times.append(None)
            continue
        if is_malicuous_code(code):
            scores.append(format_score)
            exec_times.append(None)
            continue
        
        cnt += 1
        reward = 0
        solution=f"{import_str}\n\n{code}"
        encodeed_solution = base64.b64encode(solution.encode('utf-8')).decode('utf-8')
        try:
            test_function_str = '\n\n'.join(test_functions)
            if '()' not in entry_point_str:
                final_code = f"""
{import_str}
from solution import {entry_point_str}

{test_function_str}

check({entry_point_str})
                """
            else:
                part1 = entry_point_str.split('().')[0]
                part2 = entry_point_str.split('().')[1]
                final_code = f"""
{import_str}
from solution import {part1}

{test_function_str}


check({part1}().{part2})
                """
            result = run_code_in_sandbox(RunCodeRequest(code=final_code, language=language,
                                                        files={'solution.py': encodeed_solution},
                                                        run_timeout=run_timeout if run_timeout else DEFAULT_RUNTIMEOUT), 
                                            connection_timeout=connection_timeout if connection_timeout else DEFAULT_CONNECTION_TIMEOUT,
                                            run_timeout=run_timeout if run_timeout else DEFAULT_RUNTIMEOUT)
            exec_time = result.run_result.execution_time
            if result is None:
                reward = format_score
            else:
                pass_rate = 1 if result.status == RunStatus.Success else 0
                if pass_rate == 0:
                    pass
                reward = pass_rate*correct_score+format_score
        except requests.exceptions.ConnectionError as e:
            reward = format_score
        except Exception as e:
            logger.warning(f'oj_in_sandbox failed: {e}')
            reward = format_score
        
        scores.append(reward)
        exec_times.append(exec_time)
    
    valid_exec_times = [exec_time for exec_time, score in zip(exec_times, scores) if score==correct_score + format_score]
    return max(scores), min(valid_exec_times) if valid_exec_times else None

