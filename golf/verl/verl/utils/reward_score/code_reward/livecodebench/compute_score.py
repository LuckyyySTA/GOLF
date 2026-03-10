# Copyright 2024 PRIME team and/or its affiliates
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

import json
import re
import traceback
from livecodebench import lcb_compute_score, prepare_unit_test_data
import os, pickle
from livecodebench.lcb_runner.benchmarks.code_generation import CodeGenerationProblem, Platform, Difficulty, TestType, Test
from livecodebench.lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics, check_correctness
from livecodebench.lcb_runner.evaluation.pass_k_utils import extract_instance_results
import tempfile
import subprocess
from contextlib import contextmanager
import signal
import ast
import time

IMPORT_PROMPT='''from typing import *

from functools import *
from collections import *
from itertools import *
from heapq import *
from bisect import *
from string import *
from operator import *
from math import *
import math
import datetime
inf = float('inf')

'''

# livecodebench_dir = os.environ.get("LIVECODEBENCH_DATA_PATH", "/cpfs/user/mjren/projects/Skywork-OR1/or1_data/eval/livecodebench/livecodebench_2408_2502")
livecodebench_dir = os.environ.get("LIVECODEBENCH_DATA_PATH", "/cpfs/user/mjren/projects/Skywork-OR1/or1_data/eval/livecodebench/livecodebench_2408_2505")
# if livecodebench_dir is None:
#     raise ValueError("LIVECODEBENCH_DATA_PATH is not set")


@contextmanager
def timeout_run(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("代码执行超时")
    
    # 注册信号处理器
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)

def convert_function_to_class_method(raw_code: str, function_name: str) -> str:
    # 解析原始代码为 AST
    tree = ast.parse(raw_code)
    target_func = None
    new_body = []
    # 遍历顶层节点，保留非目标函数的代码
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            target_func = node
        else:
            new_body.append(node)
    
    if target_func is None:
        return None

    if not (target_func.args.args and target_func.args.args[0].arg == "self"):
        self_arg = ast.arg(arg="self", annotation=None)
        target_func.args.args.insert(0, self_arg)    
    class_def = ast.ClassDef(
        name="Solution",
        bases=[],
        keywords=[],
        body=[target_func],
        decorator_list=[]
    )
    
    new_body.append(class_def)
    tree.body = new_body
    
    # 使用 ast.unparse 将 AST 转换为代码字符串（Python 3.9+支持）
    new_code = ast.unparse(tree)
    return new_code


_MAIN_CLASS_MAP = {
        "CodeGenerationProblem": CodeGenerationProblem,
        "Platform": Platform,
        "Difficulty": Difficulty,
        "TestType": TestType,
        "Test": Test,
    }
class _RedirectingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name in _MAIN_CLASS_MAP:
            return _MAIN_CLASS_MAP[name]
        return super().find_class(module, name)

def compute_score(completion, test_cases, timeout=6, is_binary_reward=True, is_power4_reward=False):
    # 使用r"```python\n(.*?)```"抽取并使用最后一个代码块
    if "question_id" in test_cases:
        try:
            benchmark = _RedirectingUnpickler(open(os.path.join(livecodebench_dir, "{}.pkl".format(test_cases["question_id"])), "rb")).load()
            custom_output = test_cases.copy()
            custom_output["output_list"] = [completion]
            return lcb_compute_score([custom_output], [benchmark]), None
        except:
            traceback.print_exc(10)
            return False, None
    else:
        assert False, "question_id not in test_cases"
