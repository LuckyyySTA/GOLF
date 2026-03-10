import sys
import os
# Ensure package dir is on path when run as script
_code_reward_v2_dir = os.path.dirname(os.path.abspath(__file__))
if _code_reward_v2_dir not in sys.path:
    sys.path.insert(0, _code_reward_v2_dir)

import bugduel
import numpy as np
import orjson

import sandbox_verify
import refine


def compute_score(verify_style, solution_str, ground_truth, extra_info=None,
                    thinking=False, zero=False, 
                    negative_format_score=-2,
                    format_score=0.1, correct_score=1., 
                    extract_all=False,
                    fix_language=False,
                    run_timeout = 20,
                    connection_timeout=60):
    # 可以按照不同类型的题目进行分流，如基于调用的、标准输入输出的题、代码理解题等
    if 'fix' in verify_style:
        tag = {'fix-correct': True, 'fix-incorrect': False}[verify_style]
        score = refine.compute_score_fix(
            solution_str, tag, ground_truth, extra_info,
            thinking, zero, negative_format_score,
            format_score, correct_score,
            extract_all,fix_language, run_timeout,
            connection_timeout,
        )
        print(f'fix score is {score}')
    elif 'dots_code_editing' in verify_style:
        score = sandbox_verify.compute_score_aider_format(solution_str,
                        tests={item[0]: item[1] for item in ground_truth['tests']},
                        already_solutions = {item[0]: item[1] for item in ground_truth['already_solutions']},
                        test_language=ground_truth['language'],
                        run_timeout=run_timeout,
                        correct_score=correct_score,
                        connection_timeout=connection_timeout)[0]
    elif 'sandbox[assertions]' in verify_style:
        score = sandbox_verify.compute_score_assertions(solution_str, 
                                                       entry_point_str=ground_truth['entry_point'],
                                                       assertions=ground_truth['assertions'].tolist() if isinstance(ground_truth['assertions'], np.ndarray) else ground_truth['assertions'],
                                                       import_str=ground_truth['pre_import'],
                                                       thinking=thinking, zero=zero,
                                                       negative_format_score=negative_format_score,
                                                       format_score=format_score, correct_score=correct_score,
                                                       extract_all=extract_all,
                                                       run_timeout=run_timeout,
                                                       connection_timeout=connection_timeout)[0]
    elif 'sandbox[standard_in_out]' in verify_style:
        score = sandbox_verify.compute_score_stdio(solution_str, 
                                                ground_truth['unitests'].tolist() if isinstance(ground_truth['unitests'], np.ndarray) else ground_truth['unitests'],
                                                import_str=ground_truth['pre_import'],
                                                thinking=thinking, zero=zero,
                                                negative_format_score=negative_format_score,
                                                format_score=format_score, correct_score=correct_score,
                                                extract_all=extract_all,
                                                run_timeout=run_timeout,
                                                fix_language=ground_truth['language'] if fix_language else None,
                                                connection_timeout=connection_timeout)[0]
    elif 'sandbox[pytest]' in verify_style:
        score = sandbox_verify.compute_score_pytest(solution_str, 
                                                   ground_truth['pytest'],
                                                   import_str=ground_truth['pre_import'],
                                                   thinking=thinking, zero=zero,
                                                   negative_format_score=negative_format_score,
                                                   format_score=format_score, correct_score=correct_score,
                                                   extract_all=extract_all,
                                                   run_timeout=run_timeout,
                                                   connection_timeout=connection_timeout)[0]
    elif 'sandbox[call-based]' in verify_style:
        score = sandbox_verify.compute_score_call(solution_str, 
                                                 test_functions=ground_truth['check_functions'].tolist() if isinstance(ground_truth['check_functions'], np.ndarray) else ground_truth['check_functions'],
                                                 entry_point_str=ground_truth['entry_point'],
                                                 import_str=ground_truth['pre_import'],
                                                 thinking=thinking, zero=zero,
                                                 negative_format_score=negative_format_score,
                                                 format_score=format_score, correct_score=correct_score,
                                                 extract_all=extract_all,
                                                 run_timeout=run_timeout,
                                                 connection_timeout=connection_timeout)[0]
    elif 'sandbox[hack]' in verify_style:
        # TODO 这里需要调用代理模型, 考察生成的buggy code对代理模型的难度, 然后根据难度来给reward
        # TODO extra_info需要包含原始的problem
        score = bugduel.compute_score_hack(solution_str, ground_truth, extra_info,
                       thinking, zero, negative_format_score,
                       format_score, correct_score,
                       extract_all,fix_language, run_timeout,
                       connection_timeout)
    elif 'sandbox[debug]' in verify_style:
        # TODO 这里不需要代理奖励了, 直接根据抽取的执行结果得到reward
        # 在extra_info['prompt']放置原始的question
        score = bugduel.compute_score_debug(solution_str, ground_truth, extra_info,
                       thinking, zero, negative_format_score,
                       format_score, correct_score,
                       extract_all,fix_language, run_timeout,
                       connection_timeout)
    elif 'understanding[codeio]' in verify_style:
        # TODO
        raise NotImplementedError
    else:
        score = 1 if score else 0
    # return {
    #     "score": float(score),
    #     "acc_score": float(score),
    #     "format": 0
    # }
    return score
