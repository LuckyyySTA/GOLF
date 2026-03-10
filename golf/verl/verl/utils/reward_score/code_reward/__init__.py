import time

from livecodebench.compute_score import compute_score as code_compute_score


def compute_score(solution_str, ground_truth, extra_info, timeout=6, is_binary_reward=True, is_power4_reward=False):
    score = code_compute_score(completion=solution_str, test_cases=ground_truth, timeout=timeout, is_binary_reward=is_binary_reward, is_power4_reward=is_power4_reward)[0]
    if type(score) is not bool:
        print("=== Wrong Code ===", extra_info)
        score = 0
    else:
        score = 1 if score else 0
    return {
        "score": score,
        "acc_score": score,
        "format": 0
    }

