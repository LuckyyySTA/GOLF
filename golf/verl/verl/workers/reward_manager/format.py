import re

def format_reward(predict_str: str, format_mode='R1_nothink') -> float:
    def _validate_tags(input_string):
        if format_mode == 'R1':
            tags = ['<think>', '</think>', '<answer>', '</answer>']
        elif format_mode == 'R1_nothink':
            tags = ['<answer>', '</answer>']
        else:
            raise ValueError(f"Unsupported format mode: {format_mode}")
        for tag in tags:
            if input_string.count(tag) != 1:
                return 0.0
        return 1.0

    if _validate_tags(predict_str) == 0.0:
        return 0.0
    if format_mode == 'R1':
        pattern = re.compile(r'<think>.*</think>.*<answer>.*</answer>.*', re.DOTALL)
    elif format_mode == 'R1_nothink':
        pattern = re.compile(r'.*<answer>.*</answer>.*', re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)

    return 1.0 if match_result else 0.0