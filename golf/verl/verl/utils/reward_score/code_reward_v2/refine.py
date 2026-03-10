import re
import os


def extract_response_components(text):
    """
    从文本中抽取按指定格式输出的各个组件
    
    Args:
        text (str): 包含格式化响应的文本
        
    Returns:
        dict: 包含抽取出的各组件的字典
    """
    components = {
        'analysis_process': '',
        'judgment_result': {
            'status': '',
            'reason': ''
        },
        'final_answer': ''
    }
    
    # 抽取分析过程
    analysis_match = re.search(r'## Analysis Process\s*\n(.*?)(?=## Judgment Result)', text, re.DOTALL)
    if analysis_match:
        components['analysis_process'] = analysis_match.group(1).strip()
    
    # 抽取判断结果部分
    judgment_section = re.search(r'## Judgment Result\s*\n(.*?)(?=## Final Answer|$)', text, re.DOTALL)
    if judgment_section:
        judgment_text = judgment_section.group(1)
        
        # # 抽取状态
        # status_match = re.search(r'\*\*Status\*\*:\s*\[(.*?)\]', judgment_text)
        # if status_match:
        #     components['judgment_result']['status'] = status_match.group(1).strip()
        
        # # 抽取原因
        # reason_match = re.search(r'\*\*Reason\*\*:\s*\[(.*?)\]', judgment_text)
        # if reason_match:
        #     components['judgment_result']['reason'] = reason_match.group(1).strip()
        
        # 抽取状态
        status_match = re.search(r'\*\*Status\*\*:\s*(.*)', judgment_text)
        if status_match:
            components['judgment_result']['status'] = status_match.group(1).strip()

        # 抽取原因
        reason_match = re.search(r'\*\*Reason\*\*:\s*(.*)', judgment_text)
        if reason_match:
            components['judgment_result']['reason'] = reason_match.group(1).strip()

    
    # 抽取最终答案（只在状态为Incorrect时存在）
    final_answer_match = re.search(r'## Final Answer\s*\n(.*?)$', text, re.DOTALL)
    if final_answer_match:
        components['final_answer'] = final_answer_match.group(1).strip()
    
    return components

def pretty_print_components(components):
    """
    美观地打印抽取出的组件
    """
    print("=" * 50)
    print("EXTRACTED COMPONENTS")
    print("=" * 50)
    
    print("\n📋 ANALYSIS PROCESS:")
    print("-" * 20)
    print(components['analysis_process'] or "[Not found]")
    
    print("\n⚖️  JUDGMENT RESULT:")
    print("-" * 20)
    print(f"Status: {components['judgment_result']['status'] or '[Not found]'}")
    print(f"Reason: {components['judgment_result']['reason'] or '[Not found]'}")
    
    print("\n💡 FINAL ANSWER:")
    print("-" * 15)
    print(components['final_answer'] or "[Not provided - Status is Correct or not found]")


def compute_score_fix(solution_str, tag, ground_truth, extra_info,
                       thinking, zero, negative_format_score,
                       format_score, correct_score,
                       extract_all,fix_language, run_timeout,
                       connection_timeout):
    solution_str = solution_str.split('[If status is "Incorrect", provide the corrected complete solution. Needed only when status is "Incorrect"]')[-1]
    components = extract_response_components(solution_str)

    # **Status**: [Correct/Incorrect]
    if components['judgment_result']['status'] not in ['Correct', 'Incorrect']:
        return 0

    if components['judgment_result']['status'] == 'Correct' and tag is True:
        if components['final_answer'] == '':
            # 避免copy
            return 1
        else:
            return 0
    elif components['judgment_result']['status'] == 'Incorrect' and tag is True:
        return 0
    elif components['judgment_result']['status'] == 'Correct' and tag is False:
        return 0
    elif components['judgment_result']['status'] == 'Incorrect' and tag is False:
        from verl.utils.reward_score.code_reward_v2.__init__ import \
            compute_score
        
        correct_judge_score = float(os.getenv('correct_judge_score', 0))
        
        score = compute_score(
            ground_truth['real_style'], components['final_answer'], 
            {**ground_truth, 'style': ground_truth['real_style']}, extra_info=extra_info, 
            thinking=thinking, zero=zero, 
            negative_format_score=negative_format_score,
            format_score=format_score, correct_score=correct_score, 
            extract_all=extract_all,
            fix_language=fix_language,
            run_timeout = run_timeout,
            connection_timeout=connection_timeout
        )['score']
        
        return score if score == 1 else correct_judge_score
