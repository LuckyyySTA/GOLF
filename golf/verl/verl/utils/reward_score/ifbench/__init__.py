import ast
import json
import random
import re
import urllib.request
import urllib.error
from . import instructions_registry

# Prompt template for IFEval critique generation
IFEVAL_CRITIQUE_TEMPLATE = """You are an expert at evaluating responses for instruction-following tasks.

Given a user's question, an assistant's response, required constraints, and the automated checking score, provide constructive feedback for improvement.

**User Question:**
{question}

**Required Constraints:**
{constraints_description}

**Assistant's Response:**
{response}

**Automated Score:** {score}/1.0 ({pass_rate}% of constraints satisfied)

Your task:
- Briefly acknowledge what was done correctly
- Explain WHY it failed based on the requirement and HOW to fix it
- Provide specific, actionable suggestions (not a full rewrite)

Format your feedback as:
<critique>
Strengths: [What constraints were satisfied and why]
Issues: [Which constraints failed and specific reasons]
Improvements: [Concrete steps to satisfy each failed constraint]
</critique>

Be concise, specific, and constructive. Focus on what needs to change, not on rewriting the entire response."""

# GRM service URLs (shared with reward_function_pointwise.py)
GRM_URLS = [
    "http://10.146.231.172:18901", "http://10.146.225.31:18901", "http://10.146.231.110:18901", "http://10.146.225.14:18901",
    "http://10.146.225.13:18901", "http://10.146.235.208:18901", "http://10.146.232.32:18901", "http://10.146.231.101:18901",
    "http://10.146.235.247:18901", "http://10.146.229.194:18901", "http://10.146.228.226:18901", "http://10.146.234.22:18901",
    "http://10.146.230.233:18901", "http://10.146.225.44:18901", "http://10.146.231.19:18901", "http://10.146.233.32:18901"
]

def _extract_critique(text: str) -> str:
    """Extract critique content from response."""
    critique_pattern = r'<critique>(.*?)</critique>'
    match = re.search(critique_pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()

def _make_grm_request(prompt: str, model_name: str = "nginx", timeout: int = 30) -> str:
    """Make a synchronous request to GRM service with random URL selection."""
    base_url = random.choice(GRM_URLS)
    
    try:
        data = json.dumps({
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}]
        }).encode('utf-8')
        
        req = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                result = json.loads(response.read().decode('utf-8'))
                return result["choices"][0]["message"]["content"]
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        # Silently handle errors - critique is optional
        pass
    except Exception:
        pass
    
    return ""

def request_critique(question: str, solution_str: str, constraints_description: str, score: float) -> str:
    """
    Request critique from GRM service for IFEval tasks.
    
    Args:
        question: The user's original question
        solution_str: The assistant's response
        constraints_description: Human-readable description of required constraints
        score: Automated constraint checking score (0.0 to 1.0)
        
    Returns:
        Critique text, or empty string if request fails
    """
    if not solution_str.strip():
        return ""
    
    pass_rate = int(score * 100)
    
    # Format the prompt with constraints description and score
    prompt = IFEVAL_CRITIQUE_TEMPLATE.format(
        question=question,
        constraints_description=constraints_description,
        response=solution_str,
        score=score,
        pass_rate=pass_rate
    )
    
    # Make request and extract critique
    response = _make_grm_request(prompt)
    return _extract_critique(response) if response else ""

def compute_score(solution_str, ground_truth, extra_info):
    constraint = extra_info.get("constraint", "")
    
    instruction_dict = instructions_registry.INSTRUCTION_DICT
    constraint_dict = ast.literal_eval(ground_truth)
    constraint_dict = constraint_dict[0]
    if isinstance(constraint_dict, str):
        constraint_dict = json.loads(constraint_dict)
    
    instruction_keys = constraint_dict["instruction_id"]
    args_list = constraint_dict["kwargs"]
    
    # Check constraints
    if len(solution_str) == 0:
        return {"score": 0.0, "critique": "Response is empty."}
    
    rewards = []
    critiques = []
    
    for instruction_key, args in zip(instruction_keys, args_list):
        if args is None:
            args = {}
        args = {k: v for k, v in args.items() if v is not None}
        
        instruction_cls = instruction_dict[instruction_key]
        instruction_instance = instruction_cls(instruction_key)
        instruction_instance.build_description(**args)
        
        try:
            passed = solution_str.strip() and instruction_instance.check_following(solution_str.strip())
        except (RecursionError, Exception):
            passed = False
        
        rewards.append(1.0 if passed else 0.0)
        
        # Get detailed critique for each constraint
        try:
            critique_msg = instruction_instance.get_critique(solution_str.strip(), passed)
            critiques.append(critique_msg)
        except Exception:
            # Fallback to basic feedback if get_critique fails
            if passed:
                critiques.append(f"Constraint satisfied: {instruction_key}")
            else:
                critiques.append(f"Constraint not satisfied: {instruction_key}")

    # Only give score of 1.0 if ALL constraints are satisfied, otherwise 0.0
    score = 1.0 if sum(rewards) == len(rewards) else 0.0
    
    # Generate comprehensive natural language feedback
    num_passed = sum(rewards)
    num_total = len(rewards)
    if score == 1.0:
        critique = "All constraints satisfied!\n\n" + "\n".join(critiques)
    elif num_passed == 0:
        critique = "No constraints satisfied. Please review the requirements:\n\n" + "\n".join(critiques)
    else:
        critique = f"Not all constraints satisfied ({int(num_passed)}/{num_total} constraints met). Score: {score:.3f}\n\n" + "\n".join(critiques)
    
    return {"score": score, "critique": critique}

if __name__ == "__main__":
    # For debugging: add parent directory to path to enable imports
    import sys
    import os
    _verl_root = os.environ.get("VERL_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
    sys.path.insert(0, _verl_root)
    
    # Now import with absolute path
    from verl.utils.reward_score.ifbench import compute_score
    
    # Test case
    solution_str = "algae-reduce-light-and-avoid-direct-sunlight-keep-nutrients-low-with-regular-water-changes-and-avoid-overfeeding-use-live-plants-and-appropriate-filtration-to-outcompete-and-limit-algae-growth-algae"
    ground_truth = "[{'instruction_id': ['detectable_format:sentence_hyphens', 'keywords:start_end'], 'kwargs': [None, None]}]"
    extra_info = {
        "question": "How to prevent algae growth in aquarium?",
        "constraint": "2 constraints: sentence with hyphens, start and end keywords"
    }
    
    print("="*60)
    print("Testing IFEval compute_score with critique generation")
    print("="*60)
    
    result = compute_score(solution_str, ground_truth, extra_info)
    
    print(f"\n✅ Score: {result['score']}")
    print(f"\n📝 Critique:\n{result['critique']}")
    print("\n" + "="*60)