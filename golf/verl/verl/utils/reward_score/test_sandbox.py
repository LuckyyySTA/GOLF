import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from code_reward_v2.sandbox_verify import (
    extract_solutions,
    compute_score_stdio,
    compute_score_pytest,
    compute_score_assertions,
    compute_score_call,
    compute_score_aider_format,
    normalize_language,
    is_malicuous_code,
    RunCodeResponse,
    RunStatus,
    CommandRunStatus,
    CommandRunResult,
    OJResponse,
)


class TestExtractSolutions(unittest.TestCase):
    """Test solution extraction from different formats"""
    
    def test_extract_python_code_basic(self):
        """Test basic Python code extraction"""
        solution_str = """Assistant: Here's the solution:
```python
def hello():
    print("Hello, World!")
```
"""
        result = extract_solutions(solution_str, thinking=False, zero=False)
        self.assertEqual(len(result), 1)
        language, code = result[0]
        self.assertEqual(language, 'python')
        self.assertIn('def hello()', code)
    
    def test_extract_multiple_languages(self):
        """Test extraction with different language tags"""
        test_cases = [
            ('javascript', 'nodejs'),
            ('js', 'nodejs'),
            ('python3', 'python'),
            ('c++', 'cpp'),
            ('java', 'java'),
            ('rust', 'rust'),
        ]
        
        for input_lang, expected_lang in test_cases:
            solution_str = f"```{input_lang}\ncode here\n```"
            result = extract_solutions(solution_str, thinking=False, zero=False)
            language, code = result[0]
            self.assertEqual(language, expected_lang, f"Failed for {input_lang}")
    
    def test_extract_with_thinking_tags(self):
        """Test extraction with thinking tags"""
        solution_str = """<think>
Let me think about this...
</think>
```python
def solution():
    return 42
```
"""
        result = extract_solutions(solution_str, thinking=True, zero=False)
        language, code = result[0]
        self.assertEqual(language, 'python')
        self.assertIn('def solution()', code)
    
    def test_extract_invalid_thinking_format(self):
        """Test extraction with invalid thinking format"""
        solution_str = """<think>
Missing closing tag
```python
def solution():
    return 42
```
"""
        result = extract_solutions(solution_str, thinking=True, zero=False)
        self.assertEqual(result, [(None, None)])
    
    def test_extract_last_code_block(self):
        """Test that only the last code block is extracted by default"""
        solution_str = """
```python
def first():
    pass
```

```python
def second():
    pass
```
"""
        result = extract_solutions(solution_str, thinking=False, zero=False, extract_all=False)
        self.assertEqual(len(result), 1)
        language, code = result[0]
        self.assertIn('def second()', code)
        self.assertNotIn('def first()', code)
    
    def test_extract_all_code_blocks(self):
        """Test extracting all code blocks"""
        solution_str = """
```python
def first():
    pass
```

```python
def second():
    pass
```
"""
        result = extract_solutions(solution_str, thinking=False, zero=False, extract_all=True)
        self.assertEqual(len(result), 2)
        self.assertIn('def first()', result[0][1])
        self.assertIn('def second()', result[1][1])


class TestNormalizeLanguage(unittest.TestCase):
    """Test language normalization"""
    
    def test_normalize_javascript(self):
        self.assertEqual(normalize_language('javascript'), 'nodejs')
        self.assertEqual(normalize_language('js'), 'nodejs')
    
    def test_normalize_python(self):
        self.assertEqual(normalize_language('python3'), 'python')
        self.assertEqual(normalize_language('py'), 'python')
    
    def test_normalize_cpp(self):
        self.assertEqual(normalize_language('c++'), 'cpp')
        self.assertEqual(normalize_language('cxx'), 'cpp')
    
    def test_normalize_case_insensitive(self):
        self.assertEqual(normalize_language('Python'), 'python')
        self.assertEqual(normalize_language('JAVA'), 'java')


class TestIsMaliciousCode(unittest.TestCase):
    """Test malicious code detection"""
    
    def test_detect_malicious_code(self):
        malicious = "import os\nos.killpg(0, 9)"
        self.assertTrue(is_malicuous_code(malicious))
    
    def test_safe_code(self):
        safe = "def hello():\n    print('Hello')"
        self.assertFalse(is_malicuous_code(safe))


class TestComputeScoreStdio(unittest.TestCase):
    """Test compute_score_stdio function"""
    
    @patch('code_reward_v2.sandbox_verify.oj_in_sandbox')
    def test_successful_execution(self, mock_oj):
        """Test successful code execution"""
        # Mock successful OJ response
        mock_oj.return_value = OJResponse(
            extracted_code="test_code",
            accepted=True,
            tests=[{
                'exec_info': {'run_result': {'execution_time': 0.5}},
                'test_info': 'test1'
            }]
        )
        
        solution_str = """```python
def add(a, b):
    return a + b
```"""
        ground_truth = [{"input": "1 2", "output": "3"}]
        
        score, exec_time = compute_score_stdio(
            solution_str=solution_str,
            ground_truth=ground_truth,
            thinking=False,
            zero=False,
            import_str="",
            run_timeout=30,
            format_score=0.1,
            negative_format_score=0.0,
            correct_score=1.0,
            extract_all=False,
            fix_language=None,
            connection_timeout=60
        )
        
        self.assertEqual(score, 1.1)  # correct_score + format_score
        self.assertIsNotNone(exec_time)
    
    @patch('code_reward_v2.sandbox_verify.oj_in_sandbox')
    def test_failed_execution(self, mock_oj):
        """Test failed code execution"""
        mock_oj.return_value = OJResponse(
            extracted_code="test_code",
            accepted=False,
            tests=[{
                'exec_info': {'run_result': {'execution_time': 0.1}},
                'test_info': 'test1'
            }]
        )
        
        solution_str = """```python
def add(a, b):
    return a - b  # Wrong implementation
```"""
        ground_truth = [{"input": "1 2", "output": "3"}]
        
        score, exec_time = compute_score_stdio(
            solution_str=solution_str,
            ground_truth=ground_truth,
            thinking=False,
            zero=False,
            import_str="",
            run_timeout=30,
            format_score=0.1,
            negative_format_score=0.0,
            correct_score=1.0,
            extract_all=False,
            fix_language=None,
            connection_timeout=60
        )
        
        self.assertEqual(score, 0.1)  # only format_score
    
    def test_unsupported_language(self):
        """Test handling of unsupported language"""
        solution_str = """```unsupported_lang
code here
```"""
        ground_truth = [{"input": "1 2", "output": "3"}]
        
        score, exec_time = compute_score_stdio(
            solution_str=solution_str,
            ground_truth=ground_truth,
            thinking=False,
            zero=False,
            import_str="",
            run_timeout=30,
            format_score=0.1,
            negative_format_score=0.0,
            correct_score=1.0,
            extract_all=False,
            fix_language=None,
            connection_timeout=60
        )
        
        self.assertEqual(score, 0.0)  # negative_format_score
        self.assertIsNone(exec_time)
    
    def test_malicious_code(self):
        """Test handling of malicious code"""
        solution_str = """```python
import os
os.killpg(0, 9)
```"""
        ground_truth = [{"input": "1 2", "output": "3"}]
        
        score, exec_time = compute_score_stdio(
            solution_str=solution_str,
            ground_truth=ground_truth,
            thinking=False,
            zero=False,
            import_str="",
            run_timeout=30,
            format_score=0.1,
            negative_format_score=0.0,
            correct_score=1.0,
            extract_all=False,
            fix_language=None,
            connection_timeout=60
        )
        
        self.assertEqual(score, 0.1)  # format_score
        self.assertIsNone(exec_time)


class TestComputeScorePytest(unittest.TestCase):
    """Test compute_score_pytest function"""
    
    @patch('code_reward_v2.sandbox_verify.run_code_in_sandbox')
    def test_successful_pytest(self, mock_run):
        """Test successful pytest execution"""
        mock_run.return_value = RunCodeResponse(
            status=RunStatus.Success,
            message="All tests passed",
            run_result=CommandRunResult(
                status=CommandRunStatus.Finished,
                execution_time=0.5,
                return_code=0
            )
        )
        
        solution_str = """```python
def add(a, b):
    return a + b
```"""
        pytest_code = """
def test_add():
    assert add(1, 2) == 3
"""
        
        score, exec_time = compute_score_pytest(
            solution_str=solution_str,
            pytest_code=pytest_code,
            thinking=False,
            zero=False,
            import_str="",
            run_timeout=30,
            format_score=0.1,
            negative_format_score=0.0,
            correct_score=1.0,
            extract_all=False,
            connection_timeout=60
        )
        
        self.assertEqual(score, 1.1)
        self.assertEqual(exec_time, 0.5)
    
    @patch('code_reward_v2.sandbox_verify.run_code_in_sandbox')
    def test_failed_pytest(self, mock_run):
        """Test failed pytest execution"""
        mock_run.return_value = RunCodeResponse(
            status=RunStatus.Failed,
            message="Test failed",
            run_result=CommandRunResult(
                status=CommandRunStatus.Finished,
                execution_time=0.1,
                return_code=1
            )
        )
        
        solution_str = """```python
def add(a, b):
    return a - b
```"""
        pytest_code = """
def test_add():
    assert add(1, 2) == 3
"""
        
        score, exec_time = compute_score_pytest(
            solution_str=solution_str,
            pytest_code=pytest_code,
            thinking=False,
            zero=False,
            import_str="",
            run_timeout=30,
            format_score=0.1,
            negative_format_score=0.0,
            correct_score=1.0,
            extract_all=False,
            connection_timeout=60
        )
        
        self.assertEqual(score, 0.1)


class TestComputeScoreAssertions(unittest.TestCase):
    """Test compute_score_assertions function"""
    
    @patch('code_reward_v2.sandbox_verify.run_code_in_sandbox')
    def test_successful_assertions(self, mock_run):
        """Test successful assertion execution"""
        mock_run.return_value = RunCodeResponse(
            status=RunStatus.Success,
            message="Success",
            run_result=CommandRunResult(
                status=CommandRunStatus.Finished,
                execution_time=0.3,
                return_code=0
            )
        )
        
        solution_str = """```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```"""
        assertions = [
            "assert fibonacci(0) == 0",
            "assert fibonacci(1) == 1",
            "assert fibonacci(5) == 5"
        ]
        
        score, exec_time = compute_score_assertions(
            solution_str=solution_str,
            assertions=assertions,
            entry_point_str="fibonacci",
            thinking=False,
            zero=False,
            import_str="",
            run_timeout=30,
            format_score=0.1,
            negative_format_score=0.0,
            correct_score=1.0,
            extract_all=False,
            connection_timeout=60
        )
        
        self.assertEqual(score, 1.1)
        self.assertEqual(exec_time, 0.3)


class TestComputeScoreCall(unittest.TestCase):
    """Test compute_score_call function"""
    
    @patch('code_reward_v2.sandbox_verify.run_code_in_sandbox')
    def test_successful_function_call(self, mock_run):
        """Test successful function call execution"""
        mock_run.return_value = RunCodeResponse(
            status=RunStatus.Success,
            message="Success",
            run_result=CommandRunResult(
                status=CommandRunStatus.Finished,
                execution_time=0.2,
                return_code=0
            )
        )
        
        solution_str = """```python
def is_palindrome(s):
    return s == s[::-1]
```"""
        test_functions = ["""
def check(candidate):
    assert candidate("racecar") == True
    assert candidate("hello") == False
"""]
        
        score, exec_time = compute_score_call(
            solution_str=solution_str,
            test_functions=test_functions,
            entry_point_str="is_palindrome",
            thinking=False,
            zero=False,
            import_str="",
            run_timeout=30,
            format_score=0.1,
            negative_format_score=0.0,
            correct_score=1.0,
            extract_all=False,
            connection_timeout=60
        )
        
        self.assertEqual(score, 1.1)
        self.assertEqual(exec_time, 0.2)


class TestComputeScoreAiderFormat(unittest.TestCase):
    """Test compute_score_aider_format function"""
    
    @patch('code_reward_v2.sandbox_verify.autotest_in_sandbox')
    def test_successful_aider_format(self, mock_autotest):
        """Test successful aider format execution"""
        mock_autotest.return_value = RunCodeResponse(
            status=RunStatus.Success,
            message="Success",
            run_result=CommandRunResult(
                status=CommandRunStatus.Finished,
                execution_time=0.4,
                return_code=0
            )
        )
        
        response = """main.py
```python
def main():
    print("Hello, World!")

if __name__ == "__main__":
    main()
```"""
        
        tests = {
            "test_main.py": "def test_main():\n    assert True"
        }
        already_solutions = {}
        
        score, exec_time = compute_score_aider_format(
            response=response,
            tests=tests,
            already_solutions=already_solutions,
            test_language="python",
            run_timeout=30,
            correct_score=1.0,
            connection_timeout=60
        )
        
        self.assertEqual(score, 1.0)
        self.assertEqual(exec_time, 0.4)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows"""
    
    @patch('code_reward_v2.sandbox_verify.oj_in_sandbox')
    def test_end_to_end_stdio_workflow(self, mock_oj):
        """Test complete stdio workflow"""
        mock_oj.return_value = OJResponse(
            extracted_code="def solution():\n    pass",
            accepted=True,
            tests=[{
                'exec_info': {'run_result': {'execution_time': 0.1}},
                'test_info': 'test1'
            }]
        )
        
        solution_str = """Assistant: Here's my solution:

```python
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
```
"""
        ground_truth = [
            {"input": "[2,7,11,15]\n9", "output": "[0,1]"}
        ]
        
        score, exec_time = compute_score_stdio(
            solution_str=solution_str,
            ground_truth=ground_truth,
            thinking=False,
            zero=False,
            import_str="",
            run_timeout=30,
            format_score=0.1,
            negative_format_score=0.0,
            correct_score=1.0,
            extract_all=False,
            fix_language=None,
            connection_timeout=60
        )
        
        self.assertGreater(score, 0.0)
        mock_oj.assert_called_once()


if __name__ == '__main__':
    # Set environment variables for testing
    os.environ['API_POOLS'] = 'https://agi-mcp.devops.xiaohongshu.com/api/code/code_sandbox'
    os.environ['OJ_API_POOLS'] = 'https://agi-mcp.devops.xiaohongshu.com/api/code/code_sandbox'
    
    unittest.main(verbosity=2)