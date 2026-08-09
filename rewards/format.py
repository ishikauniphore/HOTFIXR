import json
import re

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125

    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001

    if text.count("\n<question>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</question>\n")[-1])*0.001
    if text.count("\n</question>") == 1:
        count += 0.125
        count -= (len(text.split("\n</question>")[-1]) - 1)*0.001
        
    return count

# def format_reward_func(completions, **kwargs):
#     """Reward function that checks if the completion has a specific format."""
#     pattern = r"^<question>.*?</question><reasoning>.*?</reasoning><answer>.*?</answer>$"
#     completion_contents = [completion[0]["content"] for completion in completions]
#     matches = [re.match(pattern, content) for content in completion_contents]
#     return [1.0 if match else 0.0 for match in matches]
    
def parse(solution_str):
    # is the html parse
    data = parse_html(solution_str)
    if data != None:
        return (data, count_xml(solution_str))

    return None, None

def parse_html(solution_str):
    try:
        sample_pattern = re.compile(
            r'<question>(.*?)</question>\s*'
            r'<reasoning>(.*?)</reasoning>\s*'
            r'<answer>(.*?)</answer>\s*',
            re.DOTALL | re.IGNORECASE
        )
        
        match = sample_pattern.search(solution_str)
        if not match:
            return None
        
        return {
            "question": match.group(1).strip(),
            "reasoning": match.group(2).strip(),
            "answer": match.group(3).strip(),
        }
    except:
        return None
    

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if parse(solution_str) != None:
        return 1.0
    else:
        return 0.0