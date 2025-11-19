"copied from https://github.com/WeiminXiong/MPO/blob/main/prompt/templates.py"
import os
import json


PROMPT_WITH_ICL_TEMPLATE = """
{icl_prompt}

{examples}

Now, it's your turn and here is the task.
{task}
"""


def prompt_with_icl(task, raw_icl, icl_num=2):
    """
    Generate a prompt with ICL based on instructions, ICL examples, and current task
    
    Args:
        raw_icl (list): List of ICL examples, each example is a list of dialogues
        icl_num (int, optional): Number of ICL examples to use. Defaults to 2.
        workflow (str, optional): Workflow description. Defaults to None.
    
    Returns:
        tuple: prompt Generated prompt string and message list
    """
    examples = ""
    for i in range(min(icl_num, len(raw_icl))):
        for j in range(len(raw_icl[i])):
            cur_content = raw_icl[i][j]['content']
            if i == 0 and j == 0:
                if icl_num > 1:
                    examples += f"Example task {i + 1}:\n"
                examples += cur_content + '\n'
                continue
            elif i != 0 and j == 0:
                if icl_num > 1:
                    examples += f"\nExample task {i + 1}:\n"
                    examples += cur_content + '\n'
                else:
                    examples += '\n' + cur_content + '\n'
                continue
            # user
            if j % 2 == 0:
                examples += cur_content + '\n\n'
            # assistant
            else:
                examples += cur_content + '\n'
    icl_prompt = f"Here are {icl_num} examples." if icl_num > 1 else f"Here is an example."
    prompt = PROMPT_WITH_ICL_TEMPLATE.format(
        icl_prompt=icl_prompt, 
        examples=examples,
        task=task
    )
    return prompt


def load_icl_examples(icl_path):
    """
    Load ICL example files
    
    Args:
        icl_path (str): Path to ICL example file
    
    Returns:
        dict: Loaded ICL examples
    """
    if not os.path.exists(icl_path):
        raise FileNotFoundError(f"ICL example file does not exist: {icl_path}")
    
    with open(icl_path, 'r', encoding='utf-8') as f:
        icl_examples = json.load(f)
    
    return icl_examples