from seea.configs.alfworld import ALFWORLD_CONFIG
from seea.prompt.templates import prompt_with_icl, load_icl_examples
from seea.envs.alfworld.alfworld_task import identify_task_type


def get_alfworld_icl_prompt(task: str, name: str, format='react') -> str:
    """
    Generate ALFWorld prompt with ICL examples
    
    Args:
        task (str): ALFWorld task description
        name (str): ALFWorld task name
        format (str): ALFWorld format, 'react' or 'xml'
    Returns:
        Tuple[str, List[Dict[str, str]]]: Generated prompt and message list
    """
    # Get configuration
    icl_path = ALFWORLD_CONFIG.get("react_icl_path", "") if format == 'react' else ALFWORLD_CONFIG.get("icl_path", "")
    icl_num = ALFWORLD_CONFIG.get("icl_num", 2)
    
    # Load ICL examples
    icl_examples = load_icl_examples(icl_path)
    
    # Identify task type
    task_type = identify_task_type(name)
    
    # Get corresponding task type ICL examples
    raw_icl = icl_examples.get(task_type, [])
    
    # Generate prompt with ICL
    prompt = prompt_with_icl(task, raw_icl, icl_num)
    
    return prompt