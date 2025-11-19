import re
import json

def format_reward_function(content, format_version="tag_json"):
    """
    Checks the tag format in the content and calculates the reward score.
    Also checks if the content consists entirely of tags; if there is content outside the tags, it is considered a format error and given a negative score.
    
    Args:
        content (str): The text content to be checked.

    Returns:
        float: The reward score calculated based on the correctness of the tag format.
              - Initial reward: 0.1 points
              - Content outside tags: -0.05 points
              - Multiple actions or JSON format error: -0.05 points
    """
    # Initialize reward score to 0.1
    reward = 0.1

    # Define regular expression patterns
    think_pattern = r"<think>.*?</think>"
    action_pattern = r"<action>.*?</action>"
    answer_pattern = r"<answer>.*?</answer>"
    
    # Check if the content consists entirely of tags
    # Extract all tag content
    all_tags = re.findall(think_pattern + "|" + action_pattern + "|" + answer_pattern, content, re.DOTALL)
    
    # Remove all tag content from the original content
    content_without_tags = content
    for tag in all_tags:
        content_without_tags = content_without_tags.replace(tag, '')
    
    # Check if there is any remaining content after removing tags (ignoring whitespace)
    if content_without_tags.strip() != '':
        # Content exists outside the tags, format error, no points awarded
        reward = -0.05

    # Check action format
    action_matches = re.findall(action_pattern, content)
    if "json" in format_version and action_matches: 
        # Check if there is only one action
        if len(action_matches) == 1:
            # Only one action, check JSON parsability
            try:
                json.loads(action_matches[0].strip())
            except Exception as e:
                reward -= 0.05  # JSON parsing failed
        else:
            # Multiple actions, reward -0.05
            reward -= 0.05

    return reward

def react_format_reward_function(content):
    """
    Checks if the content conforms to the ReAct format and calculates the reward score.
    The ReAct format requires the output to follow the "Thought: your thoughts.\\n Action: your next action"
    or "Action: your next action" format.
    Supports multiple consecutive Actions, such as "Thought: xxx\\n Action: xxx\\n Action: xxx".
    
    Args:
        content (str): The text content to be checked.

    Returns:
        float: The reward score calculated based on the correctness of the ReAct format.
              - Initial reward: 0.1 points
              - Format does not meet requirements: -0.05 points
              - Both Thought and Action exist but format is incorrect: -0.05 points
              - Multiple consecutive Actions: -0.05 points
    """
    # Initialize reward score to 0.1
    reward = 0.1
    
    # If content is empty, return a negative score directly
    if not content or content.strip() == '':
        return 0
    
    # Define regular expression patterns
    thought_pattern = r"Thought:\\s*(.*?)(?=\\n\\s*Action:|$)"
    action_pattern = r"Action:\\s*(.*?)(?=\\n\\s*Thought:|$|\\n\\s*Action:)"
    
    # Extract Thought and Action parts
    thought_matches = re.findall(thought_pattern, content, re.DOTALL)
    action_matches = re.findall(action_pattern, content, re.DOTALL)
    
    # Check if there is Thought or Action
    has_thought = len(thought_matches) > 0
    has_action = len(action_matches) > 0
    
    # If there is neither Thought nor Action, format error
    if not has_thought and not has_action:
        reward = -0.05
        return reward
    
    # Check for multiple consecutive Actions
    if len(action_matches) > 1:
        reward -= 0.05
    
    # Check if the format is correct
    if has_thought and has_action:
        # Should be "Thought: xxx\\n Action: xxx" format
        # Check if the first Action is after Thought
        first_thought_pos = content.find("Thought:")
        first_action_pos = content.find("Action:")
        
        if first_thought_pos > first_action_pos and first_action_pos != -1:
            # Action appears before Thought, incorrect format
            reward -= 0.05
    elif has_thought and not has_action:
        # Only Thought, no Action, incomplete format
        reward -= 0.05
    
    # Check if the content only contains Thought and Action
    # Remove all Thought and Action content
    content_without_format = content
    for thought in thought_matches:
        content_without_format = content_without_format.replace(f"Thought: {thought}", '')
    
    # For multiple Actions, replace them one by one
    for action in action_matches:
        content_without_format = content_without_format.replace(f"Action: {action}", '')
    
    # Remove newline characters and whitespace
    content_without_format = re.sub(r'\\s+', '', content_without_format)
    
    # If there is other content, format error
    if content_without_format.strip() != '':
        reward -= 0.05
    
    # Ensure reward score is not less than 0
    reward = max(0, reward)
    
    return reward