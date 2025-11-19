import re
import json
import traceback
from seea.utils.reward_function import format_reward_function, react_format_reward_function
from seea.utils.logger import get_logger
logger = get_logger(__name__)


def xml_stream_parser(response_stream, action_type="str"):
    """
    Streaming parser for parsing tool calls and responses in streaming responses.
    Suitable for scenarios where <action>...</action> tags appear alternately.
    """
    def extract_multiple_actions(buffer):
        """Extract content from multiple action tags.
        
        Args:
            buffer (str): Text containing one or more action tags.
            
        Returns:
            list: List of all valid extracted action data.
            int: End position of the last processed action.
        """
        actions = []
        last_end = 0
        
        # Use non-greedy mode *? to match the nearest end tag
        action_pattern = re.compile(r"<action>(.*?)</action>", re.DOTALL)
        
        while True:
            match = action_pattern.search(buffer, last_end)
            if not match:
                break
                
            try:
                action_data = match.group(1).strip()
                actions.append(action_data)
                last_end = match.end()
            except:
                # If JSON parsing fails, continue to find the next action
                last_end = match.end()
                logger.info(traceback.format_exc())
                continue
        
        return actions, last_end

    buffer = ""
    complete_raw_response = ""
    temp_buffer = {"think": "", "answer": ""}
    
    for chunk in response_stream:
        if not chunk.choices[0].delta.content:
            continue
            
        content = chunk.choices[0].delta.content
        buffer += content
        complete_raw_response += content
        print(content, end='', flush=True)
        
        # Process think tag
        if "<think>" in buffer:
            think_end = buffer.find("</think>")
            if think_end != -1:
                think_start = buffer.find("<think>") + len("<think>")
                think_content = buffer[think_start:think_end].strip()
                if think_content != temp_buffer["think"]:
                    temp_buffer["think"] = think_content
                    yield {"type": "think", "data": think_content}
                buffer = buffer[think_end + len("</think>"):]
        
        # Process answer tag
        if "<answer>" in buffer:
            answer_end = buffer.find("</answer>")
            if answer_end != -1:
                answer_start = buffer.find("<answer>") + len("<answer>")
                answer_content = buffer[answer_start:answer_end].strip()
                if answer_content != temp_buffer["answer"]:
                    temp_buffer["answer"] = answer_content
                    yield {"type": "answer", "data": answer_content}
                buffer = buffer[answer_end + len("</answer>"):]
        
        # Process multiple action tags
        if "<action>" in buffer:
            actions, end_idx = extract_multiple_actions(buffer)
            if actions:
                for action_data in actions:
                    try:
                        if action_type == "str":
                            action_data = {"name": "embodied_operation", "arguments":{'instruction': action_data.strip()}}
                        else:
                            action_data = json.loads(action_data)
                        yield {"type": "action", "data": action_data}
                    except:
                        logger.info(traceback.format_exc())
                        continue
                buffer = buffer[end_idx:]
    
    # Return complete response at the end of the stream
    yield {
        "type": "complete_response",
        "data": complete_raw_response
    }


def react_stream_parser(response_stream):
    """
    Streaming parser for parsing tool calls and responses in streaming responses.
    Suitable for scenarios where Action: and Observation: appear alternately.
    """
    def extract_action(buffer):
        """
        Extracts the complete JSON data after 'Action:', supporting nested curly braces.
        """
        action_pattern = re.compile(r"Action:\s*(\{.*)", re.DOTALL)  # Initially match Action and the start of the JSON part
        match = action_pattern.search(buffer)

        if match:
            brace_count = 0
            start_idx = match.start(1)
            # end_idx = match.end(1) # This line is removed as end_idx is determined by brace matching

            # Iterate through the extracted part, manually handling parenthesis nesting
            for i in range(start_idx, len(buffer)):
                char = buffer[i]

                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1

                # If parenthesis matching is complete, return the complete JSON content
                if brace_count == 0:
                    end_idx = i + 1
                    break
            else: # If loop finishes without brace_count becoming 0 (e.g. incomplete JSON)
                return None, None
            
            # Return the complete JSON data
            return buffer[start_idx:end_idx].strip(), end_idx
        return None, None
    
    buffer = ""
    complete_raw_response = ""  # Used to store the complete original response string
    temp_answer_buffer = ""  # Used to temporarily store encountered text until a tool_call is encountered

    for chunk in response_stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            buffer += content  # Accumulate character stream
            complete_raw_response += content  # Record the complete original response
            print(content, end='', flush=True)  # Append and print streaming content
        else:
            continue

        # Check if buffer contains Action
        if "Action:" in buffer:
            # Extract text before Action as answer
            index = buffer.index("Action:")
            temp_answer_buffer += buffer[:index]

            # Return current answer
            if temp_answer_buffer.strip():
                yield {"type": "answer", "data": temp_answer_buffer.strip()}
                temp_answer_buffer = ""  # Clear temporary answer
                buffer = buffer[index:]

            # Extract JSON content after Action
            action_data, end_idx = extract_action(buffer)
            if action_data:
                # Parse and return tool call
                try:
                    tool_call_data = json.loads(action_data)
                    yield {"type": "action", "data": tool_call_data}  # Immediately return tool call
                    buffer = buffer[end_idx:]
                except json.JSONDecodeError:
                    pass  # If parsing fails, skip
        else:
            # If no Action, continue accumulating text content
            temp_answer_buffer += buffer
            buffer = ""  # Clear buffer to prevent reprocessing

    # After the stream ends, return the remaining answer
    if temp_answer_buffer.strip():
        yield {"type": "answer", "data": temp_answer_buffer.strip()}

    # At the end of the stream, return the complete response and original data
    yield {
        "type": "complete_response",
        "data": complete_raw_response
    }


# Added non-streaming parser
def non_stream_parser(content, format_version="tag"):
    """
    Non-streaming parser for parsing tool calls and responses in complete replies.
    Suitable for scenarios with <action>...</action>, <think>...</think>, and <answer>...</answer> tags.
    
    Args:
        content: Reply content
    Returns:
        dict: Parsed results dictionary, containing think, answer, action, and complete_response fields
    """
    format_reward = format_reward_function(content, format_version)
    
    results = {
        "think": "",
        "answer": "",
        "action": [],
        "complete_response": content,
        "format_reward": format_reward
    }
    if not content:
        return results
    
    # Extract think tag content
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    think_matches = think_pattern.findall(content)
    for think_content in think_matches:
        results["think"] += think_content.strip()
    
    # Extract answer tag content
    answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    answer_matches = answer_pattern.findall(content)
    for answer_content in answer_matches:
        results["answer"] += answer_content.strip()
    
    # Extract action tag content
    action_pattern = re.compile(r"<action>(.*?)</action>", re.DOTALL)
    action_matches = action_pattern.findall(content)
    actions = []
    for action_data in action_matches:
        try:
            # Verify if it is valid JSON
            if format_version == 'tag_json':
                new_action = json.loads(action_data.strip())
            elif format_version == 'tag':
                new_action = {"name": "embodied_operation", "arguments":{'instruction': action_data.strip()}}
            else:
                raise NotImplementedError

            # Ensure each action has complete_response and format_reward fields
            if isinstance(new_action, dict):
                if "complete_response" not in new_action:
                    new_action["complete_response"] = content
                if "format_reward" not in new_action:
                    new_action["format_reward"] = format_reward
            # Add parsed action to the list
            actions.append(new_action)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in action tag: {action_data}")
            continue
        except Exception as e:
            logger.warning(f"Error parsing action: {traceback.format_exc()}")
            continue
    
    # If there are actions, set the first action as the result
    if actions:
        results["action"] = actions
    else:
        results["action"] = [{"name": "", "arguments": {}, "complete_response": content, "format_reward": format_reward}]
    
    return results


def react_format_parser(content, format_version="tag"):
    """
    ReAct format parser for parsing reply content that conforms to the ReAct format.
    Suitable for scenarios with "Thought: your thoughts.\n Action: your next action" or "Action: your next action" format.
    Supports multiple consecutive Actions, such as "Thought: xxx\n Action: xxx\n Action: xxx".
    
    Args:
        content: Reply content
        format_version: Format version, supports "tag" and "tag_json"
        
    Returns:
        dict: Parsed results dictionary, containing think, action, and complete_response fields
    """
    format_reward = react_format_reward_function(content)
    
    results = {
        "think": "",
        "action": [],
        "answer": "",
        "complete_response": content,
        "format_reward": format_reward
    }
    
    if not content:
        return results
    
    # Extract Thought part
    thought_pattern = re.compile(r"Thought:\s*(.*?)(?=\n\s*Action:|$)", re.DOTALL)
    thought_match = thought_pattern.search(content)
    if thought_match:
        results["think"] = thought_match.group(1).strip()
    
    # Extract Action part - modify regular expression to support multiple consecutive Actions
    action_pattern = re.compile(r"Action:\s*(.*?)(?=\n\s*Thought:|$|\n\s*Action:)", re.DOTALL)
    action_matches = action_pattern.findall(content)
    
    actions = []
    for action_data in action_matches:
        try:
            action_data = action_data.strip()
            if format_version == 'tag_json':
                try:
                    new_action = json.loads(action_data)
                except json.JSONDecodeError:
                    # If JSON parsing fails, use default format
                    new_action = {"name": "embodied_operation", "arguments":{'instruction': action_data}}
            elif format_version == 'tag':
                new_action = {"name": "embodied_operation", "arguments":{'instruction': action_data}}
            else:
                raise NotImplementedError
            
            # Ensure each action has complete_response and format_reward fields
            new_action["complete_response"] = content
            new_action["format_reward"] = format_reward
            
            # Add parsed action to the list
            actions.append(new_action)
        except Exception as e:
            logger.warning(f"Error parsing action: {traceback.format_exc()}")
            continue
    
    # If there are actions, set the action list
    if actions:
        results["action"] = actions
    else:
        results["action"] = [{"name": "", "arguments": {}, "complete_response": content, "format_reward": format_reward}]
    
    return results