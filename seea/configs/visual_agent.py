import os
import json
from seea.agents.agent import FunctionCallAgent
from seea.configs.dreamer import get_dreamer_agent
from seea.utils.common import langraph_tool_to_schema
from seea.utils.paser import xml_stream_parser, react_stream_parser, non_stream_parser, react_format_parser
from seea.tools.langgraph_tools import robotic_arm_operation_wrapper, perception_wrapper, \
    robotic_arm_operation_alfworld_wrapper
from seea.configs.config import *


def get_visual_agent(config: dict):
    """
    Get the Agent based on the multi-modal large model.
    Args:
        config: Dictionary containing all parameters.
    Returns:
        FunctionCallAgent: The Agent based on the multi-modal large model.
    """
    model = config.get("model", {})
    temperature = config.get("temperature", 0.35)
    max_tokens = config.get("max_tokens", 2048)
    top_p = config.get("top_p", 0.9)
    repetition_penalty = config.get("repetition_penalty", 1.05)
    image_size = config.get("image_size", [600, 600])
    save_folder = config.get("save_folder", "assets/data")
    alfworld_env = config.get("alfworld_env", None)
    visual_world = config.get("visual_world", True)
    stream = config.get("stream", False)
    parser_tag = config.get("parser_tag", ["<action>", "</action>"])
    visual_perception = config.get("visual_perception", False)
    wo_image_tool_result = config.get("wo_image_tool_result", False)
    fixed_perception = config.get("fixed_perception", False)
    max_steps = config.get("max_steps", 30)
    visual_som = config.get("visual_som", False)

    # Initialize robot operation tool
    if alfworld_env:
        robotic_arm_operation = robotic_arm_operation_alfworld_wrapper(alfworld_env, using_admissible_commands=False, enable_visual=not wo_image_tool_result, visual_som=visual_som)
    elif visual_world:
        visual_dreamer = get_dreamer_agent(system_prompt=META_PROMPT_AGENT_PROMPT, model=model, video_generation_model=VIDEO_GENERATION_MODELS["kling"], reflection_prompt=VIDEO_REFLECTION_PROMPT)
        robotic_arm_operation = robotic_arm_operation_wrapper(visual_dreamer)
    else:
        robotic_arm_operation = robotic_arm_operation_wrapper()

    tools = [robotic_arm_operation]
    tools_prompt = ""
    if visual_perception:
        def response():
            return GET_REAL_IMAGE_RESPONSE
        perception = perception_wrapper(response)
        tools.append(perception)
        tools_prompt = TOOLS_PROMPT

    save_folder = os.path.join(save_folder, "visual_agent") if save_folder else save_folder

    if stream:
        parser = react_stream_parser if "Action:" in parser_tag else xml_stream_parser
    else:
        parser = react_format_parser if "Action:" in parser_tag else non_stream_parser

    agent_executor = FunctionCallAgent(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        system_prompt=ROLE_PROMPT,
        image_size=image_size,
        model=model["model"],
        api_key=model["api_key"],
        api_base=model["base_url"],
        tool_choice='auto' if model["model"] in ["gpt-4o", 'o1', 'o1-mini'] else None,
        tools=tools,
        parser=parser,
        observation_format="Observation: {}" if "Action:" in parser_tag else "<observation>{}</observation>",
        camera_client=None,
        stop=["observation:", "Observation:", "<observation>", "<|im_start|>", "\nThought:"],
        save_folder=save_folder,
        function_to_schema=langraph_tool_to_schema,
        max_steps=max_steps,
        stream=stream,
        fixed_perception=fixed_perception,
    )

    # Get tool list
    tools_decs = agent_executor.get_tools_list()
    # Initialize tool description
    TOOL_DESC = "" if alfworld_env else f"\nYou can use the following tools:\n{json.dumps(tools_decs, ensure_ascii=False)}" 
    # Output format
    format_prompt = ""
    if "<action>" in parser_tag:
        format_prompt = THINK_ACTION_ANSWER_FORMAT_PROMPT
    elif "Action:" in parser_tag:
        format_prompt = REACT_FORMAT
    else:
        raise ValueError(f"Invalid paser: {parser_tag}")

    SCENE_PROMPT = ALFWORLD_SCENE_PROMPT if alfworld_env else DREAMER_SCENE_PROMPT
    system_prompt = ROLE_PROMPT + format_prompt + TOOL_DESC  + tools_prompt + SCENE_PROMPT
    agent_executor.set_system_prompt(system_prompt)
    if alfworld_env:
        # Set alfworld_env attribute dynamically
        setattr(agent_executor, 'alfworld_env', alfworld_env)
    return agent_executor