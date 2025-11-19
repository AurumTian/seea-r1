import os
import json
from seea.configs.config import *
from seea.utils.common import langraph_tool_to_schema
from seea.agents.models.models import *
from seea.agents.mcts.robot_mcts import RobotMCTSWrapper
from seea.agents.agent import ChatAgent
from seea.configs.dreamer import get_dreamer_agent
from seea.utils.paser import non_stream_parser, react_format_parser
from seea.tools.langgraph_tools import robotic_arm_operation_wrapper, perception_wrapper, \
    robotic_arm_operation_alfworld_wrapper
from seea.utils.logger import get_logger
logger = get_logger(__name__)


def get_robot_mcts(config: dict) -> RobotMCTSWrapper:
    """
    Get MCTS agent
    Args:
        config: Dictionary containing all parameters
    """
    model = config.get("model", {})
    critic_model = config.get("critic_model", model)
    if critic_model is None:
        critic_model = model
    temperature = config.get("temperature", 0.35)
    top_p = config.get("top_p", 0.90)
    repetition_penalty = config.get("repetition_penalty", 1.05)
    max_tokens = config.get("max_tokens", 2048)
    image_size = config.get("image_size", [600, 600])
    sort_action = config.get("sort_action", False)
    alfworld_env = config.get("alfworld_env", None)
    n_iterations = config.get("n_iterations", 3)
    depth_limit = config.get("depth_limit", 3)
    num_proposed_action = config.get("num_proposed_action", 3)
    exploration_weight = config.get("exploration_weight", 1.0)
    visual_perception = config.get("visual_perception", False)
    wo_image_tool_result = config.get("wo_image_tool_result", False)
    visual_world = config.get("visual_world", False)
    parser_tag = config.get("parser_tag", ["<action>", "</action>"])
    enable_reflection = config.get("enable_reflection", False)
    save_folder = config.get("save_folder", "")
    visual_som = config.get("visual_som", False)
    enable_critic = config.get("enable_critic", False)
    enable_ttrl_reward = config.get("enable_ttrl_reward", False)
    ttrl_vote_num = config.get("ttrl_vote_num", 10)
    trust_critic = config.get("trust_critic", False)

    # Initialize robot operation tool
    if alfworld_env:
        robotic_arm_operation = robotic_arm_operation_alfworld_wrapper(alfworld_env, using_admissible_commands=False, enable_visual=not wo_image_tool_result, visual_som=visual_som)
    elif visual_world:
        visual_dreamer = get_dreamer_agent(system_prompt=VIDEO_GENERATION_PROMPT,
                                model=model,
                                video_generation_model=VIDEO_GENERATION_MODELS["kling"],
                                reflection_prompt=VIDEO_REFLECTION_PROMPT,
                                kling_version="v1-6")
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

    tools_call = dict()
    for tool in tools:
        tools_call[tool.name] = tool

    save_folder = os.path.join(save_folder, "robot_mcts") if save_folder else save_folder

    agent_actor = ChatAgent(
        name='Actor',
        system_prompt=ROLE_PROMPT,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        image_size=image_size,
        model=model["model"],
        api_key=model["api_key"],
        api_base=model["base_url"],
        tool_choice='auto' if model["model"] in ["gpt-4o", 'o1', 'o1-mini'] else None,
        tools=tools,
        stop=["observation:", "Observation:", "<observation>", "<|im_start|>", "\nThought:"],
        keep_message_history=False,
        save_folder=save_folder,
        function_to_schema=langraph_tool_to_schema,
    )

    # Get tool list and tool description
    tools_list = agent_actor.get_tools_list()
    # Build tool description string, use f-string formatting
    TOOL_DESC = "" if alfworld_env else f"\nYou can use the following tools:\n{json.dumps(tools_list, ensure_ascii=False)}"
    # Output format
    format_prompt = ""
    if "<action>" in parser_tag:
        format_prompt = THINK_ACTION_ANSWER_FORMAT_PROMPT
    elif "Action:" in parser_tag:
        format_prompt = REACT_FORMAT
    else:
        raise ValueError(f"Invalid paser: {parser_tag}")
    # Scene description
    scene_prompt = ALFWORLD_SCENE_PROMPT if alfworld_env else DREAMER_SCENE_PROMPT
    system_prompt = ROLE_PROMPT + format_prompt + TOOL_DESC  + tools_prompt + scene_prompt
    agent_actor.set_system_prompt(system_prompt)

    basic_prompt = TOOL_DESC + tools_prompt + scene_prompt

    agent_sortor = ChatAgent(
        name='Sortor',
        system_prompt=ROLE_PROMPT + SORTOR_PROMPT + basic_prompt,
        model=model["model"],
        api_key=model["api_key"],
        api_base=model["base_url"],
        keep_message_history=False,
        save_folder=save_folder,
    )
    
    agent_critic = ChatAgent(
        name='Critic',
        system_prompt=OUTPUT_REWARD_MODEL_PROMPT if alfworld_env else CRITIC_PROMPT,
        model=critic_model["model"],
        api_key=critic_model["api_key"],
        api_base=critic_model["base_url"],
        keep_message_history=False,
        save_folder=save_folder,
    )
    
    robot_mcts = RobotMCTSWrapper(
        actor=agent_actor,
        sortor=agent_sortor if sort_action else None,
        critic=agent_critic if enable_critic else None,
        trust_critic=trust_critic,
        tools_call=tools_call,
        alfworld_env=alfworld_env,
        n_iterations=n_iterations,
        num_proposed_action=num_proposed_action,
        depth_limit=depth_limit,
        exploration_weight=exploration_weight,
        enable_reflection=enable_reflection,
        enable_ttrl_reward=enable_ttrl_reward,
        ttrl_vote_num=ttrl_vote_num,
        parser=react_format_parser if "Action:" in parser_tag else non_stream_parser,
        action_tag=parser_tag[0],
        observation_format="Observation: {}" if "Action:" in parser_tag else "<observation>{}</observation>",
    )
    logger.info(f"[INFO] Robot MCTS initialized")
    return robot_mcts