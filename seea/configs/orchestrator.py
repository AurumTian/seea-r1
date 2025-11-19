import os
import json
from seea.tools.langgraph_tools import *
from seea.configs.config import *
from seea.utils.common import langraph_tool_to_schema
from seea.utils.openai_support import function_to_schema
from seea.agents.models.models import *
from seea.agents.orchestrator.orchestrator import Orchestrator
from seea.agents.agent import  ChatAgent, FunctionCallAgent
from seea.utils.paser import react_stream_parser


def perception():
    """
    Get images to perceive and understand the environment.
    parameters:
    """
    return GET_REAL_IMAGE_RESPONSE


def get_orchestrator(model,
                     simulation=False,
                     save_folder="assets/data"):
    save_folder = os.path.join(save_folder, "visual_brain_agent") if save_folder else save_folder

    robotic_arm_operation_plan = robotic_arm_operation_plan_wrapper()
    tools = [robotic_arm_operation_plan]
    agent_executor = FunctionCallAgent(
        name='Executor',
        system_prompt=ROLE_PROMPT,
        model=model["model"],
        api_key=model["api_key"],
        api_base=model["base_url"],
        image_size=[448, 448],
        tool_choice='auto' if model["model"] in ["gpt-4o", 'o1', 'o1-mini'] else None,
        tools=tools,
        parser=react_stream_parser,
        camera_client=None,
        stop=["\nObservation:", "Observation:"],
        keep_message_history=False, 
        recursive=False,
        save_folder=save_folder,
        function_to_schema=langraph_tool_to_schema,
    )

    agent_executor.add_tool(perception, function_to_schema(perception))
    tools_list = agent_executor.get_tools_list()
    tools_decs = "\n\n###Available Tools###\n{}".format(json.dumps(tools_list, ensure_ascii=False))
    TOOL_DESC = tools_decs if model["model"] not in ["gpt-4o", 'o1', 'o1-mini'] else ""
    format_prompt = FUNCTION_CALL_PROMPT  if model["model"] not in ["gpt-4o", 'o1', 'o1-mini'] else ""
    basic_prompt = TOOLS_PROMPT + WORLD_PROMPT
    agent_executor.set_system_prompt(ROLE_PROMPT + format_prompt + TOOL_DESC + basic_prompt)
    agent_actor = ChatAgent(
        name='Actor',
        system_prompt=ROLE_PROMPT + ACTOR_PROMPT + tools_decs + basic_prompt,
        model=model["model"],
        api_key=model["api_key"],
        api_base=model["base_url"],
        keep_message_history=False,
        save_folder=save_folder,
    )

    agent_critic = ChatAgent(
        name='Critic',
        system_prompt=ROLE_PROMPT + CRITIC_PROMPT + basic_prompt,
        model=model["model"],
        api_key=model["api_key"],
        api_base=model["base_url"],
        keep_message_history=False,
        save_folder=save_folder,
    )

    agent_score = ChatAgent(
        name='Score',
        system_prompt=SORTOR_PROMPT,
        model=model["model"],
        api_key=model["api_key"],
        api_base=model["base_url"],
        keep_message_history=False,
        save_folder=save_folder,
        stop=["\nProposed", "\nProposed actions:", "Proposed actions:"]
    )

    stage_to_agent_map = {
        Stage.ACTOR: agent_actor,
        Stage.CRITIC: agent_critic,
        Stage.STEP: agent_executor,
        Stage.SORTOR: agent_score,
    }

    stage_transition_graph = {
        Stage.ACTOR: Stage.CRITIC,
        Stage.SORTOR: Stage.STEP,
        Stage.STEP: Stage.CRITIC,
        Stage.CRITIC: Stage.ACTOR,
    }
    orchestrator = Orchestrator(stage_to_agent_map, stage_transition_graph)
    return orchestrator