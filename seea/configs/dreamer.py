from seea.agents.agent.dreamer import Dreamer
from seea.agents.agent.chat_agent import ChatAgent
from seea.agents.agent.video_agent import VideoChatAgent
from seea.agents.models.models import VideoQualityResult


# Instruction Agent, generates reasonable embodied scene instructions based on image information, preparing for subsequent automated generation of dreamer datasets
def get_instruction_agent(system_prompt, model):
    instruction_agent = ChatAgent(
        system_prompt=system_prompt,
        model=model["model"],
        api_key=model["api_key"],
        api_base=model["base_url"],
    )
    return instruction_agent

# Video Quality Assessment Agent, judges whether the generated video is qualified and whether the instruction is successfully completed based on video quality
def get_video_quality_agent(system_prompt, model, frame_extraction_mode, frame_interval):
    video_quality_agent = VideoChatAgent(
        system_prompt=system_prompt,
        model=model["model"],
        api_key=model["api_key"],
        api_base=model["base_url"],
        frame_extraction_mode=frame_extraction_mode,
        frame_interval=frame_interval,
        result_model=VideoQualityResult,
    )
    return video_quality_agent

def get_dreamer_agent(system_prompt, model, video_generation_model, reflection_prompt, kling_version="v1-6"):
    dreamer_agent = Dreamer(
        system_prompt=system_prompt,
        model=model["model"],
        api_key=model["api_key"],
        api_base=model["base_url"],
        ak=video_generation_model["ak"],
        sk=video_generation_model["sk"],
        base_url=video_generation_model["base_url"],
        reflection_prompt=reflection_prompt,
	kling_version=kling_version
    )
    return dreamer_agent
