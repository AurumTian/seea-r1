import os
from typing import Dict, Any, Union
from seea.configs.visual_agent import get_visual_agent
from seea.configs.robot_mcts import get_robot_mcts
from seea.utils.logger import get_logger

logger = get_logger(__name__)

def create_agent(config: dict) -> Any:
    """
    Unified agent creation function, supports visual and mcts types
    Args:
        config: Configuration object, contains all parameters
    Returns:
        Initialized agent instance
    """
    # Merge mcts_params into the main config (if it exists)
    if "mcts_params" in config and isinstance(config["mcts_params"], dict):
        for k, v in config["mcts_params"].items():
            if k not in config:
                config[k] = v
    # Ensure save_folder exists
    save_folder = config.get("save_folder", "")
    if save_folder and not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    try:
        version = config.get("version", "mcts")
        if version == "visual":
            agent = get_visual_agent(config)
            logger.info("Successfully initialized Visual agent")
            return agent
        elif version == "mcts":
            agent = get_robot_mcts(config)
            logger.info("Successfully initialized MCTS agent")
            return agent
        else:
            logger.error(f"Unknown agent version: {version}")
            return None
    except Exception as e:
        logger.error(f"Failed to initialize agent: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None