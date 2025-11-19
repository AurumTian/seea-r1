import argparse
from typing import Dict, Any, Optional, Tuple
from seea.utils.logger import get_logger
from seea.utils.config_loader import load_yaml_config, save_yaml_config, merge_configs

logger = get_logger(__name__)

def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add common command line arguments for the seea system"""
    # Add config file parameter
    parser.add_argument('--config', type=str, default=None,
                       help='Path to the YAML configuration file')
    parser.add_argument('--generate-config', type=str, default=None,
                       help='Path to generate the default configuration file')
    
    # Add common parameters
    parser.add_argument('--version', type=str, default='mcts',
                       choices=['visual', 'mcts'], help='Agent version')
    parser.add_argument('--model', type=str, default='Qwen2_5-VL-72B-Instruct',
                       help='Name of the model to use')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Model name with higher priority than model')
    parser.add_argument('--base_url', type=str, default=None,
                       help='Base URL for the model API')
    # Add model generation parameters
    parser.add_argument('--temperature', type=float, default=0.6,
                       help='Temperature parameter')
    parser.add_argument('--top_p', type=float, default=0.9,
                       help='Top-p sampling parameter')
    parser.add_argument('--repetition_penalty', type=float, default=1.05,
                       help='Repetition penalty parameter')
    parser.add_argument('--max_tokens', type=int, default=2048,
                       help='Maximum number of tokens to generate')
    parser.add_argument('--image_size', type=int, nargs=2, default=[300, 300],
                       help='Image size, default is [300, 300]')
    
    # Add feature switch parameters
    parser.add_argument('--visual_perception', action='store_true',
                       help='Whether to enable visual perception')
    parser.add_argument('--wo_image_tool_result', action='store_true',
                       help='Whether to not use image tool results')
    parser.add_argument('--sort_action', action='store_true',
                       help='Whether to sort actions')
    parser.add_argument('--visual_som', action='store_true',
                       help='Whether to visualize SoM for images in ALFWorld environment, drawing bounding boxes, object categories, and serial numbers')

    # Add MCTS related parameters
    parser.add_argument('--num-proposed-action', type=int, default=5,
                       help='Number of actions proposed in MCTS each time, default is 5')
    parser.add_argument('--n-iterations', type=int, default=10,
                       help='Number of MCTS iterations, default is 10')
    parser.add_argument('--depth-limit', type=int, default=30,
                       help='MCTS search depth limit, default is 30')
    parser.add_argument('--exploration-weight', type=float, default=1.0,
                       help='MCTS exploration weight, default is 1.0')
    parser.add_argument('--enable_reflection', action='store_true',
                       help='Use reflection in MCTS')
    parser.add_argument('--enable_ttrl_reward', action='store_true',
                       help='Use Test-Time Reinforce Learning for Reward Model')
    parser.add_argument('--ttrl_vote_num', type=int, default=10,
                       help='Number of votes in Test-Time Reinforce Learning, default is 10')        
    
    # Add evaluation model related parameters
    parser.add_argument("--critic_model", type=str, default="",
                       help="Evaluation model name")
    parser.add_argument("--critic_model_name", type=str, default="",
                       help="Evaluation model name (if using a custom model)")
    parser.add_argument("--critic_base_url", type=str, default="",
                       help="Evaluation model API address")
    parser.add_argument("--critic_temperature", type=float, default=1.0,
                       help="Evaluation model temperature parameter")
    parser.add_argument("--critic_top_p", type=float, default=0.95,
                       help="Evaluation model top_p parameter")
    parser.add_argument("--critic_repetition_penalty", type=float, default=1.05,
                       help="Evaluation model repetition penalty parameter")
    parser.add_argument("--critic_max_tokens", type=int, default=1024,
                       help="Evaluation model maximum token count")
    
    return parser

def process_config_args(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Process configuration files and command line arguments, return merged configuration and user-provided arguments
    
    Args:
        args: Parsed command line arguments
    Returns:
        merged_config: Merged configuration dictionary
        user_provided_args: Dictionary of arguments explicitly provided by the user
    """
    # Load configuration file (if specified)
    config_dict = {}
    if args.config:
        config_dict = load_yaml_config(args.config)
        if not config_dict:
            logger.error(f"Failed to load configuration file: {args.config}, will use command line arguments")
        else:
            logger.debug(f"Configuration loaded from file: {config_dict}")
    
    # Convert command line arguments to a dictionary
    cmd_args = vars(args)
    # Get the value of generate_config parameter and remove it from cmd_args
    generate_config = cmd_args.pop("generate_config", None)
    # Remove configuration file related parameters
    cmd_args.pop("config", None)
    
    # Get the original form of command line arguments to determine which parameters were actually provided by the user
    import sys
    argv = sys.argv[1:]
    provided_arg_keys = set()
    
    # Parse command line arguments to find user-provided parameter names
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg.startswith('--'):
            arg_name = arg[2:]  # Remove the prefix '--'
            # Handle hyphen to underscore conversion
            arg_name = arg_name.replace('-', '_')
            provided_arg_keys.add(arg_name)
            
            # If the next argument does not start with '--', it is the value of the current parameter
            if i + 1 < len(argv) and not argv[i + 1].startswith('--'):
                i += 2  # Skip parameter name and value
            else:
                i += 1  # Skip only parameter name (boolean flag parameter)
        else:
            i += 1  # Skip non-parameter options
    
    # Remove config parameter as it is specially handled
    provided_arg_keys.discard('config')
    provided_arg_keys.discard('generate_config')
    
    logger.debug(f"Parameter keys provided in the command line: {provided_arg_keys}")
    
    # Extract parameters explicitly provided by the user in the command line
    user_provided_args = {key: value for key, value in cmd_args.items() if key in provided_arg_keys}
    
    logger.debug(f"User-provided arguments: {user_provided_args}")
    
    # Directly merge configuration file parameters and user-provided parameters (user parameters have the highest priority)
    merged_config = merge_configs(config_dict, user_provided_args)
    
    logger.debug(f"Merged configuration: {merged_config}")
    
    # If generate_config is not None, save the merged configuration
    if generate_config:
        if save_yaml_config(merged_config, generate_config):
            logger.info(f"Saved merged configuration to: {generate_config}")
        else:
            logger.error(f"Failed to save merged configuration: {generate_config}")
    
    return merged_config, user_provided_args

def extract_mcts_params(merged_config: Dict[str, Any], 
                        user_provided_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract MCTS parameters from the merged configuration
    
    Args:
        merged_config: Merged configuration dictionary
        user_provided_args: Dictionary of arguments explicitly provided by the user
    
    Returns:
        mcts_params: MCTS parameter dictionary
    """
    # Set default MCTS parameters
    mcts_params = {
        "num_proposed_action": 5,  # Default value
        "n_iterations": 10,        # Default value
        "depth_limit": 30,         # Default value
        "exploration_weight": 1.0,  # Default value
        "enable_reflection": False,  # Default value
        "enable_ttrl_reward": False,  # Default value
        "vote_num": 10,  # Default value
    }
    
    # 1. Apply MCTS parameters from the configuration file (if they exist)
    if "mcts" in merged_config and isinstance(merged_config["mcts"], dict):
        logger.debug(f"Applying MCTS parameters from configuration file: {merged_config['mcts']}")
        for key, value in merged_config["mcts"].items():
            normalized_key = key.replace("-", "_")
            if normalized_key in mcts_params:
                mcts_params[normalized_key] = value
                logger.debug(f"Applied MCTS parameter: {normalized_key}={value}")
    
    # 2. Apply MCTS parameters from the command line (highest priority)
    for key in list(mcts_params.keys()):
        if key in user_provided_args:
            mcts_params[key] = user_provided_args[key]
            logger.debug(f"Applied MCTS parameter from command line: {key}={user_provided_args[key]}")
        # Check hyphenated version of the parameter name
        cmd_key = key.replace("_", "-")
        if cmd_key in user_provided_args:
            mcts_params[key] = user_provided_args[cmd_key]
            logger.debug(f"Applied MCTS parameter from command line: {key}={user_provided_args[cmd_key]}")
    
    # Remove processed MCTS related parameters from the merged configuration
    merged_config.pop("mcts", None)
    for key in ["num_proposed_action", "n_iterations", "depth_limit", 
                "exploration_weight", "enable_reflection",
                "num-proposed-action", "n-iterations", "depth-limit", 
                "exploration-weight"]:
        merged_config.pop(key, None)
    
    return mcts_params