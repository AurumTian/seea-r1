import os
import yaml
from typing import Dict, Any, Optional
from seea.utils.logger import get_logger

logger = get_logger(__name__)

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        return {}
        
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded configuration file: {config_path}")
        return config or {}
    except Exception as e:
        logger.error(f"Failed to load configuration file: {str(e)}")
        return {}

def save_yaml_config(config: Dict[str, Any], config_path: str) -> bool:
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        config_path: Path to the YAML configuration file
        
    Returns:
        Whether the save was successful
    """
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"Successfully saved configuration file: {config_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save configuration file: {str(e)}")
        return False

def merge_configs(base_config, override_config):
    """
    Merge two configuration dictionaries, values in override_config will overwrite values in base_config
    Supports merging of nested dictionaries
    
    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration dictionary
    
    Returns:
        Merged configuration dictionary
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        # If both configurations contain the same key and are both dictionaries, merge recursively
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            # Otherwise, directly overwrite
            result[key] = value
    
    return result