import logging
import os
from datetime import datetime
from types import MethodType
from typing import Optional

def get_logger(log_file: Optional[str] = None, log_level: Optional[int] = None, file_mode: str = 'w'):
    import colorlog
    """Get logging logger with colored output

    Args:
        log_file: Log pathname, if specified, file handler will be added to logger
        log_level: Logging level.
        file_mode: Specifies the mode to open the file
    """
    if log_level is None:
        log_level = os.getenv('LOG_LEVEL', 'DEBUG').upper()  # Default to DEBUG level
        log_level = getattr(logging, log_level, logging.DEBUG)  # Allow console to control level
    
    logger_name = __name__.split('.')[0]
    logger = logging.getLogger(logger_name)
    logger.propagate = False

    # Clear existing handlers to avoid duplication
    if logger.handlers:
        logger.handlers.clear()

    # Create colored console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)  # Explicitly allow DEBUG level

    # Add thread information to log format
    color_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(threadName)s - [%(pathname)s:%(lineno)d] %(message)s%(reset)s",  
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        },
        reset=True,
        style='%'
    )
    stream_handler.setFormatter(color_formatter)
    logger.addHandler(stream_handler)

    # Create log directory
    log_dir = os.path.join('assets', 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # If log_file is not specified, use default path
    if log_file is None:
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'{logger_name}_{current_time}.log')
    elif not os.path.isabs(log_file):
        # If a relative path is provided, place it in the logs directory
        log_file = os.path.join(log_dir, log_file)

    # Create file handler
    file_handler = logging.FileHandler(log_file, file_mode)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(threadName)s - [%(levelname)s:%(pathname)s:%(lineno)d] %(message)s'))
    logger.addHandler(file_handler)

    # Ensure logger level does not restrict DEBUG
    logger.setLevel(logging.DEBUG)

    # Add one-time print methods
    def info_once(self, msg, *args, **kwargs):
        if not hasattr(self, "_info_logged"):
            self._info_logged = set()
        if msg not in self._info_logged:
            self._info_logged.add(msg)
            self.info(msg, *args, **kwargs)

    def warning_once(self, msg, *args, **kwargs):
        if not hasattr(self, "_warning_logged"):
            self._warning_logged = set()
        if msg not in self._warning_logged:
            self._warning_logged.add(msg)
            self.warning(msg, *args, **kwargs)

    logger.info_once = MethodType(info_once, logger)
    logger.warning_once = MethodType(warning_once, logger)

    return logger
