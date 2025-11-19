# alfworld.py - ALFWorld environment related configurations
import os
from seea.configs.config import PROJECT_ROOT

# ALFWorld environment configurations
ALFWORLD_CONFIG = {
    # ICL example file path
    "icl_path": os.path.join(PROJECT_ROOT, "seea/prompt/icl_examples/alfworld_icl_xml.json"),
    "react_icl_path": os.path.join(PROJECT_ROOT, "seea/prompt/icl_examples/alfworld_icl_react.json"),
    # Align with MPO using 1 example
    "icl_num": 1,
}