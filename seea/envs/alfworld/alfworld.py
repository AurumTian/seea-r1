import cv2
import yaml
import copy
import traceback
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from seea.utils.base import StateManager
from alfworld.agents.environment import get_environment
from alfworld.agents.environment.alfred_thor_env import AlfredThorEnv
from alfworld.env.thor_env import ThorEnv
from seea.utils.logger import get_logger
logger = get_logger(__name__)


def load_config(env_type) -> Dict[str, Any]:
    """Load ALFWorld environment configuration"""
    if env_type == "AlfredThorEnv":
        config_path = Path("seea/envs/alfworld/configs/thor_config.yaml")
    elif env_type == "AlfredTWEnv":
        config_path = Path("seea/envs/alfworld/configs/tw_config.yaml")
    else:
        raise ValueError(f"Unknown environment type: {env_type}")
    return yaml.safe_load(config_path.read_text())


def get_alfworld_environment(env_type="AlfredThorEnv", config: Optional[Dict[str, Any]] = None, train_eval: str = 'train'):
    """Get ALFWorld environment and initialize visual information
    
    Args:
        env_type: Environment type, default is "AlfredThorEnv"
        config: Optional environment configuration dictionary, use default configuration if None
        train_eval: Environment mode, 'train','eval_in_distribution' and 'eval_out_of_distribution'
    """
    if config is None:
        config = load_config(env_type)
    
    def save_scene_state(self) -> Optional[Dict[str, Any]]:
        """Save scene state"""
        if self.last_event is None:
            logger.warning("Saving ALFWorld env failed! No event recorded.")
            return None
            
        object_poses = [
            {
                'objectName': obj['name'].split('(Clone)')[0],
                'position': obj['position'],
                'rotation': obj['rotation']
            }
            for obj in self.last_event.metadata['objects']
            if obj['pickupable']
        ]
        
        object_toggles = [
            {
                'objectType': obj['objectType'],
                'isOn': obj['isToggled']
            }
            for obj in self.last_event.metadata['objects']
            if obj['toggleable']
        ]
        
        dirty_and_empty = any(
            obj['dirtyable'] and obj['isDirty']
            for obj in self.last_event.metadata['objects']
        )
        
        return {
            'object_poses': object_poses,
            'object_toggles': object_toggles,
            'dirty_and_empty': dirty_and_empty
        }
    
    # Add scene state management method to ThorEnv class
    setattr(ThorEnv, 'save_scene_state', save_scene_state)
    
    # Modify environment initialization part
    alfred_thor_env = get_environment(config["env"]["type"])(config, train_eval=train_eval)
    alfred_thor_env = alfred_thor_env.init_env(batch_size=1)

    def get_state(self) -> Optional[Dict[str, Any]]:
        """Get the current state of the environment, including the task file and history actions"""
        if not hasattr(self, 'envs') or not self.envs:
            logger.warning("Getting ALFWorld env state failed! No env instance found.")
            return None
            
        env_thread = self.envs[0]
        if not env_thread.env:
            logger.warning("Getting ALFWorld env state failed! No env instance found.")
            return None
        
        # Save task file path
        task_file = getattr(env_thread, 'task_file', None)
        if not task_file:
            logger.warning("Getting ALFWorld env state failed! No task file found.")
            return None
        
        # Save history actions
        action_history = copy.deepcopy(getattr(env_thread, 'action_history', []))
        action_history = StateManager().get("action_history", []) if not action_history or action_history is None else action_history
        
        # Save basic information of current environment
        return {
            'task_file': task_file,
            'action_history': action_history,
            'steps': getattr(env_thread, 'steps', 0),
            'prev_command': getattr(env_thread, 'prev_command', ""),
            '_done': getattr(env_thread, '_done', False),
            '_feedback': getattr(env_thread, '_feedback', ""),
            'controller_type': getattr(env_thread, 'controller_type', "oracle"),
            'goal_desc_human_anns_prob': getattr(env_thread, 'goal_desc_human_anns_prob', 0.0),
        }
        
    def set_state(self, state: Dict[str, Any]) -> bool:
        """Restore environment state by resetting environment and replaying history actions"""
        if not state or not hasattr(self, 'envs') or not self.envs:
            logger.warning("Restoring ALFWorld env state failed! Invalid state or no env instance found.")
            return False
            
        try:
            env_thread = self.envs[0]
            if not env_thread.env:
                logger.warning("Restoring ALFWorld env state failed! No env instance found.")
                return False
            
            # 1. Get task file and history actions
            task_file = state.get('task_file')
            action_history = state.get('action_history', [])
            
            if not task_file:
                logger.warning("Restoring ALFWorld env state failed! No task file in state.")
                return False
            
            # 2. Reset environment to initial state
            logger.info(f"Reset environment to task: {task_file}")
            env_thread.reset(task_file)
            
            # 3. Replay history actions
            logger.info(f"Start replaying {len(action_history)} history actions")
            last_obs, last_rewards, last_dones, last_infos = None, None, None, None
            
            for i, action in enumerate(action_history):
                logger.info(f"Replay action {i+1}/{len(action_history)}: {action}")
                obs, rewards, dones, infos = self.step([action])
                last_obs, last_rewards, last_dones, last_infos = obs, rewards, dones, infos
            
            # 4. Restore basic state variables
            env_thread.steps = state.get('steps', 0)
            env_thread.prev_command = state.get('prev_command', "")
            env_thread._done = state.get('_done', False)
            env_thread._feedback = state.get('_feedback', "")
            
            # 5. Update the last information of the environment, ensure the goal_condition_success_rate etc. state is correct
            if last_infos:
                # Save the last information to the environment
                self.last_info = last_infos
                
                # Update the related state in StateManager
                state_manager = StateManager()
                
                # Update the basic environment state
                if last_obs:
                    state_manager.set("alfworld_env_obs", last_obs[0])
                if last_dones:
                    state_manager.set("alfworld_env_dones", last_dones[0])
                
                # Update the reward information
                if 'goal_condition_success_rate' in last_infos:
                    reward = float(last_infos['goal_condition_success_rate'][0])
                    state_manager.set("reward", reward)
                    logger.info(f"The goal_condition_success_rate after restoring: {reward}")
                
                # Update the evaluation related information
                if last_obs:
                    state_manager.set("eval_observation", last_obs[0])
                if last_dones:
                    state_manager.set("eval_done", last_dones[0])
                if 'won' in last_infos:
                    state_manager.set("eval_reward", float(last_infos['won'][0]))
                if 'goal_condition_success_rate' in last_infos:
                    state_manager.set("eval_gc_reward", float(last_infos['goal_condition_success_rate'][0]))
                    state_manager.set("eval_current_gc_success", float(last_infos['goal_condition_success_rate'][0]))
        
            logger.info("Restored ALFWorld env state successfully!")
            return True
        except Exception as e:
            logger.error(f"Restore environment state failed: {str(e)}")
            traceback.print_exc()
            return False

    # Add state management method
    setattr(alfred_thor_env, '__getstate__', get_state.__get__(alfred_thor_env))
    setattr(alfred_thor_env, '__setstate__', set_state.__get__(alfred_thor_env))
    
    def get_visual_obs(self, visual_som=False) -> Optional[str]:
        """Get the visual observation of the current environment"""
        if not isinstance(self, AlfredThorEnv):
            raise NotImplementedError(f"Environment {self.__class__.__name__} is not supported.")
        
        try:
            frame = None
            if visual_som:
                frame = self.get_exploration_frames()[0]
            if frame is None:
                frame = self.get_frames()[0]
            if frame is None:
                logger.warning("Failed to get visual observation! No frame data.")
                return None
                
            prev_frame = StateManager().get("env_image", None)

            if prev_frame is None or not np.array_equal(frame, prev_frame):
                # Modify here, ensure to get sample_save_dir from thread local storage
                sample_save_dir = StateManager().get("sample_save_dir", "./visual_obs", thread_local=True)
                frame_path = save_visual_obs(frame, sample_save_dir)
                StateManager().set("env_image_path", frame_path)
                StateManager().set("env_image", frame)
                return frame_path
            return None
        except AttributeError as e:
            # Handle the case where the environment doesn't have the required attributes
            print(f"Warning: Could not get visual observation - {str(e)}")
            return None

    # Add the method to get visual observation to the environment object
    setattr(alfred_thor_env.__class__, 'get_visual_obs', get_visual_obs)
    return alfred_thor_env


def save_visual_obs(frame: np.ndarray, save_dir: str = "./visual_obs") -> str:
    """Save the visual observation as a picture file"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Use datetime to get microsecond timestamp
    now = datetime.now()
    timestamp = now.strftime('%Y%m%d_%H%M%S') + f'_{now.microsecond:06d}'
    frame_path = str(save_path / f"frame_{timestamp}.png")
    
    try:
        # Ensure the image data is valid
        if frame is None or frame.size == 0:
            print("Warning: Empty frame data, cannot save image")
            return None
            
        # Convert BGR to RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        # Use PIL to save image
        from PIL import Image
        Image.fromarray(frame).save(frame_path)
        
        # Verify if the file is successfully written
        if not Path(frame_path).exists() or Path(frame_path).stat().st_size == 0:
            print(f"Warning: Failed to save image to {frame_path}")
            return None
            
        return frame_path
    except Exception as e:
        print(f"Error saving visual observation: {str(e)}")
        return None
