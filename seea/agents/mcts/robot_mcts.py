import os
import re
import json
import copy
import traceback
import numpy as np
from collections import deque
import threading
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Callable
from langchain_core.tools import BaseTool
from seea.agents.models.models import *
from seea.utils.base import StateManager
from seea.utils.paser import non_stream_parser
from seea.agents.mcts.core.mcts import MCTS, MCTSResult, MCTSNode
from seea.agents.mcts.core.base import Reasoner, SearchConfig, WorldModel
from seea.agents.agent.chat_agent import ChatAgent
from seea.utils.logger import get_logger
logger = get_logger(__name__)

# ANSI color codes
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"

def get_default_memory()-> Memory:
    return Memory(history=[],
                  current_stage=Stage.ACTOR,
                  think=None,
                  proposed_actions=[],
                  consideration=None,
                  best_action_index=None,
                  observation=None,
                  reflection=None,
                  state=State.CONTINUE,
                  critic_state=None,
                  )

class RobotWorldModel(WorldModel[RobotState, RobotAction]):
    def __init__(self, 
                 tools_call: Dict,
                 critic: Optional[ChatAgent] = None,
                 actor: Optional[ChatAgent] = None,
                 alfworld_env: object = None,
                 trust_critic: bool = False,
                 enable_ttrl_reward: bool = False,
                 ttrl_vote_num: int = 10,
                 depth_limit: int = 10,
                 action_tag: str = "<action>",
                 observation_format: str = "<observation>{}</observation>",
                 ) -> None:
        super().__init__()
        self.tools_call = tools_call
        self.critic = critic
        self.actor = actor
        self.depth_limit = depth_limit
        self.state_manager = StateManager()
        self.alfworld_env = alfworld_env
        self.trust_critic = trust_critic
        self.enable_ttrl_reward = enable_ttrl_reward
        self.ttrl_vote_num = ttrl_vote_num
        # Dictionary for storing environment states
        self.env_states = {}
        self.current_node_id = 0 # Tracks the env_state key for the current StateManager context
        self.active_fully_loaded_env_key = 0 # Tracks the env_state key for which alfworld_env is fully loaded
        self.action_history_to_state_map = {}
        self.node_id_to_state_map = {}
        self.action_tag = action_tag
        self.observation_format = observation_format

    def _save_env_state(self, node_id_to_save: int, update_active_full_env_key_if_alfworld_saved: bool = True) -> None:
        """Save the current environment state"""
        try:
            state_manager_values_to_save = {}
            if self.alfworld_env:
                # Save relevant state values from StateManager
                state_manager_values_to_save = {
                    # Basic environment state
                    "env_steps": self.state_manager.get("env_steps", 0),
                    "current_action": self.state_manager.get("current_action", ""),
                    "alfworld_env_obs": self.state_manager.get("alfworld_env_obs", ""),
                    "alfworld_env_dones": self.state_manager.get("alfworld_env_dones", False),
                    "reward": self.state_manager.get("reward", 0.0),
                    
                    # Evaluation related state
                    "eval_action": self.state_manager.get("eval_action", ""),
                    "eval_observation": self.state_manager.get("eval_observation", ""),
                    "eval_done": self.state_manager.get("eval_done", False),
                    "eval_reward": self.state_manager.get("eval_reward", 0.0),
                    "eval_gc_reward": self.state_manager.get("eval_gc_reward", 0.0),
                    "eval_completed": self.state_manager.get("eval_completed", False),
                    "eval_max_reward": self.state_manager.get("eval_max_reward", 0.0),
                    "eval_max_gc_success": self.state_manager.get("eval_max_gc_success", 0.0),
                    "eval_current_reward": self.state_manager.get("eval_current_reward", 0.0),
                    "eval_current_gc_success": self.state_manager.get("eval_current_gc_success", 0.0),
                    
                    # History
                    "eval_history": self.state_manager.get("eval_history", []),
                    "action_history": self.state_manager.get("action_history", []),
                    "eval_rewards_history": self.state_manager.get("eval_rewards_history", []),
                    "eval_gc_history": self.state_manager.get("eval_gc_history", []),
                    
                    # Vision related
                    "env_image_path": self.state_manager.get("env_image_path", None),
                    "env_image": self.state_manager.get("env_image", False),
                }
            else:
                # Save relevant state values from StateManager
                state_manager_values_to_save = { 
                    "reward": self.state_manager.get("reward", 0.0),
                    "env_image_path": self.state_manager.get("env_image_path", None),
                }

            current_alfworld_snapshot = None
            if self.alfworld_env:
                current_alfworld_snapshot = self.alfworld_env.__getstate__()

            # Determine the alfworld snapshot to save for node_id_to_save
            alfworld_snapshot_for_node = None
            if update_active_full_env_key_if_alfworld_saved:
                alfworld_snapshot_for_node = current_alfworld_snapshot
            else:
                # Preserve existing alfworld state if not updating active full env key
                existing_combined_state = self.env_states.get(node_id_to_save)
                if existing_combined_state and "env_state" in existing_combined_state:
                    alfworld_snapshot_for_node = existing_combined_state["env_state"]
                # else: if no existing state, it will be None, which is fine
            
            combined_state_to_save = {
                "env_state": alfworld_snapshot_for_node,
                "state_manager": state_manager_values_to_save
            }
            self.env_states[node_id_to_save] = copy.deepcopy(combined_state_to_save)
            
            if update_active_full_env_key_if_alfworld_saved and self.alfworld_env and current_alfworld_snapshot is not None:
                 self.active_fully_loaded_env_key = node_id_to_save

            logger.debug(f"{CYAN}Saved env state for Node ID: {node_id_to_save} (FullUpdate: {update_active_full_env_key_if_alfworld_saved}){RESET}")
        except Exception as e:
            logger.error(f"{RED}Failed to save environment state: {str(e)}{RESET}")
            traceback.print_exc()

    def _load_env_state(self, node_id_to_load: int, restore_full_alfworld_env: bool = False) -> bool:
        """Load the environment state of the specified node, including the environment itself and relevant state values in StateManager"""
        try:
            if node_id_to_load in self.env_states:
                combined_state = self.env_states[node_id_to_load]

                # Restore relevant state values in StateManager
                state_manager_values = combined_state["state_manager"]
                for key, value in state_manager_values.items():
                    self.state_manager.set(key, copy.deepcopy(value))
                logger.debug(f"{CYAN}Loaded StateManager for Node ID: {node_id_to_load}{RESET}")
                
                if self.alfworld_env:
                    if restore_full_alfworld_env:
                        if combined_state["env_state"] is not None:
                            self.alfworld_env.__setstate__(copy.deepcopy(combined_state["env_state"]))
                            self.active_fully_loaded_env_key = node_id_to_load
                            logger.debug(f"{CYAN}Fully restored ALFWorld env for Node ID: {node_id_to_load}{RESET}")
                        else:
                            # This case means node_id_to_load was saved when self.alfworld_env was None or __getstate__ failed for it.
                            # Or it was saved with update_active_full_env_key_if_alfworld_saved=False and had no prior alfworld state.
                            logger.warning(f"{YELLOW}Attempted to fully restore ALFWorld for Node ID {node_id_to_load}, but no env_state snapshot was found in storage for it. ALFWorld live state and active_fully_loaded_env_key remain unchanged.{RESET}")
                    # else: alfworld_env object is not touched by this call.
                    # self.active_fully_loaded_env_key is also not changed.

                return True
            logger.warning(f"{YELLOW}Environment state for the specified node not found - Node ID: {node_id_to_load}{RESET}")
            return False
        except Exception as e:
            logger.error(f"{RED}Failed to load environment state: {str(e)}{RESET}")
            traceback.print_exc()
            return False

    def init_state(self, instruction=None, image_path=None, **kwargs) -> RobotState:
        """Initialize robot state"""
        logger.info(f"{GREEN}Initializing robot state and resetting caches{RESET}")

        # Resetting caches for a new MCTS search
        self.env_states = {}
        self.action_history_to_state_map = {}
        self.node_id_to_state_map = {}
        # self.current_node_id is implicitly reset as the flow progresses from root.
        # Setting to 0 as root node (ID 0) state will be saved.
        self.current_node_id = 0 
        
        # Handle both positional and keyword arguments
        if instruction is None:
            instruction = kwargs.get('instruction', '')
        if image_path is None:
            image_path = kwargs.get('image_path', None)
        
        memory = get_default_memory()
        if image_path:
            # Handle regular initialization
            message = {"role": "user", "content": {"text": instruction}}
            if isinstance(image_path, str):
                image_path = [image_path]
            message["content"]["image"] = image_path
            self.state_manager.set("env_image_path", image_path[0])
        else:
            # Handle regular initialization
            message = {"role": "user", "content": instruction}
        system_message = {"role": "system", "content": self.actor.get_system_prompt()}
        memory.history.append(system_message)
        memory.history.append(message)

        if memory.observation is None and image_path:
            memory.observation = Observation(images=image_path, video=None, description=None)

        self.state = RobotState(
            memory=memory,
            status=RobotStatus.Available,
            task=instruction,
            hands_status=[{"hand": "left", "status": "idle"}, {"hand": "right", "status": "idle"}],
            skill_status=[],
            plan=[],
            id=0,  # Root node ID is 0
            action_history=[],
            expand_times=0,
        )
        self._save_env_state(0, update_active_full_env_key_if_alfworld_saved=True) # Save initial state for node 0
        self.node_id_to_state_map[0] = 0 # Map MCTS root node 0 to env_state key 0
        return self.state

    def step(self, state: RobotState, action: RobotAction, node_id: int, **kwargs) -> Tuple[RobotState, Observation]:
        """Execute robot action and get new state"""
        state.memory.current_stage = Stage.STEP
        new_state = RobotState(
            memory=copy.deepcopy(state.memory),
            status=state.status,
            task=state.task,
            hands_status=copy.deepcopy(state.hands_status),
            skill_status=copy.deepcopy(state.skill_status),
            plan=copy.deepcopy(state.plan),
            id=node_id, # current_mcts_node_id
            action_history=copy.deepcopy(state.action_history),
            expand_times=state.expand_times,
        )
        new_state.memory.critic_state = None
        result = "Task stopped."
        action_detail = None # Initialize action_detail

        if len(action.action) == 1 and hasattr(action.action[0], 'name') and hasattr(action.action[0], 'arguments'):
             new_state.action_history.append({"name": action.action[0].name, "arguments": action.action[0].arguments})
      
        current_mcts_node_id = node_id 
        parent_mcts_node_id = getattr(state, 'id', None)

        # env_key_of_parent_state is the env_states key for the parent MCTS node's state context.
        # It defaults to self.current_node_id, which tracks the active StateManager context.
        env_key_of_parent_state = self.current_node_id 
        if parent_mcts_node_id is not None:
            env_key_of_parent_state = self.node_id_to_state_map.get(parent_mcts_node_id, self.current_node_id)
        
        # The new MCTS node will initially point to its parent's env state key for its StateManager content.
        self.node_id_to_state_map[current_mcts_node_id] = env_key_of_parent_state

        # If current StateManager (self.current_node_id) is not for the parent state context, load SM for parent.
        if self.current_node_id != env_key_of_parent_state:
            logger.debug(f"{YELLOW}StateManager diverged. Loading SM from {env_key_of_parent_state} (parent's context) for node {current_mcts_node_id}. Active SM was for {self.current_node_id}{RESET}")
            self._load_env_state(env_key_of_parent_state, restore_full_alfworld_env=False)
            self.current_node_id = env_key_of_parent_state # SM is now for parent's context.
        
        if len(action.action) > 0 and hasattr(action.action[0], 'complete_response'):
            assitant_message = {"role": "assistant", "content": action.action[0].complete_response}
            new_state.memory.history.append(assitant_message)

        tool_message = {"role": "tool"}
        tool_call_was_executed_this_step = False

        if not action.action or not hasattr(action.action[0], 'name'):
            new_state.status = RobotStatus.Error
            result = "Invalid action structure."
        elif len(action.action) > 1:
            new_state.status = RobotStatus.Error
            result = "Format error: not support multiple actions!"
        else:
            action_detail = action.action[0]
            action_name = action_detail.name
            action_args = action_detail.arguments

            if not action_name:
                if self.action_tag in action_detail.complete_response:
                    result = "Invalid JSON in action tag."
                else: 
                    result = "No action performed." 
            elif str(new_state.action_history) in self.action_history_to_state_map: # Cache Hit
                history_entry = self.action_history_to_state_map[str(new_state.action_history)]
                result = copy.deepcopy(history_entry.get("tool_result", "unknown error."))
                key_of_cached_resulting_state = history_entry.get("node_id")

                if key_of_cached_resulting_state is not None:
                    self.node_id_to_state_map[current_mcts_node_id] = key_of_cached_resulting_state
                    # Load ONLY StateManager for this cached state. ALFWorld is NOT restored here.
                    self._load_env_state(key_of_cached_resulting_state, restore_full_alfworld_env=False)
                    self.current_node_id = key_of_cached_resulting_state # SM is now for this cached state.
                else: 
                    logger.warning(f"{RED}History entry for {str(new_state.action_history)} missing node_id!{RESET}")
                tool_call_was_executed_this_step = False

            elif action_name in self.tools_call: # New Tool Call
                # Validate 'instruction' argument if present (specific business logic)
                if "instruction" in action_args and action_args.get("instruction"):
                    instruction_text = action_args.get("instruction").lower().strip()
                    valid_actions_prefixes = ["go to", "take", "put", "open", "close", "use", "clean", "heat", "cool"]
                    if not any(instruction_text.startswith(prefix) for prefix in valid_actions_prefixes):
                        result = f"Error: '{instruction_text}' is not a valid action instruction. Valid actions include: go to, take, put, open, close, use, clean, heat, cool."
                        new_state.status = RobotStatus.Error
                        description_for_obs = self.observation_format.format(result)
                        tool_message["content"] = description_for_obs
                        new_state.memory.history.append(tool_message)
                        new_state.memory.observation = Observation(images=None, video=None, description=description_for_obs)
                        # Save current SM state (which is parent's context) before returning
                        self._save_env_state(self.current_node_id, update_active_full_env_key_if_alfworld_saved=True) 
                        return new_state, None
                
                # Ensure full alfworld_env (if used) for the parent state (env_key_of_parent_state) is active before tool call.
                # self.current_node_id should already be env_key_of_parent_state due to earlier sync.
                if self.alfworld_env and self.active_fully_loaded_env_key != env_key_of_parent_state:
                    logger.debug(f"{YELLOW}Tool call '{action_name}' on node {current_mcts_node_id} needs parent's ({parent_mcts_node_id}, env key {env_key_of_parent_state}) full env. Restoring. Live alfworld_env was for {self.active_fully_loaded_env_key}{RESET}")
                    self._load_env_state(env_key_of_parent_state, restore_full_alfworld_env=True)
                    # self.active_fully_loaded_env_key is now env_key_of_parent_state
                    # SM is also for env_key_of_parent_state
                self.current_node_id = env_key_of_parent_state # Ensure SM context is parent's for tool call.

                try:
                    tool_function = self.tools_call[action_name]
                    if isinstance(tool_function, BaseTool):
                        tool_output = tool_function.invoke(input=action_args)
                    else:
                        tool_output = tool_function(**action_args)
                    result = tool_output
                    tool_call_was_executed_this_step = True
                except Exception as e:
                    result = f"Encountered an error when calling the function {action_name}: {e}."
                    logger.error(f"{RED}Error in tool call {action_name}: {traceback.format_exc()}{RESET}")
                    tool_call_was_executed_this_step = False

                if tool_call_was_executed_this_step:
                    # The tool call modified self.alfworld_env (if exists) and self.state_manager.
                    # This new state corresponds to current_mcts_node_id.
                    # Save this new state into a new slot, keyed by current_mcts_node_id itself.
                    self.node_id_to_state_map[current_mcts_node_id] = current_mcts_node_id 
                    self.current_node_id = current_mcts_node_id # SM context is now for this new node.
                                                                # active_fully_loaded_env_key will be updated by _save_env_state later.
                    self.action_history_to_state_map[str(new_state.action_history)] = {
                        "node_id": current_mcts_node_id, 
                        "tool_result": copy.deepcopy(result)
                    }
            else: 
                result = f"Action '{action_name}' is not an available tool."
                new_state.status = RobotStatus.Error
                tool_call_was_executed_this_step = False
        
        description_for_obs = None
        if isinstance(result, dict):
            if "text" in result:
                result["text"] = self.observation_format.format(result.get("text", ""))
            else:
                result["text"] = self.observation_format.format("")
            if "image" in result and tool_message["role"] == "tool":
                tool_message["role"] = "user"
            description_for_obs = str(result.get("text", result))
        else:
            description_for_obs = self.observation_format.format(result)
        
        current_image_path = self.state_manager.get("env_image_path", None)
        observation = Observation(
            images=[current_image_path] if current_image_path else None,
            video=None,
            description=description_for_obs
        )
        new_state.memory.observation = observation

        logger.info(f"[Step] Node {current_mcts_node_id}, Action: {action_detail.name if action_detail and hasattr(action_detail, 'name') else 'N/A'}, Result: {result}")
        tool_message["content"] = result
        new_state.memory.history.append(tool_message)
        
        # Save the state of self.current_node_id. 
        # If a tool was executed, this saves the new state of current_mcts_node_id (now self.current_node_id).
        # If loaded from cache, this saves the SM state of key_of_cached_resulting_state (now self.current_node_id).
        # update_active_full_env_key_if_alfworld_saved=True ensures active_fully_loaded_env_key is updated if a new alfworld state was created by a tool or fully loaded.
        self._save_env_state(self.current_node_id, update_active_full_env_key_if_alfworld_saved=True)
        return new_state, None

    def is_terminal(self, node: MCTSNode) -> tuple[bool, float]:
        """检查状态是否为终止状态"""
        state = node.state
        terminal, reward = False, 0.0
        
        if not hasattr(state.memory, 'critic_state') or state.memory.critic_state is None:
            state.memory.critic_state = {}
        
        env_state_key_for_this_node = self.node_id_to_state_map.get(node.id, self.current_node_id)
        original_active_sm_key = self.current_node_id # Save current SM context

        if env_state_key_for_this_node != original_active_sm_key:
            logger.debug(f"{YELLOW}[is_terminal] Node {node.id} (env key {env_state_key_for_this_node}) SM differs from active SM ({original_active_sm_key}). Loading its SM.{RESET}")
            self._load_env_state(env_state_key_for_this_node, restore_full_alfworld_env=False)
            # self.current_node_id is NOT changed here, SM is temporarily for env_state_key_for_this_node

        # Check depth limit
        if node.depth >= self.depth_limit:
            terminal = True
            current_reward_in_context = self.state_manager.get("reward", 0.0)
            reward = -1.0 if current_reward_in_context <= 0.0 else current_reward_in_context
            logger.info(f"[is_terminal] Node {node.id} reached depth limit {self.depth_limit}. Reward: {reward}")
            if reward >= 1.0: 
                 state.memory.critic_state["solution"] = "Success"
            elif reward < 0: 
                 state.memory.critic_state["solution"] = "Failure"
            else: 
                 state.memory.critic_state["solution"] = "Continue"
        elif self.alfworld_env:
            terminal = self.state_manager.get("alfworld_env_dones", False)
            reward = self.state_manager.get("reward", 0.0)
            if terminal:
                if reward < 1.0: 
                    state.memory.critic_state["solution"] = "Failure"
                    if reward == 0.0: reward = -1.0
                else: 
                    state.memory.critic_state["solution"] = "Success"
            else: 
                state.memory.critic_state["solution"] = "Continue"
            logger.info(f"[is_terminal] Node {node.id} alfworld_env_dones: {terminal}, reward: {reward}")
        
        if not terminal and state.status is RobotStatus.Error: 
            terminal = True
            current_reward_in_context = self.state_manager.get("reward", 0.0)
            reward = -1.0 if current_reward_in_context <= 0.0 else current_reward_in_context
            state.memory.critic_state["solution"] = "Failure"
            logger.info(f"[is_terminal] Node {node.id} has RobotStatus.Error. Reward: {reward}")

        if self.critic:
            task_state_from_critic = run_critic(self.critic, state, enable_ttrl_reward=self.enable_ttrl_reward, ttrl_vote_num=self.ttrl_vote_num) 
            state.memory.critic_state["prediction"] = task_state_from_critic.value

            if self.trust_critic: 
                state.memory.state = task_state_from_critic 
                logger.debug(f"{CYAN}[is_terminal] Node {node.id} critic state: {task_state_from_critic}{RESET}")
                if task_state_from_critic == State.FAILURE:
                    reward = -1.0
                    terminal = True
                elif task_state_from_critic == State.SUCCESS:
                    reward = 1.0
                    terminal = True
                else:
                    reward = 0.0
                    terminal = False

        self.state_manager.set("reward", reward) 
        # Save the updated StateManager for this node, but DO NOT alter its stored alfworld_env snapshot or the active_fully_loaded_env_key
        self._save_env_state(env_state_key_for_this_node, update_active_full_env_key_if_alfworld_saved=False)
        
        # Restore the original StateManager context if it was changed for this evaluation
        if env_state_key_for_this_node != original_active_sm_key:
            logger.debug(f"{YELLOW}[is_terminal] Restoring original active SM context {original_active_sm_key} after evaluating node {node.id}{RESET}")
            self._load_env_state(original_active_sm_key, restore_full_alfworld_env=False)
            # self.current_node_id is NOT changed here as it should reflect original_active_sm_key if it was different.
            # Actually, if we loaded original_active_sm_key, current_node_id should be set to it.
            # However, self.current_node_id was never changed if env_state_key_for_this_node != original_active_sm_key.
            # The self.current_node_id variable tracks the SM context from the step() perspective.
            # Inside is_terminal, self.state_manager is temporarily pointed. The original self.current_node_id is implicitly restored
            # because is_terminal does not modify self.current_node_id.
            # The load above ensures self.state_manager is back to original_active_sm_key's content.
            # No, self.current_node_id should be restored if it was virtually changed by loading a different SM.
            # The issue is self.current_node_id is the global tracker for step. is_terminal should not modify it.
            # The _load_env_state calls inside is_terminal are for its local StateManager instance only.
            # The _load_env_state correctly uses self.state_manager.set(), so it acts on the shared SM object.

        return terminal, reward


class RobotMCTSSearchConfig(SearchConfig[RobotState, RobotAction]):
    def __init__(self,
                 actor: ChatAgent,
                 sortor: Optional[ChatAgent] = None,
                 critic: Optional[ChatAgent] = None,
                 num_proposed_action: int = 3,
                 parser: Optional[Callable] = None,
                 action_tag: str = "Action:",
                 ) -> None:
        super().__init__()
        self.actor = actor  # Generates N actions based on historical information
        self.sortor = sortor  # Sorts the N actions generated by the actor to get a sorted_action_list
        self.critic = critic  # Judges the status of the current task execution
        self.num_proposed_action = num_proposed_action # Number of actions generated each time
        self.parser = parser
        self.action_tag = action_tag
        logger.debug(f"{BLUE}RobotMCTSSearchConfig initialized{RESET}")

    def stable_softmax_for_multiple_responses(self, logprobs_list):
        """
        Stable Softmax calculation for a list of logprobs from multiple responses, avoiding numerical underflow.
        
        Args:
            logprobs_list: List[List[float]], list of logprobs for multiple responses, where each element is the logprobs for a response.
        
        Returns:
            List[float]: Softmax probability for each response
        """
        if len(logprobs_list) == 0:
            return []

        # Calculate the sum of logprobs for each response
        sums_per_response = []
        for logprobs in logprobs_list:
            # Filter out logprobs with a value of 0
            filtered_logprobs = [logprob for logprob in logprobs if logprob != 0]
            sums_per_response.append(sum(filtered_logprobs) / len(filtered_logprobs) if len(filtered_logprobs) > 0 else 0)

        # Calculate softmax
        exp_values = np.exp([sum_val for sum_val in sums_per_response])
        softmax_values = exp_values / np.sum(exp_values)

        return softmax_values

    def get_actions(self, robot_state: RobotState, logprobs: bool = False, force_prefix_think: bool = False, reflection_prefix: str = "") -> List[RobotAction]:
        logger.debug(f"{YELLOW}Getting actions for current state{RESET}")
        
        # Update current stage
        robot_state.memory.current_stage = Stage.ACTOR
        history = copy.deepcopy(robot_state.memory.history)
        robot_state.memory.proposed_actions = []
        
        try:
            # Create a deep copy of the history to avoid interference between threads
            processed_history = copy.deepcopy(history)
            
            # Initialize the number of actions to the configured number
            n_actions = self.num_proposed_action
            
            # If it's the first expansion/forced thinking/evaluation, keep the number of actions
            if force_prefix_think or robot_state.expand_times == 0 or StateManager().get("evaluate", False, thread_local=False):
                n_actions = self.num_proposed_action
            # With a 50% probability, reduce the number of actions to 1 to avoid over-expansion
            elif np.random.random() < 0.5:
                logger.debug(f"{YELLOW}Random probability is less than 0.5, reducing the number of actions to 1 to avoid over-expansion{RESET}")
                n_actions = 1
            # If the number of expansions exceeds 5, generate only 2 actions to avoid over-expansion
            elif robot_state.expand_times > 5:
                logger.debug(f"{YELLOW}Number of expansions: {robot_state.expand_times} > 5, reducing expansion width, generating only 2 actions{RESET}")
                n_actions = 2
            # If the number of expansions exceeds 3, generate only 1 action to avoid over-expansion
            elif robot_state.expand_times > 3:
                logger.debug(f"{YELLOW}Number of expansions: {robot_state.expand_times} > 3, stopping expansion, generating only 3 actions{RESET}") # Note: Original comment said 1 action, but code uses 3. Kept code's logic.
                n_actions = 3
            
            # Use actor.run method to request n actions at once, and control whether to enable logprobs
            response = self.actor.batch_run(processed_history, n=n_actions, logprobs=logprobs)
            
            if not response:
                logger.debug(f"{RED}Empty content in message, skipping...{RESET}")
                return []
            
            logprobs_list = []
            proposed_actions = []

            for res in response:
                response_content = res.get('response', '')
                current_logprobs = res.get('logprobs', []) # Renamed to avoid conflict

                # Parse response content
                if response_content:
                    result = self.parser(response_content)
                    actor_output = ActorOutput.model_validate(result)
                    proposed_action = actor_output.action

                    if proposed_action and current_logprobs: # Use renamed variable
                        # Add action's logprobs
                        proposed_actions.append(proposed_action)
                        logprobs_list.append(current_logprobs) # Use renamed variable
            
            if logprobs_list:
                # Perform stable Softmax calculation on logprobs of multiple responses
                softmax_values = self.stable_softmax_for_multiple_responses(logprobs_list)

                # Add the generated actions and their Softmax probabilities together with the actions to proposed_actions
                for action, softmax_value in zip(proposed_actions, softmax_values):
                    logger.info(f"Action: {action}, Probability: {softmax_value}")
                    robot_state.memory.proposed_actions.append([action, softmax_value])
                        
        except Exception as e:
            logger.error(f"{RED}Error validating actor output: {traceback.format_exc()}{RESET}")
        
        # —— Only call reflection_run when force_prefix_think is True —— 
        if force_prefix_think:
            try:
                logger.info(f"[Reflection] Starting reflection action generation...")
                reflection_text = self.actor.reflection_run(
                    history,
                    force_prefix_think=True,
                    reflection_prefix="Thought: Wait..." if self.action_tag == "Action:" else reflection_prefix
                )
                if reflection_text:
                    result = self.parser(reflection_text)
                    actor_output = ActorOutput.model_validate(result)
                    if actor_output.action:
                        logger.warning(f"Reflection action: {actor_output.action}")
                        # Give a default weight of 1.0 for now
                        robot_state.memory.proposed_actions.append([actor_output.action, 1.0])
            except Exception:
                logger.error(f"{RED}Error during reflection_run: {traceback.format_exc()}{RESET}")

        # Sort the generated actions (including reflection actions)
        logger.info(f"{GREEN}Obtained a total of {len(robot_state.memory.proposed_actions)} candidate actions{RESET}")
        ranked_actions = self._rank_actions(robot_state, robot_state.memory.proposed_actions)
        return ranked_actions
    
    def reward(
        self, robot_state: RobotState, robot_action: RobotAction, **kwargs
    ) -> Tuple[float, dict]:
        if not self.critic:
            logger.error(f"{RED}Critic is not initialized, cannot evaluate action{RESET}")
            return -0.01, {}
        state = run_critic(self.critic, robot_state)
        terminal =  False if state == State.CONTINUE else True
        logger.debug(f"{CYAN}is_terminal: {terminal}{RESET}")
        if terminal:
            logger.debug(f"{GREEN}Terminal state reached, reward: 1.0{RESET}")
            return 1.0, {}
        else:
            logger.debug(f"{RED}Non-terminal state, reward: -0.01{RESET}")
            return -0.01, {}

    def fast_reward(
        self, robot_state: RobotState, robot_action: RobotAction, **kwargs
    ) -> tuple[float, dict]:
        return robot_action.rank, {}

    def _rank_actions(
        self, robot_state: RobotState, proposed_actions: List[Any]
    ) -> List[RobotAction]:
        # Update current stage
        robot_state.memory.current_stage = Stage.SORTOR

        ranked_actions = []
        remaining_actions = proposed_actions.copy()
        logger.info(f"{GREEN} Sorting actions via sortor now...")

        # First, remove duplicate actions
        unique_actions = []
        for action_item in remaining_actions: # Renamed 'action' to 'action_item' to avoid conflict with outer scope
            # Check if the action already exists in unique_actions based on the complete_response field
            is_duplicate = False
            for ua in unique_actions: # Renamed 'a' to 'ua' for clarity
                if action_item[0][0].complete_response.strip() == ua[0][0].complete_response.strip():
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_actions.append(action_item)
        
        if len(unique_actions) < len(remaining_actions):
            logger.info(f"{YELLOW} Detected duplicate actions, duplicates have been removed. Original {len(remaining_actions)} actions, {len(unique_actions)} actions after deduplication{RESET}")
            remaining_actions = unique_actions

        if len(remaining_actions) > 1:
            robot_state.expand_times += 1

        if not self.sortor:
            # If there is no sortor, sort by the format_reward field
            logger.info(f"{YELLOW}No sortor provided, sorting by format_reward (softmax probability){RESET}")
            # Get the softmax probability of each action, default to 0 if not present
            action_probabilities = [(action[0], action[1]) for action in remaining_actions]  # action[0] is the action object, action[1] is the softmax probability
            # Sort by softmax probability in descending order
            sorted_actions = sorted(action_probabilities, key=lambda x: x[1], reverse=True)  # reverse=True ensures descending order
            # Return the sorted list of actions, with rank set based on format_reward (inverse of index)
            return [RobotAction(action=act, rank=1.0 / (i + 1), prob=prob) for i, (act, prob) in enumerate(sorted_actions)] # Renamed 'action' to 'act'

        # TODO: Use sortor to rank actions
        return ranked_actions


def convert_messages_to_state(messages):
    msg = {
        "role": "user",
        "content": []
    }
    # Process conversation history
    is_multi_modal = False
    for i, message in enumerate(messages):
        if message["role"] in ("user", "assistant", "tool"):
            end_fix = "\nThe Current State is: " if i == 1 else "\n"
            role_prefix = "" # Renamed 'role' to 'role_prefix'
            if message["role"] == "user":
                role_prefix = "Task: " if i == 1 else "\n"
            elif message["role"] == "assistant":
                role_prefix = "Actor: "
            elif message["role"] == "tool":
                role_prefix = "\n"
            
            if isinstance(message["content"], dict):
                # Process messages with images
                msg["content"].append({
                    "type": "text", "text": role_prefix + message["content"].get("text", "")
                })
                if "image" in message["content"]:
                    is_multi_modal = True
                    image_path = message["content"]["image"]
                    if isinstance(image_path, list):
                        image_path = image_path[0]
                    msg["content"].append({"type": "image_url", "image_url": image_path})
                msg["content"].append({
                    "type": "text", "text": end_fix
                })
            else:
                # Process text-only messages
                msg["content"].append({
                    "type": "text",
                    "text": role_prefix + str(message["content"]) + end_fix
                })
    msg["content"].append({
            "type": "text",
            "text": "\nNow, it's your turn to evaluate the state."
    })
    if not is_multi_modal:
        # Merge text content into a single string
        msg["content"] = "\n".join([item["text"] for item in msg["content"]])
    return msg


def _process_critic_message_content(message_content):
    """Helper function to process message content into a string representation."""
    if isinstance(message_content, list):
        text_parts = []
        for item in message_content:
            if isinstance(item, dict):
                if item.get("type") == "image_url":
                    text_parts.append("<image>")
                elif item.get("type") == "text" and "text" in item:
                    text_parts.append(item["text"])
        return " ".join(text_parts)
    elif isinstance(message_content, dict):
        if "image" in message_content:
            text = message_content.get("text", "")
            return f"{text} <image>".strip() if text else "<image>"
        elif "text" in message_content:
            return message_content["text"]
    return str(message_content) # Fallback


def run_critic(critic: ChatAgent, robot_state: RobotState, enable_ttrl_reward: bool = False, ttrl_vote_num: int = 10):
    robot_state.memory.current_stage = Stage.CRITIC
    msg_for_critic = convert_messages_to_state(robot_state.memory.history)
    
    # Initialize critic_state if it's None
    if not hasattr(robot_state.memory, 'critic_state') or robot_state.memory.critic_state is None:
        robot_state.memory.critic_state = {}

    # Collect all image paths and store them
    all_images = []
    for message in robot_state.memory.history:
        if isinstance(message.get("content", {}), dict) and "image" in message["content"]:
            image_paths = message["content"]["image"]
            if isinstance(image_paths, str):
                all_images.append(image_paths)
            elif isinstance(image_paths, list):
                all_images.extend(image_paths)
    robot_state.memory.critic_state["images"] = all_images

    if enable_ttrl_reward:
        responses = critic.batch_run([msg_for_critic], n=ttrl_vote_num)
        
        state_votes = {State.SUCCESS: 0, State.FAILURE: 0, State.CONTINUE: 0}
        
        if responses:
            for res in responses:
                content = res.get('response', '')
                if content:
                    parsed_result = non_stream_parser(content)
                    answer_text = parsed_result.get("answer", "").lower()
                    if answer_text == "success":
                        state_votes[State.SUCCESS] += 1
                    elif answer_text == "failure":
                        state_votes[State.FAILURE] += 1
                    elif answer_text == "continue":
                        state_votes[State.CONTINUE] += 1
            
            final_state = State.FAILURE # Default
            if any(state_votes.values()): # If there were any votes
                max_vote_count = 0
                # Find the maximum number of votes
                if state_votes[State.SUCCESS] > 0: max_vote_count = max(max_vote_count, state_votes[State.SUCCESS])
                if state_votes[State.CONTINUE] > 0: max_vote_count = max(max_vote_count, state_votes[State.CONTINUE])
                if state_votes[State.FAILURE] > 0: max_vote_count = max(max_vote_count, state_votes[State.FAILURE])

                if max_vote_count > 0:
                    # Apply priority: SUCCESS > CONTINUE > FAILURE for ties
                    if state_votes[State.SUCCESS] == max_vote_count:
                        final_state = State.SUCCESS
                    elif state_votes[State.CONTINUE] == max_vote_count:
                        final_state = State.CONTINUE
                    elif state_votes[State.FAILURE] == max_vote_count: # Only if others didn't match with max_vote_count
                        final_state = State.FAILURE
            
            # Store voting information in critic_state
            processed_prompt_content = _process_critic_message_content(msg_for_critic.get("content"))
            batch_messages_for_critic_state = [
                {"role": msg_for_critic.get("role", "user"), "content": processed_prompt_content},
                {"role": "assistant", "content": f"Voted Answer: {final_state.value}. Votes: {state_votes}."}
            ]
            robot_state.memory.critic_state["messages"] = batch_messages_for_critic_state
            robot_state.memory.critic_state["vote_counts"] = {k.value: v for k, v in state_votes.items()}
            robot_state.memory.critic_state["voted_answer"] = final_state.value
            logger.info(f"TTRL vote_counts: {robot_state.memory.critic_state['vote_counts']}{RESET}")
            logger.info(f"TTRL Reward: {final_state}{RESET}")
            return final_state
        else: # No responses from batch_run
            logger.info(f"TTRL Reward: No responses from batch_run")
            robot_state.memory.critic_state["messages"] = [msg_for_critic]
            robot_state.memory.critic_state["vote_counts"] = {k.value: 0 for k in State} # Assuming State is an enum
            robot_state.memory.critic_state["voted_answer"] = State.FAILURE.value
            return State.FAILURE
    else: # Original single run logic
        max_tries = 3
        count = 0
        while count < max_tries:
            count += 1
            # critic.run expects a list of message dicts. msg_for_critic is one such dict.
            content = critic.run([msg_for_critic]) 
            
            complete_messages_from_critic = critic.complete_messages
            
            processed_critic_messages = []
            if complete_messages_from_critic:
                for m_orig in complete_messages_from_critic:
                    new_m = copy.deepcopy(m_orig)
                    # Process the content of the message
                    new_m["content"] = _process_critic_message_content(new_m.get("content"))
                    processed_critic_messages.append(new_m)
            
            robot_state.memory.critic_state["messages"] = processed_critic_messages
            
            result = non_stream_parser(content)
            if result.get("answer", ""):
                answer_text = result.get("answer", "").lower()
                if answer_text == "success":
                    return State.SUCCESS
                elif answer_text == "failure":
                    return State.FAILURE
                elif answer_text == "continue":
                    return State.CONTINUE
            
        return State.FAILURE


class RobotMCTSWrapper(Reasoner[RobotState, RobotAction]):
    def __init__(
        self,
        actor: ChatAgent,
        tools_call: dict,
        sortor: Optional[ChatAgent] = None,
        critic: Optional[ChatAgent] = None,
        trust_critic: bool = False,
        n_iterations: int = 1,
        depth_limit: int = 3,
        num_proposed_action: int = 3,
        exploration_weight: float = 1.0,
        alfworld_env: object = None,
        enable_reflection: bool = False,
        parser: Callable = None,
        action_tag: str = "<action>",
        observation_format: str = "<observation>{}</observation>",
        enable_ttrl_reward: bool = False,
        ttrl_vote_num: int = 10,
    ):
        self.actor = actor
        self.sortor = sortor
        self.critic = critic
        self.trust_critic = trust_critic
        self.enable_reflection = enable_reflection
        self.enable_ttrl_reward = enable_ttrl_reward
        self.ttrl_vote_num = ttrl_vote_num
        world_model = RobotWorldModel(tools_call=tools_call, critic=critic, actor=actor, 
                alfworld_env=alfworld_env, depth_limit=depth_limit, trust_critic=trust_critic,
                enable_ttrl_reward=enable_ttrl_reward, ttrl_vote_num=ttrl_vote_num,
                action_tag=action_tag, observation_format=observation_format)
        search_config = RobotMCTSSearchConfig(actor, sortor, critic, 
                num_proposed_action=num_proposed_action, parser=parser, action_tag=action_tag)
        search_algo = MCTS(
            n_iters=n_iterations,
            w_exp=exploration_weight,
            cum_reward=sum,
            calc_q=np.mean,
            simulate_strategy="max",
            output_strategy="max_reward",
            depth_limit=depth_limit,
            enable_reflection=enable_reflection,
        )
        self.alfworld_env = alfworld_env
        self.write_lock = threading.Lock()  # 写入锁
        super().__init__(world_model, search_config, search_algo)

    def __call__(self, 
                 instruction: str,
                 image_path: Optional[str] = None) -> MCTSResult:
        logger.debug(f"{YELLOW}Starting MCTS search{RESET}")
        mcts_result = super().__call__(instruction, image_path=image_path) # Renamed 'result' to 'mcts_result'
        self.print_result(mcts_result) # Use renamed variable
        return mcts_result # Use renamed variable

    def print_result(self, result: MCTSResult):
        if result.trace is None or len(result.trace) == 0:
            logger.debug(f"{RED}No valid path found{RESET}")
            return

        states, actions = result.trace
        logger.debug(f"{GREEN}Path found:{RESET}")
        for i, (state, action) in enumerate(zip(states, actions)):
            logger.debug(f"{CYAN}Step {i}{RESET}")
            logger.debug(f"{CYAN}Action Name: {action.action[0].name}{RESET}")
            logger.debug(f"{CYAN}Action Arguments: {action.action[0].arguments}{RESET}")
            logger.debug(f"{CYAN}Action Rank: {action.rank}{RESET}")
            
            # Print detailed Memory information
            logger.debug(f"{MAGENTA}Task State: {state.memory.state}{RESET}")
            
            if state.memory.think:
                logger.debug(f"{MAGENTA}Think: {state.memory.think}{RESET}")
            
            if state.memory.consideration:
                logger.debug(f"{MAGENTA}Consideration: {state.memory.consideration}{RESET}")
            
            if state.memory.best_action_index is not None:
                logger.debug(f"{MAGENTA}Best Action Index: {state.memory.best_action_index}{RESET}")
            
            if state.memory.reflection:
                logger.debug(f"{MAGENTA}Reflection: {state.memory.reflection}{RESET}")
            
            if state.memory.observation:
                logger.debug(f"{CYAN}Observation:{RESET}")
                if state.memory.observation.images:
                    logger.debug(f"{CYAN}Images: {state.memory.observation.images}{RESET}")
                if state.memory.observation.video:
                    logger.debug(f"{CYAN}Video: {state.memory.observation.video}{RESET}")
                if state.memory.observation.description:
                    logger.debug(f"{CYAN}Description: {state.memory.observation.description}{RESET}")
            
            # Print history length
            logger.debug(f"{MAGENTA}History Length: {len(state.memory.history)}{RESET}")

        logger.debug(f"{GREEN}Final Status: {states[-1].status}{RESET}")
        logger.debug(f"{GREEN}Cumulative reward: {result.cum_reward}{RESET}")
        logger.debug(f"{GREEN}Total steps: {len(actions)}{RESET}")
        
    def _extract_all_paths(self, root_node):
        """
        Extract all paths from the root node to leaf nodes
        
        Args:
            root_node: Root node of the MCTS tree
            
        Returns:
            list: List of all paths from the root node to leaf nodes
        """
        if not root_node:
            return []
            
        all_paths = []
        
        def dfs_extract_paths(node, current_path=None):
            if current_path is None:
                current_path = []
            
            path = current_path + [node]
            
            # Leaf node condition check
            if node.is_terminal or not node.children:
                all_paths.append(path)
                return
            
            # Recursively process child nodes
            for child in node.children:
                if child.state is not None:  # Only consider visited nodes
                    dfs_extract_paths(child, path)
        
        dfs_extract_paths(root_node)
        return all_paths
    
    def _serialize_memory(self, memory):
        """
        Serialize Memory object
        
        Args:
            memory: Memory object
            
        Returns:
            dict: Serialized Memory data
        """
        if not memory:
            return None
            
        memory_data = {
            "current_stage": str(memory.current_stage),
            "state": str(memory.state),
            "think": memory.think,
            "consideration": memory.consideration,
            "best_action_index": memory.best_action_index,
            "reflection": memory.reflection,
        }
        
        # Extract observation information
        if memory.observation:
            memory_data["observation"] = {
                "images": memory.observation.images,
                "video": memory.observation.video,
                "description": memory.observation.description
            }
        
        # Extract proposed actions
        if memory.proposed_actions:
            memory_data["proposed_actions"] = []
            for prop_action in memory.proposed_actions: # Renamed 'action' to 'prop_action'
                # Check if prop_action is a list type
                if isinstance(prop_action, list):
                    # If it is a list, process the first element in the list
                    if len(prop_action) > 0:
                        action_item = prop_action[0]
                        # Check if action_item has a name attribute
                        if hasattr(action_item, 'name'):
                            memory_data["proposed_actions"].append({
                                "name": action_item.name,
                                "arguments": getattr(action_item, 'arguments', {}),
                                "complete_response": getattr(action_item, 'complete_response', None)
                            })
                        else:
                            # If there is no name attribute, try to convert the entire object to a string
                            memory_data["proposed_actions"].append({
                                "name": "unknown",
                                "arguments": {},
                                "complete_response": str(action_item)
                            })
                else:
                    # If it is not a list, directly check if it has a name attribute
                    if hasattr(prop_action, 'name'):
                        memory_data["proposed_actions"].append({
                            "name": prop_action.name,
                            "arguments": getattr(prop_action, 'arguments', {}),
                            "complete_response": getattr(prop_action, 'complete_response', None)
                        })
                    else:
                        # If there is no name attribute, try to convert the entire object to a string
                        memory_data["proposed_actions"].append({
                            "name": "unknown",
                            "arguments": {},
                            "complete_response": str(prop_action)
                        })
        
        # Extract history length
        memory_data["history_length"] = len(memory.history) if hasattr(memory, 'history') else 0
        memory_data["history"] = memory.history
        return memory_data
    
    def _serialize_action(self, action: RobotAction):
        """
        Serialize Action object
        
        Args:
            action: Action object
            
        Returns:
            dict: Serialized Action data
        """
        if not action:
            return None
        
        try:
            # Check if action.action is a list
            if not isinstance(action.action, list) or len(action.action) == 0:
                return {
                    "name": "unknown",
                    "arguments": {},
                    "complete_response": None,
                    "rank": float(action.rank) if hasattr(action, 'rank') else 0.0
                }
            
            # Check if action.action[0] is a list
            if isinstance(action.action[0], list):
                # If it is a list, try to extract useful information
                return {
                    "name": "list_action",
                    "arguments": {"action_list": str(action.action[0])},
                    "complete_response": str(action.action[0]),
                    "rank": float(action.rank) if hasattr(action, 'rank') else 0.0
                }
            
            # Normal case: action.action[0] is an object
            return {
                "name": getattr(action.action[0], 'name', 'unknown'),
                "arguments": getattr(action.action[0], 'arguments', {}),
                "complete_response": getattr(action.action[0], 'complete_response', None),
                "rank": float(action.rank) if hasattr(action, 'rank') else 0.0
            }
        except Exception as e:
            logger.error(f"{RED}Error serializing action: {traceback.format_exc()}{RESET}")
            return {
                "name": "error",
                "arguments": {"error": str(e)},
                "complete_response": str(action.action) if hasattr(action, 'action') else None,
                "rank": float(action.rank) if hasattr(action, 'rank') else 0.0
            }
    
    def _serialize_node(self, node, include_action=True):
        """
        Serialize a single node
        
        Args:
            node: MCTS node
            include_action: Whether to include action information
            
        Returns:
            dict: Serialized node data
        """
        try:
            node_data = {
                "node_id": node.id,
                "depth": node.depth,
                "is_terminal": node.is_terminal,
                "reward": float(node.reward) if hasattr(node, 'reward') else 0.0,
                "fast_reward": float(node.fast_reward) if hasattr(node, 'fast_reward') else 0.0,
                "visits": node.N if hasattr(node, 'N') else 0,
                "q_value": float(node.Q) if hasattr(node, 'Q') else 0.0,
            }
            
            # Add state information
            if node.state:
                if hasattr(node.state, "memory"):
                    node_data["memory"] = self._serialize_memory(node.state.memory)
                
                # Add other state attributes
                for attr in ["status", "task", "hands_status", "skill_status", "plan"]:
                    if hasattr(node.state, attr):
                        value = getattr(node.state, attr)
                        node_data[attr] = str(value) if attr == "status" else value # Ensure status is string
            
            # Add action information
            if include_action and node.action:
                node_data["action"] = self._serialize_action(node.action)
            
            return node_data
        except Exception as e:
            logger.error(f"{RED}Error serializing node: {traceback.format_exc()}{RESET}")
            return {"error": str(e), "node_id": getattr(node, 'id', 'unknown')}
    
    def _get_all_tree_nodes(self, root_node):
        """
        Get all nodes in the entire tree (using BFS traversal)
        
        Args:
            root_node: Root node of the tree
            
        Returns:
            list: List of all nodes
        """        
        all_nodes = []
        if not root_node:
            return all_nodes
            
        queue = deque([root_node])
        visited = set()
        
        while queue:
            node = queue.popleft()
            if node.id in visited:
                continue
                
            visited.add(node.id)
            all_nodes.append(node)
            
            # Add all child nodes to the queue
            for child in node.children:
                if child.id not in visited:
                    queue.append(child)
                    
        return all_nodes
    
    def save_result_to_jsonl(self, result: MCTSResult, save_path: str) -> None:
        """
        Save MCTS search results to a jsonl file, including all nodes in the entire search tree
        
        Args:
            result: MCTS search result
            save_path: Path to save the file
        """ 
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Extract all paths from the root node to leaf nodes
            all_paths = self._extract_all_paths(result.tree_state)
            
            # Extract all nodes in the entire tree
            all_nodes = self._get_all_tree_nodes(result.tree_state)
            
            # Serialize all nodes
            serialized_nodes = []
            for node_item in all_nodes: # Renamed 'node' to 'node_item'
                node_data = self._serialize_node(node_item, include_action=True) # Use renamed variable
                
                # Add parent and child node ID information
                node_data["parent_id"] = node_item.parent.id if node_item.parent else None # Use renamed variable
                node_data["children_ids"] = [child.id for child in node_item.children] # Use renamed variable
                
                serialized_nodes.append(node_data)
            
            # Serialize paths
            serialized_paths = []
            for path_item in all_paths: # Renamed 'path' to 'path_item'
                path_data = []
                for i, node_in_path in enumerate(path_item): # Renamed 'node' to 'node_in_path'
                    # For the root node, do not include action
                    include_action = i > 0
                    node_data_for_path = self._serialize_node(node_in_path, include_action) # Use renamed variable, renamed 'node_data' to 'node_data_for_path'
                    path_data.append(node_data_for_path) # Use renamed variable
                
                # Calculate the cumulative reward of the path
                path_reward = sum(getattr(n, 'reward', 0) for n in path_item) # Renamed 'node' to 'n'
                
                serialized_paths.append({
                    "path_length": len(path_item), # Use renamed variable
                    "path_reward": float(path_reward),
                    "is_terminal_path": path_item[-1].is_terminal if path_item else False, # Use renamed variable
                    "nodes": path_data
                })
            
            # Add metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "total_paths": len(all_paths),
                "total_nodes": len(all_nodes),
                "best_path_reward": float(result.cum_reward) if result.cum_reward is not None else None,
                "max_node_id": result.tree_state.id + 1 if result.tree_state else 0, # Assuming result.tree_state.id is the max ID
            }
            
            # Serialize main result trace
            main_trace_serialized = None # Renamed 'main_trace' to 'main_trace_serialized'
            if result.trace_of_nodes:
                main_trace_serialized = [self._serialize_node(node, True) for node in result.trace_of_nodes] # Use renamed variable
            
            # Build final data
            final_data = {
                "metadata": metadata,
                "main_trace": main_trace_serialized, # Use renamed variable
                "all_nodes": serialized_nodes,
                "all_paths": serialized_paths
            }
            
            # Save as jsonl file
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"{GREEN}MCTS result saved to: {save_path}{RESET}")
            logger.info(f"{GREEN}Saved {len(all_nodes)} nodes and {len(all_paths)} paths from root to leaf nodes{RESET}")
        except Exception as e:
            logger.error(f"{RED}Error saving MCTS result: {str(e)}{RESET}")
    
    def _get_valid_conversation_nodes(self, root_node):
        """
        Get all valid conversation nodes (nodes with state and history)
        
        Args:
            root_node: Root node of the tree
            
        Returns:
            list: List of valid conversation nodes
        """
        all_nodes = self._get_all_tree_nodes(root_node)
        valid_nodes = []
        
        for node_item in all_nodes: # Renamed 'node' to 'node_item'
            if node_item.state and hasattr(node_item.state, "memory") and node_item.state.memory.history: # Use renamed variable
                valid_nodes.append(node_item) # Use renamed variable
                
        return valid_nodes
    
    def _calculate_q_statistics(self, nodes):
        """
        Calculate Q value statistics for nodes
        
        Args:
            nodes: List of nodes
            
        Returns:
            tuple: (mean, standard deviation)
        """
        if nodes is None or len(nodes) == 0:
            return 0, 0
            
        q_values = []
        format_rewards = []
        for node_item in nodes: # Renamed 'node' to 'node_item'
            q_value = 0
            format_reward_val = 0 # Renamed 'format_reward' to 'format_reward_val'
            if hasattr(node_item, 'Q') and node_item.Q is not None: # Use renamed variable
                q_value += float(node_item.Q) # Use renamed variable
            if hasattr(node_item, 'format_reward') and node_item.format_reward is not None: # Use renamed variable
                q_value += float(node_item.format_reward) # Use renamed variable
                format_reward_val += float(node_item.format_reward) # Use renamed variable, renamed variable
            q_values.append(q_value)
            format_rewards.append(format_reward_val) # Use renamed variable
            
        if q_values:
            if nodes[0].advantage_calc_method == "parent":
                q_mean = (nodes[0].parent.Q - nodes[0].parent.reward) / nodes[0].gamma
                q_mean += np.mean(format_rewards)
            else:
                q_mean = np.mean(q_values)
            q_std = np.sqrt(np.mean((np.array(q_values) - q_mean) ** 2))
            return q_mean, q_std
        else:
            return 0.0, 0.0

    def save_conversation_format(self, result: MCTSResult, save_dir: str) -> None:
        """
        Save MCTS trace as conversation format jsonl files, saving the conversation history of all nodes in the entire tree
        
        Args:
            result: MCTS search result
            save_dir: Save directory
        """
        try:
            # Check if there is a valid tree state
            if not result.tree_state:
                logger.warning(f"{YELLOW}No valid tree state, cannot save conversation format{RESET}")
                return
            
            # Get all valid conversation nodes
            valid_nodes = self._get_valid_conversation_nodes(result.tree_state)
            # Filter out the root node
            valid_nodes = [node for node in valid_nodes if node.parent is not None]
            
            logger.info(f"{GREEN}Found {len(valid_nodes)} valid nodes to save in conversation format{RESET}")
            
            if not valid_nodes:
                logger.warning(f"{YELLOW}No valid conversation nodes found{RESET}")
                return
            
            reward_model_data = []
            reward_model_file = os.path.join(save_dir, "reward_model_data.jsonl")
            for node_item in valid_nodes: # Renamed 'node' to 'node_item'
                node_id = node_item.id # Use renamed variable
                if (hasattr(node_item, 'state') and  # Use renamed variable
                    hasattr(node_item.state, 'memory') and 
                    hasattr(node_item.state.memory, 'critic_state') and 
                    node_item.state.memory.critic_state is not None):
                    
                    critic_state = node_item.state.memory.critic_state # Use renamed variable
                    # Check if both "messages" and "solution" keys are present
                    if "messages" in critic_state and "solution" in critic_state:
                        reward_model_data.append(critic_state)

            # Revised code for saving reward_model_data
            if reward_model_data:  # Only proceed if there's data to write
                try:
                    with open(reward_model_file, "w", encoding="utf-8") as f:
                        for data_dict in reward_model_data:
                            if not isinstance(data_dict, dict):
                                # Consider using a proper logger (e.g., self.logger.warning) if available.
                                print(f"WARNING: Item in reward_model_data is not a dict, skipping: {str(data_dict)[:200]}")
                                continue
                            try:
                                json.dump(data_dict, f, ensure_ascii=False)
                                f.write("\n")  # Newline for JSONL format
                            except TypeError as e:
                                # Proper logging should be used here.
                                print(f"ERROR: JSON serialization failed for an item in robot_mcts.py: {e}. Data (partial): {str(data_dict)[:200]}")
                                # Skipping problematic item to allow other valid data to be saved.
                        f.flush()  # Ensure data is passed to OS buffers after the loop.
                except IOError as e:
                    print(f"ERROR: IOError while writing reward data to {reward_model_file} in robot_mcts.py: {e}")
                except Exception as e:
                    print(f"ERROR: Unexpected error while writing reward data to {reward_model_file} in robot_mcts.py: {e}")

            # Group all nodes by parent ID
            nodes_by_parent = {}
            for node_item in valid_nodes: # Renamed 'node' to 'node_item'
                parent_id = node_item.parent.id # Use renamed variable
                if parent_id not in nodes_by_parent:
                    nodes_by_parent[parent_id] = []
                nodes_by_parent[parent_id].append(node_item) # Use renamed variable

            # Construct dpo_pairs.json file in conversation format
            dpo_file = os.path.join(os.path.dirname(save_dir), "dpo_pairs.json")
            dpo_delta_reward_file = os.path.join(os.path.dirname(save_dir), "dpo_pairs_delta_reward.json")
            
            # Collect all data to be written
            dpo_pairs = []
            dpo_delta_pairs = []
            
            # Pairwise combination of child nodes under each parent_id
            for parent_id, node_list in nodes_by_parent.items():
                if len(node_list) < 2:
                    continue
                # Sort by Q+format_reward in descending order
                sorted_nodes = sorted(
                    node_list,
                    key=lambda n: float(getattr(n, "Q", 0.0)) + float(getattr(n, "format_reward", 0.0)), # Renamed 'node' to 'n'
                    reverse=True
                )
                # Pairwise combination: high Q is chosen, low Q is rejected
                for i in range(len(sorted_nodes)):
                    for j in range(i + 1, len(sorted_nodes)):
                        high_node = sorted_nodes[i] # Renamed 'high' to 'high_node'
                        low_node  = sorted_nodes[j] # Renamed 'low' to 'low_node'

                        # Calculate delta_reward
                        high_q = float(getattr(high_node, "Q", 0.0)) + float(getattr(high_node, "format_reward", 0.0)) # Use renamed variable
                        low_q = float(getattr(low_node, "Q", 0.0)) + float(getattr(low_node, "format_reward", 0.0)) # Use renamed variable
                        delta_reward = high_q - low_q

                        # If delta_reward is 0, skip this pair
                        if abs(delta_reward) < 1e-6:
                            continue

                        # --- Extract chosen conversation ---
                        hist = getattr(high_node.state, "memory", None).history or [] # Use renamed variable
                        # Truncate to the last assistant message
                        last_idx = max((idx for idx, m in enumerate(hist) if m.get("role") == "assistant"), default=-1)
                        messages = hist[: last_idx + 1] if last_idx >= 0 else []

                        # --- Extract rejected last assistant response ---
                        low_hist = getattr(low_node.state, "memory", None).history or [] # Use renamed variable
                        rejected_response = ""
                        for msg_item in reversed(low_hist): # Renamed 'msg' to 'msg_item'
                            if msg_item.get("role") == "assistant": # Use renamed variable
                                rejected_response = msg_item.get("content", "") # Use renamed variable
                                break

                        mem = getattr(high_node.state, "memory", None) # Use renamed variable
                        images = getattr(mem, "images", []) if mem else []

                        # Collect original dpo data
                        pair = {
                            "messages": messages,
                            "rejected_response": rejected_response,
                            "images": images,
                        }
                        dpo_pairs.append(pair)

                        # Collect dpo data with delta_reward
                        pair_delta = {
                            "messages": messages,
                            "rejected_response": rejected_response,
                            "images": images,
                            "delta_reward": delta_reward
                        }
                        dpo_delta_pairs.append(pair_delta)
            
            # Thread-safe file writing
            with self.write_lock:
                # Write original dpo file
                with open(dpo_file, "a", encoding="utf-8") as f:
                    for pair_item in dpo_pairs: # Renamed 'pair' to 'pair_item'
                        f.write(json.dumps(pair_item, ensure_ascii=False) + "\n") # Use renamed variable
                
                # Write dpo file with delta_reward
                with open(dpo_delta_reward_file, "a", encoding="utf-8") as f_delta:
                    for pair_delta_item in dpo_delta_pairs: # Renamed 'pair_delta' to 'pair_delta_item'
                        f_delta.write(json.dumps(pair_delta_item, ensure_ascii=False) + "\n") # Use renamed variable
            
            # Print information about saving dpo files
            logger.debug(f"DPO conversations saved to: {dpo_file}")
            logger.debug(f"DPO conversations with delta_reward saved to: {dpo_delta_reward_file}")
            
            # Save conversation files
            conv_path_list = []
            # Calculate Q value statistics for child nodes of each parent node and save conversations
            for parent_id, node_list in nodes_by_parent.items():
                # Calculate Q value statistics
                q_mean, q_std = self._calculate_q_statistics(node_list)
                if q_mean != 0.0 or q_std != 0.0:
                    logger.info(f"{CYAN}Q value statistics for child nodes of parent node {parent_id}: Mean={q_mean:.4f}, StdDev={q_std:.4f}, Number of child nodes={len(node_list)}{RESET}")
                
                # Process each child node
                for node_item in node_list: # Renamed 'node' to 'node_item'
                    # Get Q value of the current node
                    q_value = float(node_item.Q) if hasattr(node_item, 'Q') and node_item.Q is not None else 0.0 # Use renamed variable
                    format_reward_val = float(node_item.format_reward) if hasattr(node_item, 'format_reward') and node_item.format_reward is not None else 0.0 # Use renamed variable, renamed 'format_reward'
                    # Calculate Advantage value
                    advantage = 0.0
                    if q_std > 0 or q_mean != 0: # Avoid division by zero if q_std is close to zero but q_mean is also zero
                        advantage = (q_value + format_reward_val - q_mean) / (q_std + 1e-4) # Use renamed variable
                    
                    # Check if it is valid data and count
                    if abs(advantage) > 1e-4:  # Non-zero advantage is considered valid data
                        current_count = StateManager().get("num_valid_data", 0, thread_local=False)
                        StateManager().set("num_valid_data", current_count + 1, thread_local=False)
                        logger.info(f"{GREEN}Number of valid data points: {current_count + 1}{RESET}")
                                        
                    # Get conversation history
                    history_list = [] # Renamed 'history' to 'history_list'
                    if hasattr(node_item.state, "memory") and node_item.state.memory.history: # Use renamed variable
                        history_list = node_item.state.memory.history.copy() # Use renamed variable
                    
                    # Find the last assistant message and truncate messages after it
                    last_assistant_idx = -1
                    for idx, msg_item in enumerate(history_list): # Use renamed variable, renamed 'msg' to 'msg_item'
                        if msg_item.get("role") == "assistant": # Use renamed variable
                            last_assistant_idx = idx
                    
                    if last_assistant_idx >= 0:
                        # Truncate messages after the last assistant message
                        history_list = history_list[:last_assistant_idx + 1] # Use renamed variable
                        
                        # Add reward field and advantage field in the last assistant message step
                        history_list[last_assistant_idx]["reward"] = q_value # Use renamed variable
                        history_list[last_assistant_idx]["advantage"] = float(advantage) # Use renamed variable
                    
                    # Get node depth and ID
                    node_depth = node_item.depth if hasattr(node_item, 'depth') else 0 # Use renamed variable
                    node_id_val = node_item.id if hasattr(node_item, 'id') else 0 # Use renamed variable, renamed 'node_id'
                    node_visited_N = node_item.N if hasattr(node_item, 'N') else 0 # Use renamed variable
                    node_reward_val = node_item.reward if hasattr(node_item, 'reward') else 0 # Use renamed variable, renamed 'node_reward'
                    
                    # Create save file path, including node ID and parent node ID information
                    conv_filename = f"parent{parent_id}_depth{node_depth}_node{node_id_val}_N{node_visited_N}_Q{q_value:.2f}_R{node_reward_val:.2f}_F{format_reward_val:.2f}_conversation.jsonl" # Use renamed variables
                    conv_path = os.path.join(save_dir, conv_filename)
                    
                    # Save conversation file
                    with open(conv_path, 'w', encoding='utf-8') as f:
                        json.dump(history_list, f, ensure_ascii=False, indent=2) # Use renamed variable
                    
                    conv_path_list.append(conv_path)
                    logger.info(f"{GREEN}Conversation format for node {node_id_val}(parent node {parent_id}) saved to: {conv_path}{RESET}") # Use renamed variable
                    logger.info(f"{GREEN}Q value: {q_value:.4f}, Advantage: {advantage:.4f}, Saved {len(history_list)} messages{RESET}") # Use renamed variable
            
            logger.info(f"{GREEN}Saved a total of {len(conv_path_list)} conversation files{RESET}")
        except Exception as e:
            logger.error(f"{RED}Error saving conversation format: {str(e)}{RESET}")
            logger.error(traceback.format_exc())
    
    def save_mcts_metadata(self, result: MCTSResult, save_path: str) -> None:
        """
        Save MCTS metadata
        
        Args:
            result: MCTS search result
            save_path: Save path
        """      
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Calculate maximum depth
            max_depth = 0
            if result.trace_of_nodes:
                max_depth = max((node.depth for node in result.trace_of_nodes), default=0)
            
            # Build metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "total_nodes": result.tree_state.id + 1 if result.tree_state else 0, # Assuming result.tree_state.id is max_node_id
                "max_depth": max_depth,
                "best_path_reward": float(result.cum_reward) if result.cum_reward is not None else 0.0,
                "exploration_rate": getattr(self.search_algo, 'w_exp', None),
                "search_config": {
                    "n_iterations": getattr(self.search_algo, 'n_iters', None),
                    "depth_limit": getattr(self.search_algo, 'depth_limit', None),
                    "simulate_strategy": getattr(self.search_algo, 'simulate_strategy', None),
                    "output_strategy": getattr(self.search_algo, 'output_strategy', None),
                }
            }
            
            # Save metadata - use jsonl format
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
                
            logger.info(f"{GREEN}MCTS metadata saved to: {save_path}{RESET}")
        except Exception as e:
            logger.error(f"{RED}Error saving MCTS metadata: {str(e)}{RESET}")
    
    def save_all_results(self, result: MCTSResult, save_dir: str) -> None:
        """
        Save results in all formats simultaneously
        
        Args:
            result: MCTS search result
            save_dir: Save directory
        """        
        # Original format
        original_path = os.path.join(save_dir, "mcts_result.jsonl")
        self.save_result_to_jsonl(result, original_path)
        
        # Conversation format
        self.save_conversation_format(result, save_dir)
        
        # Save GC success rate data
        self.save_gc_success_data(result, save_dir)

        # Metadata format
        metadata_path = os.path.join(save_dir, "mcts_meta_data.jsonl")
        self.save_mcts_metadata(result, metadata_path)

    def save_gc_success_data(self, result: MCTSResult, save_dir: str) -> None:
        """
        Save GC success rate data to a file
        
        Args:
            result: MCTS search result
            save_dir: Save directory
        """
        max_gc_success_per_iter = result.max_gc_success_per_iter
        if not max_gc_success_per_iter:
            logger.warning("max_gc_success_per_iter data not found, cannot save")
            return
            
        # Save data to json file
        data_path = os.path.join(save_dir, "gc_success_data.json")
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(max_gc_success_per_iter, f, ensure_ascii=False, indent=2)
        
        logger.info(f"GC success rate data saved to: {data_path}")