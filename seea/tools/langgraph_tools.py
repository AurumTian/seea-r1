import os
import re
import traceback
import time
from typing import Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from seea.utils.common import get_video_frame
from seea.utils.base import StateManager
from seea.agents.agent.dreamer import Dreamer
from seea.utils.logger import get_logger
logger = get_logger(__name__)


class ManipulationInput(BaseModel):
    instruction: str = Field(description="""Json format English operation instruction like {"pick": <object>, "place": <target>}. \
<object> and <target> must be clear, explicitly detectable targets in English without using directional terms. \
The output should prioritize both pick and place. Example: {"pick": "red apple", "place": "user's hand"} {"pick": "banana"} place": "blue plate"}""")


class RoboticArmOperationWithActionModeInput(BaseModel):
    instruction: str = Field(description="""Generate only one simple operation instruction in English for robotic arms to complete a specific task.
A instruction should consist of:
1. Action: Use action verbs like wipe and open to describe each step.
2. Object Interaction: Clearly identify objects involved, such as bottle or door. Ensure interactions are logical and resemble typical human actions.
3. Action Mode: using the left arm or right arm, not supporting both arms.
Examples 1: Grip the bottle using the left arm.
Examples 2: Wipe the bottle with a cloth, moving from top to bottom with the left arm.
Examples 3: Place the bottle in the center of the table using the right arm.""")

class ALFWorldActionModeInput(BaseModel):
    instruction: str = Field(description='''Generate a single, clear English operation instruction for a specific task. The instruction must follow these guidelines:

1. Basic Structure:
   - Use a single action verb (e.g., "go", "take", "put", "open", "close", "heat", "cool", "clean", "slice", "examine", "look", "inventory")
   - Specify the target object and/or receptacle clearly with their IDs
   - Keep the instruction simple and focused on one action

2. Valid Action Templates:
   - Movement: "go to {receptacle}<id>" (e.g., "go to kitchen counter 1")
   - Pickup: "take {object}<id> from {receptacle}<id>" (e.g., "take apple 1 from table 2")
   - Placement: "put {object}<id> in/on {receptacle}<id>" (e.g., "put apple 1 in fridge 1")
   - Container: "open/close {receptacle}<id>" (e.g., "open microwave 2")
   - Heating: "heat {object}<id> with {receptacle}<id>" (e.g., "heat soup 1 with microwave 1") - This will automatically open microwave, place object, turn on, wait, turn off, and retrieve object
   - Cleaning: "clean {object}<id> with {receptacle}<id>" (e.g., "clean apple 1 with sink 1") - This will automatically place object in sink, turn on water, wait, turn off water, and retrieve object
   - Cooling: "cool {object}<id> with {receptacle}<id>" (e.g., "cool soup 1 with fridge 1") - This will automatically open fridge, place object, wait, and retrieve object
   - Slicing: "slice {object}<id> with knife 1" (e.g., "slice apple 1 with knife 1") - Requires holding a knife
   - Examination: "examine {object/receptacle}<id>" (e.g., "examine book 2") - Provides detailed information about an object or contents of a receptacle
   - Looking: "look around" - Provides information about the current surroundings
   - Inventory: "inventory" - Lists objects currently being carried

3. Important Rules:
   - ALWAYS start with "go to" when approaching a new object or location
   - Objects, receptacles and lamps must include their numeric IDs (e.g., apple 1, table 2, lamp 3)
   - Complex actions like heating, cleaning, and cooling are automatically broken down into their component steps
   - Avoid using relative directions or ambiguous terms
   - When carrying an object, you cannot perform other interactions since the agent can only hold one item at a time. You must first put down the currently held object before interacting with other objects or receptacles

Examples:
✓ "go to kitchen counter 1"
✓ "take apple 2 from table 1"
✓ "put apple 2 in fridge 1"
✓ "heat soup 1 with microwave 1"
✓ "clean apple 1 with sink 1"
✓ "examine fridge 1"
✓ "inventory"
✗ "take apple" (missing ID and source location)
✗ "put pan 1 in/on countertop 1 near sink 1" (ambiguous placement and unnecessary location reference)
✗ "heat soup" (missing IDs and appliance)''')
    

class RoboticArmOperationPlanInput(BaseModel):
    task: str = Field(description="""A specific task for robotic arms to perform in the same language as the user, such as toast, pour milk, grab apple.""")
    plan: str = Field(description="""Generate a complete plan in English for the task.
Each step in the plan should include the following:
1. Action: Actions including pick up, place x on/near y, press, open, close, push and pull.
2. Object Interaction: Clearly identify objects involved, such as bottle or door. Ensure interactions are logical and resemble typical human actions.
3. Action Mode: Using the left arm or right arm, not supporting both arms.
4. Location: Location or Destination of the object.
Conditional constraints:
1. The picking up action must be followed by the placing action, because other actions require that there is nothing in the arm.
2. All actions can only be used to operate on a single object.
Example plan for roasting corn with plate from plate rack using oven rack:
Step 1: Pick up the gray plate from plate rack and place it on the left of the table using the left arm.
Step 2: Pick up the corn from the basket and place it on the gary plate using the left arm.
Step 3: Push the gary plate to the right of the table using the left arm.
Step 4: Using the right arm to open the oven, and put the gray plate in the oven rack, then close the oven.
Step 5: Using the right arm to open the oven, then pick up the gray plate from the oven rack and place it on the right side of the table, and close the oven.""")


def perception_wrapper(func):
    """Environment perception"""
    @tool("perception", return_direct=False)
    def perception():
        """Obtain images to perceive and understand the environment."""
        result = func()
        return str(result)
    return perception


def robotic_arm_operation_plan_wrapper():
    """Operation planning tool for TGControlAgentV2"""
    @tool("embodied_operation", args_schema=RoboticArmOperationPlanInput, return_direct=False)
    def embodied_operation(task: str, plan:str):
        """Break down and plan tasks for robotic arm operations based on environmental perception information."""
        # !TODO: Connect to simulation environment or world model
        return "success"
    return embodied_operation


def robotic_arm_operation_wrapper(dreamer_agent: Optional[Dreamer] = None):
    @tool("embodied_operation", args_schema=RoboticArmOperationWithActionModeInput, return_direct=False)
    def embodied_operation(instruction: str):
        """Control the robotic arm to complete the operation task."""
        if dreamer_agent:
            state = StateManager()
            image_path = state.get("env_image_path")
            assert image_path, "No environment image path provided."
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file does not exist: {image_path}")
            result = dreamer_agent.infer(image_path=image_path, instruction=instruction)
            if result["status"] == "success" and "file_path" in result and result["file_path"]:
                last_frame_path = get_video_frame(result["file_path"], -1)
                if last_frame_path:
                    state.set("env_image_path", last_frame_path)
                    return {"text": "success", "image": last_frame_path}
        return "failure"
    return embodied_operation

def robotic_arm_operation_alfworld_wrapper(env, using_admissible_commands=True, enable_visual=True, visual_som=True):
    """Operation tool based on ALFWorld environment"""
    @tool("embodied_operation", args_schema=ALFWorldActionModeInput, return_direct=False)
    def embodied_operation(instruction: str):
        """Embodied operation and navigation tool for controlling robotic agents in the environment.
        This tool enables physical interaction and movement through the environment by executing
        commands for manipulation and navigation tasks."""
        try:
            # Get list of executable commands
            admissible_commands = getattr(env, 'last_info', {}).get('admissible_commands', [[]])[0]
            instruction = instruction.rstrip('.')
            
            action = _select_best_action(instruction, admissible_commands) if admissible_commands and using_admissible_commands else instruction.strip()
            
            put_action = re.findall(r"put\s+(.*)\s+[io]n\s+(.*)", action)
            if put_action:
                action = f"put {put_action[0][0]} in/on {put_action[0][1]}"
            logger.info(f"[Step] action: {action}")
            
            # Record current action to state manager
            state = StateManager()
            state.set("current_action", action)
            
            # Get current task ID
            current_task_id = state.get("current_task_id", None)
            if current_task_id is not None:
                # Update heartbeat for this task
                heartbeat_key = f"task_heartbeat_{current_task_id}"
                state.set(heartbeat_key, time.time(), thread_local=False)
                state.set(f"task_status_{current_task_id}", f"Preparing to execute action: {action}", thread_local=False)
            
            return _execute_action(action, env, enable_visual)
            
        except Exception as e:
            traceback.print_exc()
            # Record error information to state manager
            state = StateManager()
            state.set("action_error", str(e))
            state.set("action_traceback", traceback.format_exc())
            return f"Error: {str(e)}\n{traceback.format_exc()}"

    def _execute_action(action: str, env, enable_visual: bool):
        """Execute action and process results"""
        try:
            def process_ob(ob):
                if ob.startswith('You arrive at loc '):
                    ob = ob[ob.find('. ')+2:]
                return ob
            state = StateManager()
            
            # Get current steps and task ID from state manager
            current_steps = state.get("env_steps", 0)
            current_task_id = state.get("current_task_id", None)
            start_steps = current_steps
            state.set("start_steps", start_steps)
            
            # Update global heartbeat - Add global heartbeat counter
            global_heartbeat_key = "global_task_heartbeat"
            state.set(global_heartbeat_key, time.time(), thread_local=False)
            state.set("global_task_status", f"Executing action: {action}", thread_local=False)
            
            # If current task ID is known, also update task-specific heartbeat
            if current_task_id is not None:
                heartbeat_key = f"task_heartbeat_{current_task_id}"
                state.set(heartbeat_key, time.time(), thread_local=False)
                state.set(f"task_status_{current_task_id}", f"Executing action: {action}", thread_local=False)
            
            # Execute action
            obs, rewards, dones, infos = env.step([action])
            obs = [process_ob(ob) for ob in obs]
            
            # Update environment state
            env.last_info = infos
            
            # Update step count and save to state manager
            current_steps += 1
            state.set("env_steps", current_steps)
            
            # Update global heartbeat again - mark action as completed
            state.set(global_heartbeat_key, time.time(), thread_local=False)
            state.set("global_task_status", f"Action completed: {action}, observation: {obs[0][:50]}...", thread_local=False)
            
            # If current task ID is known, also update task-specific heartbeat
            if current_task_id is not None:
                heartbeat_key = f"task_heartbeat_{current_task_id}"
                state.set(heartbeat_key, time.time(), thread_local=False)
                state.set(f"task_status_{current_task_id}", f"Action completed: {action}, observation: {obs[0][:50]}...", thread_local=False)
            
            # Update environment state to state manager
            _update_environment_state(state, obs, dones, infos, start_steps, action)
            
            # Record evaluation-related information
            state.set("eval_action", action)
            state.set("eval_observation", obs[0] if obs else "")
            state.set("eval_done", dones[0] if dones else False)
            state.set("eval_gc_reward", float(infos.get('goal_condition_success_rate', [0])[0]))
            
            # Accumulate evaluation information
            _update_evaluation_history(state, action, obs, dones, infos)
            
            if dones[0]:
                logger.info("[Step] Task completed, environment state ended")
                state.set("eval_completed", True)
                
            if enable_visual:
                try:
                    # Get visual observation result
                    frame_path = env.get_visual_obs(visual_som=visual_som)
                    
                    # Validate image path
                    if frame_path and frame_path is not None and os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
                        state.set("env_image_path", frame_path)
                        return {"text": "", "image": frame_path}
                    else:
                        logger.info(f"[Step] Invalid image path or empty file: {frame_path}")
                        # Ensure evaluation history is updated even without image
                        return "Nothing happens."
                except Exception as img_err:
                    logger.error(f"[Step] Error getting visual observation: {str(img_err)}")
                    # Ensure text observation is returned even if image processing fails
                    return "Nothing happens."
            
            return f"{obs[0]}"
            
        except ValueError as ve:
            logger.error(f"[Step] Environment step failed with ValueError: {ve}")
            _handle_environment_error(env)
            state = StateManager()
            state.set("eval_error", str(ve))
            # Ensure evaluation history is initialized even on error
            if not state.get("eval_history"):
                state.set("eval_history", [{"action": action, "observation": f"Error: {str(ve)}", "done": False, "reward": 0.0, "goal_condition_success": 0.0}])
            return f"Error executing action '{action}': The environment returned an invalid response. The environment may be in an invalid state."
            
        except Exception as e:
            state = StateManager()
            start_steps = state.get("start_steps", 0)
            _log_error(state, e, action, start_steps)
            # Ensure evaluation history is initialized even on error
            if not state.get("eval_history"):
                state.set("eval_history", [{"action": action, "observation": f"Error: {str(e)}", "done": False, "reward": 0.0, "goal_condition_success": 0.0}])
            raise e

    def _update_environment_state(state: StateManager, obs, dones, infos, start_steps: int, action: str):
        """Update environment state"""
        state.set("alfworld_env_obs", obs[0])
        state.set("alfworld_env_dones", dones[0] if dones else False)
        
        if 'goal_condition_success_rate' in infos:
            reward = float(infos['goal_condition_success_rate'][0])
        else:
            reward = 1.0 if dones and dones[0] else 0.0
        logger.info(f"[Step] goal_condition_success_rate: {reward}")
        state.set("reward", reward)
        
        # Use step count from state manager
        current_steps = state.get("env_steps", 0)
        result_data = {
            "observation": obs[0] if obs else "",
            "success": dones[0] if dones else False,
            "steps": current_steps - start_steps,
            "reward": reward,
            "action": action
        }
        state.set("eval_data", result_data)

    def _update_evaluation_history(state: StateManager, action: str, obs, dones, infos):
        """Update evaluation history records"""
        # Get existing history or create new
        history = state.get("eval_history", [])
        
        # Create current step record
        step_record = {
            "action": action,
            "observation": obs[0] if obs else "",
            "done": dones[0] if dones else False,
            "reward": float(infos.get('won', [0])[0]),
            "goal_condition_success": float(infos.get('goal_condition_success_rate', [0])[0])
        }
        
        # Add to history
        history.append(step_record)
        state.set("eval_history", history)
        
        # Update cumulative statistics
        state.set("eval_total_steps", len(history))
        state.set("eval_current_reward", step_record["reward"])
        state.set("eval_current_gc_success", step_record["goal_condition_success"])
        
        # Add additional information needed for evaluation
        # Record all action sequences for printing
        action_history = [step.get("action", "") for step in history]
        state.set("action_history", action_history)
        
        # Record rewards for each step to calculate maximum reward
        rewards_history = [step.get("reward", 0.0) for step in history]
        state.set("eval_rewards_history", rewards_history)
        
        # Record goal condition completion rate for each step to calculate maximum completion rate
        gc_history = [step.get("goal_condition_success", 0.0) for step in history]
        state.set("eval_gc_history", gc_history)
        
        # Calculate maximum reward and maximum goal condition completion rate
        state.set("eval_max_reward", max(rewards_history) if rewards_history else 0.0)
        state.set("eval_max_gc_success", max(gc_history) if gc_history else 0.0)

        iter = state.get("mcts_iter", 0)
        iter2max_gc_success = state.get("max_gc_success_per_iter", {})
        iter2max_gc_success[iter] = max(gc_history)
        state.set("max_gc_success_per_iter", iter2max_gc_success)
        logger.info(f"iter: {iter}, max_gc_success_per_iter: {iter2max_gc_success[iter]}")

    def _handle_environment_error(env):
        """Handle environment error"""
        try:
            logger.info("[Step] Attempting to reset the environment...")
            env.reset()
        except Exception as reset_err:
            logger.error(f"[Step] Failed to reset environment: {reset_err}")

    def _log_error(state: StateManager, error: Exception, action: str, start_steps: int):
        """Log error information"""
        # Use step count from state manager
        current_steps = state.get("env_steps", 0)
        error_data = {
            "error": str(error),
            "traceback": traceback.format_exc(),
            "action": action,
            "steps": current_steps - start_steps
        }
        state.set("eval_error", error_data)

    def _select_best_action(instruction: str, commands: list) -> str:
        """Select the command that best matches the instruction"""
        instruction = instruction.lower()
        best_command = commands[0]
        max_overlap = 0
        
        for command in commands:
            command_words = set(command.lower().split())
            instruction_words = set(instruction.split())
            overlap = len(command_words.intersection(instruction_words))
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_command = command
                
        return best_command
    
    return embodied_operation  