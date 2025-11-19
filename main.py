import os
import time
import json
import shutil
import argparse
import traceback
import numpy as np
import concurrent.futures
from seea.utils.base import StateManager
from seea.agents.agent import ChatAgent
from seea.configs.robot_mcts import RobotMCTSWrapper
from seea.configs.visual_agent import FunctionCallAgent
from seea.configs.config import *
from seea.utils.common import clean_json_output
from seea.agents.models.models import AvailabilityCheck
from seea.envs.alfworld.alfworld import get_alfworld_environment
from seea.configs.config import LLM_MODELS, VLM_MODELS
from seea.utils.agent_factory import create_agent
from seea.utils.dataset_extractor import extract_samples, save_dataset_to_file
from seea.envs.alfworld.alfworld_icl import get_alfworld_icl_prompt
from seea.utils.logger import get_logger
from seea.utils.config_loader import save_yaml_config
from seea.utils.cleanup import perform_cleanup, cleanup_all_unity_processes
from seea.utils.config_utils import add_common_args, process_config_args, extract_mcts_params
from seea.utils.data_analyzer import aggregate_reward_data

# Use the custom get_logger function instead of logging.basicConfig
logger = get_logger(log_file=f"seea_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}.log")


def filter_dataset(dataset_path, output_dir):
    """Filter the dataset, remove data where advantage is 0 or 1"""
    logger.info(f"Start filtering dataset: {dataset_path}")
    
    # Read the original dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    original_count = len(dataset)
    logger.info(f"Original dataset contains {original_count} records")
    
    # Filter data
    filtered_dataset = []
    for item in dataset:
        if 'advantage' in item and (item['advantage'] == 0 or item['advantage'] == 1):
            continue
        if 'messages' in item:
            has_invalid_advantage = False
            for msg in item['messages']:
                if 'advantage' in msg and (msg['advantage'] == 0 or msg['advantage'] == 1):
                    has_invalid_advantage = True
                    break
            if has_invalid_advantage:
                continue
        filtered_dataset.append(item)
    
    filtered_count = len(filtered_dataset)
    logger.info(f"Filtered dataset contains {filtered_count} records, removed {original_count - filtered_count}")
    
    # Use the output directory to save the filtered dataset
    filtered_path = f"{output_dir}/dataset_filtered.json"
    with open(filtered_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_dataset, f, ensure_ascii=False, indent=4)
    
    logger.info(f"Filtered dataset saved to: {filtered_path}")
    return filtered_path


def load_dataset(dataset_root="/media/data/benchmark1_0_release"):
    if not dataset_root:
        image_video_paths = [["assets/data/kling/franka/pick_up_the_apple/first_frame.jpg", "assets/data/kling/franka/pick_up_the_apple/demo.mp4"],
                             ]
        tasks = ["Take the gray plate from the holder and place it on the left side of the table, put the apple in the blue plate, and put a corn on the gray plate"]
    else:
        image_video_paths, tasks = acquire_path(dataset_root=dataset_root)

    # Extract first frame image and video paths
    # Only keep images and videos from front, top, right cameras
    if 'h5_franka_1rgb' in image_video_paths[0][0]:
        image_video_paths = [path for path in image_video_paths if 'camera_top' in path[0]]
    elif 'h5_franka_3rgb' in image_video_paths[0][0]:
        image_video_paths = [path for path in image_video_paths if 'camera_right' in path[0]]
    elif 'h5_ur_1rgb' in image_video_paths[0][0]:
        image_video_paths = [path for path in image_video_paths if 'camera_top' in path[0]]
    elif 'h5_tienkung_gello_1rgb' in image_video_paths[0][0]:
        image_video_paths = [path for path in image_video_paths if 'camera_top' in path[0]]
    elif 'h5_tienkung_xsens_1rgb' in image_video_paths[0][0]:
        image_video_paths = [path for path in image_video_paths if 'camera_top' in path[0]]
    elif 'h5_agilex_3rgb' in image_video_paths[0][0]:
        image_video_paths = [path for path in image_video_paths if 'camera_front' in path[0]]
    elif 'h5_simulation' in image_video_paths[0][0]:
        image_video_paths = [path for path in image_video_paths if 'camera_front' in path[0]]
    
    logger.debug(f"[DEBUG] Paths: {image_video_paths}")

    image_paths = [path[0] for path in image_video_paths]
    video_paths = [path[1] for path in image_video_paths]
    return tasks, image_paths, video_paths

def run(config):
    # Extract parameters from config
    version = config.get("version", "mcts")
    model = config.get("model", {})
    dataset_root = config.get("dataset_root", "/media/data/benchmark1_0_release")
    enable_alfworld_env = config.get("enable_alfworld_env", True)
    num = config.get("num", 1)
    save_dir = config.get("save_dir", "./samples")
    wo_image_tool_result = config.get("wo_image_tool_result", False)
    num_threads = config.get("num_threads", 0)
    thread_agent_mode = config.get("thread_agent_mode", False)
    max_valid_data = config.get("max_valid_data", -1)
    evaluate = config.get("evaluate", False)
    StateManager().set("evaluate", evaluate, thread_local=False)
    # Initialize valid data counter
    StateManager().set("num_valid_data", 0, thread_local=False)
    StateManager().set("max_valid_data", max_valid_data, thread_local=False)

    # Set save path
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    save_dir = os.path.join(save_dir, version, model["model"], "alfworld" if enable_alfworld_env else "world_model", time_str)
    os.makedirs(save_dir, exist_ok=True)
    
    # Add environment instance cache in the main function scope
    thread_envs = {}
    
    def init_robot_agent(thread_id=None):
        """Initialize robot agent"""
        thread_prefix = f"[Thread-{thread_id}] " if thread_id is not None else ""
        logger.info(f"{thread_prefix}Initializing {version} agent, using model: {model['model']}")
        
        # Only create environment if ALFWorld is enabled and it's thread mode or main thread
        thread_alfworld_env = None
        if enable_alfworld_env and (thread_agent_mode or thread_id is not None or thread_id is None):
            try:
                # Create independent environment instance for each thread
                # env_type = "AlfredThorEnv" if not wo_image_tool_result else "AlfredTWEnv"
                env_type = "AlfredThorEnv"
                if config['split'] == "train":
                    alfworld_split = "train"
                elif config['split'] in ["dev", "eval_in_distribution"]:
                    alfworld_split = "eval_in_distribution"
                elif config['split'] in ["test", "eval_out_of_distribution"]:
                    alfworld_split = "eval_out_of_distribution"
                else:
                    raise ValueError(f"Invalid split: {config['split']}. Must be one of: train, dev, test")
                thread_alfworld_env = get_alfworld_environment(env_type=env_type, train_eval=alfworld_split)
                thread_alfworld_env.seed(np.random.randint(0, 10000))  # Use random seed to avoid parallel task duplication
                logger.info(f"{thread_prefix}Created independent ALFWorld environment: {id(thread_alfworld_env)}")
                thread_envs[thread_id] = thread_alfworld_env
            except Exception as e:
                logger.error(f"{thread_prefix}Failed to create ALFWorld environment: {str(e)}")
                return None

        # Initialize robot agent using factory function
        try:
            # Directly use the config object, add necessary parameters
            config_copy = config.copy()
            # Add thread-specific parameters
            config_copy.update({
                "alfworld_env": thread_alfworld_env,
                "visual_world": not enable_alfworld_env,
                "save_folder": StateManager().get("sample_save_dir", "") if thread_id is not None else save_dir,
            })
                
            robot_agent = create_agent(config_copy)
            
            if robot_agent:
                logger.info(f"{thread_prefix}Successfully initialized {version} agent")
                return robot_agent
            else:
                logger.error(f"{thread_prefix}Failed to initialize {version} agent")
                return None
        except Exception as e:
            logger.error(f"{thread_prefix}Failed to initialize robot agent: {str(e)}")
            traceback.print_exc()
            return None

    # Save run configuration
    config_path = os.path.join(save_dir, "config.json")
    # Create a copy of the configuration and add a timestamp
    save_config = config.copy()
    save_config["timestamp"] = time_str
    
    # Save as JSON format
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(save_config, f, ensure_ascii=False, indent=4)
    logger.info(f"Configuration saved to: {config_path}")
    
    # Also save as YAML format for future use
    yaml_config_path = os.path.join(save_dir, "config.yaml")
    save_yaml_config(save_config, yaml_config_path)
    logger.info(f"YAML configuration saved to: {yaml_config_path}")

    def save_result(robot_agent, result, save_path):
        """Save result to file"""
        try:
            if result is None:
                logger.warning(f"Result is empty, saving empty result to: {save_path}")
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump({"error": "Task execution failed, result is empty"}, f, ensure_ascii=False, indent=4)
                return
                
            if isinstance(robot_agent, FunctionCallAgent):
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(robot_agent.complete_messages, f, ensure_ascii=False, indent=4)
                logger.info(f"FunctionCallAgent result saved to: {save_path}")
            elif isinstance(robot_agent, RobotMCTSWrapper):
                save_dir = os.path.dirname(save_path)
                robot_agent.save_all_results(result, save_dir)
                logger.info(f"RobotMCTSWrapper result saved to: {save_path}")
            else:
                logger.warning(f"Unknown agent type: {type(robot_agent).__name__}, cannot save result")
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump({"error": f"Unknown agent type: {type(robot_agent).__name__}"}, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"Error saving result: {str(e)}")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump({"error": f"Error saving result: {str(e)}"}, f, ensure_ascii=False, indent=4)

    def copy_file_safe(src, dst):
        """Safely copy file, handle possible errors"""
        try:
            if not os.path.exists(src):
                logger.warning(f"Source file does not exist: {src}")
                return False
                
            # Use shutil.copy2 instead of os.system to preserve file metadata
            shutil.copy2(src, dst)
            return True
        except Exception as e:
            logger.error(f"Error copying file: {str(e)}")
            return False

    def process_task(i, task_data=None, thread_id=None):
        """Process a single task
        
        Args:
            i: Task index
            task_data: Task data
            thread_id: Thread ID, for logging identification
        """
        # Check if the maximum number of valid data has been reached before starting the task
        current_count = StateManager().get("num_valid_data", 0, thread_local=False)
        max_count = StateManager().get("max_valid_data", -1, thread_local=False)
        if max_count > 0 and current_count >= max_count:
            logger.info(f"[Thread-{thread_id}] Maximum number of valid data {max_count} reached, skipping task {i}")
            return None

        thread_prefix = f"[Thread-{thread_id}] " if thread_id is not None else ""
        sample_save_dir = os.path.join(save_dir, f"sample_{i}")
        os.makedirs(sample_save_dir, exist_ok=True)
        StateManager().set("sample_save_dir", sample_save_dir, thread_local=True)
        
        # Record task start time
        start_time = time.time()
        logger.info(f"{thread_prefix}Start processing task {i}")
        
        # Create heartbeat identifier
        heartbeat_key = f"task_heartbeat_{i}"
        expected_key = f"task_expected_duration_{i}"  # Used to store expected completion time
        StateManager().set(heartbeat_key, time.time(), thread_local=False)
        StateManager().set(f"task_status_{i}", "Initializing", thread_local=False)
        
        # Record current task ID to the global variable of the state manager, so other modules (like langgraph_tools) can get it
        StateManager().set("current_task_id", i, thread_local=False)
        logger.info(f"{thread_prefix}Set current task ID: {i}")
        
        # Set expected duration based on task type, to dynamically adjust heartbeat timeout
        # Initially set a shorter expected duration
        StateManager().set(expected_key, 60, thread_local=False)  # Expected 60 seconds in initialization phase
        
        # Create an independent robot_agent for each task
        robot_agent = init_robot_agent(thread_id)
        if not robot_agent:
            logger.error(f"{thread_prefix}Task {i} failed to initialize agent")
            return None

        result = None
        try:
            if enable_alfworld_env:
                # Get the environment instance for the current thread
                thread_alfworld_env = thread_envs.get(thread_id)
                if thread_alfworld_env is None:
                    logger.error(f"{thread_prefix}Cannot get ALFWorld environment instance")
                    return None

                # Reset environment before each task
                obs, info = thread_alfworld_env.reset()
                StateManager().set(f"task_status_{i}", "Environment reset", thread_local=False)
                StateManager().set(heartbeat_key, time.time(), thread_local=False)
                
                reward = float(info.get('goal_condition_success_rate', [0])[0]) if 'goal_condition_success_rate' in info else 0.0
                logger.info(f"[Init] goal_condition_success_rate: {reward}")
                StateManager().set("initial_reward", reward)
                setattr(thread_alfworld_env, 'last_info', info)
                logger.info(f"{thread_prefix}ALFWorld task: {obs}")
                if 'facts' in info:
                    del info['facts']
                logger.info(f"{thread_prefix}ALFWorld info: {info}")
                instruction = "\n".join(obs[0].split("\n\n")[1:])
                game_file = info["extra.gamefile"][0]
                name = "/".join(game_file.split("/")[-3:-1])
                logger.info(f"obs: {obs}\n{game_file}: game_file")
                instruction = get_alfworld_icl_prompt(instruction, name, config.get("format", "react"))
                
                # Update heartbeat
                StateManager().set(heartbeat_key, time.time(), thread_local=False)
                StateManager().set(f"task_status_{i}", "Prompt generated", thread_local=False)
                
                if not wo_image_tool_result:
                    frame_path = thread_alfworld_env.get_visual_obs(config.get("visual_som", False))
                    logger.info(f"{thread_prefix}ALFWorld first frame image path: {frame_path}")
                    StateManager().set(heartbeat_key, time.time(), thread_local=False)
                    StateManager().set(f"task_status_{i}", "Image acquired", thread_local=False)
                    
                    if frame_path is not None and frame_path != "" and os.path.exists(frame_path):
                        instruction += "\nThe current visual observation is shown below:"
                    
                    # Model inference phase may take a long time, increase expected duration
                    StateManager().set(expected_key, 600, thread_local=False)  # Expected 10 minutes for model inference
                    result = robot_agent(instruction, frame_path)
                    # Task completed, update heartbeat
                    StateManager().set(heartbeat_key, time.time(), thread_local=False)
                    StateManager().set(f"task_status_{i}", "Task completed", thread_local=False)
                else:
                    # Model inference phase may take a long time, increase expected duration
                    StateManager().set(expected_key, 600, thread_local=False)  # Expected 10 minutes for model inference
                    result = robot_agent(instruction)
                    # Task completed, update heartbeat
                    StateManager().set(heartbeat_key, time.time(), thread_local=False)
                    StateManager().set(f"task_status_{i}", "Task completed", thread_local=False)
            else:
                task, image_path, video_path = task_data
                image_dst_path = os.path.join(sample_save_dir, os.path.basename(image_path))
                video_dst_path = os.path.join(sample_save_dir, os.path.basename(video_path))
                
                # Safely copy files
                image_copied = copy_file_safe(image_path, image_dst_path)
                video_copied = copy_file_safe(video_path, video_dst_path)
                
                # Update heartbeat
                StateManager().set(heartbeat_key, time.time(), thread_local=False)
                StateManager().set(f"task_status_{i}", "Files copied", thread_local=False)
                
                if not image_copied:
                    logger.warning(f"{thread_prefix}Cannot copy image file: {image_path}")
                if not video_copied:
                    logger.warning(f"{thread_prefix}Cannot copy video file: {video_path}")
                
                logger.info(f"{thread_prefix}World model task: {task}")
                logger.info(f"{thread_prefix}First frame image path: {image_dst_path if image_copied else '(copy failed)'}")
                logger.info(f"{thread_prefix}Video path: {video_dst_path if video_copied else '(copy failed)'}")
                
                # VLM processing image takes some time
                StateManager().set(expected_key, 300, thread_local=False)  # Expected 5 minutes for VLM image processing
                
                # Update heartbeat
                StateManager().set(heartbeat_key, time.time(), thread_local=False)
                StateManager().set(f"task_status_{i}", "Instruction processed", thread_local=False)
                
                if not wo_image_tool_result:
                    # Model inference phase may take a long time
                    StateManager().set(expected_key, 600, thread_local=False)  # Expected 10 minutes for model inference
                    result = robot_agent(task, image_dst_path if image_copied else None)
                    # Task completed, update heartbeat
                    StateManager().set(heartbeat_key, time.time(), thread_local=False)
                    StateManager().set(f"task_status_{i}", "Task completed", thread_local=False)
                else :
                    result = None
                    # Task failed, update heartbeat
                    StateManager().set(heartbeat_key, time.time(), thread_local=False)
                    StateManager().set(f"task_status_{i}", "Task failed", thread_local=False)
        except Exception as e:
            logger.error(f"{thread_prefix}Error executing task: {traceback.format_exc()}")
            result = None
        finally:
            # Use new cleanup tool
            perform_cleanup(
                env=thread_alfworld_env if enable_alfworld_env else None, 
                agent=robot_agent, 
                thread_prefix=thread_prefix
            )
            # Clean up heartbeat data
            StateManager().remove(heartbeat_key, thread_local=False)
            StateManager().remove(f"task_status_{i}", thread_local=False)
            StateManager().remove(expected_key, thread_local=False)
            
            # Clean up task ID if current global task ID equals this task ID
            current_id = StateManager().get("current_task_id", None, thread_local=False)
            if current_id == i:
                StateManager().remove("current_task_id", thread_local=False)
                logger.info(f"{thread_prefix}Cleaned up task ID: {i}")

            # New log point
            logger.info(f"{thread_prefix}Task {i} is about to exit finally block")

        # Record task end time and duration
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"{thread_prefix}Task {i} processing completed, duration: {duration:.2f}s")
        
        # Save task metadata
        task_metadata = {
            "task_id": i,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "success": result is not None,
            "thread_id": thread_id
        }
        with open(os.path.join(sample_save_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(task_metadata, f, ensure_ascii=False, indent=4)

        # Save result
        save_path = os.path.join(sample_save_dir, "conversation.jsonl")
        save_result(robot_agent, result, save_path)
        
        # Clean thread-local state
        StateManager().clear_thread_local()
        logger.info(f"{thread_prefix}Task {i} process_task function is about to return") # New log
        
        return {
            "path": save_path,
            "success": result is not None,
            "duration": duration,
            "thread_id": thread_id
        }
    
    # Initialize robot agent
    robot_agent = init_robot_agent()
    
    if not robot_agent:
        logger.error("Failed to initialize robot agent, exiting program")
        return

    # Process tasks
    samples = []
    total_start_time = time.time()

    
    try:
        # Get number of threads parameter
        max_workers = min(num_threads, num) if num_threads > 0 else min(os.cpu_count() or 4, num)
        logger.info(f"Using {max_workers} threads to process tasks")
        # Determine if multithreading is used
        if thread_agent_mode and num_threads > 0:
            # Create thread pool
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                current_count = StateManager().get("num_valid_data", 0, thread_local=False)
                max_count = StateManager().get("max_valid_data", -1, thread_local=False)

                if enable_alfworld_env:
                    # Submit ALFWorld tasks
                    for i in range(num):
                        # Check if maximum number is reached before each submission
                        if max_count > 0 and current_count >= max_count:
                            logger.info(f"Maximum number of valid data {max_count} reached, stopping task submission")
                            break
                        
                        thread_id = i % max_workers
                        future = executor.submit(process_task, i, None, thread_id)
                        futures.append(future)
                else:
                    # Load and submit world model tasks
                    tasks, image_paths, video_paths = load_dataset(dataset_root=dataset_root)
                        
                    for i, task_data in enumerate(zip(tasks, image_paths, video_paths)):
                        # Check if maximum number is reached
                        if max_count > 0 and current_count >= max_count:
                            logger.info(f"Maximum number of valid data {max_count} reached, stopping task submission")
                            break

                        if i >= num:
                            break
                        thread_id = i % max_workers
                        future = executor.submit(process_task, i, task_data, thread_id)
                        futures.append(future)

                # Collect results
                valid_samples = []
                heartbeat_timeout = config.get("heartbeat_timeout", 300)  # Default heartbeat timeout is 5 minutes
                logger.info(f"Set heartbeat timeout to: {heartbeat_timeout}s")
                
                # Record all task future objects
                pending_futures = list(futures)
                task_ids = {}  # Record the task ID corresponding to each future
                
                for i, future in enumerate(futures):
                    task_ids[future] = i  # Save task ID, can be used for debugging
                
                # Loop until all tasks are completed or timed out
                while pending_futures:
                    # Use as_completed to process completed tasks, with timeout
                    try:
                        # Wait for the next completed task, up to 30 seconds (short polling interval for heartbeat check)
                        done_futures, pending_futures = concurrent.futures.wait(
                            pending_futures,
                            timeout=30,  # Short polling interval
                            return_when=concurrent.futures.FIRST_COMPLETED
                        )
                        
                        # Process completed tasks
                        for future in done_futures:
                            try:
                                result = future.result(timeout=0)  # Get result immediately
                                if result is not None:  # Only add valid results
                                    valid_samples.append(result)
                                    logger.info(f"Completed task: {result['path']}, success: {result['success']}, duration: {result['duration']:.2f}s")
                            except concurrent.futures.TimeoutError:
                                logger.error(f"Task timed out while getting result, skipped")
                            except Exception as e:
                                logger.error(f"Error getting task result: {traceback.format_exc()}")
                            
                            # Remove completed task from task ID record
                            if future in task_ids:
                                task_id = task_ids[future]
                                del task_ids[future]
                                logger.info(f"Task ID {task_id} completed and removed from tracking list")
                    
                    except concurrent.futures.TimeoutError:
                        # No task completed within 30 seconds, check heartbeat
                        pass
                    
                    # Check if tasks in pending_futures have timed out on heartbeat
                    current_time = time.time()
                    heartbeat_timed_out_futures = []
                    
                    for future in pending_futures:
                        if future in task_ids:
                            task_id = task_ids[future]
                            # Get task heartbeat time and status
                            heartbeat_key = f"task_heartbeat_{task_id}"
                            expected_key = f"task_expected_duration_{task_id}"  # Expected duration key
                            last_heartbeat = StateManager().get(heartbeat_key, None, thread_local=False)
                            task_status = StateManager().get(f"task_status_{task_id}", "Unknown", thread_local=False)
                            
                            # Get current task expected completion time, use default timeout if not exists
                            expected_duration = StateManager().get(expected_key, heartbeat_timeout, thread_local=False)
                            # Actual timeout used is the maximum of expected duration and base timeout
                            actual_timeout = max(expected_duration, heartbeat_timeout)
                            
                            # Also check global heartbeat - if global heartbeat exists and is newer than task heartbeat, use global heartbeat
                            global_heartbeat = StateManager().get("global_task_heartbeat", None, thread_local=False)
                            global_status = StateManager().get("global_task_status", None, thread_local=False)
                            
                            # If task heartbeat is empty but global heartbeat exists, use global heartbeat
                            if last_heartbeat is None and global_heartbeat is not None:
                                last_heartbeat = global_heartbeat
                                task_status = global_status or task_status
                                logger.info(f"Task ID {task_id} using global heartbeat: {global_status}")
                            # If both exist, use the newer one
                            elif last_heartbeat is not None and global_heartbeat is not None:
                                if global_heartbeat > last_heartbeat:
                                    last_heartbeat = global_heartbeat
                                    task_status = global_status or task_status
                                    logger.info(f"Task ID {task_id} using updated global heartbeat: {global_status}")
                            
                            if last_heartbeat is not None:
                                time_since_heartbeat = current_time - last_heartbeat
                                # If heartbeat timed out, consider the task stuck
                                if time_since_heartbeat > actual_timeout:
                                    heartbeat_timed_out_futures.append(future)
                                    logger.warning(f"Task ID {task_id} heartbeat timeout: last status '{task_status}', {time_since_heartbeat:.2f}s since last heartbeat, exceeding dynamic threshold {actual_timeout}s")
                                elif time_since_heartbeat > actual_timeout / 2:
                                    # Issue a warning but do not cancel the task
                                    logger.warning(f"Task ID {task_id} heartbeat approaching timeout: last status '{task_status}', {time_since_heartbeat:.2f}s since last heartbeat (dynamic threshold {actual_timeout}s)")
                    
                    # Cancel tasks with heartbeat timeout
                    if heartbeat_timed_out_futures:
                        for future in heartbeat_timed_out_futures:
                            future.cancel()
                            pending_futures.remove(future)
                            
                            # Get and clean task information
                            if future in task_ids:
                                task_id = task_ids[future]
                                del task_ids[future]
                                
                                # Get last heartbeat and status information
                                heartbeat_key = f"task_heartbeat_{task_id}"
                                expected_key = f"task_expected_duration_{task_id}"
                                last_heartbeat = StateManager().get(heartbeat_key, None, thread_local=False)
                                task_status = StateManager().get(f"task_status_{task_id}", "Unknown", thread_local=False)
                                expected_duration = StateManager().get(expected_key, heartbeat_timeout, thread_local=False)
                                actual_timeout = max(expected_duration, heartbeat_timeout)
                                
                                # Check global heartbeat
                                global_heartbeat = StateManager().get("global_task_heartbeat", None, thread_local=False)
                                global_status = StateManager().get("global_task_status", None, thread_local=False)
                                
                                # If global heartbeat is newer than task heartbeat, use global heartbeat
                                if global_heartbeat is not None and (last_heartbeat is None or global_heartbeat > last_heartbeat):
                                    last_heartbeat = global_heartbeat
                                    task_status = global_status or task_status
                                
                                # Clear heartbeat data from state manager (if still exists)
                                StateManager().remove(heartbeat_key, thread_local=False)
                                StateManager().remove(f"task_status_{task_id}", thread_local=False)
                                StateManager().remove(expected_key, thread_local=False)
                                
                                # Calculate heartbeat timeout duration
                                time_since_heartbeat = current_time - (last_heartbeat or current_time)
                                
                                logger.warning(f"Cancelled task ID {task_id} due to heartbeat timeout, last status: '{task_status}', timeout duration: {time_since_heartbeat:.2f}s")
                                
                                # Create error information file for heartbeat timeout task
                                try:
                                    timeout_save_dir = os.path.join(save_dir, f"sample_{task_id}")
                                    os.makedirs(timeout_save_dir, exist_ok=True)
                                    
                                    # Save task metadata
                                    timeout_metadata = {
                                        "task_id": task_id,
                                        "last_heartbeat_time": int(last_heartbeat) if last_heartbeat else None,
                                        "cancel_time": int(current_time),
                                        "heartbeat_timeout": heartbeat_timeout,
                                        "actual_timeout": actual_timeout,
                                        "expected_duration": expected_duration,
                                        "time_since_heartbeat": time_since_heartbeat,
                                        "last_status": task_status,
                                        "success": False,
                                        "error": f"Task ID {task_id} cancelled due to heartbeat timeout, base heartbeat timeout threshold {heartbeat_timeout}s, dynamic timeout threshold {actual_timeout}s, last status '{task_status}', timeout duration {time_since_heartbeat:.2f}s"
                                    }
                                    with open(os.path.join(timeout_save_dir, "metadata.json"), "w", encoding="utf-8") as f:
                                        json.dump(timeout_metadata, f, ensure_ascii=False, indent=4)
                                    
                                    # Save empty result
                                    error_save_path = os.path.join(timeout_save_dir, "conversation.jsonl")
                                    with open(error_save_path, "w", encoding="utf-8") as f:
                                        json.dump({
                                            "error": f"Task ID {task_id} cancelled due to heartbeat timeout, base heartbeat timeout threshold {heartbeat_timeout}s, dynamic timeout threshold {actual_timeout}s, last status '{task_status}', timeout duration {time_since_heartbeat:.2f}s"
                                        }, f, ensure_ascii=False, indent=4)
                                    
                                    logger.info(f"Created error information file for heartbeat timeout task ID {task_id}: {error_save_path}")
                                    
                                    # Try to clean up environment and agent related to the task
                                    thread_id = task_id % max_workers if max_workers > 0 else None
                                    if thread_id is not None and thread_id in thread_envs:
                                        try:
                                            # Get the environment instance corresponding to the thread
                                            env_to_cleanup = thread_envs.get(thread_id)
                                            if env_to_cleanup:
                                                # Use perform_cleanup function to clean environment and agent
                                                logger.info(f"Cleaning up environment and agent resources for heartbeat timeout task ID {task_id} (thread ID {thread_id})")
                                                perform_cleanup(
                                                    env=env_to_cleanup if enable_alfworld_env else None,
                                                    agent=None,  # Agent instance is no longer accessible
                                                    thread_prefix=f"[Thread-{thread_id}] "
                                                )
                                                # Remove from environment cache
                                                del thread_envs[thread_id]
                                        except Exception as e:
                                            logger.error(f"Error cleaning up heartbeat timeout task resources: {str(e)}")
                                except Exception as e:
                                    logger.error(f"Error creating error information file for heartbeat timeout task: {str(e)}")
                        
                        logger.warning(f"Cancelled {len(heartbeat_timed_out_futures)} tasks due to heartbeat timeout")
                
                # After all tasks are completed, ensure all resources are cleaned up
                for thread_id, env in list(thread_envs.items()):
                    try:
                        logger.info(f"Cleaning up environment resources for thread ID {thread_id} after task completion")
                        if env:
                            perform_cleanup(env=env, agent=None, thread_prefix=f"[Thread-{thread_id}] ")
                    except Exception as e:
                        logger.error(f"Error cleaning up environment resources for thread ID {thread_id}: {str(e)}")
                thread_envs.clear()
                
                samples = valid_samples
        else:
            # Non-multithreaded mode, process tasks directly
            logger.info("Using single-threaded mode to process tasks")
            valid_samples = []
            
            if enable_alfworld_env:
                # Process ALFWorld tasks
                for i in range(num):
                    # Check if maximum number of valid data is reached
                    current_count = StateManager().get("num_valid_data", 0, thread_local=False)
                    max_count = StateManager().get("max_valid_data", -1, thread_local=False)
                    if max_count > 0 and current_count >= max_count:
                        logger.info(f"Maximum number of valid data {max_count} reached, stopping task submission")
                        break
                    
                    # Do not pass thread_id in single-threaded mode
                    result = process_task(i, None, None)
                    if result is not None:
                        valid_samples.append(result)
                        logger.info(f"Completed task: {result['path']}, success: {result['success']}, duration: {result['duration']:.2f}s")
            else:
                # Process world model tasks
                tasks, image_paths, video_paths = load_dataset(dataset_root=dataset_root)
                
                for i, task_data in enumerate(zip(tasks, image_paths, video_paths)):
                    # Check if maximum number of valid data is reached
                    current_count = StateManager().get("num_valid_data", 0, thread_local=False)
                    max_count = StateManager().get("max_valid_data", -1, thread_local=False)
                    if max_count > 0 and current_count >= max_count:
                        logger.info(f"Maximum number of valid data {max_count} reached, stopping task submission")
                        break
                    
                    if i >= num:
                        break
                    
                    # Do not pass thread_id in single-threaded mode
                    result = process_task(i, task_data, None)
                    if result is not None:
                        valid_samples.append(result)
                        logger.info(f"Completed task:. {result['path']}, success: {result['success']}, duration: {result['duration']:.2f}s") # Corrected log message
            
            samples = valid_samples
                
    except Exception as e:
        logger.error(f"Error processing tasks: {traceback.format_exc()}")
    
    # Calculate total duration and success rate
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # Ensure samples is not empty and only contains valid results
    if samples:
        success_count = sum(1 for sample in samples if isinstance(sample, dict) and sample.get("success", False))
        success_rate = success_count / len(samples) if len(samples) > 0 else 0.0 # Avoid division by zero
    else:
        success_count = 0
        success_rate = 0.0
    
    # Save overall statistics
    stats = {
        "total_tasks": len(samples),
        "success_count": success_count,
        "success_rate": success_rate,
        "total_duration": total_duration,
        "average_duration": total_duration / len(samples) if samples and len(samples) > 0 else 0, # Avoid division by zero
        "samples": samples
    }
    
    stats_path = os.path.join(save_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)
    logger.info(f"Statistics saved to: {stats_path}")
    
    # Add global cleanup code at the end of the main function
    # Save sample paths
    save_path = os.path.join(save_dir, "data.jsonl")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=4)
    
    logger.info(f"All tasks processed! Total duration: {total_duration:.2f}s, success rate: {success_rate:.2%}")

    logger.info("ThreadPoolExecutor context has exited. Proceeding to global cleanup.")

    # Use imported cleanup_all_unity_processes function instead of redefining
    logger.info("Attempting to call cleanup_all_unity_processes().")
    cleanup_all_unity_processes()
    logger.info("cleanup_all_unity_processes() call completed.")
    
    # Force garbage collection
    logger.info("Attempting to call gc.collect().")
    import gc
    gc.collect()
    logger.info("gc.collect() call completed.")

    grpo_data_path = os.path.join(save_dir, 'dataset.json')
    # Add error handling to prevent dataset_extractor from failing
    try:
        dataset = extract_samples(save_dir)
        if dataset:  # Ensure dataset is not None
            save_dataset_to_file(dataset, output_file=grpo_data_path)
            logger.info(f"Dataset saved to: {grpo_data_path}")
        else:
            logger.warning("Extracted dataset is empty, skipping saving dataset.json")
    except Exception as e:
        logger.error(f"Error extracting or saving dataset: {traceback.format_exc()}") 
    dpo_file_path = os.path.join(save_dir, "dpo_pairs.json")
    dpo_with_reward_file = os.path.join(save_dir, "dpo_pairs_delta_reward.json")
    aggregate_reward_data(save_dir, config.get("enable_ttrl_reward", False))
    reward_file = os.path.join(save_dir, "reward_model_data.jsonl")
    if not os.path.exists(dpo_file_path):
        logger.warning(f"DPO samples not generated or path does not exist: {dpo_file_path}")

    # Filter GRPO dataset
    if os.path.exists(grpo_data_path): # Check if grpo_data_path exists before filtering
        filtered_path = filter_dataset(grpo_data_path, config["save_dir"])
        logger.info(f"Filtered result: {filtered_path}")
    else:
        logger.warning(f"GRPO dataset not found at {grpo_data_path}, skipping filtering.")
        filtered_path = "" # Set to empty if not found
    
    logger.info(f"Sampling completed, original result: {grpo_data_path}")
    
    # Write filtered GRPO dataset path to specified output file
    output_file_grpo = os.path.join(config["save_dir"], "sample_output.txt") # Renamed for clarity
    with open(output_file_grpo, "w") as f: # Renamed for clarity
        f.write(filtered_path)

    output_file_policy = os.path.join(config["save_dir"], "sample_output_policy.txt") # Renamed for clarity
    with open(output_file_policy, "w") as f: # Renamed for clarity
        f.write(filtered_path)
    
    # Write DPO dataset path to specified output file
    output_file_dpo = os.path.join(config["save_dir"], "sample_output_dpo.txt") # Renamed for clarity
    with open(output_file_dpo, "w") as f: # Renamed for clarity
        f.write(dpo_file_path)

    # Write DPO dataset with reward path to specified output file
    output_file_dpo_reward = os.path.join(config["save_dir"], "sample_output_dpo_reward.txt") # Renamed for clarity
    with open(output_file_dpo_reward, "w") as f: # Renamed for clarity
        f.write(dpo_with_reward_file)

    # Write reward model dataset path to specified output file
    output_file_reward = os.path.join(config["save_dir"], "sample_output_reward.txt") # Renamed for clarity
    with open(output_file_reward, "w") as f: # Renamed for clarity
        f.write(reward_file)
    
    return output_file_grpo, dpo_file_path, dpo_with_reward_file, reward_file # Return renamed variables


def main():
    parser = argparse.ArgumentParser(description="seea data sampling parameters")
    
    # Add common arguments
    parser = add_common_args(parser)
    
    # Add main.py specific arguments
    parser.add_argument("--dataset_root", type=str, default="",
                       help="Dataset root directory")
    parser.add_argument("--num", type=int, default=1,
                       help="Number of tasks")
    parser.add_argument("--enable_alfworld_env", action="store_true",
                       help="Enable ALFWorld environment, use ALFWorld environment as robot operation tool.")
    parser.add_argument("--save_dir", type=str, default="./samples",
                       help="Save data directory")
    parser.add_argument("--max_valid_data", type=int, default=-1,
                        help="Maximum number of valid data, stop sampling after reaching, -1 means no limit. \
                        For online MCTS sampling and GRPO training, max_valid_data = per_device_train_batch_size * nproc_per_node * gradient_accumulation_steps.")
    # Add multithreading related arguments
    parser.add_argument("--num_threads", type=int, default=0,
                       help="Number of parallel threads, default is 0 for automatic selection (use CPU core count)")
    parser.add_argument("--thread_agent_mode", action="store_true",
                       help="Create independent agent instance for each thread, improve parallel performance but increase memory usage")
    parser.add_argument("--task_timeout", type=int, default=900,
                       help="Timeout for each task (seconds), default is 900s (15 minutes). Tasks exceeding this time will be cancelled")
    parser.add_argument("--heartbeat_timeout", type=int, default=300,
                       help="Task heartbeat timeout (seconds), default is 300s (5 minutes). If a task does not update its heartbeat within this time, it will be cancelled")
    
    args = parser.parse_args()
    
    # Process configuration and arguments
    merged_config, user_provided_args = process_config_args(args)
    if merged_config is None:  # Exit if only generating config file
        exit(0)
    
    # Prioritize model_name and base_url
    if merged_config.get('model_name'): # Use .get for safer access
        # Create custom model configuration
        model_config = {
            "model": merged_config['model_name'],
            "base_url": merged_config.get('base_url'), # Use .get for safer access
            "api_key": "EMPTY"
        }
        logger.info(f"Using custom model: {merged_config['model_name']}, base_url: {merged_config.get('base_url') or 'default'}")
    elif merged_config.get('model') in VLM_MODELS: # Use .get for safer access
        model_config = VLM_MODELS[merged_config["model"]]
    elif merged_config.get('model') in LLM_MODELS: # Use .get for safer access
        model_config = LLM_MODELS[merged_config["model"]]
    else:
        raise ValueError(f"Invalid model: {merged_config.get('model')}. Must be one of: {list(VLM_MODELS.keys()) + list(LLM_MODELS.keys())}")
    
    # Process critic model configuration
    if 'critic_model_name' in merged_config and merged_config.get('critic_model_name'): # Use .get for safer access
        critic_model_config = {
            "model": merged_config['critic_model_name'],
            "base_url": merged_config.get('critic_base_url'), # Use .get for safer access
            "api_key": "EMPTY"
        }
        logger.info(f"Using custom critic model: {merged_config['critic_model_name']}, base_url: {merged_config.get('critic_base_url') or 'default'}")
    elif 'critic_model' in merged_config and merged_config.get('critic_model'): # Use .get for safer access
        if merged_config['critic_model'] in VLM_MODELS:
            critic_model_config = VLM_MODELS[merged_config["critic_model"]]
        elif merged_config['critic_model'] in LLM_MODELS:
            critic_model_config = LLM_MODELS[merged_config["critic_model"]]
        else:
            raise ValueError(f"Invalid critic model: {merged_config['critic_model']}. Must be one of: {list(VLM_MODELS.keys()) + list(LLM_MODELS.keys())}")
    else:
        critic_model_config = None
        logger.warning("Critic model not configured, will use default model")
    
    # Extract MCTS parameters
    mcts_params = extract_mcts_params(merged_config, user_provided_args)
    
    # Integrate all parameters into one configuration object
    final_config = merged_config.copy()
    final_config["model"] = model_config
    final_config["critic_model"] = critic_model_config
    final_config["mcts_params"] = mcts_params
    return run(config=final_config)


if __name__ == '__main__':
    main()