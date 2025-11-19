import os
import json
import time
import copy
import argparse
import traceback
import threading
import numpy as np
import concurrent.futures
from seea.utils.base import StateManager
from seea.utils.logger import get_logger
from seea.utils.agent_factory import create_agent
from seea.envs.alfworld.alfworld import get_alfworld_environment
from seea.envs.alfworld.alfworld_icl import get_alfworld_icl_prompt
from seea.utils.cleanup import perform_cleanup, cleanup_all_unity_processes
from seea.utils.config_utils import add_common_args, process_config_args
logger = get_logger(__name__)


def create_agent_and_env(save_folder, config: dict):
    """Create agent and environment instance"""
    if config['split'] == "train":
        alfworld_split = "train"
    elif config['split'] in ["dev", "eval_in_distribution"]:
        alfworld_split = "eval_in_distribution"
    elif config['split'] in ["test", "eval_out_of_distribution"]:
        alfworld_split = "eval_out_of_distribution"
    else:
        raise ValueError(f"Invalid split: {config['split']}. Must be one of: train, dev, test")
    
    env_type = "AlfredThorEnv" if not config['wo_image_tool_result'] else "AlfredTWEnv"
    env = get_alfworld_environment(env_type=env_type, train_eval=alfworld_split)
    env.seed(np.random.randint(0, 10000))  # Use random seed to avoid duplicate parallel tasks
    new_config = copy.deepcopy(config)
    new_config.update({
        'alfworld_env': env,
        'save_folder': save_folder,
        'visual_world': False
    })

    agent = create_agent(new_config)
    
    return agent, env

def evaluate_single_game(game_idx: int, config: dict, agent=None, env=None):
    """Evaluate a single game, use existing agent and env or create new instances"""
    thread_prefix = f"[Game-{game_idx}] "
    logger.info(f"{thread_prefix}Start evaluating game")
    # Create a game-specific save directory
    save_folder = os.path.join(config["output_dir"], f"game_{game_idx}")
    os.makedirs(save_folder, exist_ok=True)
    StateManager().set("sample_save_dir", save_folder, thread_local=True)
    
    # If agent and env are not provided, create new instances
    created_locally = False
    try:
        if agent is None or env is None:
            created_locally = True
            agent, env = create_agent_and_env(save_folder, config)
        
        # Execute evaluation
        game_names, game_points, game_steps, game_gcs, actions = evaluate_episode(
            agent, env, config
        )

        # Format game name
        game_name = game_names[0] if isinstance(game_names, list) else game_names
        if isinstance(game_name, str) and '/' in game_name:
            game_name = "/".join(game_name.split("/")[-3:-1])
        
        return {
            "game_idx": game_idx,
            "game_name": game_name,
            "game_names": game_names,
            "points": game_points,
            "steps": game_steps,
            "gc_points": game_gcs,
            "actions": actions,
            "success": game_points > 0
        }
    except Exception as e:
        logger.error(f"{thread_prefix}Error occurred during evaluation: {traceback.format_exc()}")
        return {
            "game_idx": game_idx,
            "game_name": f"game_{game_idx}",
            "game_names": [f"game_{game_idx}"],
            "points": 0,
            "steps": 0,
            "gc_points": 0,
            "actions": [],
            "success": False,
            "error": str(e)
        }
    finally:
        # Only resources created locally need to be cleaned up
        if created_locally:
            # Use cleanup tool to clean up resources
            perform_cleanup(
                env=env,
                agent=agent,
                thread_prefix=thread_prefix
            )
        
        # Clean up thread local state
        StateManager().clear_thread_local()


def run_evaluation(config: dict):
    """Run evaluation process - support multi-threaded parallel execution of tasks"""
    # Initialize log level
    logger.setLevel(config['log_level'])
    
    # Set the number of tasks based on split
    if config['split'] == 'train':
        N_TASKS = 3553
    elif config['split'] in ['dev', 'eval_in_distribution']:
        N_TASKS = 140
    elif config['split'] in ['test', 'eval_out_of_distribution']:
        N_TASKS = 134
    else:
        raise ValueError(f"Invalid split: {config['split']}. Must be one of: train, dev, test")
    
    # If num_games is not specified, use the default number for the corresponding split
    if config["num_games"] is None:
        config["num_games"] = N_TASKS
    
    # Consider the repeats parameter
    total_games = config["num_games"] * config["repeats"]
    
    # Create task list
    tasks = []
    for i in range(config["num_games"]):
        for _ in range(config["repeats"]):
            tasks.append(i)
    tasks = tasks[:total_games]  # Ensure it does not exceed the total number
    
    # Initialize result recording
    res_points, res_steps, res_gcs = [], [], []
    res_info = []
    
    # Determine if multi-threading is used
    use_multithread = config.get("max_workers", 1) > 1
    
    if use_multithread:   
        # Thread environment cache - use thread ID as key
        thread_envs = {}
        thread_lock = threading.Lock()

        def process_game(game_idx):
            """Thread function for processing a single game"""
            # Get the current thread ID
            thread_name = threading.current_thread().name
            # Extract worker_id from ThreadPoolExecutor-1_5 format (5)
            if '_' in thread_name:
                thread_id = int(thread_name.split('_')[-1]) % config["max_workers"]
            else:
                # If the format does not match, use the hash value of the thread ID modulo
                thread_id = hash(threading.get_ident()) % config["max_workers"]
                
            thread_config = config.copy()
            thread_prefix = f"[Thread-{thread_id}-Game-{game_idx}] "
            logger.info(f"{thread_prefix}Start evaluating game")
            
            # Create a game-specific save directory
            save_folder = os.path.join(thread_config["output_dir"], f"game_{game_idx}")
            os.makedirs(save_folder, exist_ok=True)
            StateManager().set("sample_save_dir", save_folder, thread_local=True)
            
            # Check if the current thread already has an environment instance, if not, create one
            with thread_lock:
                if thread_id not in thread_envs:
                    logger.info(f"{thread_prefix}Creating a new environment instance for the thread")
                    thread_env_folder = os.path.join(thread_config["output_dir"], f"thread_{thread_id}")
                    os.makedirs(thread_env_folder, exist_ok=True)
                    agent, env = create_agent_and_env(thread_env_folder, thread_config)
                    thread_envs[thread_id] = (agent, env)
            
            # Get the environment instance for the current thread
            agent, env = thread_envs[thread_id]
            
            try:
                # Set timeout processing
                if thread_config["timeout"]:
                    import signal
                    
                    def timeout_handler(signum, frame):
                        raise TimeoutError(f"Task {game_idx} execution timed out")
                    
                    # Set timeout signal
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(int(thread_config["timeout"]))
                
                # Execute evaluation - use the environment instance shared by the thread
                result = evaluate_single_game(game_idx, thread_config, agent, env)
                
                # If timeout is set, cancel the timeout signal
                if thread_config["timeout"]:
                    signal.alarm(0)
                
                return result
            except Exception as e:
                logger.error(f"{thread_prefix}Error occurred during evaluation: {traceback.format_exc()}")
                return {
                    "game_idx": game_idx,
                    "game_name": f"game_{game_idx}",
                    "game_names": [f"game_{game_idx}"],
                    "points": 0,
                    "steps": 0,
                    "gc_points": 0,
                    "actions": [],
                    "success": False,
                    "error": str(e)
                }
            finally:
                # Clean up thread local state
                StateManager().clear_thread_local()
        
        try:
            # Create a thread pool
            max_workers = min(config["max_workers"], total_games)
            logger.info(f"Using {max_workers} threads to process evaluation tasks")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                futures = [executor.submit(process_game, game_idx) for game_idx in tasks]
                
                # Collect results
                completed = 0
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        
                        # Record results
                        res_points.append(result["points"])
                        res_steps.append(result["steps"])
                        res_gcs.append(result["gc_points"])
                        res_info.append(f"{result['game_name']}, score: {result['points']}, step: {result['steps']}")
                        
                        completed += 1
                        
                        # Print progress
                        if completed % 10 == 0 or config["debug"]:
                            logger.info(f"Completed {completed}/{total_games} evaluation tasks")
                        
                        # Save intermediate results
                        if config["save_intermediate"] and completed % 10 == 0:
                            try:
                                interim_results = {
                                    "config": config,
                                    "success_rate": np.mean(res_points),
                                    "avg_steps": np.mean(res_steps),
                                    "avg_points": np.mean(res_points),
                                    "avg_gc_points": np.mean(res_gcs),
                                    "res_info": res_info,
                                    "completed": completed,
                                    "total": total_games
                                }
                                save_results(interim_results, config["output_dir"], interim=True)
                            except Exception as save_err:
                                logger.error(f"Error saving intermediate results: {str(save_err)}")
                    except Exception as e:
                        logger.error(f"Error processing evaluation results: {str(e)}")
                        logger.error(f"Error details: {traceback.format_exc()}")
        finally:
            # Clean up all thread environment instances
            for thread_id, (agent, env) in thread_envs.items():
                perform_cleanup(
                    env=env,
                    agent=agent,
                    thread_prefix=f"[Thread-{thread_id}] "
                )
            thread_envs.clear()
            
            # If multi-threading is used, perform a global Unity process cleanup once
            if use_multithread:
                cleanup_all_unity_processes(logger)
                
    else: # Single-thread execution
        logger.info(f"Processing {total_games} evaluation tasks with a single thread")
        for game_idx in tasks:
            try:
                # Create independent agent and env for each task
                save_folder = os.path.join(config["output_dir"], f"game_{game_idx}")
                os.makedirs(save_folder, exist_ok=True)
                StateManager().set("sample_save_dir", save_folder, thread_local=True)
                
                agent, env = create_agent_and_env(save_folder, config)
                
                result = evaluate_single_game(game_idx, config, agent, env)
                
                res_points.append(result["points"])
                res_steps.append(result["steps"])
                res_gcs.append(result["gc_points"])
                res_info.append(f"{result['game_name']}, score: {result['points']}, step: {result['steps']}")
                
                # In single-thread mode, clean up resources after each task
                perform_cleanup(
                    env=env,
                    agent=agent,
                    thread_prefix=f"[Game-{game_idx}] "
                )
                StateManager().clear_thread_local()

            except Exception as e:
                logger.error(f"Error occurred during evaluation of game {game_idx}: {traceback.format_exc()}")
                res_points.append(0)
                res_steps.append(0)
                res_gcs.append(0)
                res_info.append(f"game_{game_idx}, score: 0, step: 0, error: {str(e)}")
            
            if (game_idx + 1) % 10 == 0 or config["debug"]:
                logger.info(f"Completed {game_idx + 1}/{total_games} evaluation tasks")

    # Ensure the output directory exists
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Calculate final results
    avg_points = np.mean(res_points) if res_points else 0
    avg_steps = np.mean(res_steps) if res_steps else 0
    avg_gcs = np.mean(res_gcs) if res_gcs else 0
    success_rate = np.sum([1 for p in res_points if p > 0]) / len(res_points) if res_points else 0
    
    results = {
        "avg_points": avg_points,
        "avg_steps": avg_steps,
        "avg_gc_points": avg_gcs,
        "success_rate": success_rate,
        "num_games": total_games,
        "game_details": res_info,
        "config": config,
        "model": config["model"],
        "model_name": config["model_name"],
        "base_url": config["base_url"],
        "split": config["split"],
    }
    
    save_results(results, config["output_dir"])
    save_average_scores_and_steps(os.path.join(config["output_dir"], "results.json"))

    logger.info(f"Evaluation completed. Average score: {avg_points:.4f}, Average steps: {avg_steps:.4f}, Average GC score: {avg_gcs:.4f}, Success rate: {success_rate:.4f}")
    return results


def evaluate_episode(agent, env, config):
    """Evaluate a single episode"""
    # Initialize StateManager
    state_manager = StateManager()
    state_manager.clear_thread_local()
    
    # Ensure the evaluation history is initialized
    state_manager.set("eval_history", [])
    state_manager.set("eval_actions_history", [])
    state_manager.set("eval_rewards_history", [])
    state_manager.set("eval_gc_history", [])
    
    # Reset the environment
    obs, infos = env.reset()
    game_names = infos.get("extra.gamefile", [f"game_{obs}"])

    # Set the initial state of the environment
    env.last_info = infos
    
    # Process observations
    observation_strings = list(obs)
    
    # Debug output
    if config["debug"]:
        print(f"Initial observation: {observation_strings[0]}")
    
    try:
        agent.initialize()
        instruction = "\n".join(obs[0].split("\n\n")[1:])
        game_file = infos["extra.gamefile"][0]
        name = "/".join(game_file.split("/")[-3:-1])
        logger.info(f"obs: {obs}\ngame_file: {game_file}")
        instruction = get_alfworld_icl_prompt(instruction, name, config.get("format", "react"))

        if not config["wo_image_tool_result"]:
            frame_path = env.get_visual_obs()
            logger.info(f"ALFWorld first frame image path: {frame_path}")
            if frame_path is not None and frame_path != "" and os.path.exists(frame_path):
                instruction += "\nThe current visual observation is shown below:"
            _ = agent(instruction, [frame_path])
        else:
            _ = agent(instruction)
         
        # After the task is completed, get the evaluation history from StateManager
        eval_history = state_manager.get("eval_history")
        if not eval_history:
            logger.warning("No evaluation history was obtained from StateManager, it may be due to the agent not calling the tool correctly.")
            # Create a basic history to avoid subsequent processing errors
            eval_history = [{"action": "unknown", "observation": "", "done": False, "reward": 0.0, "goal_condition_success": 0.0}]
            state_manager.set("eval_history", eval_history)
        
        # Get the final results
        final_reward = state_manager.get("eval_max_reward", 0.0)
        final_gc = state_manager.get("eval_max_gc_success", 0.0)
        actions_history = state_manager.get("eval_actions_history", [])
        
        # If actions_history is empty, extract it from eval_history
        if not actions_history and eval_history:
            actions_history = [step.get("action", "") for step in eval_history]
            state_manager.set("eval_actions_history", actions_history)

        return game_names, final_reward, len(eval_history), final_gc, actions_history
        
    except Exception as e:
        logger.error(f"Error occurred during evaluation of episode: {traceback.format_exc()}")
        # Ensure valid results are returned even if an error occurs
        return game_names, 0.0, 0, 0.0, []


def save_results(results: dict, output_dir: str = "eval_results", interim: bool = False):
    """Save evaluation results to a JSON file"""
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Build the filename
    model_name_safe = results["model_name"].replace("/", "_") if results["model_name"] else results["model"].replace("/", "_")
    filename_parts = [
        model_name_safe,
        results["split"],
        f"games_{results['num_games']}",
        f"avg_score_{results['avg_points']:.2f}",
        f"success_{results['success_rate']:.2f}",
        time.strftime("%Y%m%d-%H%M%S")
    ]
    if interim:
        filename_parts.append("interim")
    filename = "_vs_".join(filename_parts) + ".json"
    
    file_path = os.path.join(output_dir, filename)
    
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        logger.info(f"Evaluation results have been saved to: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save evaluation results to {file_path}: {e}")

    # Save a fixed results.json file for easy script reading
    fixed_file_path = os.path.join(output_dir, "results.json")
    try:
        with open(fixed_file_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        logger.info(f"Latest evaluation results have been synchronized to: {fixed_file_path}")
    except Exception as e:
        logger.error(f"Failed to synchronize evaluation results to {fixed_file_path}: {e}")


def save_average_scores_and_steps(result_path: str):
    """Read the result.json file, calculate the average score and average steps, and save to average_scores_steps.json."""
    try:
        with open(result_path, 'r') as f:
            data = json.load(f)
        
        if not data or 'game_details' not in data or not data['game_details']:
            logger.warning(f"No game details found in {result_path}, cannot calculate average values.")
            return

        scores = []
        steps_list = []
        success_count = 0

        for detail_str in data['game_details']:
            try:
                parts = detail_str.split(',')
                score_part = next((p for p in parts if 'score:' in p), None)
                step_part = next((p for p in parts if 'step:' in p), None)
                
                if score_part:
                    score = float(score_part.split(':')[1].strip())
                    scores.append(score)
                    if score > 0:
                        success_count += 1
                else:
                    logger.warning(f"Failed to parse score from '{detail_str}'")

                if step_part:
                    steps = int(step_part.split(':')[1].strip())
                    steps_list.append(steps)
                else:
                    logger.warning(f"Failed to parse steps from '{detail_str}'")
                    
            except Exception as e:
                logger.error(f"Error parsing game details '{detail_str}': {e}")
                continue  # Continue processing the next item

        if not scores:
            logger.warning(f"No valid game scores found in {result_path}, cannot calculate average values.")
            avg_scores = 0
            success_rate = 0
        else:
            avg_scores = sum(scores) / len(scores)
            success_rate = success_count / len(scores)
            
        if not steps_list:
            logger.warning(f"No valid game steps found in {result_path}, cannot calculate average values.")
            avg_steps = 0
        else:
            avg_steps = sum(steps_list) / len(steps_list)

        output_data = {
            'average_score': avg_scores,
            'average_steps': avg_steps,
            'success_rate': success_rate,
            'total_games': len(scores) # or use data['num_games'] if it is more accurate
        }

        output_path = os.path.join(os.path.dirname(result_path), "average_scores_steps.json")
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        logger.info(f"Average scores and steps have been saved to: {output_path}")

    except FileNotFoundError:
        logger.error(f"Result file not found: {result_path}")
    except json.JSONDecodeError:
        logger.error(f"Error parsing result file JSON: {result_path}")
    except Exception as e:
        logger.error(f"Unknown error occurred while calculating or saving average scores and steps: {e}")


def main():
    parser = argparse.ArgumentParser(description='SEEA evaluation system')
    # Add common parameters
    parser = add_common_args(parser)
    
    # Parse parameters
    args = parser.parse_args()
    config = process_config_args(args)
    
    # Add evaluation-specific parameters to config
    eval_params = {
        "output_dir": config.get("output_dir", "eval_results"),
        "split": config.get("split", "dev"),
        "num_games": config.get("num_games", None),
        "repeats": config.get("repeats", 1),
        "max_steps": config.get("max_steps", 50),
        "max_workers": config.get("max_workers", 1),
        "save_intermediate": config.get("save_intermediate", False),
        "timeout": config.get("timeout", None),
        "log_level": config.get("log_level", "INFO"),
        "wo_image_tool_result": config.get("wo_image_tool_result", False),
        "visual_perception": config.get("visual_perception", False),
        "evaluate": True
    }
    config.update(eval_params)

    # Print the final configuration for debugging
    logger.info("Final evaluation configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    # Ensure output directory exists
    os.makedirs(config["output_dir"], exist_ok=True)

    # Run evaluation
    try:
        run_evaluation(config)
    except Exception as e:
        logger.error(f"Critical error occurred during evaluation: {traceback.format_exc()}")
    finally:
        # Perform final cleanup before program ends
        cleanup_all_unity_processes(logger)
        logger.info("Evaluation program completed.")


if __name__ == "__main__":
    main()