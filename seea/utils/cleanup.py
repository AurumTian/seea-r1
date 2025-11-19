import os
import time
import signal
import traceback
from typing import Any, Optional
from seea.utils.logger import get_logger

logger = get_logger(__name__)

def cleanup_alfworld_env(env: Any, thread_prefix: str = "") -> None:
    """Clean up ALFWorld environment resources
    
    Args:
        env: ALFWorld environment instance
        thread_prefix: Thread prefix for log identification
    """
    if env is None:
        return
        
    try:
        # Close the environment
        if hasattr(env, 'close'):
            env.close()
            
        # More thoroughly release Unity process
        if hasattr(env, 'controller'):
            # Stop Unity thread
            if hasattr(env.controller, '_unity_thread') and env.controller._unity_thread:
                try:
                    env.controller._unity_thread = None
                except:
                    pass
                    
            # Close WebSocket connection
            if hasattr(env.controller, 'ws'):
                try:
                    env.controller.ws.close()
                    env.controller.ws = None
                except:
                    pass
                    
            # Terminate Unity process
            if hasattr(env.controller, 'unity_proc') and env.controller.unity_proc:
                try:
                    # First try to terminate normally
                    env.controller.unity_proc.terminate()
                    # Give the process some time to terminate
                    time.sleep(0.5)
                    # If the process is still running, force terminate
                    if env.controller.unity_proc.poll() is None:
                        env.controller.unity_proc.kill()
                        # May need an additional SIGKILL signal on macOS
                        os.kill(env.controller.unity_proc.pid, signal.SIGKILL)
                except Exception as kill_e:
                    logger.error(f"{thread_prefix}Error terminating Unity process: {str(kill_e)}")
                finally:
                    env.controller.unity_proc = None
                    
            # Explicitly set controller to None
            env.controller = None
            
        logger.info(f"{thread_prefix}ALFWorld environment closed")
    except Exception as env_e:
        logger.error(f"{thread_prefix}Error closing ALFWorld environment: {str(env_e)}")

def cleanup_robot_agent(agent: Any, thread_prefix: str = "") -> None:
    """Clean up robot agent resources
    
    Args:
        agent: Robot agent instance
        thread_prefix: Thread prefix for log identification
    """
    if agent is None:
        return
        
    try:
        # Import necessary classes, but do not import at module level to avoid circular dependencies
        from seea.configs.robot_mcts import RobotMCTSWrapper
        from seea.configs.visual_agent import FunctionCallAgent
        
        core_llm_agent = None
        # Determine the core LLM interaction agent instance
        if isinstance(agent, RobotMCTSWrapper):
            if hasattr(agent, 'actor') and agent.actor is not None and \
               hasattr(agent.actor, 'llm') and agent.actor.llm is not None:
                core_llm_agent = agent.actor.llm
        elif isinstance(agent, FunctionCallAgent):
            # FunctionCallAgent may be a subclass of ChatAgent directly, or hold ChatAgent via .llm
            if hasattr(agent, 'llm') and agent.llm is not None:
                core_llm_agent = agent.llm
            elif hasattr(agent, 'close') and callable(getattr(agent, 'close')): # If agent itself is closable
                core_llm_agent = agent 
        elif hasattr(agent, 'close') and callable(getattr(agent, 'close')): # If agent itself is ChatAgent or similar closable object
            core_llm_agent = agent

        # Try to close the core LLM agent
        if core_llm_agent is not None:
            if hasattr(core_llm_agent, 'close') and callable(getattr(core_llm_agent, 'close')):
                logger.info(f"{thread_prefix}Attempting to close core_llm_agent: {type(core_llm_agent).__name__}")
                try:
                    core_llm_agent.close()
                    logger.info(f"{thread_prefix}Successfully closed core_llm_agent: {type(core_llm_agent).__name__}")
                except Exception:
                    logger.error(f"{thread_prefix}Error closing core_llm_agent {type(core_llm_agent).__name__}: {traceback.format_exc()}")
            else:
                logger.warning(f"{thread_prefix}core_llm_agent ({type(core_llm_agent).__name__}) does not have a callable 'close' method.")
        
        # After attempting to close, release top-level references (helps garbage collection)
        if isinstance(agent, RobotMCTSWrapper):
            if hasattr(agent, 'actor') and agent.actor is not None:
                if hasattr(agent.actor, 'llm'):
                    agent.actor.llm = None
                agent.actor = None
            if hasattr(agent, 'mcts'):
                agent.mcts = None
        elif isinstance(agent, FunctionCallAgent):
            if hasattr(agent, 'llm'):
                agent.llm = None
        
        # Explicitly set to None to help garbage collection (original part)
        if hasattr(agent, 'alfworld_env'):
            agent.alfworld_env = None
            
        logger.info(f"{thread_prefix}Robot agent resources cleaned up (LLM close attempted)") # Modify log to reflect attempt to close
    except Exception: # Maintain broad exception capture from original, but log more detailed traceback
        logger.error(f"{thread_prefix}Error cleaning up robot agent: {traceback.format_exc()}")

def cleanup_all_unity_processes() -> None:
    """Clean up all Unity related processes"""
    try:
        import psutil
        
        # Find all Unity related processes
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                # Check if process name contains Unity related strings
                if 'Unity' in proc.info['name'] or 'unity' in proc.info['name']:
                    logger.info(f"Terminating orphaned Unity process: {proc.info['pid']} - {proc.info['name']}")
                    try:
                        # Try to terminate normally
                        os.kill(proc.info['pid'], signal.SIGTERM)
                        # Give the process some time to terminate
                        time.sleep(0.5)
                        # If the process is still running, force terminate
                        if psutil.pid_exists(proc.info['pid']):
                            os.kill(proc.info['pid'], signal.SIGKILL)
                    except Exception as e:
                        logger.error(f"Error terminating Unity process {proc.info['pid']}: {str(e)}")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    except Exception as e:
        logger.error(f"Error cleaning up Unity processes: {str(e)}")

def perform_cleanup(env: Optional[Any] = None, 
                   agent: Optional[Any] = None, 
                   thread_prefix: str = "",
                   clear_thread_local: bool = True) -> None:
    """Perform complete resource cleanup
    
    Args:
        env: ALFWorld environment instance
        agent: Robot agent instance
        thread_prefix: Thread prefix for log identification
        clear_thread_local: Whether to clear thread-local state
    """
    try:
        # 1. Clean up ALFWorld environment
        if env is not None:
            cleanup_alfworld_env(env, thread_prefix)
        
        # 2. Clean up robot agent
        if agent is not None:
            cleanup_robot_agent(agent, thread_prefix)
        
        # 3. Clean up thread-local state
        if clear_thread_local:
            from seea.utils.base import StateManager
            StateManager().clear_thread_local()
        
        # 4. Force garbage collection
        import gc
        gc.collect()
        
        logger.info(f"{thread_prefix}Resource cleanup complete")
    except Exception as cleanup_e:
        logger.error(f"{thread_prefix}Error occurred during resource cleanup: {str(cleanup_e)}")