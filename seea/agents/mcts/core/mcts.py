import math
import itertools
from abc import ABC
from copy import deepcopy
from collections import defaultdict
from seea.utils.base import StateManager
from typing import Callable, Generic, Hashable, NamedTuple, Optional
from seea.utils.logger import get_logger
logger = get_logger(__name__)

import numpy as np
from tqdm import trange

from seea.agents.mcts.core.base import (
    Action,
    SearchAlgorithm,
    SearchConfig,
    State,
    Trace,
    WorldModel,
)


import threading

class MCTSNode(Generic[State, Action]):
    # Use thread-local storage to maintain the counter
    _local_data = threading.local()

    @classmethod
    def _get_id_iter(cls):
        if not hasattr(cls._local_data, 'id_iter'):
            cls._local_data.id_iter = itertools.count()
        return cls._local_data.id_iter

    @classmethod
    def reset_id(cls):
        cls._local_data.id_iter = itertools.count()

    def __init__(
        self,
        state: "Optional[State]" = None,
        action: "Optional[Action]" = None,
        parent: "Optional[MCTSNode]" = None,
        fast_reward: float = 0.0,
        fast_reward_details=None,
        is_terminal: bool = False,
        calc_q: Callable[[list[float]], float] = np.mean,
        advantage_calc_method: str = "mean",
    ):
        """
        Node constructor in MCTS search tree

        :param state: The state of the current node
        :param action: The action taken from the parent node to the current node
        :param parent: Parent node; if None, it means this node is the root node
        :param fast_reward: Quick estimate of the reward for the previous action
        :param fast_reward_details: Detailed information of the fast reward, defaults to an empty dictionary if not provided
        :param is_terminal: Whether the current state is a terminal state (True means terminal)
        :param calc_q: Method for calculating Q value from historical rewards, defaults to np.mean
        """
        # Print key information for initialization start
        logger.debug("Starting to initialize MCTSNode, %s, parent.id=%s", action, parent.id if parent is not None else None)
        
        # Assign a unique ID to the current node using a thread-local counter
        self.id = next(self._get_id_iter())
        
        # If fast_reward_details is not provided, initialize it as an empty dictionary
        if fast_reward_details is None:
            fast_reward_details = {}
        
        # Initialize cumulative reward list
        self.cum_rewards: list[float] = []
        
        # Set initial values for fast reward and actual reward
        self.fast_reward = fast_reward
        
        # Initialize reward to 0
        self.reward = 0.0

        # Set format reward
        self.format_reward = 0.0

        # Set goal reward
        self.goal_reward = 0.0

        # Cost per step = -0.01
        self.cost_per_step = -0.01

        # Advantage calculation method: "parent" or "mean"
        self.advantage_calc_method = advantage_calc_method
        self.gamma = 0.99
        
        # Save detailed information of fast reward
        self.fast_reward_details = fast_reward_details
                
        # Mark whether the current state is a terminal state
        self.is_terminal = is_terminal
        
        # Save action, state, and parent node information
        self.action = action
        self.state = state
        self.parent = parent

        # Initialize child node set to empty
        self.children: "list[MCTSNode]" = []

        # Save the function used to calculate the Q value
        self.calc_q = calc_q
        
        # Initialize visit count and cumulative reward (Q value)
        self.N = 0  # Visit count
        self._Q = 0  # Cumulative reward (Q value)

        
        # Determine the depth of the current node based on the depth of the parent node. The root node depth is 0, otherwise it is the parent node depth + 1
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1
        
        logger.debug(f"New MCTSNode: id={self.id}, depth={self.depth}")

    def __str__(self):
        return f"MCTSNode(id={self.id}, state={self.state}, action={self.action}, reward={self.reward}, is_terminal={self.is_terminal})"
    # noinspection PyPep8Naming
    # @property
    # def Q(self) -> float:
    #     if self.state is None:
    #         return self.fast_reward
    #     else:
    #         return self.calc_q(self.cum_rewards)

    @property
    def Q(self) -> float:
        if self.N == 0:
            return 0
        return self._Q  # Getter
    
    @Q.setter
    def Q(self, value: float):
        self._Q = value  # Setter


class MCTSResult(NamedTuple):
    """
    Named tuple for MCTS search results
    Attributes:
        terminal_state (State): State when the search algorithm terminates
        cum_reward (float): Cumulative reward from the root node to the terminal state
        trace (Trace): Trajectory from the root node to the terminal state
        trace_of_nodes (list[MCTSNode]): Trajectory from the root node to the terminal state, composed of MCTSNode
        tree_state (MCTSNode): State of the tree when the search algorithm terminates
        trace_in_each_iter (list[list[MCTSNode]]): Trajectory of each iteration, composed of MCTSNode
        tree_state_after_each_iter (list[MCTSNode]): State of the tree at the end of each iteration, composed of MCTSNode
        aggregated_result (Optional[Hashable]): Aggregated result
        max_gc_success_per_iter (Dict): Maximum GC success rate per iteration
    """
    terminal_state: State
    cum_reward: float
    trace: Trace
    trace_of_nodes: list[MCTSNode]
    tree_state: MCTSNode
    trace_in_each_iter: list[list[MCTSNode]] = []
    tree_state_after_each_iter: list[MCTSNode] = []
    aggregated_result: Optional[Hashable] = None
    max_gc_success_per_iter: dict = {}


class MCTSAggregation(Generic[State, Action], ABC):
    def __init__(
        self, retrieve_answer: Callable[[State], Hashable], weight_policy: str = "edge"
    ):
        assert weight_policy in ["edge", "edge_inverse_depth", "uniform"]
        self.retrieve_answer = retrieve_answer
        self.weight_policy = weight_policy

    def __call__(
        self, tree_state: MCTSNode[State, Action]
    ) -> Optional[Hashable]:
        answer_dict = defaultdict(lambda: 0)

        def visit(cur: MCTSNode[State, Action]):
            if cur.state is None:
                return []
            if cur.is_terminal:
                answer = self.retrieve_answer(cur.state)
                if answer is None:
                    logger.info("MCTSAggregation: no answer retrieved.")
                    return []
                if self.weight_policy == "edge":
                    answer_dict[answer] = answer_dict[answer] + int(cur.reward)
                elif self.weight_policy == "edge_inverse_depth":
                    answer_dict[answer] = answer_dict[answer] + int(cur.reward / cur.depth)
                elif self.weight_policy == "uniform":
                    answer_dict[answer] += 1
                return [(answer, cur.depth)]
            depth_list = defaultdict(list)
            cur_list = []
            # Check if children is None, if so use an empty list
            for child in (cur.children or []):
                cur_list.extend(child_info := visit(child))
                for answer, depth in child_info:
                    depth_list[answer].append(depth)
            for answer, depths in depth_list.items():
                if self.weight_policy == "edge":
                    answer_dict[answer] += int(cur.reward)
                elif self.weight_policy == "edge_inverse_depth":
                    answer_dict[answer] += int(cur.reward / np.mean(depths))
            return cur_list

        visit(tree_state)

        if len(answer_dict) == 0:
            return None
        return max(answer_dict, key=lambda answer: answer_dict[answer])


class MCTS(SearchAlgorithm, Generic[State, Action]):
    def __init__(
        self,
        output_trace_in_each_iter: bool = True,
        w_exp: float = 1.0,
        depth_limit: int = 5,
        n_iters: int = 10,
        cum_reward: Callable[[list[float]], float] = sum,
        calc_q: Callable[[list[float]], float] = np.mean,
        simulate_strategy: str | Callable[[list[float]], int] = "random",
        output_strategy: str = "max_reward",
        uct_with_fast_reward: bool = True,
        aggregator: Optional[MCTSAggregation] = None,
        disable_tqdm: bool = True,
        node_visualizer: Callable[[MCTSNode], dict] = lambda x: x.__dict__,
        enable_reflection: bool = False,
        enable_ttrl_reward: bool = False,
        ttrl_vote_num: int = 10,
    ):
        """
        MCTS algorithm

        :param output_trace_in_each_iter: whether to output the trace of the chosen trajectory in each iteration ; the trace is *deepcopy*-ed
                                          will also output *tree_state_after_each_iter*, which is the *deepcopy*-ed root
        :param w_exp: the weight of exploration in UCT
        :param cum_reward: the way to calculate the cumulative reward from each step. Defaults: sum
        :param calc_q: the way to calculate the Q value from histories. Defaults: np.mean
        :param simulate_strategy: simulate strategy. Options: 'max', 'sample', 'random', or use a custom function
        :param output_strategy: the way to output the result. The nodes are not *deepcopy*-ed, so the information is after all iterations
                                Options: 'max_reward': dfs on the final tree to find a trajectory with max reward using :param cum_reward:
                                         'follow_max': starting from root, choose the maximum reward child at each step. May output a non-terminal node if dead end
                                         'max_visit': the terminal node with maximum number of visits
                                         'max_iter': the trajectory with a terminal node and max reward among those in each iteration
                                         'last_iter': the last trajectory. May output a non-terminal node if the last iteration leads to a dead end
                                         'last_terminal_iter': the last trajectory with a terminal node
                                Outputs *None* if no trajectory with terminal node but required
        :param uct_with_fast_reward: if True, use fast_reward instead of reward for unvisited children in UCT
                                     Otherwise, visit the *unvisited* children with maximum fast_reward first
        """
        super().__init__()
        self.world_model = None
        self.search_config = None
        self.output_trace_in_each_iter = output_trace_in_each_iter
        self.w_exp = w_exp
        self.depth_limit = depth_limit
        self.n_iters = n_iters
        self.enable_reflection = enable_reflection
        self.enable_ttrl_reward = enable_ttrl_reward
        self.ttrl_vote_num = ttrl_vote_num
        self.cum_reward = cum_reward
        self.calc_q = calc_q
        default_simulate_strategies: dict[str, Callable[[list[float]], int]] = {
            "max": lambda x: int(np.argmax(x)),
            "sample": lambda x: np.random.choice(len(x), p=x),
            "random": lambda x: np.random.choice(len(x)),
        }
        self.simulate_choice: Callable[[list[float]], int] = (
            default_simulate_strategies[simulate_strategy] if isinstance(simulate_strategy, str) else simulate_strategy
        )
        assert output_strategy in [
            "max_reward",
            "follow_max",
            "max_visit",
            "max_iter",
            "last_iter",
            "last_terminal_iter",
        ]
        self.output_strategy = output_strategy
        self.uct_with_fast_reward = uct_with_fast_reward
        self._output_iter: list[MCTSNode] = []
        self._output_cum_reward = -math.inf
        self.trace_in_each_iter: list[list[MCTSNode]] = []
        self.root: Optional[MCTSNode] = None
        self.disable_tqdm = disable_tqdm
        self.node_visualizer = node_visualizer
        self.aggregator = aggregator
        self.reflection_root: bool = False  # Reflection breakpoint flag
        
        logger.debug("Initializing MCTS tree, key information is as follows:")
        logger.debug("Whether to output the trace of each iteration: %s", output_trace_in_each_iter)
        logger.debug("Exploration weight (w_exp): %f", w_exp)
        logger.debug("Tree depth limit: %d", depth_limit)
        logger.debug("Number of iterations: %d", n_iters)
        logger.debug("Cumulative reward calculation method: %s", cum_reward.__name__ if hasattr(cum_reward, '__name__') else str(cum_reward))
        logger.debug("Q value calculation method: %s", calc_q.__name__ if hasattr(calc_q, '__name__') else str(calc_q))
        logger.debug("Simulation strategy: %s", simulate_strategy)
        logger.debug("Output strategy: %s", output_strategy)
        logger.debug("Whether to use fast_reward: %s", uct_with_fast_reward)
        logger.debug("Whether to disable tqdm: %s", disable_tqdm)
        if aggregator is not None:
            logger.debug("Aggregator: %s", aggregator)
        logger.debug("Node visualizer function: %s", node_visualizer.__name__ if hasattr(node_visualizer, '__name__') else str(node_visualizer))


    def iterate(self, node: MCTSNode) -> list[MCTSNode]:
        path = self._select(node)
        logger.info(f"Selected Node ID: {path[-1].id}")
        if not self._is_terminal_with_depth_limit(path[-1]):
            self._expand(path[-1])
            self._simulate(path)
        cum_reward = self._back_propagate(path)        
        if (
            self.output_strategy == "max_iter"
            and path[-1].is_terminal
            and cum_reward > self._output_cum_reward
        ):
            self._output_cum_reward = cum_reward
            self._output_iter = path
        if self.output_strategy == "last_iter":
            self._output_cum_reward = cum_reward
            self._output_iter = path
        if self.output_strategy == "last_terminal_iter" and path[-1].is_terminal:
            self._output_cum_reward = cum_reward
            self._output_iter = path
        return path

    def _is_terminal_with_depth_limit(self, node: MCTSNode):
        if node.is_terminal:
            logger.info(f"Node {node.id} is a terminal node")
            return True
        elif node.depth >= self.depth_limit:
            logger.info(f"Node {node.id} depth is {node.depth}, reached limit {self.depth_limit}, terminating search")
            node.is_terminal = True
            return True
        return False

    def _select(self, node: MCTSNode) -> list[MCTSNode]:
        path = []
        while True:
            logger.debug("[DEBUG] Current node id: %s", node.id)
            path.append(node)
            # Current path
            if node.children:
                logger.debug(f"[DEBUG] Current node\'s children ids: {[child.id for child in node.children]}", )
            if (
                node.children is None # Current node has not been expanded, self.children = None
                or len(node.children) == 0 # Current node is a leaf node, children list is empty
                or self._is_terminal_with_depth_limit(node) # Current node is a terminal node or depth limit reached
            ):
                return path
            node = self._uct_select(node) # Use UCT to select the next node
            
    # def _uct(self, node: MCTSNode) -> float:
    #     return node.Q + self.w_exp * np.sqrt(
    #         np.log(len(node.parent.cum_rewards)) / max(1, len(node.cum_rewards))
    #     )

    def _uct(self, node: MCTSNode) -> float:
        return node.Q + self.w_exp * math.sqrt(math.log(node.parent.N) / (1 + node.N))

    # def _uct_select(self, node: MCTSNode) -> MCTSNode:
    #     if self.uct_with_fast_reward or all(x.state is not None for x in node.children):
    #         return max(node.children, key=self._uct)
    #     else:
    #         unvisited_children = filter(lambda x: x.state is None, node.children)
    #         return max(unvisited_children, key=lambda x: x.fast_reward)

    def _uct_select(self, node: MCTSNode) -> MCTSNode:
        # First, filter out terminal nodes
        non_terminal_children = [child for child in node.children if not child.is_terminal]

        if not non_terminal_children:
            # If all child nodes are terminal nodes, select the first child node (or you can return None or raise an exception)
            return node.children[0] if node.children else node # Fallback to node itself if no children

        # If non-terminal child nodes exist, perform UCT selection among them
        return max(non_terminal_children, key=self._uct)

    def _expand(self, node: MCTSNode):
        logger.info(f"Expanding node {node.id}")

        if node.state is None and self.world_model is not None:
            node.state, _ = self.world_model.step(
                node.parent.state, node.action, node.id
            )
            is_terminal, goal_reward = self.world_model.is_terminal(node)
            
            # Get format_reward
            # node.action is of type Action (e.g. RobotAction)
            # node.action.action can be list[ActionChoice] or a single ActionChoice
            if node.action and hasattr(node.action, 'action'):
                action_detail_or_list = node.action.action
                if isinstance(action_detail_or_list, list):
                    if action_detail_or_list:
                        node.format_reward = action_detail_or_list[0].format_reward
                    else:
                        node.format_reward = 0.0
                        logger.warning(f"Node {node.id}: node.action.action is an empty list. Setting format_reward to 0.0.")
                else:
                    node.format_reward = action_detail_or_list.format_reward
            else:
                node.format_reward = 0.0
                logger.warning(f"Node {node.id}: node.action or node.action.action is missing. Setting format_reward to 0.0.")

            node.is_terminal = is_terminal
            node.goal_reward = goal_reward
            node.reward = goal_reward - node.parent.goal_reward

            logger.info(f"[Expanding] Node {node.id} has format reward {node.format_reward}")
            logger.info(f"[Expanding] Node {node.id} has goal reward {goal_reward - node.parent.goal_reward}")
            logger.info(f"[Expanding] Node {node.id} has reward {node.reward}")

        if node.is_terminal:
            return

        logger.info(f"[Expanding] Getting action list for node {node.id}")

        if self.enable_reflection and self._should_reflect(node):
            logger.info(f"[Reflection triggered] Node {node.id} triggered reflection, calling get_actions to generate reflection action")
            actions = self.search_config.get_actions(node.state, logprobs=True, force_prefix_think=True, reflection_prefix="<think>Wait...")
            node.reflection_root = True 
        else:
            actions = self.search_config.get_actions(node.state, logprobs=True)

        if actions is None or len(actions) == 0:
            logger.info(f"[Expanding] Failed to get action list for node {node.id}, marking as terminal node")
            node.is_terminal = True
            node.reward = -1.0
            return

        children = []
        for action in actions:
            fast_reward, fast_reward_details = self.search_config.fast_reward(node.state, action)
            child = MCTSNode(
                state=None,
                action=action,
                parent=node,
                fast_reward=fast_reward,
                fast_reward_details=fast_reward_details,
                calc_q=self.calc_q,
            )
            children.append(child)
        node.children = children


    def _simulate(self, path: list[MCTSNode]):
        node = path[-1]
        logger.info("Simulation Start.")
        while True:
            logger.info(f"[Simulating] Current node id: {node.id}")
            if node.state is None:
                self._expand(node)
            if self._is_terminal_with_depth_limit(node) or len(node.children) == 0:
                return
            fast_rewards = [child.fast_reward for child in node.children]
            logger.info(f"[Simulating] fast rewards: {fast_rewards}")
            node = node.children[self.simulate_choice(fast_rewards)]
            path.append(node)

    # def _back_propagate(self, path: list[MCTSNode]):
    #     rewards = []
    #     cum_reward = -math.inf
    #     for node in reversed(path):
    #         rewards.append(node.reward)
    #         cum_reward = self.cum_reward(rewards[::-1])
    #         node.cum_rewards.append(cum_reward)
    #     return cum_reward

    def _back_propagate(self, path: list[MCTSNode]):
        logger.info("Back Propagating")
        logger.info(f"[Back Propagating] Backpropagation started, path length: {len(path)}")
        state_return = 0
        for node in reversed(path):
            logger.info(f"[Back Propagating] Current node ID: {node.id}, old Q value: {node.Q}, old visit count: {node.N}")
            # Update the Q value of the node, Q = (Q * N + action_reward + V_{i+1}) / (N + 1)
            logger.info(f"[Back Propagating] Current node ID: {node.id}, reward: {node.reward}")
            logger.info(f"[Back Propagating] Total reward of previous path: {state_return}")
            # G_{i} = action_reward + gamma * G_{i+1}
            state_return = node.reward + node.gamma * state_return
            # Q = (Q * N + G_{i}) / (N + 1)
            node.Q = round((node.Q * node.N + state_return) / (node.N + 1), 3)
            # Increment the visit count of the node
            node.N += 1
            logger.info(f"[Back Propagating] Current node ID: {node.id}, new Q value: {node.Q}, new visit count: {node.N}")
        # Return the updated Q value of the root node
        logger.info(f"[Back Propagating] Backpropagation ended, root node's updated Q value: {path[0].Q}")
        return path[0].Q

    def _dfs_max_reward(self, path: list[MCTSNode]) -> tuple[float, list[MCTSNode]]:
        cur = path[-1]
        if cur.is_terminal:
            return self.cum_reward([node.reward for node in path[1:]]), path
        if cur.children is None:
            return -math.inf, path
        visited_children = [x for x in cur.children if x.state is not None]
        if len(visited_children) == 0:
            return -math.inf, path
        return max(
            (self._dfs_max_reward(path + [child]) for child in visited_children),
            key=lambda x: x[0],
        )

    def search(self, instruction, image_path):
        """
        Execute the MCTS search algorithm
        
        Args:
            instruction: User instruction, used to initialize the root node state
            image_path: Image path, used to initialize the root node state
        """
        # Initialize the output cumulative reward to negative infinity, indicating that no valid path has been found yet
        self._output_cum_reward = -math.inf
        # Initialize the output iteration path to empty
        self._output_iter = []
        if self.world_model is None or self.search_config is None:
            raise ValueError("World model and search config must be set before search")
        # Create the root node, initialize the state
        self.root = MCTSNode(
            state=self.world_model.init_state(instruction, image_path),  # Use the world model to initialize the state
            action=None,  # The root node has no corresponding action
            parent=None,  # The root node has no parent
            calc_q=self.calc_q,  # Use the configured Q value calculation method
        )
        # If trace for each iteration needs to be recorded, initialize the trace list
        if self.output_trace_in_each_iter:
            self.trace_in_each_iter = []

        # Execute n_iters times of MCTS iterations
        for iter in trange(
            self.n_iters, disable=self.disable_tqdm, desc="MCTS iteration", leave=False
        ):
            state = StateManager()
            if state.get("evaluate", False, thread_local=False) and state.get("max_gc_success", 0.0) == 1.0:
                break
            state.set("mcts_iter", iter)
            logger.info(f"Starting {iter}-th iteration")
            
            # Check if there is only one path
            def has_single_path(node: MCTSNode) -> bool:
                if not node.children:
                    return True
                if len(node.children) != 1:
                    return False
                return has_single_path(node.children[0])
            
            # Check if the current iteration has only one path
            if has_single_path(self.root):
                _single_path_count = state.get("_single_path_count", 0)
                _single_path_count += 1
                state.set("_single_path_count", _single_path_count)
                logger.info(f"Detected that the tree has only one path, this is the {_single_path_count}-th consecutive time")
                # Stop early only if there is only one path for two consecutive times
                if _single_path_count >= 2:
                    logger.info("Only one path for two consecutive iterations, stopping iteration early")
                    break
            else:
                # Reset counter
                state.set("_single_path_count", 0)
                
            # Execute one iteration, get the path from the root node to the leaf node
            path = self.iterate(self.root)
            # If trace for each iteration needs to be recorded, deep copy the current path and add it to the trace list
            if self.output_trace_in_each_iter:
                self.trace_in_each_iter.append(deepcopy(path))
            
        # Process search results based on the output strategy
        # If the strategy is "follow_max", start from the root node and select the child node with the highest reward each time
        if self.output_strategy == "follow_max":
            self._output_iter = []
            cur = self.root
            while True:
                self._output_iter.append(cur)  # Add the current node to the output path
                if cur.is_terminal:  # If the current node is a terminal node, end the loop
                    break
                # Get all visited child nodes (state is not None)
                visited_children = [x for x in cur.children if x.state is not None]
                if len(visited_children) == 0:  # If there are no visited child nodes, end the loop
                    break
                # Select the child node with the highest reward as the next node
                cur = max(visited_children, key=lambda x: x.reward)
            # Calculate the cumulative reward of the output path
            self._output_cum_reward = self.cum_reward(
                [node.reward for node in self._output_iter[1::-1]]
            )
        # If the strategy is "max_reward", use depth-first search to find the path with the highest reward
        if self.output_strategy == "max_reward":
            self._output_cum_reward, self._output_iter = self._dfs_max_reward(
                [self.root]
            )
            # If no valid path is found (cumulative reward is negative infinity), set the output path to None
            if self._output_cum_reward == -math.inf:
                self._output_iter = []
                # If no valid path is found, set the strategy to "max_visit"
                self.output_strategy = "max_visit"
        # If the strategy is "max_visit", select the child node with the most visits as the output path
        if self.output_strategy == "max_visit":
            # Check if the root node has child nodes
            if self.root.children:
                self._output_iter = [
                    max(self.root.children, key=lambda x: x.N)
                ]  # The node with the most visits among the child nodes of the root node
            else:
                self._output_iter = []  # If there are no child nodes, set the output path to an empty list

    def __call__(
        self,
        instruction: str,
        world_model: WorldModel[State, Action],
        search_config: SearchConfig[State, Action],
        image_path: Optional[str] = None,
        **kwargs,
    ) -> MCTSResult:
        MCTSNode.reset_id()
        self.world_model = world_model
        self.search_config = search_config
        logger.info(f"[MCTS] Starting task: {instruction}")

        self.search(instruction, image_path)
        result = self._build_result()
        return result

    def _should_reflect(self, node: MCTSNode, window_size: int = 5) -> bool:
        """Checks if goal_reward has not improved in the last `window_size` steps; stops backtracking upon encountering a reflection breakpoint."""
        recent_nodes = []
        cur = node
        for _ in range(window_size):
            # Stop if a reflection breakpoint is encountered or the root node is reached
            if cur.parent is None or getattr(cur, "reflection_root", False):
                break
            cur = cur.parent
            recent_nodes.append(cur)

        if len(recent_nodes) < window_size:
            return False

        first_reward = recent_nodes[-1].goal_reward
        return all(n.goal_reward <= first_reward for n in recent_nodes[:-1])


    def _build_result(self) -> MCTSResult:
        if not self._output_iter:
            terminal_state = trace = None
        else:
            terminal_state = self._output_iter[-1].state
            trace = (
                [node.state for node in self._output_iter],
                [node.action for node in self._output_iter[1:]],
            )

        trace_in_each_iter = self.trace_in_each_iter if self.output_trace_in_each_iter else None
        tree_state_after_each_iter = (
            [trace[0] for trace in self.trace_in_each_iter]
            if self.output_trace_in_each_iter else None
        )

        return MCTSResult(
            terminal_state=terminal_state,
            cum_reward=self._output_cum_reward,
            trace=trace,
            trace_of_nodes=self._output_iter,
            tree_state=self.root,
            trace_in_each_iter=trace_in_each_iter,
            tree_state_after_each_iter=tree_state_after_each_iter,
            aggregated_result=self.aggregator(self.root) if self.aggregator else None,
            max_gc_success_per_iter=StateManager().get("max_gc_success_per_iter", {})
        )
    
    