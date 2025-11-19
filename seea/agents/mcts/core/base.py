from abc import ABC, abstractmethod
from typing import Generic, Protocol, Tuple, TypeVar, Union, runtime_checkable

State = TypeVar("State")
Action = TypeVar("Action")
Observation = TypeVar("Observation")
Trace = tuple[list[State], list[Action]]


class WorldModel(ABC, Generic[State, Action]):
    def __init__(self) -> None: ...

    @abstractmethod
    def init_state(self, **kwargs) -> State: ...

    @abstractmethod
    def step(
        self, state: State, action: Action, **kwargs
    ) -> Union[State, Tuple[State, Observation]]:
        """Returns the next state and optionally the observation

        :param state: The current state
        :param action: The action to take
        :param kwargs: Additional arguments
        :return: The next state and optionally the observation
        """
        ...

    @abstractmethod
    def is_terminal(self, node) -> tuple[bool, float]: ...



class SearchConfig(ABC, Generic[State, Action]):
    def __init__(self) -> None:
        ...

    @abstractmethod
    def get_actions(self, state: State) -> list[Action]:
        pass

    @abstractmethod
    def fast_reward(self, state: State, action: Action) -> tuple[float, dict]:
        return 0, {}

    @abstractmethod
    def reward(self, state, action, **kwargs) -> tuple[float, dict]:
        pass



@runtime_checkable
class AlgorithmOutput(Protocol[State]):
    terminal_state: State
    trace: Trace


class SearchAlgorithm(ABC):
    def __init__(self, **kwargs): ...

    @abstractmethod
    def __call__(
        self, instruction: str, world_model: WorldModel, search_config: SearchConfig, **kwargs
    ) -> AlgorithmOutput: ...


class Reasoner(ABC, Generic[State, Action]):
    def __init__(
        self,
        world_model: WorldModel[State, Action],
        search_config: SearchConfig[State, Action],
        search_algo: SearchAlgorithm,
    ) -> None:
        self.world_model = world_model
        self.search_config = search_config
        self.search_algo = search_algo

    def __call__(
        self, instruction: str, **kwargs
    ) -> AlgorithmOutput[State]:
        return self.search_algo(instruction, self.world_model, self.search_config, **kwargs)
