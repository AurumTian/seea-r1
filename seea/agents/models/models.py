from enum import Enum
from typing import List, Literal, Optional, Union, Any

from pydantic import BaseModel
from pydantic.fields import Field

class VideoQualityResult(BaseModel):
    """Video quality evaluation result model"""
    success: bool = Field(description="Whether the video is successfully completed")
    quality_score: float = Field(description="Video quality score (0-10)")
    feedback: str = Field(description="Detailed evaluation feedback")

# Global
class Stage(str, Enum):
    COMPLETED = "completed"
    ACTOR = "actor"
    CRITIC = "critic"
    STEP = "step"
    SORTOR = "sortor"


class State(str, Enum):
    SUCCESS = 'success'
    FAILURE = 'failure'
    CONTINUE = 'continue'


class Observation(BaseModel):
    images: Optional[List[str]]
    video: Optional[str]
    description: Optional[str]

# Actor
class Action(BaseModel):
    name: str
    arguments: dict
    complete_response: Optional[str]
    format_reward: Optional[float] = Field(description="The reward for this action, higher is better")


class ActorOutput(BaseModel):
    think: Optional[str]
    action: List[Action]
    answer: Optional[str]
    complete_response: Optional[str]
    format_reward: Optional[float] = Field(description="The reward for this action, higher is better")


# Sortor
class SortorOutput(BaseModel):
    consideration: str
    best_action_index: int

# Executor
class ExecutorOutput(BaseModel):
    observation: Observation


# Dreamer
class DreamerIntput(BaseModel):
    images: Optional[List[str]]
    action: str
    state_after_action: State


class DreamerOutput(BaseModel):
    imagination: str


# Critic
class CriticOutput(BaseModel):
    reflection: str
    state: State
    reward: float

class AvailabilityCheck(BaseModel):
    available: Literal["yes", "no"]
    instruction: str

# Monte-Carlo
class RobotStatus(str, Enum):
    Available = "Available",
    Running = "Running",
    Error = "Error"


class Memory(BaseModel):
    history: List[dict]
    current_stage: Stage
    think: Optional[str]
    proposed_actions: List[Any]
    consideration: Optional[str]
    best_action_index: Optional[int]
    observation: Optional[Observation]
    reflection: Optional[str]
    state: State
    reflection_count: int = 0
    critic_state: Optional[dict]


class RobotState(BaseModel):
    """Robot state model"""
    memory: Memory
    status: RobotStatus
    task: str
    hands_status: List[dict]
    skill_status: List[dict]
    plan: List[dict]
    id: Optional[int]
    action_history: List[dict]
    expand_times: int = 0

class RobotAction(BaseModel):
    action: List[Action] = Field(description="The action to be taken")
    rank: float = Field(description="The rank of this action, higher is better")
    prob: float = Field(description="The probability of this action, higher is better")