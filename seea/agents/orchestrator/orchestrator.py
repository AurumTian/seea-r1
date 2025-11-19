import json
from typing import Dict, List
from seea.agents.agent.chat_agent import ChatAgent
from seea.agents.models.models import *
from seea.utils.common import clean_json_output

class Orchestrator:
    def __init__(
        self,
        stage_to_agent_map: Dict[Stage, ChatAgent],
        stage_transition_graph: Dict[Stage, Stage]
    ):
        self.stage_to_agent_map = stage_to_agent_map
        self.stage_transition_graph = stage_transition_graph
        self.memory = None

    def run(self, insturction):
        return self.execute_command(insturction)

    def execute_command(self, insturction: str):
        try:
            # Create initial memory
            self.memory = Memory(
                history=[{"role": "user", "content": insturction}],
                current_stage=Stage.ACTOR,
                thought='',
                proposed_actions=[],
                consideration='',
                best_action_index=-1,
                observation=None,
                reflection='',
                state=State.CONTINUE,
            )

            print(f"Orchestrator User insturction {insturction}")
            while self.memory.state != State.SUCCESS:
                message = self._handle_Stage()
            return message
        except Exception as e:
            print(f"Orchestrator Error processing the insturction {insturction}: {e}")

    def _handle_Stage(self):
        if self.memory.current_stage not in self.stage_to_agent_map:
            raise ValueError(f"Orchestrator Unhandled stage! No agent for {self.memory.current_stage}")
        agent = self.stage_to_agent_map[self.memory.current_stage]
        messages = agent.run(self.memory.history)
        self.update_memory(messages)
        return messages[-1]
    
    def update_memory(self, messages):
        self.memory.history.extend(messages)
        output = messages[-1].get('content', '')
        if isinstance(output, list):
            flag = False
        else:
            output, flag = clean_json_output(output)
        output = json.loads(output) if flag else output
        if self.memory.current_stage == Stage.ACTOR and flag:
            output = ActorOutput.model_validate(output)
            self.memory.thought = output.thought
            self.memory.proposed_actions = output.proposed_actions
        elif self.memory.current_stage == Stage.CRITIC and flag:
            output = CriticOutput.model_validate(output)
            self.memory.consideration = output.consideration
            self.memory.best_action_index = output.best_action_index
        elif self.memory.current_stage == Stage.SCORE and flag:
            output = ScoreOutput.model_validate(output)
            self.memory.reflection = output.reflection
            self.memory.state = output.state
        elif self.memory.current_stage == Stage.EXECUTOR:
            if flag:
                output = ExecutorOutput.model_validate(output)
                self.memory.observation = output
            elif isinstance(output, list):
                description = ''
                images = []
                video = None
                for element in output:
                    if element.get('type') == "text":
                        description += element.get("text")
                    elif element.get('type') == "image_url":
                        images.append(element.get("image_url").get("url"))
                    elif element.get('type') == "video":
                        video = element.get("video")
                self.memory.observation = Observation(description=description, images=images or None, video=video)
            else:
                self.memory.observation = Observation(description=output, images=None, video=None)
        self.memory.current_stage = self.stage_transition_graph[self.memory.current_stage]