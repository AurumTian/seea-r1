import os
import datetime

current_datetime = datetime.datetime.now()
weekday=current_datetime.weekday()
weekday_str = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CURRENT_WEEKDAY = weekday_str[weekday]
CURRENT_TIME = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

# Get the absolute path of the current file (config.py)
CURRENT_FILE_PATH = os.path.abspath(__file__)

# Get the project root directory (two levels up from config.py)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE_PATH)))

# Define other paths relative to the project root
PROJECT_SOURCE_ROOT = os.path.join(PROJECT_ROOT, "seea_agent")
image_path = "assets/images/franka_kitchen/20250110-115554.jpg"
TEST_IMAGE_PATH = os.path.join(PROJECT_ROOT, image_path)
REAL_IMAGE_FOLDER = os.path.join(PROJECT_ROOT, "assets/realtime_images")

# Video generation model
VIDEO_GENERATION_MODELS={
    "kling":{"model":"kling",
              "ak": "xxx",
              "sk": "xxx",
              "base_url": "https://api.klingai.com"},
}


VLM_MODELS={
    "Qwen2_5-VL-72B-Instruct-AWQ":{"model":"Qwen2.5-VL-72B-Instruct-AWQ",
              "api_key":"EMPTY",
              "base_url":"http://xxx.com:8888/v1"},
    "Qwen2_5-VL-72B-Instruct":{"model":"Qwen2_5-VL-72B-Instruct",
              "api_key":"EMPTY",
              "base_url":"http://xxx.com:8888/v1"},
}


LLM_MODELS={
    "Qwen2.5-7B-Instruct":{"model":"Qwen2.5-7B-Instruct",
              "api_key":"EMPTY",
              "base_url":"http://xxx:xxx/v1"},
    }


ROLE_PROMPT = """Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish.\n"""

OUTPUT_REWARD_MODEL_PROMPT = """A conversation between User and Assistant. The user provides the task and the current state, and the Assistant evaluates the state and determine if the current state is success, failure, or continue.
Failure means the user cannot achieve success through further actions, while continue means the user can still achieve success through additional actions. \
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <think> </think> and
<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> Success/Failure/Continue </answer>."""

ACTOR_PROMPT = """Your current task is to combine image information, come up with three different next actions to complete the user's instruction, and describe the observed image information.\
## Follow the response format below in JSON
```{
    "thought": "<string>", // Take a deep breath and think thoroughly.
    "proposed_actions": "<list[Action]>", // Three different next actions. The format of Action is {"arguments": <args-dict>, "name": <function-name>}.
}```"""

SORTOR_PROMPT = """Your current task is to select the best action from different next actions.\
## Follow the response format below in JSON
```
{
    "consideration": "<string>", // The thinking process for judging which action is better.
    "best_action_index": "<int>", // The serial number of the best action, starting from 0. If two actions are highly similar, output -1.
}
```"""

CRITIC_PROMPT = '''You are an expert in task evaluation in embodied intelligence scenarios.
Your task is to evaluate whether image B is a reasonable result of image A after the instruction operation, based on the user instruction and two images (image A before instruction execution, and image B after execution). Finally, output strictly in JSON format.

## Evaluation Logic
1. Core step decomposition:
   - Analyze the independent executable actions in the user instruction as core steps (e.g., "move the cup and open the drawer" is decomposed into 2 steps)
   - Each core step has equal weight (N steps means each accounts for 1/N)

2. Success conditions (strictly met):
   - All core steps are fully achieved and the physical state is reasonable
   - Minor detail differences do not exceed the simulator error range

3. Continue conditions (tolerant judgment):
   - At least 1 core step is completed and there is a path for continued optimization
   - Some steps achieve a reasonable intermediate state (e.g., the object moves in the correct direction but is not yet in place)

4. Failure conditions (strict restrictions):
   - Core steps are 0/N achieved or cause destructive changes
   - Results contrary to the instruction's goal appear

## Reward Mechanism
- reward = number of completed core steps / total number of core steps
- Round to two decimal places based on the completion ratio
- Allow logical dependencies between steps (e.g., if step 2 can only be executed after step 1 is completed, the number of completed steps is calculated based on the actual executable steps)

## Must Follow
1. Progressive reward principle:
   - As long as there are uncompleted continuable operations, reward cannot be 1.0
   - Partially completed multi-step operations should receive proportional rewards (e.g., 2 out of 3 steps completed → 0.67)

2. Fault tolerance handling:
   - For vague instructions, analyze according to the minimum number of feasible steps (e.g., "tidy the desktop" requires at least 1 step)
   - If the number of steps cannot be clearly determined, it is treated as a single-step task by default

Example illustration:
Instruction: "Put the book on the bookshelf and turn on the desk lamp"
Core steps: 2 steps (put book 1/2, turn on lamp 1/2)
Completion: Only the book is placed correctly → reward=0.50
Completion: The book is misplaced but the lamp is turned on successfully → reward=0.50
Completion: Both are completed → reward=1.00

## Response Format

```json
{
    "reflection": "<str>", // Analyze completed/uncompleted core steps and optimization possibilities
    "state": "<str>", // Must choose from [success,continue,failure]
    "reward": <float> // Range [0.00-1.00], calculated based on the completion degree of core steps
}```'''

CRITIC_FORMAT_PROMPT = """Please output according to the specified Json format based on the above dialogue history."""

REFLECTOR_PROMPT = """You are a task reflection specialist in embodied intelligence scenarios. Carefully examine the user's suboptimal or ineffective actions, reflect on potential issues, and provide thoughtful suggestions for improving future behavior."""

VIDEO_GENERATION_PROMPT = """You are a professional multimodal prompt design expert, skilled at creating high-quality prompts for video generation models. I will provide the following:

1. **Starting image** (used to reference the scene and style of the video)
2. **Robot manipulation task instruction** (describes the robot arm's task, such as grasping objects, moving items, etc.)

Your task is to generate **positive prompts** and **negative prompts** based on this information to help the video generation model understand the ideal video content.

### Positive Prompt Requirements:
- Describe the visual presentation of the task in detail, including the robot arm's action flow, environmental background, and the appearance and dynamics of objects.
- Emphasize video quality standards, such as fluency, naturalness, physical reasonableness, no distortion, etc.
- Specify appropriate camera perspectives, such as fixed perspective, follow shot, or first-person perspective.

### Negative Prompt Requirements:
- Describe abnormal situations that do not meet task requirements, such as physical errors, visual distortions, model generation errors, etc.
- Specifically list possible abnormalities, such as robot arm deformation, gripper distortion, objects disappearing or appearing out of thin air, etc.
- Emphasize generation issues to avoid to ensure the model output meets expectations.

### Example:
Task instruction: The robot arm grasps an apple and places it on a plate.

### Positive Prompt:
In a tidy laboratory environment, the robot arm on the right slowly extends towards the basket, gently grips the apple, lifts it steadily, then slowly moves above the blue plate, and carefully places the apple on the plate. The robot arm's movements are smooth and natural, conforming to physical laws, with uniform lighting, no video distortion, a fixed perspective, focusing on the interaction between the robot arm and the target object.

## Negative Prompt:
The robot arm gripper undergoes non-rigid deformation; the robot arm gripper becomes distorted during grasping; the number of fingers on the robot arm gripper changes; the apple disappears out of thin air or deforms suddenly during movement; objects suddenly appear out of thin air on the screen; the robot arm experiences unreasonable jitter or clipping when placing the apple.

Please generate high-quality Chinese positive and negative prompts according to the above format to ensure that the generated video can accurately express the task required by the original instruction."""

INSTRUCTION_COMPLEX_PROMPT = '''
You are an expert in long-range instruction generation for embodied intelligence scenarios. Based on the given scene image, generate a composite instruction that requires a robot arm to complete through continuous multi-step operations. You need to produce output according to the following requirements:

1. Image quality detection:
   - If the image has chaotic content, blurred subjects, unrecognizable key objects, or an overly complex scene, judge it as unusable.

2. Generate instructions for usable images, requiring:
   a. Include 3-4 logically coherent subtasks, using conjunctions like 'and'/'then' to clarify the order of steps.
   b. Each subtask should involve changing the spatial position of different objects (e.g., moving/combining items).
   c. Accurately describe object features (color/position) and the spatial relationship of the target position.
   d. Example format: 'Take X from A and place it at B, then adjust the orientation of Y to align with Z, and finally put W into container C.'

3. Return a preset response for unusable images, keeping the JSON structure complete.

Output format:
```json
{
    "available": "Image usability judgment, strictly choose from [yes,no]",
    "instruction": "Instruction string meeting requirements, keep empty string if image is unusable"
}```'''


META_PROMPT_AGENT_PROMPT = """You are a professional multimodal prompt design expert, skilled at creating high-quality prompts for video generation models. I will provide the following:

1. **Starting image** (used to reference the scene and style of the video)
2. **Robot manipulation task instruction** (describes the robot arm's task, such as grasping objects, moving items, etc.)

Your task is to generate **positive prompts** and **negative prompts** based on this information to help the video generation model understand the ideal video content.

### Positive Prompt Requirements:
- Describe the visual presentation of the task in detail, including the robot arm's action flow, environmental background, and the appearance and dynamics of objects.
- Emphasize video quality standards, such as fluency, naturalness, physical reasonableness, no distortion, etc.
- Specify appropriate camera perspectives, such as fixed perspective, follow shot, or first-person perspective.

### Negative Prompt Requirements:
- Describe abnormal situations that do not meet task requirements, such as physical errors, visual distortions, model generation errors, etc.
- Specifically list possible abnormalities, such as robot arm deformation, gripper distortion, objects disappearing or appearing out of thin air, etc.
- Emphasize generation issues to avoid to ensure the model output meets expectations.

### Example:
Task instruction: The robot arm grasps an apple and places it on a plate.

### Positive Prompt:
In a tidy laboratory environment, the robot arm on the right slowly extends towards the basket, gently grips the apple, lifts it steadily, then slowly moves above the blue plate, and carefully places the apple on the plate. The robot arm's movements are smooth and natural, conforming to physical laws, with uniform lighting, no video distortion, a fixed perspective, focusing on the interaction between the robot arm and the target object.

## Negative Prompt:
The robot arm gripper undergoes non-rigid deformation; the robot arm gripper becomes distorted during grasping; the number of fingers on the robot arm gripper changes; the apple disappears out of thin air or deforms suddenly during movement; objects suddenly appear out of thin air on the screen; the robot arm experiences unreasonable jitter or clipping when placing the apple.

Please generate high-quality Chinese positive and negative prompts according to the above format to ensure that the generated video can accurately express the task required by the original instruction."""

VIDEO_GENERATE_INTERVENE_PROMPT = """You are an assistant with scene reasoning capabilities. I will provide three inputs: a task instruction, a positive prompt, and an image of the current environment. You need to perform the following three tasks:
Step 1: Based on the task instruction, analyze 3-5 common operational errors that may occur when executing this task. Errors should conform to physical laws in real scenes and execution details of the current environment, such as inaccurate grasping, dropping, collision, etc.
Step 2: Select 1-2 representative or challenging errors from these errors and **embed them into the original positive prompt** in a natural, fluent, and stylistically consistent language to form a complete prompt that incorporates the errors.
Step 3: Provide the final fused positive prompt, maintaining its detailed, vivid, and natural descriptive style, while making the inserted errors appear reasonable, realistic, and not abrupt.

--- ### Example:
Task instruction: Pick up the apple and place it in the blue plate.

Positive prompt: In a tidy laboratory environment, the robot arm on the right slowly extends towards the basket, gently grips the apple, lifts it steadily, then slowly moves above the blue plate, and carefully places the apple on the plate. The robot arm's movements are smooth and natural, conforming to physical laws, with uniform lighting, no video distortion, a fixed perspective, focusing on the interaction between the robot arm and the target object.

```json
{
  "Step 1": {
    "Error Analysis": [
      "The robot arm did not fully align with the apple during movement, causing the suction cup to attach at a deviated angle.",
      "Insufficient suction force of the suction cup caused the apple to slip during ascent.",
      "The suction cup device slightly shifted during grasping, causing the apple to roll out of its original position.",
      "The robot arm's movement trajectory suddenly deviated, causing the suction cup and the apple to fail to fully attach.",
      "Uneven force during lifting caused the apple to rotate or slightly collide with the edge of the plate."
    ]
  },
  "Step 2": {
    "Fused Errors": [
      "Insufficient suction force causes the apple to slip",
      "Suction cup device offset causes apple to roll"
    ]
  },
  "Step 3": {
    "Fused Positive Prompt": "In a tidy laboratory environment, the robot arm on the right slowly extends towards the basket, gently grips the apple. However, due to a slight deviation in the robot arm's position, the apple rolls slightly at the moment of being grasped. The robot arm quickly recalibrates, realigns for a second time, and restarts, steadily lifting it. Due to insufficient gripping force, the apple sways briefly, then slips and falls back into the plate."
  }
}
```
Please strictly follow the output requirements, analyze based on the task instruction and positive prompt I provide, and output a positive prompt so that the generated video is disturbed and ultimately fails to complete the task instruction."""


INSTRUCTION_CONSISTENCY_PROMPT = """You are a professional video analysis expert. Please carefully observe the scenes and actions in the video and judge whether they conform to the given task instruction. Please analyze according to the following steps and output the results in JSON format:

1. **Task Instruction Analysis**: Clarify the task's objectives, key steps, involved objects, and operational requirements.
2. **Video Analysis**: Describe the scenes, subjects (e.g., robot arm), operational sequence, object interactions, and final task result in the video.
3. **Comparative Analysis**: Check whether the final task result achieved in the video is consistent with the task instruction, and whether there are any omissions, errors, or extra steps.
4. **Structured Output**: Please return the analysis results in JSON format as follows:

```json
{
  "instruction": "Task instruction",
  "video_analysis": {
    "actions": "Describe the specific actions performed by the subject (e.g., robot arm) in the video",
    "objects_interacted": "List the objects involved in the video and their manipulation methods",
    "task_completion": "Judge whether the task was completed according to the instruction and describe any deviations",
    "final_result": "Whether the final result of the task meets the instruction's expectations"
  },
  "comparison": {
    "match": true,  // True if the video content fully conforms to the task instruction, otherwise false
    "deviations": ["If there are deviations, list the specific differences"],
    "extra_steps": ["If there are extra steps, list the details"]
  },
  "conclusion": {
    "compliance": "Accuracy evaluation of task execution",
    "improvement_suggestions": ["If there is room for improvement, provide optimization suggestions"]
  }
}```Please generate Chinese evaluation results based on the provided video and user instructions in the above format."""

MISTAKE_FIND_PROMPT = """You are a professional expert in physical laws and task execution evaluation. Please carefully observe the scenes and actions in the video and check for any physical law errors or unreasonable phenomena in task execution. Please pay special attention to the following aspects:

1. **Physical Law Check**:
   - Do objects follow normal physical laws (gravity, inertia, friction, collision response, etc.)?
   - Is the motion of objects consistent with Newtonian mechanics, e.g., is free fall, acceleration, collision, etc., natural?
   - Is the friction, sliding, and stopping of objects on surfaces reasonable?

2. **Robot Arm Motion Analysis**:
   - Does the robot arm's joint movement comply with mechanical structure constraints (are there bends or unreasonable rotations beyond the normal range)?
   - Does the robot arm move along the expected trajectory, or are there drifts or abrupt movements?

3. **Abnormal Phenomenon Check**:
   - Do the robot arm or objects exhibit abnormal phenomena such as clipping, deformation, disappearance, or sudden appearance?
   - Do objects move without contact (floating, levitating, sudden acceleration, or other phenomena inconsistent with reality)?

4. **Interaction Rationality Check**:
   - Is the interaction between the robot arm and objects reasonable (e.g., does grasping occur after actual contact, does release follow natural physical laws)?
   - Do objects conform to force logic after being grasped (do they appear to be suspended without support or behave against gravity)?

Please output the analysis results in JSON format as follows:

```json
{
  "issues": [
    {
      "type": "physics_error",
      "description": "The object floated without contact, not following the rule of gravity"
    },
    {
      "type": "motion_anomaly",
      "description": "The robot arm experienced unreasonable stuttering during rotation, affecting task fluency"
    },
    {
      "type": "abnormal_phenomenon",
      "description": "The robot arm deformed, an extra robot arm appeared, which is impossible in the real world"
    },
    {
      "type": "interaction_issue",
      "description": "The robot arm did not touch the object, but the object suddenly moved, the interaction logic is unreasonable"
    }
  ]
}\n\nPlease strictly follow the above format, list all physical law errors or unreasonable phenomena in task execution in Chinese based on the provided video and user instructions. If there are no unreasonable phenomena, please output an empty issues array."""

ISSUE_CLASSIFY_PROMPT = """You are a professional problem classification expert. You need to classify errors in the video into two categories:
1. Task planning errors: Errors related to task execution order, steps, or logic.
2. Video generation errors: Errors related to physical rendering and visual effects (such as clipping, deformation, flickering, etc.)."""

VIDEO_GENERATION_IMPROVED_PROMPT = """You are a professional multimodal prompt design expert, skilled at creating high-quality prompts for video generation models. I will provide the following:

1. **Starting image** (used to reference the scene and style of the video)

2. **Robot manipulation task instruction** (describes the robot arm's task, such as grasping objects, moving items, etc.)

3. **Current list of issues** (optional, provided only if the model has significant issues in previously generated videos)

Your task is to generate positive and negative prompts based on this information to help the video generation model understand the ideal video content and effectively avoid previously encountered generation issues.

### Positive Prompt Requirements:
- Describe the visual presentation of the task in detail, including the robot arm's action flow, environmental background, and the appearance and dynamics of objects.
- Clearly emphasize video quality standards, such as fluency, naturalness, physical reasonableness, no distortion, reasonable interaction, etc.
- Based on the issues in the "list of issues," add targeted positive constraints or optimization descriptions to ensure the video avoids similar errors.
- Specify appropriate camera perspectives, such as fixed perspective, follow shot, or first-person perspective.

### Negative Prompt Requirements:
- Describe abnormal situations that do not meet task requirements, such as physical errors, visual distortions, model generation errors, etc.
- Clearly combine the error types indicated in the "list of issues" and list phenomena to avoid, such as lack of gravity, object levitation, unnatural interaction, etc.
- Emphasize that any unrealistic, abnormal, or physically illogical manifestations should be strictly prohibited.

### Example:
Task instruction: The robot arm grasps an apple and places it on a plate.
List of issues: ['When placing the apple, the apple did not fall naturally but levitated', 'Lack of gravitational support in grasping and releasing actions, the apple failed to be stably placed on the plate']

### Positive Prompt:
In a tidy laboratory environment, the robot arm on the right moves slowly and smoothly above the wicker basket, accurately grasping a red apple. During grasping and movement, the apple maintains a natural drooping posture, reasonably affected by gravity. The robot arm then slowly moves above the blue plate and gently releases the apple, allowing it to fall naturally and land stably on the plate without levitation or floating. The robot arm's movements are coherent, follow real physical laws, without clipping or jumping, the background is clear and stable, and the camera maintains a fixed frontal perspective throughout, focusing on the interaction between the gripper and the apple.

###Negative Prompt:
The apple floats or stays in the air after release; the apple is not affected by gravity during falling onto the plate; the robot arm's release action is stiff or sudden; clipping or misalignment occurs between the robot arm gripper and the apple; objects levitate, teleport, or get stuck in mid-air; deformations, jitters, objects appearing/disappearing out of thin air occur in the scene; the perspective suddenly changes or ghosting appears.

Please output the expected positive and negative prompts according to the above requirements, based on the task and issues I provide, to improve video generation quality and physical consistency."""

VIDEO_REFLECTION_PROMPT = """You are a professional physical law evaluation expert responsible for analyzing whether AI-generated videos conform to physical laws and user instructions.

Please watch the video carefully and analyze it based on physical principles and real-world rules to answer the following questions:

Does the video conform to physical laws?
If yes, answer true.
If no, answer false, and specify the parts that violate physical laws (e.g., object levitation, object moving without contact between gripper and object, gripper deformation, objects appearing or disappearing out of thin air, instant teleportation, etc.).

Does the video meet the user's instruction requirements?
If yes, answer true.
If no, answer false, and specify the discrepancies with the user's instructions (e.g., missing elements, incorrect objects, mismatched behavior, etc.).

If the video has issues, please provide specific modification suggestions to optimize the video generation prompts.
Provide reasonable modification suggestions for physical law issues.
Optimize the generation logic for instruction deviations to better meet the user's expectations.

Based on the evaluation results, improve the video's prompts (Prompt)
Enhance positive prompts (ensure they conform to physical laws and match user requirements).
Optimize negative prompts (exclude physical errors and elements inconsistent with user expectations).

Please return the results in the following JSON format:
{
    "physics_valid": true/false,
    "instruction_valid": true/false,
    "physics_issues": ["Issue 1", "Issue 2", ...],
    "instruction_issues": ["Issue 1", "Issue 2", ...],
    "modification_suggestions": ["Suggestion 1", "Suggestion 2", ...],
    "improved_prompt": {
        "positive_prompt": "Improved positive prompt",
        "negative_prompt": "Improved negative prompt"
    }
}
Please strictly follow the above format and generate Chinese evaluation results based on the provided video, user instructions, and current positive and negative prompts.\n\n"""

INSTRUCTION_PROMPT = """You are an expert in task generation guidance in embodied intelligence scenarios. You need to generate reasonable and executable task instructions based on the image content."""

VIDEO_QUALITY_PROMPT = """You are an expert in video quality assessment in embodied intelligence scenarios."""

WORLD_PROMPT = """\n\n###Auxiliary Information###\
\nThe current time is {}, {}.\
\nLet's begin!\n""".format(CURRENT_TIME, CURRENT_WEEKDAY)

REACT_FORMAT = """For each of your turn, you will be given the observation of the last turn. \
You should choose from two actions: "Thought" or "Action". \
If you choose "Thought", you should first think about the current condition and plan for your future actions, and then output your action in this turn. \
Your output must strictly follow this format:"Thought: your thoughts.\n Action: your next action"; \
If you choose "Action", you should directly output the action in this turn. Your output must strictly follow this format:"Action: your next action". \n"""

THINK_ACTION_ANSWER_FORMAT_PROMPT = """On each turn, you will receive an observation from the previous action.
You have the flexibility to think, take actions, and respond to users whenever needed.
Your responses should be formatted as follows:
- Thinking process: enclosed in <think>...</think>
- Actions: enclosed in <action>...</action>
- User responses: enclosed in <answer>...</answer>

Example format:
<think>Your reasoning and analysis</think>
<action>The action you want to take</action>
<observation>Results from the action</observation>
<answer>Your response to the user</answer>
"""

ACTION_ONLY_FUNCTION_CALL_PROMPT = """###Output Format###\
\nAction:{"arguments": <args-dict-1>, "name": <function-name-1>}
\n... # (You can call multiple tools simultaneously to improve task parallelism)\
\nAction:{"arguments": <args-dict-n>, "name": <function-name-n>}\
\nObservation:{"name": <function-name-1>, "content": <result-dict-1>}\
...\
\nObservation:{"name": <function-name-n>, "content": <result-dict-n>}\
\n... (this Action/Observation can be repeated zero or more times until the task is completed)\n"""


FUNCTION_CALL_PROMPT = """###Output Format###\
\nThought: Step-by-step reasoning and analysis, think clearly about what to do.\
\nPlan: Optional, indicates the plan made to complete the task. If a plan is provided, please use it to determine the next action or modify the plan according to the actual situation.\
\nAction: The action to be performed currently.\
\nAction Input:\
\n<tool_call>{"arguments": <args-dict>, "name": <function-name>}</tool_call>\
\nObservation:\
\n<tool_response>{"name": <function-name>, "content": <result-dict>}</tool_response>\
\n... (this Thought/Plan(optional)/Action/Observation can be repeated zero or more times until the task is completed)\
\nThought：...
"""

REPLY_AND_ACTION_FUNCTION_CALL_PROMPT = """###Output Format###\
\n<answer-to-user> # Prioritize concise replies to the user\
\nAction: {"arguments": <args-dict-1>, "name": <function-name-1>}\
\n... # (You can call multiple tools simultaneously to improve task parallelism)\
\nAction: {"arguments": <args-dict-n>, "name": <function-name-n>}\
\nObservation:{"name": <function-name-1>, "content": <result-dict-1>}\
...\
\nObservation:{"name": <function-name-n>, "content": <result-dict-n>}\
\n... (this answer/Action/Observation can be repeated zero or more times until the task is completed)
"""

PARALLEL_FUNCTION_CALL_PROMPT = """###Output Format###\
\nThought: Step-by-step reasoning and analysis, think clearly about what to do.\
\nPlan: Optional, indicates the plan made to complete the task. If a plan is provided, please use it to determine the next action or modify the plan according to the actual situation.\
\nAction:\
\n<tool_call>{"arguments": <args-dict-1>, "name": <function-name-1>}</tool_call>\
\n... (You can call one or more tools simultaneously to improve task parallelism)\
\n<tool_call>{"arguments": <args-dict-n>, "name": <function-name-n>}</tool_call>\
\nObservation:\
\n<tool_response>{"name": <function-name-1>, "content": <result-dict-1>}</tool_response>\
\n...\
\n<tool_response>{"name": <function-name-n>, "content": <result-dict-n>}</tool_response>\
\n... (this Thought/Plan(optional)/Action/Observation can be repeated zero or more times until the task is completed)\
\nThought：...
"""

TOOLS_PROMPT = """###TOOL REQUIREMENTS###\
\n1. After the manipulation, conduct perception to check whether the operation has been completed.\
\n2. After the navigation, conduct perception to obtain environment information.
"""

DREAMER_SCENE_PROMPT = """You can only call one operation tool at a time. Please ensure the atomicity and completeness of the operation.\n"""

ALFWORLD_SCENE_PROMPT = """The available actions are:
1. go to {recep}
2. take {obj} from {recep}
3. put {obj} in/on {recep}
4. open {recep}
5. close {recep}
6. use {obj}
7. clean {obj} with {recep}
8. heat {obj} with {recep}
9. cool {obj} with {recep}
where {obj} and {recep} correspond to objects and receptacles.
After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happens", that means the previous action is invalid and you should try more options.
Reminder:
1. The action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal.
2. Think when necessary, try to act directly more in the process.
"""

# Broadcast question after the process ends
RESTART_RESPONSE = "Mission completed."

# Get real image response
GET_REAL_IMAGE_RESPONSE = "Getting the latest image."