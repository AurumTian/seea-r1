import io
import os
import re
import cv2
import jwt
import time
import json
import base64
import requests
import traceback
from PIL import Image
from openai import OpenAI
from typing import Optional
from datetime import datetime
from seea.utils.base import StateManager
from seea.agents.agent.base import BaseChatAgent
from seea.utils.logger import get_logger
from seea.configs.config import (VIDEO_GENERATION_IMPROVED_PROMPT, INSTRUCTION_CONSISTENCY_PROMPT, 
                                MISTAKE_FIND_PROMPT, ISSUE_CLASSIFY_PROMPT, VIDEO_GENERATE_INTERVENE_PROMPT)
logger = get_logger(__name__)


class Dreamer(BaseChatAgent):
    def __init__(self,
                 system_prompt: str,
                 model: str,
                 api_key: str,
                 api_base: str,
                 ak: str,
                 sk: str,
                 base_url: str,
                 save_folder: Optional[str] = None,
                 enable_only_end_frame_evalution: bool = False,
                 enable_physics_check: bool = False,
                 name: str = "Dreamer",
                 reflection_prompt: str = "",
                 kling_version: str = "v1-6"):
        """
        Initialize Dreamer class
        :param system_prompt: System prompt
        :param model: Name of the scene description model used
        :param api_key: API key
        :param api_base: Service link for the scene description model
        :param ak: Access Key for the image-to-video interface
        :param sk: Secret Key for the image-to-video interface
        :param base_url: Base URL for the image-to-video interface
        :param save_folder: Target folder for saving videos
        :param enable_only_end_frame_evalution: Whether to evaluate based only on the end frame of the video
        :param enable_physics_check: Whether to enable physics check
        :param name: Name of the agent
        :param reflection_prompt: System prompt for reflecting on the quality of the generated video
        """
        super().__init__(name=name)
        self.system_prompt = system_prompt
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.ak = ak
        self.sk = sk
        self.base_url = base_url
        self.image_path = None
        self.api_token = self.encode_jwt_token() if ak and sk else None
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
        )
        self.save_folder = save_folder
        self.enable_physics_check = enable_physics_check
        self.enable_only_end_frame_evalution = enable_only_end_frame_evalution
        self.state = StateManager()
        self.reflection_prompt = reflection_prompt
        self.kling_version = kling_version # Image-to-video model version, enum values v1-6 and v2-master
    def encode_jwt_token(self):
        """Generate API Token (JWT)"""
        payload = {
            "iss": self.ak,
            "exp": int(time.time()) + 99999,#Set a very large value to prevent timeout
            "nbf": int(time.time()) - 30,
        }
        token = jwt.encode(payload, self.sk, algorithm="HS256")
        if isinstance(token, bytes):
            token = token.decode("utf-8")
        return token
    
    def check_image_validity(self, image_path):
        """Check image validity"""
        with Image.open(image_path) as img:
            width, height = img.size
            aspect_ratio = width / height
            return width >= 300 and height >= 300 and (1 / 2.5 <= aspect_ratio <= 2.5)
        
    def resize_image_to_valid(self, image_path, output_path, target_width=640, target_height=480):
        """Resize image"""
        with Image.open(image_path) as img:
            img = img.resize((target_width, target_height))
            img.save(output_path)

    def encode_image_to_base64(self, image_path):
        """Encode image to Base64"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Image encoding failed: {e}")
            return None
        
    def generate_video_prompt(self, image_path, instruction):
        """Generate scene description"""
        try:
            # Encode image to Base64
            base64_image = self.encode_image_to_base64(image_path)
            if not base64_image:
                logger.error("Image encoding failed, cannot continue.")
                return None
            
            # Construct messages
            messages = [
                # System prompt: Define the model's task and output style
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": instruction
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]

            # Call the model to generate results
            start_time = time.time()
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            elapsed_time = time.time() - start_time
            logger.info(f"Call duration: {elapsed_time:.2f} seconds")

            # Output results
            logger.info(f"Video generation prompt: {completion.choices[0].message.content}")
            return completion.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Error during model invocation: {traceback.format_exc()}")

    def generate_video_intervene_prompt(self, image_path, instruction, positive_prompt):
        """Generate scene description"""
        try:
            # Encode image to Base64
            base64_image = self.encode_image_to_base64(image_path)
            if not base64_image:
                logger.error("Image encoding failed, cannot continue.")
                return None

            # Construct messages
            messages = [
                # System prompt: Define the model's task and output style
                {
                    "role": "system",
                    "content": VIDEO_GENERATE_INTERVENE_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Task instruction: {instruction}\n\nPositive prompt: {positive_prompt}\n\nPlease generate a new positive prompt."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]

            # Call the model to generate results
            start_time = time.time()
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            elapsed_time = time.time() - start_time
            logger.info(f"Call duration: {elapsed_time:.2f} seconds")

            # Output results
            logger.info(f"Task instruction intervention analysis: {completion.choices[0].message.content}")
            return completion.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Error during model invocation: {traceback.format_exc()}")

    def generate_video(self, image_path, prompt, instruction):
        """Call the image-to-video interface to generate a video"""
        # Image encoding
        image_base64 = self.encode_image_to_base64(image_path)
        # Positive prompt parsing
        positive_match = re.search(r'(?i)(?:### ?\*?\*?Positive Prompt[:：]?\*?\*?\n?|### ?\*?Positive Prompt[:：])(.+?)(?=\n### ?\*?\*?Negative Prompt[:：]?\*?\*?|$)', prompt, re.S)
        positive_prompt = positive_match.group(1).strip() if positive_match else ""
        
        # Negative prompt parsing
        negative_match = re.search(r'(?i)(?:### ?\*?\*?Negative Prompt[:：]?\*?\*?\n?|### ?\*?Negative Prompt[:：])(.+)', prompt, re.S)
        negative_prompt = negative_match.group(1).strip() if negative_match else ""

        logger.info(f"Positive prompt:{positive_prompt} \nNegative prompt:{negative_prompt}")

        kling_model_name = "kling-v1-6" if self.kling_version == "v1-6" else "kling-v2-master"

        # Add random interference, with a probability of swapping positive and negative prompts
        import random
        if random.random() < 0.2:  # 20% probability of triggering interference
            positive_prompt = self.generate_video_intervene_prompt(image_path, instruction, positive_prompt)
            # Remove possible code block markers
            positive_prompt = positive_prompt.replace("```json", "", 1).replace("```", "").strip()
            positive_prompt_json = json.loads(positive_prompt)
            positive_prompt = positive_prompt_json.get("Step 3", "").get("Fused Positive Prompt", "")
            logger.info(f"Interference triggered! Interfered positive prompt:{positive_prompt}")

            # Construct request data
            data = {
                "model_name": kling_model_name,
                "mode": "std",
                "duration": "10",
                "image": image_base64,
                "prompt": positive_prompt,
                "negative_prompt": "",
                "cfg_scale": 0.0,
            }
            headers = {
                "Authorization": f"Bearer {self.encode_jwt_token()}",
                "Content-Type": "application/json",
            }
            url = f"{self.base_url}/v1/videos/image2video"
            response = requests.post(url, json=data, headers=headers)
            if response.status_code == 200:
                return {"status": "success", "result": response.json(), "prompt": prompt}
            else:
                logger.error(f"Image-to-video API call failed: {response.text}")
                return {"status": "failure", "error": response.text, "prompt": prompt}
        # Construct request data
        data = {
            "model_name": kling_model_name,
            "mode": "std",
            "duration": "10",
            "image": image_base64,
            "prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "cfg_scale": 0.0,
        }
        headers = {
            "Authorization": f"Bearer {self.encode_jwt_token()}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/v1/videos/image2video"
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            return {"status": "success", "result": response.json(), "prompt": prompt}
        else:
            logger.error(f"Image-to-video API call failed: {response.text}")
            return {"status": "failure", "error": response.text, "prompt": prompt}

    def save_task_info(self, folder, filename, task_info):
        """
        Save task information to the specified folder and file
        :param folder: Folder path
        :param filename: Filename
        :param task_info: Task information to save (dictionary format)
        """
        # Ensure folder exists
        os.makedirs(folder, exist_ok=True)
        
        # Construct full path
        file_path = os.path.join(folder, filename)
        
        # Write to file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(task_info, f, ensure_ascii=False, indent=4)
        logger.info(f"Task information saved to file: {file_path}")

    def execute_command(self, image_path, instruction: str, video_prompt: str):
        """Execute instruction, generate video"""
        # Check image validity
        if not self.check_image_validity(image_path):
            output_path = "resized_image.png"
            self.resize_image_to_valid(image_path, output_path)
            image_path = output_path

        # Call image-to-video interface
        response = self.generate_video(image_path, video_prompt, instruction)
        
        if response["status"] == "success":
            logger.info(f"Image-to-video generation task successful: {response['result']}")
        else:
            logger.error(f"Image-to-video generation task failed: {response['error']}")
        
        task_id = response["result"]["data"]["task_id"]
        task_status = response["result"]["data"]["task_status"]
        logger.info(f"Task ID: {task_id}")
        logger.info(f"Task status:{task_status}")

        # Get current Beijing time
        beijing_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Format time as string

        # Generate API Token
        api_token = self.encode_jwt_token()

        # Save task information
        task_info = {
            "instruction": instruction,
            "task_id": task_id,
            "task_status": task_status,
            "api_token": api_token,
            "timestamp": int(time.time())
        }

        # Save task information
        save_folder = self.save_folder
        if save_folder is None:
            # Get data save path from state manager
            save_folder = self.state.get("sample_save_dir", None)
            # If no data save path in state manager, use default path
            if save_folder is None:
                save_folder = "assets/data/video/kling/tasks"
        filename = f"{beijing_time}_{instruction.replace(' ', '_')}.json".replace('..', '.')  # Filename: time + instruction, replace spaces with underscores
        self.save_task_info(save_folder, filename, task_info)
        return response

    def save_video_to_file(self, video_url):
        """
        Download video and save to specified folder
        :param video_url: URL of the video
        :param output_folder: Target folder for saving the video
        """
        try:
            save_folder = self.save_folder
            if save_folder is None:
                # Get data save path from state manager
                save_folder = self.state.get("sample_save_dir", None)
                # If no data save path in state manager, use default path
                if save_folder is None:
                    save_folder = "assets/data/video/kling/tasks"
            # Ensure target folder exists
            os.makedirs(save_folder, exist_ok=True)

            # Get filename (extract from URL)
            video_filename = os.path.basename(video_url)
            output_path = os.path.join(save_folder, video_filename)

            # Download video
            logger.info(f"Downloading video: {video_url}")
            response = requests.get(video_url, stream=True)
            if response.status_code == 200:
                with open(output_path, "wb") as video_file:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            video_file.write(chunk)
                logger.info(f"Video downloaded successfully to: {output_path}")
                return output_path
            else:
                logger.error(f"Download failed, status code: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error occurred while downloading video: {traceback.format_exc()}")
            return None

    def query_task_status(self, task_id, base_url, api_token):
        """
        Query image-to-video task status (single task)
        :param task_id: Task ID (can be task_id or external_task_id)
        :param base_url: Base URL of the image-to-video interface
        :param api_token: API Token for authentication
        :return: JSON response of the task status
        """
        # Construct full URL
        url = f"{base_url}/v1/videos/image2video/{task_id}"
        
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }

        # Make GET request
        response = requests.get(url, headers=headers)
        
        # Check return status code
        if response.status_code == 200:
            return response.json()  # Return JSON formatted response data
        else:
            logger.error(f"Failed to query task status: {response.status_code}, {response.text}")
            return None

    def _extract_frames_from_video(self, video_path, extract_all=False, interval=20):
        """
        Extract key frames from video
        :param video_path: Video file path
        :param extract_all: Whether to extract all frames, default False only extracts the last frame
        :param interval: Interval for extracting frames, default extract one frame every 20 frames
        :return: List of frames (base64 encoded)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            def frame_to_base64(frame):
                """Convert frame to base64 encoding"""
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                buffered = io.BytesIO()
                frame_pil.save(buffered, format="JPEG")
                return base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            if not extract_all:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame_to_base64(frame))
            else:
                for frame_count in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_count % interval == 0:
                        frames.append(frame_to_base64(frame))
                        
            cap.release()
            return frames
            
        except Exception as e:
            logger.error(f"Failed to extract frames from video: {traceback.format_exc()}")
            return []

    def _generate_improved_prompt(self, original_instruction, issues, first_frame_base64):
        """
        Use Meta-Prompt Agent to generate improved prompt
        :param original_instruction: Original instruction
        :param issues: List of issues
        :param first_frame_base64: Base64 encoding of the first frame
        :return: Improved prompt
        """
        try:
            # Construct messages to generate improved prompt
            meta_prompt_messages = [
                {
                    "role": "system", 
                    "content": VIDEO_GENERATION_IMPROVED_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Original instruction: {original_instruction}\n\nThe currently generated video has the following issues:\n{', '.join(issues)}\n\nPlease optimize the video generation prompt to accurately express the task required by the original instruction."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{first_frame_base64}"
                            }
                        }
                    ]
                }
            ]
            
            # Call model to generate improved prompt
            start_time = time.time()
            meta_prompt_completion = self.client.chat.completions.create(
                model=self.model,
                messages=meta_prompt_messages,
            )
            elapsed_time = time.time() - start_time
            logger.info(f"Time taken to generate improved prompt: {elapsed_time:.2f} seconds")
            
            # Parse result
            improved_prompt = meta_prompt_completion.choices[0].message.content
            logger.info(f"Improved prompt: {improved_prompt}")
            return improved_prompt
        except Exception as e:
            logger.error(f"Failed to generate improved prompt: {traceback.format_exc()}")
            return None

    def _check_instruction_consistency(self, original_instruction, frames):
        """
        Check consistency of video with original instruction
        :param original_instruction: Original instruction
        :param frames: List of video frames
        :return: (Whether consistent, list of inconsistent issues, reconstructed instruction)
        """
        try:
            instruction_consistency_messages = [
                {
                    "role": "system", 
                    "content": INSTRUCTION_CONSISTENCY_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Original instruction: {original_instruction}"
                        }
                    ] + [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{frames}"
                            }
                        } 
                    ]
                }
            ]

            # Call model to evaluate instruction consistency
            start_time = time.time()
            instruction_consistency_completion = self.client.chat.completions.create(
                model=self.model,
                messages=instruction_consistency_messages,
            )
            elapsed_time = time.time() - start_time
            logger.info(f"Instruction consistency evaluation time: {elapsed_time:.2f} seconds")
            
            # Parse result
            consistency_result = instruction_consistency_completion.choices[0].message.content
            logger.info(f"Instruction consistency evaluation result: {consistency_result}")
            
            # Try to parse JSON result
            try:
                # Remove possible code block markers
                consistency_result = consistency_result.replace("```json", "").replace("```", "").strip()
                consistency_json = json.loads(consistency_result)
                # Get match from comparison field as instruction_valid
                instruction_valid = consistency_json.get("comparison", {}).get("match", False)
                
                # Collect all relevant issues
                instruction_issues = []
                
                # Get deviations and extra steps from comparison
                deviations = consistency_json.get("comparison", {}).get("deviations", [])
                extra_steps = consistency_json.get("comparison", {}).get("extra_steps", [])
                instruction_issues.extend(deviations)
                instruction_issues.extend(extra_steps)
                
                # Get improvement suggestions from conclusion
                improvement_suggestions = consistency_json.get("conclusion", {}).get("improvement_suggestions", [])
                instruction_issues.extend(improvement_suggestions)
                
                # If no issues found but validation failed, add default issue description
                if not instruction_issues and not instruction_valid:
                    instruction_issues = ["Video content does not meet instruction requirements"]

            except json.JSONDecodeError:
                # Simple result parsing
                instruction_valid = "true" in consistency_result.lower() and "false" not in consistency_result.lower()
                instruction_issues = ["Unable to parse instruction consistency evaluation result"]
            
            return instruction_valid, instruction_issues
        except Exception as e:
            logger.error(f"Failed to check instruction consistency: {traceback.format_exc()}")
            return False, ["Error occurred during instruction consistency check"]

    def _detect_physical_issues(self, original_instruction, frames):
        """
        Detect physics errors in video
        :param original_instruction: Original instruction 
        :param frames: List of video frames
        :return: List of physics errors
        """
        try:
            mistake_find_messages = [
                {
                    "role": "system", 
                    "content": MISTAKE_FIND_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Task instruction: {original_instruction}"
                        }
                    ] + [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{frame}"
                            }
                        } for frame in frames
                    ]
                }
            ]
            
            # Call model to check for physics errors
            start_time = time.time()
            mistake_find_completion = self.client.chat.completions.create(
                model=self.model,
                messages=mistake_find_messages,
            )
            elapsed_time = time.time() - start_time
            logger.info(f"Physics error check time: {elapsed_time:.2f} seconds")
            
            # Parse result
            mistake_result = mistake_find_completion.choices[0].message.content
            logger.info(f"Physics error check result: {mistake_result}")
            
            # Try to parse JSON result
            try:
                # Remove possible code block markers
                mistake_result = mistake_result.replace("```json", "").replace("```", "").strip()
                mistake_json = json.loads(mistake_result)
                physics_issues = mistake_json.get("issues", [])
            except json.JSONDecodeError:
                # Simple result parsing
                physics_issues = []
                lines = mistake_result.split('\n')
                for line in lines:
                    if '-' in line or '•' in line or '*' in line:
                        physics_issues.append(line.strip().lstrip('-').lstrip('•').lstrip('*').strip())
            
            return physics_issues
        except Exception as e:
            logger.error(f"Failed to detect physics errors: {traceback.format_exc()}")
            return []

    def _classify_issues(self, original_instruction, physics_issues):
        """
        Classify detected issues into task planning errors and video generation errors
        :param original_instruction: Original instruction
        :param physics_issues: List of physics errors
        :return: (List of task planning errors, List of video generation errors)
        """
        try:
            issue_classify_messages = [
                {
                    "role": "system", 
                    "content": ISSUE_CLASSIFY_PROMPT
                },
                {
                    "role": "user",
                    "content": f"Task instruction: {original_instruction}\n\nList of detected issues: {physics_issues}\n\nPlease classify these issues into 'task planning errors' and 'video generation errors'. Please reply in JSON format, including fields 'planning_issues' and 'generation_issues'."
                }
            ]
            
            # Call model to classify issues
            start_time = time.time()
            issue_classify_completion = self.client.chat.completions.create(
                model=self.model,
                messages=issue_classify_messages,
            )
            elapsed_time = time.time() - start_time
            logger.info(f"Issue classification time: {elapsed_time:.2f} seconds")
            
            # Parse result
            classify_result = issue_classify_completion.choices[0].message.content
            logger.info(f"Issue classification result: {classify_result}")
            
            # Try to parse JSON result
            try:
                # Remove possible code block markers
                classify_result = classify_result.replace("```json", "").replace("```", "").strip()
                classify_json = json.loads(classify_result)
                planning_issues = classify_json.get("planning_issues", [])
                generation_issues = classify_json.get("generation_issues", [])
            except json.JSONDecodeError:
                # Simple result parsing
                planning_issues = []
                generation_issues = physics_issues  # Default all issues are generation errors
                
                # Try to extract planning errors from text
                if "Task planning errors" in classify_result:
                    planning_section = classify_result.split("Task planning errors")[1].split("Video generation errors")[0]
                    planning_lines = planning_section.split('\n')
                    for line in planning_lines:
                        if '-' in line or '•' in line or '*' in line:
                            planning_issues.append(line.strip().lstrip('-').lstrip('•').lstrip('*').strip())
                
                # Try to extract generation errors from text
                if "Video generation errors" in classify_result:
                    generation_section = classify_result.split("Video generation errors")[1]
                    generation_lines = generation_section.split('\n')
                    for line in generation_lines:
                        if '-' in line or '•' in line or '*' in line:
                            generation_issues.append(line.strip().lstrip('-').lstrip('•').lstrip('*').strip())
            
            return planning_issues, generation_issues
        except Exception as e:
            logger.error(f"Issue classification failed: {traceback.format_exc()}")
            return [], physics_issues  # Default all issues are generation errors

    def reflect_on_video(self, video_path, original_instruction, current_prompt):
        """
        Reflect on and evaluate the generated video, using multiple specialized agents to complete evaluation and improvement
        :param video_path: Path of the generated video
        :param original_instruction: Original user instruction
        :param current_prompt: Currently used video generation prompt
        :return: Reflection result, including evaluation and improvement suggestions
        """
        try:
            # Extract key frames from video
            if self.enable_only_end_frame_evalution:
                frames = self._extract_frames_from_video(video_path)
            else:
                frames = self._extract_frames_from_video(video_path, extract_all=True)
            if not frames:
                logger.error("Failed to extract frames from video")
                return {
                    "physics_valid": False,
                    "instruction_valid": False,
                    "physics_issues": ["Unable to extract frames from video"],
                    "instruction_issues": ["Unable to extract frames from video"],
                    "improved_prompt": current_prompt
                }
            
            # Instruction consistency check
            instruction_check_iterations = 0
            max_instruction_check_iterations = 3
            instruction_valid = False
            instruction_issues = []
            improved_prompt = current_prompt
            
            while not instruction_valid and instruction_check_iterations < max_instruction_check_iterations:
                instruction_check_iterations += 1
                logger.info(f"Starting instruction consistency check, iteration {instruction_check_iterations}")
                
                # Check instruction consistency
                instruction_valid, instruction_issues = self._check_instruction_consistency(original_instruction, frames[-1])
                if instruction_valid:
                    logger.info("Instruction consistency check passed")
                    break
                
                # If instruction is inconsistent, use Meta-Prompt Agent to generate improved prompt
                if not instruction_valid and instruction_check_iterations < max_instruction_check_iterations:
                    logger.info("Instruction inconsistent, generating improved prompt...")
                    
                    # Generate improved prompt
                    improved_prompt = self._generate_improved_prompt(original_instruction, instruction_issues, frames[0])
                    if not improved_prompt:
                        improved_prompt = current_prompt
                        break
                    
                    # Return to external function to regenerate video
                    return {
                        "physics_valid": True,  # Default to physics valid for now, as it hasn't been checked yet
                        "instruction_valid": False,
                        "physics_issues": [],
                        "instruction_issues": instruction_issues,
                        "improved_prompt": improved_prompt
                    }
            
            # If all iterations fail instruction consistency check
            if not instruction_valid:
                logger.warning(f"After {max_instruction_check_iterations} iterations, instruction consistency check still failed")
                return {
                    "physics_valid": True,  # Default to physics valid for now, as it hasn't been checked yet
                    "instruction_valid": False,
                    "physics_issues": [],
                    "instruction_issues": instruction_issues,
                    "improved_prompt": improved_prompt
                }
            
            # Physics error check
            if self.enable_physics_check:
                physics_issues = self._detect_physical_issues(original_instruction, frames)
                if not physics_issues:
                    logger.info("No physics errors found")
                    return {
                        "physics_valid": True,
                        "instruction_valid": True,
                        "physics_issues": [],
                        "instruction_issues": [],
                        "improved_prompt": current_prompt
                    }
                
                # If only video generation errors exist, optimize prompt and retry the process
                if physics_issues:
                    logger.info(f"Video generation errors exist: {physics_issues}")
                    
                    # Maximum attempts
                    max_meta_prompt_iterations = 3
                    meta_prompt_iterations = 0
                    
                    while physics_issues and meta_prompt_iterations < max_meta_prompt_iterations:
                        meta_prompt_iterations += 1
                        logger.info(f"Starting optimization of video generation prompt, iteration {meta_prompt_iterations}")
                        
                        physics_issues = [issue['description'] for issue in physics_issues]

                        # Generate improved prompt
                        improved_prompt = self._generate_improved_prompt(original_instruction, physics_issues, frames[0])
                        if not improved_prompt:
                            improved_prompt = current_prompt
                            break
                        
                        # Return to external function to regenerate video and perform full evaluation process again
                        if meta_prompt_iterations < max_meta_prompt_iterations:
                            return {
                                "physics_valid": False,
                                "instruction_valid": True,
                                "physics_issues": physics_issues,
                                "instruction_issues": [],
                                "improved_prompt": improved_prompt,
                                "need_instruction_check": True  # Indicate that instruction consistency check needs to be redone
                            }
                        else:
                            # Last iteration, return current result
                            logger.warning(f"Reached maximum optimization attempts ({max_meta_prompt_iterations}), returning current result")
                            return {
                                "physics_valid": False,
                                "instruction_valid": True,
                                "physics_issues": physics_issues,
                                "instruction_issues": [],
                                "improved_prompt": improved_prompt,
                                "warning": f"After {max_meta_prompt_iterations} optimization attempts, video generation errors still exist"
                            }
            # Return success result
            return {
                "physics_valid": True,
                "instruction_valid": True,
                "physics_issues": [],
                "instruction_issues": [],
                "improved_prompt": current_prompt
            }
                
        except Exception as e:
            logger.error(f"Video reflection evaluation failed: {traceback.format_exc()}")
            return {
                "physics_valid": True,
                "instruction_valid": True,
                "physics_issues": [],
                "instruction_issues": [],
                "improved_prompt": current_prompt
            }

    def infer(self, instruction: str, image_path: str = ""):
        """
        Unified interface, receives user instruction and calls execute_command
        Receives user instruction, submits task, then queries task status until result is obtained.
        :param instruction: User input instruction
        :param image_path: Input image path
        :return: Result of image-to-video task
        """
        logger.debug(f"image_path: {image_path}")
        if not self.image_path and not image_path:
            logger.error("self.image_path is None, please initialize it or use self.run(image_path, instruction).")
            return {"status": "failure", "error": "image_path is None!"}
        self.image_path = image_path if image_path else self.image_path

        try:
            # Maximum attempts
            max_attempts = 3
            current_attempt = 0
            current_prompt = None
            
            while current_attempt < max_attempts:
                current_attempt += 1
                logger.info(f"Starting {current_attempt}/{max_attempts} attempt to generate video")
                
                # If it's the first attempt or prompt needs to be regenerated
                if current_prompt is None:
                    # Generate scene description
                    current_prompt = self.generate_video_prompt(self.image_path, instruction)
                
                # Submit task
                # Check if current_prompt is None, use default empty string if it is
                video_prompt = current_prompt if current_prompt is not None else ""
                result = self.execute_command(self.image_path, instruction, video_prompt)
                
                # Get task ID
                task_id = result["result"]["data"]["task_id"]
                task_status = result["result"]["data"]["task_status"]

                # Query task status
                max_retries = 50 if self.kling_version == "v1-6" else 1000000 # Set maximum query attempts to prevent infinite loop
                retries = 0
                
                while retries < max_retries:
                    # Query task status
                    task_response = self.query_task_status(task_id, self.base_url, self.api_token)
                    if not task_response:
                        return {"status": "failure", "error": "Failed to query task status"}
                    # Get task status
                    task_status = task_response.get("data", {}).get("task_status", "unknown")
                    task_status_msg = task_response.get("data", {}).get("task_status_msg", "")

                    # Process based on task status
                    if task_status == "succeed":
                        # Task successful, extract video URL
                        task_result = task_response.get("data", {}).get("task_result", {})
                        videos = task_result.get("videos", [])
                        if videos and "url" in videos[0]:
                            video_url = videos[0]["url"]
                        else:
                            raise ValueError("Unable to get video URL, response data structure incomplete")
                        logger.info(f"Task successful, video URL: {video_url}")

                        # Download video
                        downloaded_video_path = self.save_video_to_file(video_url)
                        if not downloaded_video_path:
                            return {"status": "failure", "error": "Video download failed"}
                        
                        # Reflect on and evaluate the generated video
                        logger.info("Starting reflection and evaluation of the generated video...")
                        reflection_result = self.reflect_on_video(
                            downloaded_video_path, 
                            instruction, 
                            current_prompt
                        )
                        
                        # Check if video meets instruction and physics requirements
                        instruction_valid = reflection_result.get("instruction_valid", True)
                        physics_valid = reflection_result.get("physics_valid", True)
                        need_instruction_check = reflection_result.get("need_instruction_check", False)
                        
                        # If video meets requirements, return success result
                        if physics_valid and instruction_valid:
                            logger.info("Video evaluation passed, meets physics and instruction requirements")
                            return {
                                "status": "success", 
                                "video_url": video_url, 
                                "file_path": downloaded_video_path,
                                "prompt": current_prompt,
                                "reflection": reflection_result
                            }
                        
                        # Use improved prompt and determine if full evaluation process needs to be redone
                        improved_prompt = reflection_result.get("improved_prompt")
                        if improved_prompt and improved_prompt != current_prompt:
                            current_prompt = improved_prompt
                            logger.info(f"Retrying with improved prompt: {current_prompt}")
                            break  # Break out of status query loop, retry with new prompt
                        
                        # If it's the last attempt, return current result
                        if current_attempt >= max_attempts:
                            logger.warning(f"Reached maximum attempts ({max_attempts}), returning current result")
                            return {
                                "status": "partial_success", 
                                "video_url": video_url, 
                                "file_path": downloaded_video_path,
                                "prompt": current_prompt,
                                "reflection": reflection_result,
                                "warning": "Video generation did not fully meet requirements, but maximum attempts reached"
                            }
                        
                        break  # Break out of status query loop, proceed to next attempt

                    elif task_status == "failed":
                        # Task failed, return failure information
                        logger.error(f"Task failed, reason: {task_status_msg}")
                        if current_attempt >= max_attempts:
                            return {"status": "failure", "error": task_status_msg}
                        else:
                            # If not the last attempt, regenerate prompt
                            logger.info("Regenerating prompt...")
                            current_prompt = None
                            break  # Break out of status query loop, retry

                    elif task_status in ["submitted", "processing"]:
                        # Task still processing, wait for some time
                        logger.info("Waiting for task completion...")
                        time.sleep(10)  # Wait 10 seconds before querying again

                    else:
                        # Unknown status, return error
                        logger.error(f"Unknown task status: {task_status}")
                        if current_attempt >= max_attempts:
                            return {"status": "failure", "error": f"Unknown task status: {task_status}"}
                        else:
                            break  # Break out of status query loop, retry

                    retries += 1
                
                # If maximum retries reached and still in current attempt
                if retries >= max_retries and current_attempt < max_attempts:
                    logger.error("Query task status timed out, trying to regenerate")
                    current_prompt = None  # Regenerate prompt
                    continue
                elif retries >= max_retries:
                    logger.error("Query task status timed out, task may not be completed")
                    return {"status": "failure", "error": "Query task status timed out"}
            
            # If all attempts fail
            logger.error(f"Video generation still unsuccessful after {max_attempts} attempts")
            return {"status": "failure", "error": "Video generation failed after multiple attempts"}

        except Exception as e:
            logger.error(f"Run failed: {traceback.format_exc()}")
            return {"status": "failure", "error": str(traceback.format_exc())}