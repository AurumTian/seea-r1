from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any
from seea.agents.agent.chat_agent import ChatAgent
from seea.agents.models.models import VideoQualityResult

# Video quality assessment Agent based on large video understanding models, determines if the generated video is qualified and successfully completes the instruction based on video quality.
class VideoChatAgent(ChatAgent):
    def __init__(self, 
                 system_prompt: str = "You are a professional video quality assessment expert. You need to analyze the video content, determine if the video successfully completed the given instruction, and evaluate the video quality.",
                 frame_extraction_mode: str = "interval",  # New: Frame extraction mode, options "interval" (extract at intervals) or "endpoints" (only first and last frames)
                 frame_interval: int = 10,  # New: Frame extraction interval
                 save_path: Optional[str] = None,  # New: Data save path, if empty, save directly in the video path
                 result_model: Optional[BaseModel] = None,  # New: Output result model type
                 *args, **kwargs):
        """
        Initialize Video Quality Assessment Agent, inherits from ChatAgent
        
        Args:
            system_prompt: System prompt
            frame_extraction_mode: Frame extraction mode, "interval" means extract at intervals, "endpoints" means extract only first and last frames
            frame_interval: Frame extraction interval when frame_extraction_mode is "interval"
            save_path: Data save path, if empty, save directly in the video path
            result_model: Output result model type
            *args, **kwargs: Other parameters for ChatAgent
        """
        super().__init__(system_prompt=system_prompt, *args, **kwargs)
        self.frame_extraction_mode = frame_extraction_mode
        self.frame_interval = frame_interval
        self.save_path = save_path
        self.result_model = result_model
        self.last_frame_path = None
        
    def _get_last_frame_path(self):
        """Get the path of the last frame of the video"""
        return self.last_frame_path
    
    def infer(self, instruction: str, video_path: str, frame_extraction_mode=None, frame_interval=None, save_path=None):
        """
        Analyze video quality and determine if the instruction was successfully completed
        
        Args:
            instruction: The instruction the video should complete
            video_path: Video file path
            frame_extraction_mode: Optional parameter, overrides the default frame extraction mode
            frame_interval: Optional parameter, overrides the default frame extraction interval
            save_path: Optional parameter, overrides the default data save path
            
        Returns:
            VideoQualityResult: Object containing the assessment result
        """
        # Use passed parameters or default parameters
        extraction_mode = frame_extraction_mode or self.frame_extraction_mode
        interval = frame_interval or self.frame_interval
        output_path = save_path or self.save_path
        
        # Extract key frame paths from the video
        key_frames = self._extract_key_frames(video_path, extraction_mode, interval, output_path)
        
        # Get the path of the last frame
        self.last_frame_path = key_frames[-1] if key_frames else ""
        
        # Get field information of the result model
        result_fields = ""
        for field_name, field in self.result_model.__fields__.items():
            field_desc = field.field_info.description
            result_fields += f"- {field_name}: {field_desc}\n"
        
        # Construct task prompt
        prompt = f"""Please determine if the video successfully completed the instruction: "{instruction}" """
        
        format_prompt = """
        Please return the result in JSON format, including the following fields:
        {
        """

        # Dynamically construct JSON fields and descriptions
        for field_name, field in self.result_model.__fields__.items():
            field_type = "true/false" if field.type_ == bool else \
                         "float from 0-10" if field_name == "quality_score" else \
                         "\"Detailed assessment feedback\"" 
            format_prompt += f'    "{field_name}": {field_type}, // {field.field_info.description}\n'
        
        # Remove the last comma and close JSON
        format_prompt = format_prompt.rstrip(',\n') + "\n}\n"
        
        # Call the large model for analysis
        message = self.chat(self.system_prompt + prompt + format_prompt, key_frames)
        
        # Parse the response result
        analysis_result = self._parse_response(message.content)
        
        # Return the result
        return VideoQualityResult(
            success=analysis_result.get("success", False),
            quality_score=analysis_result.get("quality_score", 0.0),
            feedback=analysis_result.get("feedback", ""),
        )
    
    def _extract_key_frames(self, video_path: str, extraction_mode: str = None, interval: int = None, save_path: str = None) -> list:
        """
        Extract key frames from the video
        
        Args:
            video_path: Video file path
            extraction_mode: Frame extraction mode
            interval: Frame extraction interval
            save_path: Save path, if empty, save directly in the video path
            
        Returns:
            list: List of key frame paths
        """
        import cv2
        import os
        
        # Use passed parameters or default parameters
        mode = extraction_mode or self.frame_extraction_mode
        frame_interval = interval or self.frame_interval
        
        # Ensure video file exists
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return []
        
        # Open video file
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            print(f"Unable to open video file: {video_path}")
            return []
        
        # Create directory to save key frames
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            base_filename = os.path.basename(video_path).split('.')[0]
            output_dir = os.path.join(save_path, f"{base_filename}_frames")
        else:
            output_dir = os.path.splitext(video_path)[0] + "_frames"
        
        os.makedirs(output_dir, exist_ok=True)
        
        frame_paths = []
        frame_count = 0
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if mode == "endpoints":
            # Extract only first and last frames
            # Read the first frame
            success, frame = video.read()
            if success:
                frame_path = os.path.join(output_dir, f"frame_0.jpg")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
            
            # Jump to the last frame
            if total_frames > 1:  # Ensure video has more than one frame
                video.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
                success, frame = video.read()
                if success:
                    frame_path = os.path.join(output_dir, f"frame_{total_frames-1}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)
        else:
            # Extract frames at intervals
            while True:
                success, frame = video.read()
                if not success:
                    break
                
                if frame_count % frame_interval == 0:
                    frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)
                
                frame_count += 1
        
        # Release video resources
        video.release()
        
        print(f"Saved {len(frame_paths)} key frames to {output_dir}")
        return frame_paths
    
    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the large model's response"""
        try:
            # Try to parse JSON directly
            if isinstance(response, dict) and "content" in response:
                content = response["content"]
                # Try to extract JSON from text
                import re
                import json
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
                
                # Try to parse the entire content directly
                try:
                    return json.loads(content)
                except:
                    pass
            
            # If unable to parse, return default values
            return {
                "success": False,
                "quality_score": 0.0,
                "feedback": "Unable to parse model response"
            }
        except Exception as e:
            return {
                "success": False,
                "quality_score": 0.0,
                "feedback": f"Parsing error: {str(e)}"
            }
