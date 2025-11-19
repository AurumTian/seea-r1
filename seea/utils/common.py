import cv2
import json
import logging
import os
from typing import Callable
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

RED = "\033[1;31m"
GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
BLUE = "\033[1;34m"
MAGENTA = "\033[1;35m"
CYAN = "\033[1;36m"
RESET = "\033[0m"


def langraph_tool_to_schema(func: Callable):
    return {
            "type": "function",
            "function": {
                "name": func.name,
                "description": func.description,
                "parameters": {
                    "type": "object",
                    "properties": func.args,
                    "required": list(func.args.keys()),
                    "additionalProperties": False,
                },
            },
    }


def get_video_frame(video_path, frame_index):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video was opened successfully
    if not cap.isOpened():
        logger.info("Error: Unable to open video file")
        return None

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_index >= total_frames:
        frame_index = total_frames - 1
    elif frame_index < -total_frames:
        frame_index = 0
    elif frame_index < 0:
        frame_index = total_frames + frame_index

    # Move the read position to the specified frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    
    # Read the frame
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Generate save path, same path as the original video, with _last_frame suffix
        base_name = os.path.splitext(video_path)[0]  # Remove extension
        last_frame_path = f"{base_name}_last_frame.jpg"
        
        # Save the image
        cv2.imwrite(last_frame_path, frame)
        logger.info(f"Last frame saved as: {last_frame_path}")
        return last_frame_path
    else:
        logger.info("Error: Unable to read the last frame")
        return None


def clean_json_output(output: str, replace_nulls: bool = True) -> tuple[str, bool]:
    """
    Cleans a string containing JSON output, removing surrounding backticks and handling null-like values.

    Args:
        output: The string containing the JSON output, potentially with surrounding backticks.
        replace_nulls: Whether to replace "unknown", "na", and "null" string values with empty strings. Defaults to True.

    Returns:
        A tuple containing:
            - The cleaned JSON string.
            - A boolean indicating whether the JSON was successfully parsed and cleaned.
              Returns the original cleaned string and False if parsing fails.
    """
    output = output.strip()
    if output.startswith("```json"):
        output = output[7:]
    elif output.startswith("```"):
        output = output[4:]
    if output.endswith("```"):
        output = output[:-3]
    cleaned_output = output.strip()

    try:
        json_data = json.loads(cleaned_output)
    except json.JSONDecodeError as e:
        logger.info(f"JSON decoding error: {e}")
        return cleaned_output, False

    def clean_json(data):
        if isinstance(data, dict):
            return {key: clean_json(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [clean_json(item) for item in data]
        elif isinstance(data, str):
            return "" if replace_nulls and data.lower() in ["unknown", "na", "null"] else data # Use parameter to control whether to replace
        else:
            return data

    cleaned_json_data = clean_json(json_data)
    cleaned_output = json.dumps(cleaned_json_data, ensure_ascii=False, indent=2) # Add indent to make output more readable

    return cleaned_output, True


def encode_image(image_path, new_size=None, max_size=None):
    """Encodes an image to a base64 string.
    
    Args:
        image_path: Path to the image file.
        new_size: Target size to resize the image to (width, height).
        max_size: Maximum size of the image; if exceeded, it will be proportionally scaled down.
        
    Returns:
        Base64 encoded image string.
    """
    try:
        # Check if the file exists and its size is normal
        import os
        if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
            print(f"Warning: Image file {image_path} does not exist or is empty")
            return None
            
        import base64
        from PIL import Image
        import io
        
        # Try to open the image file, add error handling
        try:
            img = Image.open(image_path)
            
            # Resize the image
            if new_size:
                img = img.resize(new_size)
            elif max_size:
                width, height = img.size
                if width > max_size or height > max_size:
                    ratio = min(max_size / width, max_size / height)
                    new_width = int(width * ratio)
                    new_height = int(height * ratio)
                    img = img.resize((new_width, new_height))
            
            # Convert image to bytes
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            
            # Encode to base64
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return img_str
            
        except (OSError, IOError) as e:
            print(f"Error loading image {image_path}: {str(e)}")
            # If image loading fails, try to read the file directly and encode
            try:
                with open(image_path, "rb") as image_file:
                    img_data = image_file.read()
                    if img_data:
                        img_str = base64.b64encode(img_data).decode()
                        return img_str
            except Exception as file_e:
                print(f"Failed to read image file directly: {str(file_e)}")
            return None
            
    except Exception as e:
        print(f"Unexpected error encoding image {image_path}: {str(e)}")
        return None
