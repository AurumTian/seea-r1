import os
import json
import glob
import traceback
from seea.utils.logger import get_logger
logger = get_logger(__name__)
DEFAULT_SAVE_PATH = "data.json"
OUTPUT_FILE = os.path.join("assets", "data", "sft", "multi_turn.json")

def transform_user_content(content):
    """
    Transforms the content of a user message:
      - If content is a dictionary, extracts "text" and "image" fields.
        Generates "<image>" tags corresponding to the number and order of images,
        and concatenates them with the original text.
      - If text is not empty, returns text concatenated with "<image>" tags, otherwise returns only the tags.
    """
    if isinstance(content, dict):
        text = content.get("text", "").strip()
        images = content.get("image", [])
        # Ensure images is iterable; if None, convert to an empty list
        if images is None:
            images = []
        elif not isinstance(images, list):
            # If it's a single image path (string), convert to a list
            images = [images]

        image_markers = "".join(["<image>" for _ in images])
        new_content = f"{text}{image_markers}" if text else image_markers
        return new_content
    return str(content)

def extract_samples(save_path, max_samples=None):
    """
    Extracts all sample data from the save path and converts it to the target JSON format.

    Args:
        save_path: Can be one of two forms:
                  1. A JSON file path, which contains a list of sample directories.
                  2. A directory path; the function will process all *conversation.jsonl files in this directory.
        max_samples: Limits the number of samples; None means all.

    Returns:
        list: A list of sample records, each in the format:
              {
                 "messages": [ {"role": "user", "content": "..."}, ... ],
                 "images": [ "img_path1", "img_path2", ... ]  # Order based on the appearance of images in JSON messages
              }
    """
    dataset = []
    json_files = []

    # Check if save_path is a directory or a file
    if os.path.isdir(save_path):
        # If it's a directory, find all conversation.jsonl files
        logger.info(f"Extracting samples from directory: {save_path}")
        # Find conversation.jsonl files directly under the directory
        json_files.extend(glob.glob(os.path.join(save_path, "*conversation.jsonl")))

        # Find conversation.jsonl files in subdirectories
        for sample_dir in glob.glob(os.path.join(save_path, "sample_*")):
            if os.path.isdir(sample_dir):
                json_files.extend(glob.glob(os.path.join(sample_dir, "*conversation.jsonl")))

        logger.info(f"Found {len(json_files)} conversation.jsonl files")
    else:
        # If it's a file, read the list of sample directories
        logger.info(f"Reading sample directory list from file: {save_path}")
        try:
            with open(save_path, "r", encoding="utf-8") as f:
                sample_dirs = json.load(f)

            for sample_dir_entry in sample_dirs:
                # Process sample directories in dictionary or string format
                sample_dir = None
                if isinstance(sample_dir_entry, dict):
                    sample_dir = sample_dir_entry.get("sample_save_dir")
                    if not sample_dir:
                        logger.warning(f"Dictionary entry missing 'sample_save_dir': {sample_dir_entry}")
                        continue
                else:
                    sample_dir = sample_dir_entry

                if not isinstance(sample_dir, str):
                    logger.warning(f"Invalid sample directory (not a string): {sample_dir}")
                    continue

                # Find conversation.json or conversation.jsonl files
                conv_files = glob.glob(os.path.join(sample_dir, "*conversation.json"))
                if not conv_files:
                    conv_files = glob.glob(os.path.join(sample_dir, "*conversation.jsonl"))

                if conv_files:
                    json_files.extend(conv_files)
                else:
                    logger.warning(f"No conversation files found in {sample_dir}")
        except Exception as e:
            logger.error(f"Error reading sample directory list file: {e}")

    # Process each conversation file
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f_json:
                data = json.load(f_json)

            messages = []
            ordered_images = []

            # Ensure data is a list type
            if not isinstance(data, list):
                logger.warning(f"Data in file {json_file} is not in list format, skipping processing")
                continue

            for entry in data:
                # Ensure entry is a dictionary type
                if not isinstance(entry, dict):
                    logger.warning(f"Entry in file {json_file} is not in dictionary format: {entry}")
                    continue

                role = entry.get("role")
                content = entry.get("content")

                # Collect image paths
                if isinstance(content, dict) and "image" in content:
                    imgs = content.get("image", [])
                    if imgs is not None:  # Ensure imgs is not None
                        if isinstance(imgs, list):
                            ordered_images.extend(imgs)
                        else:
                            # Handle single image path case
                            ordered_images.append(imgs)

                # Process messages from different roles
                if role in ["system", "user", "tool"]:
                    messages.append({"role": role, "content": transform_user_content(content)})
                elif role == "assistant":
                    new_content = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
                    msg = {"role": "assistant", "content": new_content}

                    # If there's a reward field, keep it
                    if "reward" in entry:
                        msg["reward"] = entry["reward"]

                    # If there's an advantage field, keep it
                    if "advantage" in entry:
                        msg["advantage"] = entry["advantage"]

                    messages.append(msg)
                else:
                    logger.warning(f"Unknown role: {role}")
                    continue

            # Create record and add to dataset
            if messages:
                record = {"messages": messages}
                if ordered_images:
                    record["images"] = ordered_images

                # If the last message is from assistant and has advantage, add it to the record
                if messages and messages[-1]["role"] == "assistant" and "advantage" in messages[-1]:
                    record["advantage"] = messages[-1]["advantage"]

                dataset.append(record)

        except Exception as e:
            logger.error(f"Error reading file {json_file}: {traceback.format_exc()}")
            continue

    # Limit sample quantity
    if max_samples and len(dataset) > max_samples:
        dataset = dataset[:max_samples]
        logger.info(f"Limiting sample quantity to {max_samples}")

    logger.info(f"Extracted a total of {len(dataset)} sample records")
    return dataset


def save_dataset_to_file(dataset, output_file=OUTPUT_FILE):
    """
    Saves the dataset data to a JSON file (in JSON array format).
    If the target directory does not exist, it will be created automatically.
    """
    out_dir = os.path.dirname(output_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(output_file, "w", encoding="utf-8") as f_out:
        json.dump(dataset, f_out, ensure_ascii=False, indent=4)
    print(f"Dataset generated: {output_file}, total {len(dataset)} records")

if __name__ == "__main__":
    # Example: Assume the sample path file is data.json in the current directory
    save_path = os.path.join(os.getcwd(), "data.json")
    data = extract_samples(save_path, max_samples=10)
    save_dataset_to_file(data)