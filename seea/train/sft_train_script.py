import os
import sys
import time
import json
import argparse
from swift.llm import sft_main, TrainArguments
from seea.utils.logger import get_logger

logger = get_logger(__name__)

def filter_grpo_data(dataset_path, output_dir=None):
    """
    Filter GRPO data to create SFT dataset:
    - Keep only samples with advantage > 0
    - Keep messages and images fields
    - Remove reward and advantage from the assistant's last message
    
    Parameters:
    dataset_path: str - Path to the GRPO dataset file
    output_dir: str - Output directory, uses directory of dataset file if None
    
    Returns:
    str - Path to the filtered SFT dataset
    """
    try:
        # Determine output directory and file
        if output_dir is None:
            output_dir = os.path.dirname(dataset_path)
        os.makedirs(output_dir, exist_ok=True)
        
        sft_output_path = os.path.join(output_dir, "dataset_sft.json")
        logger.info(f"Processing GRPO dataset from: {dataset_path}")
        
        # Read the GRPO data
        with open(dataset_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                # Try reading as JSONL if JSON fails
                f.seek(0)
                data = [json.loads(line) for line in f if line.strip()]
        
        # Filter and process the data
        sft_data = []
        for item in data:
            # Skip items without advantage field or with advantage <= 0
            if 'advantage' not in item or item['advantage'] <= 0:
                continue
            
            # Create a new item
            new_item = {}
            
            # Process the messages
            if 'messages' in item:
                # Add messages field
                new_item["messages"] = []
                
                # Copy all messages, but clean the last assistant message
                for i, msg in enumerate(item['messages']):
                    if i == len(item['messages']) - 1 and msg.get('role') == 'assistant':
                        # For the last assistant message, keep only role and content
                        new_item['messages'].append({
                            "role": msg['role'],
                            "content": msg['content']
                        })
                    else:
                        # For other messages, copy as is
                        new_item['messages'].append(msg)
            
            # Also keep the images field if present
            if 'images' in item:
                new_item['images'] = item['images']
                
            sft_data.append(new_item)
        
        # Save the filtered data
        with open(sft_output_path, 'w', encoding='utf-8') as f:
            json.dump(sft_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Filtered SFT dataset saved to: {sft_output_path}")
        logger.info(f"Retained {len(sft_data)} samples with positive advantage")
        
        # Create a path file for the pipeline
        path_file = os.path.join(output_dir, "sample_out_sft.txt")
        with open(path_file, 'w') as f:
            f.write(sft_output_path)
        
        return sft_output_path
    
    except Exception as e:
        logger.error(f"Error filtering GRPO data: {str(e)}", exc_info=True)
        raise

def train_sft(dataset, model, output_dir=None, learning_rate=1e-6, train_type='full', **kwargs):
    """
    Run SFT training on the filtered dataset
    
    Parameters:
    dataset: str - Path to the SFT dataset
    model: str - Model name or path
    output_dir: str - Output directory for model checkpoints
    learning_rate: float - Learning rate for training
    train_type: str - Training type (full or lora)
    **kwargs: Additional arguments to pass to TrainArguments
    
    Returns:
    str - Path to the last model checkpoint
    """
    try:
        # Set default output directory if not provided
        if output_dir is None:
            output_dir = os.path.join("output", "sft", time.strftime('%Y%m%d_%H%M%S'))
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory created: {os.path.abspath(output_dir)}")
        
        # Set up training arguments
        is_multi_modal = "VL" in dataset or "vl" in dataset or "VL" in model or "vl" in model
        
        # Log training start
        logger.info(f"🚀 Starting SFT training, dataset: {dataset}")
        
        # Configure training arguments based on model type and training method
        train_args = TrainArguments(
            model=model,
            train_type=train_type,
            dataset=[dataset],
            num_train_epochs=1,
            per_device_train_batch_size=1 if is_multi_modal else 4,
            per_device_eval_batch_size=1 if is_multi_modal else 4,
            gradient_accumulation_steps=16 if is_multi_modal else 4,
            logging_steps=1,
            learning_rate=learning_rate,
            dataloader_num_workers=4,
            dataloader_drop_last=False,
            dataset_num_proc=4,
            output_dir=output_dir,
            deepspeed="zero3" if is_multi_modal else "zero2",
            lr_scheduler_type="constant",
            split_dataset_ratio=0,
            **kwargs
        )
        
        # Run SFT training
        result = sft_main(train_args)
        last_model_checkpoint = result.get('last_model_checkpoint')
        
        if not last_model_checkpoint:
            logger.warning("No checkpoint was returned by SFT training")
            return None
            
        logger.info(f"✅ SFT training completed, latest model checkpoint: {last_model_checkpoint}")
        return last_model_checkpoint
        
    except Exception as e:
        logger.error(f"Training error: {str(e)}", exc_info=True)
        # Try to return the last checkpoint if available
        checkpoints_dir = os.path.join(os.path.abspath("output"), "checkpoint-*")
        import glob
        checkpoints = sorted(glob.glob(checkpoints_dir), key=os.path.getmtime)
        if checkpoints:
            logger.info(f"Returning last available checkpoint: {checkpoints[-1]}")
            return checkpoints[-1]
        raise

def main():
    try:
        parser = argparse.ArgumentParser(description="SFT Training Script")
        parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Model name or path")
        parser.add_argument("--dataset", type=str, required=True, help="Path to GRPO dataset for filtering")
        parser.add_argument("--resume_from_checkpoint", type=str, help="Resume training from checkpoint")
        parser.add_argument("--output_dir", type=str, help="Model output directory")
        parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate")
        parser.add_argument("--train_type", type=str, default="full", choices=["full", "lora"], help="Training method: full or lora")
        parser.add_argument("--skip_filtering", action="store_true", help="Skip filtering and use dataset directly for SFT")
        args = parser.parse_args()

        # Check if dataset file exists
        if not os.path.exists(args.dataset):
            logger.error(f"Dataset file does not exist: {args.dataset}")
            sys.exit(1)

        # Create default output directory if not specified
        if not args.output_dir:
            args.output_dir = f"output/sft_{time.strftime('%Y%m%d_%H%M%S')}"

        # Process dataset
        dataset_path = args.dataset
        if not args.skip_filtering:
            logger.info(f"Filtering GRPO dataset: {args.dataset}")
            dataset_path = filter_grpo_data(args.dataset, args.output_dir)
            logger.info(f"Filtered dataset created: {dataset_path}")
        
        # Prepare training kwargs
        kwargs = {}
        if args.resume_from_checkpoint:
            kwargs["resume_from_checkpoint"] = args.resume_from_checkpoint

        # Run SFT training
        checkpoint = train_sft(
            dataset_path, 
            args.model, 
            output_dir=args.output_dir, 
            learning_rate=args.learning_rate,
            train_type=args.train_type,
            **kwargs
        )
        
        # Save checkpoint path for future reference
        with open("checkpoint.txt", "w") as f:
            f.write(checkpoint + "\n")
            
        # Also save a copy in the output directory
        with open(os.path.join(args.output_dir, "checkpoint.txt"), "w") as f:
            f.write(checkpoint + "\n")
            
    except KeyboardInterrupt:
        logger.info("Received user interrupt, exiting...")
    except Exception as e:
        logger.error(f"Program execution error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main() 