import os
import argparse
from seea.utils.logger import get_logger
from swift.llm import TrainArguments, sft_main

logger = get_logger(__name__)

# Set CUDA devices
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

# Training parameters
kwargs = {
    'per_device_train_batch_size': 2,
    'per_device_eval_batch_size': 2,
    'save_steps': 5,
    'gradient_accumulation_steps': 4,
    'num_train_epochs': 1,
}

def main(dataset):
    """
    Train SFT model
    """
    
    result = sft_main(
        TrainArguments(
            model='/media/users/name/models/Qwen/Qwen2.5-VL-7B-Instruct',
            dataset=dataset,
            split_dataset_ratio=0,
            deepspeed='zero2',
            **kwargs
        )
    )
    last_model_checkpoint = result['last_model_checkpoint']
    return last_model_checkpoint

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="SFT training script")
    parser.add_argument("--dataset", type=str, default="samples/visual/Qwen2_5-VL-72B-Instruct/alfworld/2025-03-18_18-00-50/multi_turn_sft.json", help="Dataset path")

    args = parser.parse_args()

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    # Handle dataset path (supports relative and absolute paths)
    if not os.path.isabs(args.dataset):
        dataset_path = os.path.abspath(os.path.join(PROJECT_ROOT, args.dataset))
    else:
        dataset_path = args.dataset

    # Ensure dataset path exists
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path does not exist: {dataset_path}")


    # Run training
    main(dataset=dataset_path)