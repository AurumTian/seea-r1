import os
import re
import sys
import time
import signal
import argparse
from swift.llm import rlhf_main, RLHFArguments
from seea.utils.logger import get_logger
logger = get_logger(__name__)

# Signal handler function
def signal_handler(signum, frame):
    logger.warning(f"Received signal {signum}, attempting graceful exit...")
    # Do not exit immediately, allow the program to complete current operations
    sys.exit(0)


def train_rlhf(dataset, model, output_dir, rlhf_type='grpo', learning_rate=1e-6, train_phase='policy', train_type='full', enable_ttrl_reward=False, **kwargs):
    try:
        os.environ["WANDB_PROJECT"] = 'seea'
        os.makedirs(output_dir, exist_ok=True)
        run_name = re.search(r"(Qwen[^/]+/[^/]+/[^/]+)", output_dir).group(1)
        is_multi_modal = "VL" in dataset or "vl" in dataset or "VL" in model or "vl" in model
        logger.info(f"Output directory created: {os.path.abspath(output_dir)}")

        # Register signal handlers
        signal.signal(signal.SIGHUP, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Set log information based on training type and phase
        phase_info = f" ({train_phase} phase)" if train_phase else ""
        logger.info(f"🚀 Starting {rlhf_type.upper()}{phase_info} training, dataset: {dataset}")
        
        # Set different parameters based on training type
        if rlhf_type == 'dpo':
            args = RLHFArguments(
                rlhf_type='dpo',
                model=model,
                train_type=train_type,
                use_vllm=False,
                dataset=[dataset],
                num_train_epochs=1,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=8,
                logging_steps=1,
                learning_rate=learning_rate,
                dataloader_num_workers=4,
                dataset_num_proc=4,
                output_dir=output_dir,
                # report_to="wandb",
                deepspeed="zero3" if is_multi_modal else "zero2",
                run_name=run_name,
                **kwargs
            )
        else:  # grpo
            if train_phase == 'reward':
                # Reward model training phase
                args = RLHFArguments(
                    rlhf_type='grpo',
                    model=model,
                    train_type=train_type,
                    dataset=[dataset],
                    num_train_epochs=1,
                    use_vllm=True,
                    num_infer_workers=2,
                    external_plugins=['./seea/train/plugin.py'],
                    reward_funcs=['external_state_acc'],
                    max_completion_length=4096,
                    num_generations=24,
                    per_device_train_batch_size=1 if is_multi_modal else 2,
                    per_device_eval_batch_size=1 if is_multi_modal else 2,
                    gradient_accumulation_steps=8 if is_multi_modal else 4,
                    max_steps=256,
                    #save_steps=32,
                    split_dataset_ratio=0,
                    logging_steps=1,
                    learning_rate=learning_rate,
                    dataloader_num_workers=4,
                    deepspeed="zero3" if is_multi_modal else "zero2",
                    output_dir=output_dir,
                    lr_scheduler_type="cosine",
                    num_iterations=1,
                    loss_type="grpo",
                    # report_to="wandb",
                    run_name=run_name,
                    use_precomputed_advantages=False,
                    async_generate=False,
                    log_completions=True,
                    sleep_level=0,
                    **({"vllm_limit_mm_per_prompt": {'image': 30, 'video': 0}} if is_multi_modal else {}),
                    **kwargs
                )
            else:
                # Policy network training phase
                args = RLHFArguments(
                    rlhf_type='grpo',
                    model=model,
                    train_type=train_type,
                    use_vllm=False,
                    dataset=[dataset],
                    num_train_epochs=1,
                    reward_funcs=['accuracy', 'format'],
                    max_completion_length=4096,
                    num_generations=1,
                    per_device_train_batch_size=1 if is_multi_modal else 4,
                    per_device_eval_batch_size=1 if is_multi_modal else 4,
                    gradient_accumulation_steps=16 if is_multi_modal else 4,
                    # save_steps=32,
                    split_dataset_ratio=0,
                    # max_steps=32,
                    logging_steps=1,
                    learning_rate=learning_rate,
                    dataloader_num_workers=4,
                    dataset_num_proc=4,
                    deepspeed="zero3" if is_multi_modal else "zero2",
                    output_dir=output_dir,
                    lr_scheduler_type="constant",
                    num_iterations=1,
                    # report_to="wandb",
                    run_name=run_name,
                    use_precomputed_advantages=True,
                    loss_type="grpo",
                    **kwargs
                )
        
        result = rlhf_main(args)
        last_model_checkpoint = result['last_model_checkpoint']
        logger.info(f"✅ {rlhf_type.upper()}{phase_info} training completed, latest model checkpoint: {last_model_checkpoint}")
        return last_model_checkpoint
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        # Try to return the last checkpoint if it exists
        checkpoints_dir = os.path.join(os.path.abspath("output"), "checkpoint-*")
        import glob
        checkpoints = sorted(glob.glob(checkpoints_dir), key=os.path.getmtime)
        if checkpoints:
            logger.info(f"Returning the last available checkpoint: {checkpoints[-1]}")
            return checkpoints[-1]
        raise

def main():
    try:
        parser = argparse.ArgumentParser(description="RLHF training script")
        parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="Model name")
        parser.add_argument("--dataset", type=str, required=True, help="Dataset path")
        parser.add_argument("--resume_from_checkpoint", type=str, help="Resume training from specified checkpoint")
        parser.add_argument("--output_dir", type=str, help="Model output directory")
        parser.add_argument("--learning_rate", type=float, default=1e-6,help="Learning rate")
        parser.add_argument("--rlhf_type", type=str, default="grpo", choices=["grpo", "dpo"], help="RLHF training type")
        parser.add_argument("--train_phase", type=str, choices=["policy", "reward"], default="policy", help="GRPO training phase: policy or reward")
        parser.add_argument("--train_type", type=str, default="full", choices=["full", "lora"], help="Training method: full or lora")
        parser.add_argument("--enable_ttrl_reward", action="store_true", help="Enable TTRL reward")
        args = parser.parse_args()

        # Check if dataset file exists
        if not os.path.exists(args.dataset):
            logger.error(f"Dataset file not found: {args.dataset}")
            sys.exit(1)

        # If output directory is not specified, create a default output directory
        if not args.output_dir:
            phase_suffix = f"_{args.train_phase}" if args.train_phase else ""
            args.output_dir = f"output/{args.rlhf_type}{phase_suffix}_{time.strftime('%Y%m%d_%H%M%S')}"

        kwargs = {}
        if args.resume_from_checkpoint:
            kwargs["resume_from_checkpoint"] = args.resume_from_checkpoint

        checkpoint = train_rlhf(
            args.dataset, 
            args.model, 
            output_dir=args.output_dir, 
            rlhf_type=args.rlhf_type, 
            learning_rate=args.learning_rate,
            train_phase=args.train_phase,
            train_type=args.train_type,
            enable_ttrl_reward=args.enable_ttrl_reward,
            **kwargs
        )
        
        # Write checkpoint path to file for subsequent deployment
        with open("checkpoint.txt", "w") as f:
            f.write(checkpoint + "\n")  # Add newline character
            
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
