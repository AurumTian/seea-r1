#!/usr/bin/env python
import os
import argparse
import logging
from swift.llm import deploy_main, DeployArguments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Reward model deployment service script")
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--port", type=int, default=8001, help="Deployment service port")
    args = parser.parse_args()

    # Set visible GPUs (use last four cards)
    os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
    deploy_args = DeployArguments(
        model=args.model,
        served_model_name=args.model_name,
        infer_backend='vllm',
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=32768,
        verbose=False,
        port=args.port,
        **({"limit_mm_per_prompt": {'image': 30, 'video': 0}} if "VL" in args.model else {})
    )
    logger.info("Starting reward model deployment service...")
    deploy_main(deploy_args)

if __name__ == '__main__':
    main() 