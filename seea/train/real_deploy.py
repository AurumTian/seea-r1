#!/usr/bin/env python
import os
import argparse
import logging
from swift.llm import deploy_main, DeployArguments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Real deployment service script")
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--port", type=int, default=8000, help="Deployment service port")
    args = parser.parse_args()

    # Set visible GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
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
    logger.info("Starting real deployment service...")
    deploy_main(deploy_args)

if __name__ == '__main__':
    main()
