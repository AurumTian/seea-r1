import os
import argparse
import logging
import subprocess
import time
import aiohttp
from swift.llm import InferClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Deployment service wrapper script")
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--port", type=int, default=8000, help="Deployment service port")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    # Use subprocess to start real_deploy.py (new process)
    command = [
        "python", "-u", "-m", "seea.train.real_deploy",
        "--model", args.model,
        "--port", str(args.port),
        "--model_name", args.model_name
    ]
    logger.info("Starting deployment service: %s", " ".join(command))
    proc = subprocess.Popen(command)
    
    # Wait for deployment service to start (health check)
    infer_client = InferClient(port=args.port)
    start_time = time.time()
    timeout = 300
    while True:
        try:
            models = infer_client.models
            logger.info("✅ Deployment service started successfully in %.1fs, available models: %s", time.time()-start_time, models)
            break
        except aiohttp.ClientConnectorError:
            if time.time()-start_time > timeout:
                proc.terminate()
                raise TimeoutError("Deployment service startup timeout")
            time.sleep(1)
    
    # Write deployment service PID to file for main process to terminate later
    with open("deploy_pid.txt", "w") as f:
        f.write(str(proc.pid))
    logger.info("Deployment service started, PID: %s", proc.pid)

if __name__ == '__main__':
    main()
