#!/bin/bash
# Add new environment variables to prevent OpenGL errors
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3
INITIAL_LR=0.000001 # 1e-6
FINAL_LR=0.0000001 # 1e-8
WARMUP_RATIO=0.05 # Add warmup ratio
WARMUP_INITIAL_LR=0.0000001 # 1e-7, WARMUP initial learning rate
# Start training from the specified model, optionally passing in the dataset path
# Usage: ./seea/train/run_self_play.sh <train_type> <train_method> <policy_model_path> <reward_model_path> [save_directory] [policy_network_dataset_path] [reward_model_dataset_path] [use_delta_reward(DPO only)] [reward_model_train_mode(true|false|ttrl)]

# First, clean up any existing AI2Thor processes
echo "🔸 Cleaning up AI2Thor processes..."
bash ./seea/utils/kill_thor.sh

# Function to kill all background processes
cleanup() {
    echo "Termination signal caught, cleaning up all background processes..."
    # Create a stop file to indicate exit is needed
    touch "$STOP_FILE"
    # Try to kill all processes created by the current script
    pkill -P $$
    # Clean up AI2Thor processes
    bash ./seea/utils/kill_thor.sh
    exit 1
}

# Set to capture more signal types
trap cleanup SIGINT SIGTERM SIGHUP

# Recursive process killing function: kill the specified PID and all its child processes
kill_tree() {
    local pid=$1
    for child in $(ps -o pid= --ppid "$pid"); do
        kill_tree "$child"
    done
    echo "Killing process $pid"
    kill -9 "$pid" 2>/dev/null
}

# Check parameters
if [ $# -lt 4 ]; then
    echo "Usage: $0 <train_type(GRPO|DPO)> <train_method(lora|full)> <policy_model_path> <reward_model_path> [save_directory] [policy_network_dataset_path] [reward_model_dataset_path] [use_delta_reward(DPO only)] [reward_model_train_mode(true|false|ttrl)]"
    echo "Example: $0 DPO lora /path/to/policy_model /path/to/reward_model /path/to/save /path/to/dataset \"\" true true"
    echo "Example: $0 GRPO full /path/to/policy_model /path/to/reward_model /path/to/save /path/to/policy_dataset /path/to/reward_dataset \"\" ttrl"
    exit 1
fi

# Get and check training type
TRAIN_TYPE=$(echo "$1" | tr '[:lower:]' '[:upper:]')
if [ "$TRAIN_TYPE" != "GRPO" ] && [ "$TRAIN_TYPE" != "DPO" ]; then
    echo "❗Training type must be GRPO or DPO"
    exit 1
fi
echo "🔸 Training type: $TRAIN_TYPE"

# Get and check training method
TRAIN_METHOD=$(echo "$2" | tr '[:upper:]' '[:lower:]')
if [ "$TRAIN_METHOD" != "lora" ] && [ "$TRAIN_METHOD" != "full" ]; then
    echo "❗Training method must be lora or full"
    exit 1
fi
echo "🔸 Training method: $TRAIN_METHOD"

# Get and check policy model path
CURRENT_MODEL="$3"
if [ ! -d "$CURRENT_MODEL" ]; then
    echo "❗Policy model path does not exist: $CURRENT_MODEL"
    exit 1
fi
echo "🔸 Policy model path: $CURRENT_MODEL"

# Get and check reward model path
INITIAL_REWARD_MODEL="$4"
if [ ! -d "$INITIAL_REWARD_MODEL" ]; then
    echo "❗Reward model path does not exist: $INITIAL_REWARD_MODEL"
    exit 1
fi
echo "🔸 Reward model path: $INITIAL_REWARD_MODEL"

# Set save directory
if [ $# -ge 5 ] && [ -n "$5" ]; then
    SAVE_DIR="$5"
else
    SAVE_DIR="outputs"
fi
echo "🔸 Save directory: $SAVE_DIR"

# Create new output directory
OUTPUT_DIR="$SAVE_DIR/${TRAIN_TYPE}_${TRAIN_METHOD}_$(date +%Y%m%d_%H%M%S)"
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR" || {
        echo "❗Failed to create output directory: $OUTPUT_DIR"
        exit 1
    }
fi

# Create files to save model paths
POLICY_MODEL_PATH_FILE="$OUTPUT_DIR/policy_model_path.txt"
REWARD_MODEL_PATH_FILE="$OUTPUT_DIR/reward_model_path.txt"

# Initially, the policy network uses the policy model, and the reward model uses the specified reward model
echo "$CURRENT_MODEL" > "$POLICY_MODEL_PATH_FILE"
echo "$INITIAL_REWARD_MODEL" > "$REWARD_MODEL_PATH_FILE"

# Get current model paths
CURRENT_POLICY_MODEL="$CURRENT_MODEL"
CURRENT_REWARD_MODEL="$INITIAL_REWARD_MODEL"

echo "🔸 Output directory: $OUTPUT_DIR"
echo "🔸 Starting iteration: 1"

# Add stop training signal file
STOP_FILE="$OUTPUT_DIR/stop_training"
echo "🔸 To stop training, create the file: $STOP_FILE (use the command: touch $STOP_FILE)"

# Set whether to use delta_reward (DPO only)
USE_DELTA_REWARD=false
if [ $# -ge 8 ] && [ -n "$8" ] && [ "$TRAIN_TYPE" = "DPO" ]; then
    DELTA_ARG=$(echo "$8" | tr '[:upper:]' '[:lower:]')
    if [ "$DELTA_ARG" = "true" ] || [ "$DELTA_ARG" = "1" ]; then
        USE_DELTA_REWARD=true
        echo "🔸 Using DPO data with delta_reward for policy training"
    else
        USE_DELTA_REWARD=false
        echo "🔸 Using normal DPO data for policy training"
    fi
elif [ "$TRAIN_TYPE" = "DPO" ]; then
    echo "🔸 DPO training did not specify the 8th parameter or it is empty, using normal DPO data for policy training"
fi

# Set reward model training mode (true|false|ttrl)
TRAIN_REWARD_MODEL=true # Default to true
ENABLE_TTRL_REWARD_ARGS="" # Default to empty

if [ $# -ge 9 ] && [ -n "$9" ]; then
    REWARD_MODE_ARG=$(echo "$9" | tr '[:upper:]' '[:lower:]')
    if [ "$REWARD_MODE_ARG" = "ttrl" ]; then
        TRAIN_REWARD_MODEL=true
        ENABLE_TTRL_REWARD_ARGS="--enable_ttrl_reward"
        echo "🔸 Performing TTRL reward model training (enable_ttrl_reward=true)"
    elif [ "$REWARD_MODE_ARG" = "true" ] || [ "$REWARD_MODE_ARG" = "1" ]; then
        TRAIN_REWARD_MODEL=true
        ENABLE_TTRL_REWARD_ARGS=""
        echo "🔸 Performing standard reward model training"
    elif [ "$REWARD_MODE_ARG" = "false" ] || [ "$REWARD_MODE_ARG" = "0" ]; then
        TRAIN_REWARD_MODEL=false
        ENABLE_TTRL_REWARD_ARGS=""
        echo "🔸 Skipping reward model training"
    else
        echo "⚠️ Invalid reward model training mode: '$9'. Using default: Performing standard reward model training."
        TRAIN_REWARD_MODEL=true
        ENABLE_TTRL_REWARD_ARGS=""
    fi
else
    echo "🔸 Reward model training mode not specified (9th parameter), defaulting to standard reward model training."
    TRAIN_REWARD_MODEL=true
    ENABLE_TTRL_REWARD_ARGS=""
fi

# Optional dataset path handling
INITIAL_TRAINING_DONE=false

# Handle DPO training dataset
if [ "$TRAIN_TYPE" = "DPO" ] && [ $# -ge 6 ] && [ -n "$6" ]; then
    DATASET="$6"
    echo "🔸 Using DPO dataset: $DATASET"
    
    # Check if dataset file exists
    if [ -f "$DATASET" ]; then
        # Perform initial training
        echo "===== 🚀 Starting DPO initial training ====="
        echo "🔸 Starting training, dataset: $DATASET"
        
        # If DPO training and using delta_reward, add extra arguments
        EXTRA_ARGS=""
        if [ "$USE_DELTA_REWARD" = "true" ]; then
            EXTRA_ARGS="--use_delta_reward"
            echo "🔸 Using DPO training with delta_reward"
        fi
        
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
            -m seea.train.train_script \
            --dataset "$DATASET" \
            --model "$CURRENT_MODEL" \
            --output_dir "$OUTPUT_DIR/initial_train" \
            --rlhf_type "$(echo "$TRAIN_TYPE" | tr '[:upper:]' '[:lower:]')" \
            --train_type "$TRAIN_METHOD" \
            $EXTRA_ARGS
        if [ $? -ne 0 ]; then
            echo "❗Error in initial training phase, exiting"
            exit 1
        fi
        
        # Update model path to new checkpoint
        CHECKPOINT=$(cat checkpoint.txt)
        echo "✅ Initial training complete, latest model checkpoint: $CHECKPOINT"
        
        # If LoRA training, merge the model
        if [ "$TRAIN_METHOD" = "lora" ]; then
            echo "🔸 Starting to merge initial LoRA model..."
            MERGED_MODEL_DIR="${CHECKPOINT}-merged"
            
            swift export \
                --adapters "$CHECKPOINT" \
                --merge_lora true \
                --output_dir "$MERGED_MODEL_DIR"
                
            if [ $? -ne 0 ]; then
                echo "❗Initial LoRA model merge failed, exiting"
                exit 1
            fi
            
            echo "✅ Initial LoRA model merge complete: $MERGED_MODEL_DIR"
            CURRENT_POLICY_MODEL="$MERGED_MODEL_DIR"
            CURRENT_REWARD_MODEL="$MERGED_MODEL_DIR"
        else
            CURRENT_POLICY_MODEL="$CHECKPOINT"
            CURRENT_REWARD_MODEL="$CHECKPOINT"
        fi
        
        # Update model path files
        echo "$CURRENT_POLICY_MODEL" > "$POLICY_MODEL_PATH_FILE"
        echo "$CURRENT_REWARD_MODEL" > "$REWARD_MODEL_PATH_FILE"
        
        INITIAL_TRAINING_DONE=true
    else
        echo "⚠️ DPO dataset file does not exist: $DATASET, skipping initial training"
    fi

# Handle GRPO training dataset
elif [ "$TRAIN_TYPE" = "GRPO" ] && [ $# -ge 6 ] && [ -n "$6" ]; then
    POLICY_DATASET="$6"
    if [ $# -ge 7 ] && [ -n "$7" ]; then
        REWARD_DATASET="$7"
        echo "🔸 Using GRPO reward model dataset: $REWARD_DATASET"
    else
        REWARD_DATASET=""
        echo "🔸 GRPO reward model dataset not provided"
    fi
    
    echo "🔸 Using GRPO policy network dataset: $POLICY_DATASET"
    
    # Check if policy network dataset file exists
    if [ -f "$POLICY_DATASET" ]; then
        # Perform policy network initial training
        echo "===== 🚀 Starting GRPO policy network initial training ====="
        echo "🔸 Starting policy network training, dataset: $POLICY_DATASET"
        
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
            -m seea.train.train_script \
            --dataset "$POLICY_DATASET" \
            --model "$CURRENT_MODEL" \
            --output_dir "$OUTPUT_DIR/initial_train_policy" \
            --rlhf_type "$(echo "$TRAIN_TYPE" | tr '[:upper:]' '[:lower:]')" \
            --train_type "$TRAIN_METHOD" \
            --train_phase policy
        if [ $? -ne 0 ]; then
            echo "❗Error in GRPO policy network initial training phase, exiting"
            exit 1
        fi
        
        # Update model path to new checkpoint
        POLICY_CHECKPOINT=$(cat "$OUTPUT_DIR/initial_train_policy/checkpoint.txt")
        echo "✅ GRPO policy network initial training complete, latest model checkpoint: $POLICY_CHECKPOINT"
        
        # If LoRA training, merge the model
        if [ "$TRAIN_METHOD" = "lora" ]; then
            echo "🔸 Starting to merge policy network LoRA model..."
            MERGED_MODEL_DIR="${POLICY_CHECKPOINT}-merged"
            
            swift export \
                --adapters "$POLICY_CHECKPOINT" \
                --merge_lora true \
                --output_dir "$MERGED_MODEL_DIR"
                
            if [ $? -ne 0 ]; then
                echo "❗Policy network LoRA model merge failed, exiting"
                exit 1
            fi
            
            echo "✅ Policy network LoRA model merge complete: $MERGED_MODEL_DIR"
            CURRENT_POLICY_MODEL="$MERGED_MODEL_DIR"
        else
            CURRENT_POLICY_MODEL="$POLICY_CHECKPOINT"
        fi
        
        # Update policy model path file
        echo "$CURRENT_POLICY_MODEL" > "$POLICY_MODEL_PATH_FILE"
        
        # Check if reward model dataset file exists
        if [ -n "$REWARD_DATASET" ] && [ -f "$REWARD_DATASET" ]; then
            # Perform reward model initial training
            echo "===== 🚀 Starting GRPO reward model initial training ====="
            echo "🔸 Starting reward model training, dataset: $REWARD_DATASET"
            
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=6 \
                -m seea.train.train_script \
                --dataset "$REWARD_DATASET" \
                --model "$INITIAL_REWARD_MODEL" \
                --output_dir "$OUTPUT_DIR/initial_train_reward" \
                --rlhf_type "$(echo "$TRAIN_TYPE" | tr '[:upper:]' '[:lower:]')" \
                --train_type "$TRAIN_METHOD" \
                --train_phase reward \
                $ENABLE_TTRL_REWARD_ARGS
            if [ $? -ne 0 ]; then
                echo "❗Error in GRPO reward model initial training phase, exiting"
                exit 1
            fi
            
            # Update model path to new checkpoint
            REWARD_CHECKPOINT=$(cat "$OUTPUT_DIR/initial_train_reward/checkpoint.txt")
            echo "✅ GRPO reward model initial training complete, latest model checkpoint: $REWARD_CHECKPOINT"
            
            # If LoRA training, merge the model
            if [ "$TRAIN_METHOD" = "lora" ]; then
                echo "🔸 Starting to merge reward model LoRA model..."
                MERGED_MODEL_DIR="${REWARD_CHECKPOINT}-merged"
                
                swift export \
                    --adapters "$REWARD_CHECKPOINT" \
                    --merge_lora true \
                    --output_dir "$MERGED_MODEL_DIR"
                    
                if [ $? -ne 0 ]; then
                    echo "❗Reward model LoRA model merge failed, exiting"
                    exit 1
                fi
                
                echo "✅ Reward model LoRA model merge complete: $MERGED_MODEL_DIR"
                CURRENT_REWARD_MODEL="$MERGED_MODEL_DIR"
            else
                CURRENT_REWARD_MODEL="$REWARD_CHECKPOINT"
            fi
            
            # Update reward model path file
            echo "$CURRENT_REWARD_MODEL" > "$REWARD_MODEL_PATH_FILE"
            
            INITIAL_TRAINING_DONE=true
        else
            echo "⚠️ GRPO reward model dataset file does not exist: $REWARD_DATASET, skipping reward model initial training"
            # Reward model continues to use the initial model
            echo "$INITIAL_REWARD_MODEL" > "$REWARD_MODEL_PATH_FILE"
            CURRENT_REWARD_MODEL="$INITIAL_REWARD_MODEL"
            INITIAL_TRAINING_DONE=true
        fi
    else
        echo "⚠️ GRPO policy network dataset file does not exist: $POLICY_DATASET, skipping initial training"
    fi
else
    echo "⚠️ Dataset not provided, skipping initial training"
fi

# Get model names
INIT_POLICY_MODEL_NAME=$(basename "$CURRENT_POLICY_MODEL")
INIT_REWARD_MODEL_NAME=$(basename "$CURRENT_REWARD_MODEL")

# Determine if it's a VL model first
IS_VL_MODEL=false
if [[ "$CURRENT_POLICY_MODEL" == *"VL"* ]] || [[ "$CURRENT_POLICY_MODEL" == *"vl"* ]] || \
   [[ "$CURRENT_REWARD_MODEL" == *"VL"* ]] || [[ "$CURRENT_REWARD_MODEL" == *"vl"* ]]; then
    IS_VL_MODEL=true
fi

# Set config paths based on VL status and TTRL mode for TRAIN_CONFIG_PATH
if [ "$IS_VL_MODEL" = true ]; then
    echo "🔸 VL model detected, using multimodal evaluation config..."
    CONFIG_PATH="configs/react_eval_config.yaml"
    if [ "$REWARD_MODE_ARG" = "ttrl" ]; then # Check if TTRL mode is active (REWARD_MODE_ARG is set earlier)
        echo "🔸 TTRL mode activated, using VL TTRL self-evolution training config..."
        TRAIN_CONFIG_PATH="configs/react_ttrl_self_evolve_config.yaml"
    else
        echo "🔸 Using standard VL multimodal training config..."
        TRAIN_CONFIG_PATH="configs/react_self_play_config.yaml"
    fi
else
    echo "🔸 Non-VL model detected, using language-only evaluation config..."
    CONFIG_PATH="configs/react_llm_eval_config.yaml"
    if [ "$REWARD_MODE_ARG" = "ttrl" ]; then # Check if TTRL mode is active
        echo "🔸 TTRL mode activated, using LLM TTRL self-evolution training config..."
        TRAIN_CONFIG_PATH="configs/react_llm_ttrl_self_evolve_config.yaml"
    else
        echo "🔸 Using standard language-only training config..."
        TRAIN_CONFIG_PATH="configs/react_llm_self_play_config.yaml"
    fi
fi

# Set number of iterations
NUM_ITERATIONS=100

# Add function to check if Python training process was terminated externally
check_external_termination() {
    # Check policy network training process
    if [ -z "$(pgrep -f 'torchrun.*seea.train.train_script')" ] && [ -f "$ITER_DIR/training_policy_started" ] && [ ! -f "$ITER_DIR/training_policy_completed" ]; then
        echo "⚠️ Detected that policy network training process may have been terminated externally, creating stop signal for graceful exit..." | tee -a "$ITER_DIR/log.txt"
        touch "$STOP_FILE"
        return 0
    fi
    
    # Check reward model training process
    if [ -z "$(pgrep -f 'torchrun.*seea.train.train_script')" ] && [ -f "$ITER_DIR/training_reward_started" ] && [ ! -f "$ITER_DIR/training_reward_completed" ]; then
        echo "⚠️ Detected that reward model training process may have been terminated externally, creating stop signal for graceful exit..." | tee -a "$ITER_DIR/log.txt"
        touch "$STOP_FILE"
        return 0
    fi
    
    return 1
}

# Create a function to check status at each key point
check_status_and_exit() {
    # Check for external termination
    check_external_termination
    
    # Check for stop signal file
    if [ -f "$STOP_FILE" ]; then
        echo "🛑 Detected stop signal file ($STOP_FILE), preparing to exit training loop..." | tee -a "$ITER_DIR/log.txt"
        return 0
    fi
    
    return 1
}

# Start iteration loop directly
for ((i=1; i<=NUM_ITERATIONS; i++)); do
    # Check if training should be stopped
    if [ -f "$STOP_FILE" ]; then
        echo "🛑 Detected stop signal file ($STOP_FILE), gracefully exiting training..."
        break
    fi
    
    echo "===== 🚀 Starting iteration $i of $TRAIN_TYPE ====="
    
    # Create output directory for current iteration
    ITER_DIR="$OUTPUT_DIR/iter_$i"
    mkdir -p "$ITER_DIR"
    
    # Read current policy and reward model paths
    CURRENT_POLICY_MODEL=$(cat "$POLICY_MODEL_PATH_FILE")
    CURRENT_REWARD_MODEL=$(cat "$REWARD_MODEL_PATH_FILE")
    
    # Get model names
    POLICY_MODEL_NAME=$(basename "$CURRENT_POLICY_MODEL")
    REWARD_MODEL_NAME=$(basename "$CURRENT_REWARD_MODEL")
    
    # 1. Deployment phase
    echo "🔸 Starting sampling model deployment service..."
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m seea.train.real_deploy --model_name "$POLICY_MODEL_NAME" --model "$CURRENT_POLICY_MODEL" --port 8000 &
    POLICY_DEPLOY_PID=$!
    echo "✅ Sampling model deployment process started, PID: $POLICY_DEPLOY_PID"

    # Wait for service to start
    echo "🔸 Waiting for service to fully start..."
    sleep 120

    if ! kill -0 $POLICY_DEPLOY_PID 2>/dev/null; then
        echo "❗Policy model deployment process not running"
        kill_tree $POLICY_DEPLOY_PID
        exit 1
    fi

    # 2. Evaluation (except for the first iteration)
    if [ $i -ne 1 ]; then
        echo "🔸 Starting evaluation for iteration $i..."
        export DISPLAY=:99
        Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
        sleep 3

        xvfb-run -a --server-args="-screen 0 1024x768x24 -ac +extension GLX +render -noreset" \
            python -m seea.eval.evaluate \
                --model_name "$POLICY_MODEL_NAME" \
                --base_url http://127.0.0.1:8000/v1 \
                --output_dir "$ITER_DIR/eval_results" \
                --config "$CONFIG_PATH"
        if [ $? -ne 0 ]; then
            echo "⚠️ Error in evaluation phase, continuing training" | tee -a "$ITER_DIR/log.txt"
        fi
        echo "✅ Evaluation for iteration $i complete" | tee -a "$ITER_DIR/log.txt"
        pkill Xvfb
    fi

    echo "🔸 Starting reward model deployment service..."
    CUDA_VISIBLE_DEVICES=4,5,6,7 python -m seea.train.reward_deploy --model_name "$REWARD_MODEL_NAME" --model "$CURRENT_REWARD_MODEL" --port 8001 &
    REWARD_DEPLOY_PID=$!
    echo "✅ Reward model deployment process started, PID: $REWARD_DEPLOY_PID"

    # Check if deployment process is running
    if ! kill -0 $REWARD_DEPLOY_PID 2>/dev/null; then
        echo "❗Reward model deployment process not running"
        kill_tree $REWARD_DEPLOY_PID
        exit 1
    fi

    sleep 120

    # 3. Sampling phase
    echo "🔸 Starting sampling..."
    
    # Ensure previous Xvfb processes are cleaned up
    pkill Xvfb || true
    sleep 2
    
    # Check status
    if check_status_and_exit; then
        break
    fi
    
    # Use random port number to avoid conflicts
    DISPLAY_NUM=$((100 + RANDOM % 900))
    export DISPLAY=:$DISPLAY_NUM
    
    # Start Xvfb and add retry mechanism
    MAX_RETRIES=3
    RETRY_COUNT=0
    
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        echo "🔸 Starting Xvfb (attempt $((RETRY_COUNT+1))/$MAX_RETRIES)..."
        Xvfb :$DISPLAY_NUM -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
        XVFB_PID=$!
        sleep 5
        
        # Check if Xvfb is running normally
        if kill -0 $XVFB_PID 2>/dev/null; then
            echo "✅ Xvfb started successfully, PID: $XVFB_PID"
            
            # Add different parameters based on training type
            if [ "$TRAIN_TYPE" = "DPO" ]; then
                xvfb-run -a --server-num=$DISPLAY_NUM --server-args="-screen 0 1024x768x24 -ac +extension GLX +render -noreset" \
                    python -m main \
                        --model_name "$POLICY_MODEL_NAME" \
                        --base_url http://127.0.0.1:8000/v1 \
                        --save_dir "$ITER_DIR" \
                        --enable_reflection \
                        --critic_model_name "$REWARD_MODEL_NAME" \
                        --critic_base_url http://127.0.0.1:8001/v1 \
                        --config "$TRAIN_CONFIG_PATH" 
            else
                xvfb-run -a --server-num=$DISPLAY_NUM --server-args="-screen 0 1024x768x24 -ac +extension GLX +render -noreset" \
                    python -m main \
                        --model_name "$POLICY_MODEL_NAME" \
                        --base_url http://127.0.0.1:8000/v1 \
                        --save_dir "$ITER_DIR" \
                        --critic_model_name "$REWARD_MODEL_NAME" \
                        --critic_base_url http://127.0.0.1:8001/v1 \
                        --config "$TRAIN_CONFIG_PATH"
            fi
            
            SAMPLE_RESULT=$?
            
            # Clean up Xvfb process regardless of success
            if kill -0 $XVFB_PID 2>/dev/null; then
                kill -TERM $XVFB_PID
                sleep 5  # Increase wait time to ensure process exits
                kill -9 $XVFB_PID 2>/dev/null || true
            fi
            pkill -f "Xvfb :$DISPLAY_NUM" || true  # Precisely match current Display
            sleep 3
            
            # If sampling is successful, break the loop
            if [ $SAMPLE_RESULT -eq 0 ]; then
                echo "✅ Sampling completed successfully"
                break
            else
                echo "⚠️ Sampling failed, error code: $SAMPLE_RESULT"
            fi
        else
            echo "⚠️ Xvfb failed to start"
        fi
        
        # Increment retry count
        RETRY_COUNT=$((RETRY_COUNT+1))
        
        # If there are still retries left, wait for a while before retrying
        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            echo "🔄 Retrying after 10 seconds..."
            sleep 10
        fi
    done
    
    # Check if all retries failed
    if [ $RETRY_COUNT -ge $MAX_RETRIES ] && [ $SAMPLE_RESULT -ne 0 ]; then
        echo "❗ Sampling phase still failed after multiple attempts" | tee -a "$ITER_DIR/log.txt"
    fi
    
    if [ $? -ne 0 ]; then
        echo "Error in sampling phase" | tee -a "$ITER_DIR/log.txt"
    fi
    pkill Xvfb

    # Select different sampling scripts based on training type
    if [ "$TRAIN_TYPE" = "DPO" ]; then
        if [ "$USE_DELTA_REWARD" = "true" ]; then
            DATASET_PATH="$ITER_DIR/sample_output_dpo_reward.txt"
            echo "🔸 Using DPO dataset with delta_reward"
        else
            DATASET_PATH="$ITER_DIR/sample_output_dpo.txt"
            echo "🔸 Using normal DPO dataset"
        fi
    else
        POLICY_DATASET_PATH="$ITER_DIR/sample_output_policy.txt"
        REWARD_DATASET_PATH="$ITER_DIR/sample_output_reward.txt"
        echo "🔸 Using GRPO policy network and reward model datasets"
    fi

    # 4. Stop deployment services
    echo "🔸 Terminating deployment services..."
    kill_tree $POLICY_DEPLOY_PID
    kill_tree $REWARD_DEPLOY_PID
    sleep 5
    
    # Check status
    if check_status_and_exit; then
        break
    fi

    # 5. Get and process dataset paths
    DATASET_FOUND=false
    POLICY_DATASET_FOUND=false
    REWARD_DATASET_FOUND=false
    
    # Handle DPO training dataset
    if [ "$TRAIN_TYPE" = "DPO" ] && [ -f "$DATASET_PATH" ]; then
        # Read path from sample_output.txt
        DATASET=$(cat "$DATASET_PATH")
        echo "🔸 Found DPO dataset path: $DATASET"
        
        # Check if file exists
        if [ -f "$DATASET" ]; then
            echo "✅ DPO dataset file confirmed: $DATASET"
            DATASET_FOUND=true
        else
            echo "❗DPO dataset file does not exist: $DATASET"
            echo "Attempting to find alternative file..."
            
            # Try to find other possible dataset files in the same directory
            DATASET_DIR=$(dirname "$DATASET")
            if [ -f "$DATASET_DIR/dataset_filtered.json" ]; then
                DATASET="$DATASET_DIR/dataset_filtered.json"
                echo "✅ Found alternative DPO dataset: $DATASET"
                DATASET_FOUND=true
            elif [ -f "$DATASET_DIR/dataset.jsonl" ]; then
                DATASET="$DATASET_DIR/dataset.jsonl"
                echo "✅ Found alternative DPO dataset: $DATASET"
                DATASET_FOUND=true
            else
                echo "⚠️ Could not find a valid DPO dataset file, skipping training for current iteration" | tee -a "$ITER_DIR/log.txt"
                # Do not exit, continue to next iteration
            fi
        fi
    # Handle GRPO training dataset
    elif [ "$TRAIN_TYPE" = "GRPO" ]; then
        # Check policy network dataset
        if [ -f "$POLICY_DATASET_PATH" ]; then
            POLICY_DATASET=$(cat "$POLICY_DATASET_PATH")
            echo "🔸 Found GRPO policy network dataset path: $POLICY_DATASET"
            
            # Check if file exists
            if [ -f "$POLICY_DATASET" ]; then
                echo "✅ GRPO policy network dataset file confirmed: $POLICY_DATASET"
                POLICY_DATASET_FOUND=true
            else
                echo "❗GRPO policy network dataset file does not exist: $POLICY_DATASET"
                echo "Attempting to find alternative file..."
                
                # Try to find other possible dataset files in the same directory
                DATASET_DIR=$(dirname "$POLICY_DATASET")
                if [ -f "$DATASET_DIR/dataset_filtered.json" ]; then
                    POLICY_DATASET="$DATASET_DIR/dataset_filtered.json"
                    echo "✅ Found alternative GRPO policy network dataset: $POLICY_DATASET"
                    POLICY_DATASET_FOUND=true
                elif [ -f "$DATASET_DIR/dataset.jsonl" ]; then
                    POLICY_DATASET="$DATASET_DIR/dataset.jsonl"
                    echo "✅ Found alternative GRPO policy network dataset: $POLICY_DATASET"
                    POLICY_DATASET_FOUND=true
                else
                    echo "⚠️ Could not find a valid GRPO policy network dataset file, skipping policy network training for current iteration" | tee -a "$ITER_DIR/log.txt"
                fi
            fi
        else
            echo "⚠️ Could not find $POLICY_DATASET_PATH file, unable to get GRPO policy network dataset path, skipping policy network training for current iteration" | tee -a "$ITER_DIR/log.txt"
        fi
        
        # Check reward model dataset
        if [ -f "$REWARD_DATASET_PATH" ]; then
            REWARD_DATASET=$(cat "$REWARD_DATASET_PATH")
            echo "🔸 Found GRPO reward model dataset path: $REWARD_DATASET"
            
            # Check if file exists
            if [ -f "$REWARD_DATASET" ]; then
                echo "✅ GRPO reward model dataset file confirmed: $REWARD_DATASET"
                REWARD_DATASET_FOUND=true
            else
                echo "❗GRPO reward model dataset file does not exist: $REWARD_DATASET"
                echo "Attempting to find alternative file..."
                
                # Try to find other possible dataset files in the same directory
                DATASET_DIR=$(dirname "$REWARD_DATASET")
                if [ -f "$DATASET_DIR/dataset_filtered.json" ]; then
                    REWARD_DATASET="$DATASET_DIR/dataset_filtered.json"
                    echo "✅ Found alternative GRPO reward model dataset: $REWARD_DATASET"
                    REWARD_DATASET_FOUND=true
                elif [ -f "$DATASET_DIR/dataset.jsonl" ]; then
                    REWARD_DATASET="$DATASET_DIR/dataset.jsonl"
                    echo "✅ Found alternative GRPO reward model dataset: $REWARD_DATASET"
                    REWARD_DATASET_FOUND=true
                else
                    echo "⚠️ Could not find a valid GRPO reward model dataset file, skipping reward model training for current iteration" | tee -a "$ITER_DIR/log.txt"
                fi
            fi
        else
            echo "⚠️ Could not find $REWARD_DATASET_PATH file, unable to get GRPO reward model dataset path, skipping reward model training for current iteration" | tee -a "$ITER_DIR/log.txt"
        fi
        
        # If at least one dataset is found, consider dataset found
        if [ "$POLICY_DATASET_FOUND" = "true" ] || [ "$REWARD_DATASET_FOUND" = "true" ]; then
            DATASET_FOUND=true
        fi
    else
        echo "⚠️ Dataset file not found, skipping training for current iteration" | tee -a "$ITER_DIR/log.txt"
    fi

    # 6. Training phase
    # Check status
    if check_status_and_exit; then
        break
    fi
    
    if [ "$DATASET_FOUND" = "true" ]; then
        # Calculate current learning rate (cosine annealing)
        CURRENT_LR=$(awk -v i=$i -v max=$NUM_ITERATIONS -v init=$INITIAL_LR -v final=$FINAL_LR 'BEGIN {
            # Cosine annealing formula: lr = final_lr + 0.5 * (init_lr - final_lr) * (1 + cos(pi * i / max))
            pi = atan2(0, -1)
            lr = final + 0.5 * (init - final) * (1 + cos(pi * i / max))
            printf "%.10f", lr
        }')
        
        echo "🔸 Current learning rate: $CURRENT_LR" | tee -a "$ITER_DIR/log.txt"
        
        # If DPO training
        if [ "$TRAIN_TYPE" = "DPO" ]; then
            echo "🔸 Starting DPO training, dataset: $DATASET"
            
            # Set extra arguments
            EXTRA_ARGS=""
            if [ "$USE_DELTA_REWARD" = "true" ]; then
                EXTRA_ARGS="--use_delta_reward"
                echo "🔸 Using DPO training with delta_reward"
            fi
            
            # Perform policy network training
            echo "🔸 Starting DPO policy network training, dataset: $DATASET"
            
            # Mark training start
            touch "$ITER_DIR/training_policy_started"

            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
                -m seea.train.train_script \
                --dataset "$DATASET" \
                --model "$CURRENT_POLICY_MODEL" \
                --output_dir "$ITER_DIR/model_policy" \
                --learning_rate $CURRENT_LR \
                --rlhf_type "$(echo "$TRAIN_TYPE" | tr '[:upper:]' '[:lower:]')" \
                --train_type "$TRAIN_METHOD" \
                $EXTRA_ARGS

            # Check training result
            TRAIN_RESULT=$?
            # Mark training complete
            touch "$ITER_DIR/training_policy_completed"

            if [ $TRAIN_RESULT -ne 0 ]; then
                echo "Error in policy network training phase, exiting loop" | tee -a "$ITER_DIR/log.txt"
                exit 1
            fi
            
            # Update policy network model path
            if [ "$TRAIN_METHOD" = "lora" ]; then
                echo "🔸 Starting to merge policy network LoRA model..."
                POLICY_CHECKPOINT=$(cat "$ITER_DIR/model_policy/checkpoint.txt")
                MERGED_MODEL_DIR="${POLICY_CHECKPOINT}-merged"
                
                swift export \
                    --adapters "$POLICY_CHECKPOINT" \
                    --merge_lora true \
                    --output_dir "$MERGED_MODEL_DIR"
                    
                if [ $? -ne 0 ]; then
                    echo "❗Policy network LoRA model merge failed, exiting loop" | tee -a "$ITER_DIR/log.txt"
                    exit 1
                fi
                
                echo "✅ Policy network LoRA model merge complete: $MERGED_MODEL_DIR" | tee -a "$ITER_DIR/log.txt"
                CURRENT_POLICY_MODEL="$MERGED_MODEL_DIR"
            else
                CURRENT_POLICY_MODEL=$(cat "$ITER_DIR/model_policy/checkpoint.txt")
            fi
            
            # Update policy model path file
            echo "$CURRENT_POLICY_MODEL" > "$POLICY_MODEL_PATH_FILE"
            
            # Perform reward model training (if enabled)
            if [ "$TRAIN_REWARD_MODEL" = "true" ]; then
                echo "🔸 Starting DPO reward model training, dataset: $DATASET"
                
                # Check status
                if check_status_and_exit; then
                    break
                fi
                
                # Mark training start
                touch "$ITER_DIR/training_reward_started"
                
                CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=6 \
                    -m seea.train.train_script \
                    --dataset "$DATASET" \
                    --model "$CURRENT_REWARD_MODEL" \
                    --output_dir "$ITER_DIR/model_reward" \
                    --learning_rate $CURRENT_LR \
                    --rlhf_type "$(echo "$TRAIN_TYPE" | tr '[:upper:]' '[:lower:]')" \
                    --train_type "$TRAIN_METHOD" \
                    --train_phase reward \
                    $ENABLE_TTRL_REWARD_ARGS
                
                # Check training result    
                TRAIN_RESULT=$?
                # Mark training complete
                touch "$ITER_DIR/training_reward_completed"
                
                if [ $TRAIN_RESULT -ne 0 ]; then
                    echo "Error in reward model training phase, exiting loop" | tee -a "$ITER_DIR/log.txt"
                    exit 1
                fi
                
                # Update reward model path
                if [ "$TRAIN_METHOD" = "lora" ]; then
                    echo "🔸 Starting to merge reward model LoRA model..."
                    REWARD_CHECKPOINT=$(cat "$ITER_DIR/model_reward/checkpoint.txt")
                    MERGED_MODEL_DIR="${REWARD_CHECKPOINT}-merged"
                    
                    swift export \
                        --adapters "$REWARD_CHECKPOINT" \
                        --merge_lora true \
                        --output_dir "$MERGED_MODEL_DIR"
                        
                    if [ $? -ne 0 ]; then
                        echo "❗Reward model LoRA model merge failed, exiting loop" | tee -a "$ITER_DIR/log.txt"
                        exit 1
                    fi
                    
                    echo "✅ Reward model LoRA model merge complete: $MERGED_MODEL_DIR" | tee -a "$ITER_DIR/log.txt"
                    CURRENT_REWARD_MODEL="$MERGED_MODEL_DIR"
                else
                    CURRENT_REWARD_MODEL=$(cat "$ITER_DIR/model_reward/checkpoint.txt")
                fi
                
                # Update reward model path file
                echo "$CURRENT_REWARD_MODEL" > "$REWARD_MODEL_PATH_FILE"
            else
                echo "🔸 Skipping reward model training" | tee -a "$ITER_DIR/log.txt"
            fi
            
            echo "✅ DPO training complete" | tee -a "$ITER_DIR/log.txt"
            echo "Policy network model: $CURRENT_POLICY_MODEL" | tee -a "$ITER_DIR/log.txt"
            if [ "$TRAIN_REWARD_MODEL" = "true" ]; then
                echo "Reward model: $CURRENT_REWARD_MODEL" | tee -a "$ITER_DIR/log.txt"
            fi
        else
            # If GRPO training, first perform policy network training
            if [ "$POLICY_DATASET_FOUND" = "true" ]; then
                echo "🔸 Starting GRPO policy network training, dataset: $POLICY_DATASET"
                
                # Mark training start
                touch "$ITER_DIR/training_policy_started"
                
                CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
                    -m seea.train.train_script \
                    --dataset "$POLICY_DATASET" \
                    --model "$CURRENT_POLICY_MODEL" \
                    --output_dir "$ITER_DIR/model_policy" \
                    --learning_rate $CURRENT_LR \
                    --rlhf_type "$(echo "$TRAIN_TYPE" | tr '[:upper:]' '[:lower:]')" \
                    --train_type "$TRAIN_METHOD" \
                    --train_phase policy
                
                # Check training result
                TRAIN_RESULT=$?
                # Mark training complete
                touch "$ITER_DIR/training_policy_completed"
                
                if [ $TRAIN_RESULT -ne 0 ]; then
                    echo "Error in GRPO policy network training phase, exiting loop" | tee -a "$ITER_DIR/log.txt"
                    exit 1
                fi
                
                # Update policy network model path
                if [ "$TRAIN_METHOD" = "lora" ]; then
                    echo "🔸 Starting to merge policy network LoRA model..."
                    POLICY_CHECKPOINT=$(cat "$ITER_DIR/model_policy/checkpoint.txt")
                    MERGED_MODEL_DIR="${POLICY_CHECKPOINT}-merged"
                    
                    swift export \
                        --adapters "$POLICY_CHECKPOINT" \
                        --merge_lora true \
                        --output_dir "$MERGED_MODEL_DIR"
                        
                    if [ $? -ne 0 ]; then
                        echo "❗Policy network LoRA model merge failed, exiting loop" | tee -a "$ITER_DIR/log.txt"
                        exit 1
                    fi
                    
                    echo "✅ Policy network LoRA model merge complete: $MERGED_MODEL_DIR" | tee -a "$ITER_DIR/log.txt"
                    CURRENT_POLICY_MODEL="$MERGED_MODEL_DIR"
                else
                    CURRENT_POLICY_MODEL=$(cat "$ITER_DIR/model_policy/checkpoint.txt")
                fi
                
                # Update policy model path file
                echo "$CURRENT_POLICY_MODEL" > "$POLICY_MODEL_PATH_FILE"
            else
                echo "⚠️ Skipping GRPO policy network training because no valid dataset was found" | tee -a "$ITER_DIR/log.txt"
            fi
            
            # Then perform reward model training (if enabled)
            if [ "$TRAIN_REWARD_MODEL" = "true" ] && [ "$REWARD_DATASET_FOUND" = "true" ]; then
                echo "🔸 Starting GRPO reward model training, dataset: $REWARD_DATASET"
                
                # Check status
                if check_status_and_exit; then
                    break
                fi
                
                # Mark training start
                touch "$ITER_DIR/training_reward_started"
                
                CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=6 \
                    -m seea.train.train_script \
                    --dataset "$REWARD_DATASET" \
                    --model "$CURRENT_REWARD_MODEL" \
                    --output_dir "$ITER_DIR/model_reward" \
                    --learning_rate $CURRENT_LR \
                    --rlhf_type "$(echo "$TRAIN_TYPE" | tr '[:upper:]' '[:lower:]')" \
                    --train_type "$TRAIN_METHOD" \
                    --train_phase reward \
                    $ENABLE_TTRL_REWARD_ARGS
                
                # Check training result
                TRAIN_RESULT=$?
                # Mark training complete
                touch "$ITER_DIR/training_reward_completed"
                
                if [ $TRAIN_RESULT -ne 0 ]; then
                    echo "Error in GRPO reward model training phase, exiting loop" | tee -a "$ITER_DIR/log.txt"
                    exit 1
                fi
                
                # Update reward model path
                if [ "$TRAIN_METHOD" = "lora" ]; then
                    echo "🔸 Starting to merge reward model LoRA model..."
                    REWARD_CHECKPOINT=$(cat "$ITER_DIR/model_reward/checkpoint.txt")
                    MERGED_MODEL_DIR="${REWARD_CHECKPOINT}-merged"
                    
                    swift export \
                        --adapters "$REWARD_CHECKPOINT" \
                        --merge_lora true \
                        --output_dir "$MERGED_MODEL_DIR"
                        
                    if [ $? -ne 0 ]; then
                        echo "❗Reward model LoRA model merge failed, exiting loop" | tee -a "$ITER_DIR/log.txt"
                        exit 1
                    fi
                    
                    echo "✅ Reward model LoRA model merge complete: $MERGED_MODEL_DIR" | tee -a "$ITER_DIR/log.txt"
                    CURRENT_REWARD_MODEL="$MERGED_MODEL_DIR"
                else
                    CURRENT_REWARD_MODEL=$(cat "$ITER_DIR/model_reward/checkpoint.txt")
                fi
                
                # Update reward model path file
                echo "$CURRENT_REWARD_MODEL" > "$REWARD_MODEL_PATH_FILE"
            else
                if [ "$TRAIN_REWARD_MODEL" = "true" ]; then
                    echo "⚠️ Skipping GRPO reward model training because no valid dataset was found" | tee -a "$ITER_DIR/log.txt"
                else
                    echo "🔸 Skipping reward model training" | tee -a "$ITER_DIR/log.txt"
                fi
            fi
            
            echo "✅ GRPO training complete" | tee -a "$ITER_DIR/log.txt"
            echo "Policy network model: $CURRENT_POLICY_MODEL" | tee -a "$ITER_DIR/log.txt"
            if [ "$TRAIN_REWARD_MODEL" = "true" ]; then
                echo "Reward model: $CURRENT_REWARD_MODEL" | tee -a "$ITER_DIR/log.txt"
            fi
        fi
    else
        echo "⚠️ Skipping training for current iteration because no valid dataset was found" | tee -a "$ITER_DIR/log.txt"
    fi

    # 8. Save key information for current iteration
    CURRENT_POLICY_MODEL=$(cat "$POLICY_MODEL_PATH_FILE")
    CURRENT_REWARD_MODEL=$(cat "$REWARD_MODEL_PATH_FILE")
    
    echo "Policy network model: $CURRENT_POLICY_MODEL" > "$ITER_DIR/status.txt"
    if [ "$TRAIN_REWARD_MODEL" = "true" ]; then
        echo "Reward model: $CURRENT_REWARD_MODEL" >> "$ITER_DIR/status.txt"
    fi
    echo "Training type: $TRAIN_TYPE" >> "$ITER_DIR/status.txt"
    echo "Training method: $TRAIN_METHOD" >> "$ITER_DIR/status.txt"
    if [ "$TRAIN_TYPE" = "GRPO" ]; then
        echo "Training phase: Policy network and reward model" >> "$ITER_DIR/status.txt"
    fi
    
    # Clean up any remaining AI2Thor processes
    echo "🔸 Cleaning up AI2Thor processes after iteration..."
    bash ./seea/utils/kill_thor.sh
    
    # Check status
    if check_status_and_exit; then
        break
    fi
    
    echo "===== ✅ Iteration $i of $TRAIN_TYPE complete =====" | tee -a "$ITER_DIR/log.txt"
done

# Save final status
CURRENT_POLICY_MODEL=$(cat "$POLICY_MODEL_PATH_FILE")
CURRENT_REWARD_MODEL=$(cat "$REWARD_MODEL_PATH_FILE")

echo "Policy network model: $CURRENT_POLICY_MODEL" > "$OUTPUT_DIR/final_status.txt"
if [ "$TRAIN_REWARD_MODEL" = "true" ]; then
    echo "Reward model: $CURRENT_REWARD_MODEL" >> "$OUTPUT_DIR/final_status.txt"
fi
echo "Training type: $TRAIN_TYPE" >> "$OUTPUT_DIR/final_status.txt"
echo "Training method: $TRAIN_METHOD" >> "$OUTPUT_DIR/final_status.txt"
if [ "$TRAIN_TYPE" = "GRPO" ]; then
    echo "Training phase: Policy network and reward model" >> "$OUTPUT_DIR/final_status.txt"
fi
if [ "$TRAIN_TYPE" = "DPO" ]; then
    echo "Using delta_reward: $USE_DELTA_REWARD" >> "$OUTPUT_DIR/final_status.txt"
fi
echo "🎉 All $TRAIN_TYPE training iterations complete! Results saved in: $OUTPUT_DIR"

# Kill all background processes
pkill -P $$
exit 0