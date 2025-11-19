#!/bin/bash
# Add environment variables to prevent OpenGL errors
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3
INITIAL_LR=0.000001 # 1e-6
FINAL_LR=0.0000001 # 1e-8
WARMUP_RATIO=0.05 # Add warmup ratio
WARMUP_INITIAL_LR=0.0000001 # 1e-7, WARMUP initial learning rate
# Start training from specified model, optionally pass dataset path
# Usage: ./seea/train/run_pipeline.sh <training_type> <training_method> <model_path> [save_directory] [dataset_path] [use_delta_reward(DPO only)] [resume]

# First clean up any existing AI2Thor processes
echo "🔸 Cleaning AI2Thor processes..."
bash ./seea/utils/kill_thor.sh

# Function to kill all background processes
cleanup() {
    echo "Caught Ctrl-C, killing all background processes..."
    pkill -P $$
    exit 1
}

# Set up the trap to catch Ctrl-C (SIGINT)
trap cleanup SIGINT

# Recursive process killing function: kill specified PID and all its child processes
kill_tree() {
    local pid=$1
    for child in $(ps -o pid= --ppid "$pid"); do
        kill_tree "$child"
    done
    echo "Killing process $pid"
    kill -9 "$pid" 2>/dev/null
}

# Check if training should be resumed
RESUME=false
if [ $# -ge 7 ]; then
    RESUME_ARG=$(echo "$7" | tr '[:upper:]' '[:lower:]')
    if [ "$RESUME_ARG" = "resume" ] || [ "$RESUME_ARG" = "true" ]; then
        RESUME=true
        echo "🔸 Checkpoint resume enabled"
    fi
fi

# Check parameters
if [ $# -lt 3 ]; then
    echo "Usage: $0 <training_type(GRPO|DPO)> <training_method(lora|full)> <model_path> [save_directory] [dataset_path] [use_delta_reward(DPO only)] [resume]"
    echo "Example: $0 DPO lora /path/to/model /path/to/save /path/to/dataset true"
    exit 1
fi

# Get training type and check
TRAIN_TYPE=$(echo "$1" | tr '[:lower:]' '[:upper:]')
if [ "$TRAIN_TYPE" != "GRPO" ] && [ "$TRAIN_TYPE" != "DPO" ]; then
    echo "❗Training type must be GRPO or DPO"
    exit 1
fi
echo "🔸 Training type: $TRAIN_TYPE"

# Get training method and check
TRAIN_METHOD=$(echo "$2" | tr '[:upper:]' '[:lower:]')
if [ "$TRAIN_METHOD" != "lora" ] && [ "$TRAIN_METHOD" != "full" ]; then
    echo "❗Training method must be lora or full"
    exit 1
fi
echo "🔸 Training method: $TRAIN_METHOD"

# Set save directory
if [ $# -ge 4 ]; then
    SAVE_DIR="$4"
else
    SAVE_DIR="outputs"
fi

# Set whether to use delta_reward (DPO only)
USE_DELTA_REWARD=false
if [ $# -ge 6 ] && [ "$TRAIN_TYPE" = "DPO" ]; then
    USE_DELTA_REWARD=$(echo "$6" | tr '[:upper:]' '[:lower:]')
    if [ "$USE_DELTA_REWARD" = "true" ] || [ "$USE_DELTA_REWARD" = "1" ]; then
        USE_DELTA_REWARD=true
        echo "🔸 Using DPO data with delta_reward"
    else
        USE_DELTA_REWARD=false
        echo "🔸 Using regular DPO data"
    fi
fi

# Get model path and check
CURRENT_MODEL="$3"
if [ ! -d "$CURRENT_MODEL" ]; then
    echo "❗Model path does not exist: $CURRENT_MODEL"
    exit 1
fi

# Set iteration count
NUM_ITERATIONS=100

# Set output directory and resume state
if [ "$RESUME" = "true" ] && [ $# -ge 4 ]; then
    # Check if training directory exists
    if [ ! -d "$SAVE_DIR" ]; then
        echo "❗Save directory to resume does not exist: $SAVE_DIR"
        exit 1
    fi
    
    # Use the input save directory as the training directory
    OUTPUT_DIR="$SAVE_DIR"
    echo "🔸 Continuing training in specified directory: $OUTPUT_DIR"
    
    # Find the latest iteration directory
    LATEST_ITER=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "iter_*" | sort -V | tail -n 1)
    if [ -z "$LATEST_ITER" ]; then
        echo "❗No iteration directory found in $OUTPUT_DIR"
        exit 1
    fi
    
    # Extract iteration number from directory name
    ITER_NUM=$(basename "$LATEST_ITER" | sed 's/iter_//')
    echo "🔸 Found latest iteration: $LATEST_ITER (iteration $ITER_NUM)"
    
    # Check status in the iteration directory
    if [ -f "$LATEST_ITER/status.txt" ]; then
        echo "🔸 Found status file, checking iteration completion..."
        grep -q "Latest model" "$LATEST_ITER/status.txt"
        if [ $? -eq 0 ]; then
            echo "✅ Iteration $ITER_NUM completed, starting from iteration $((ITER_NUM+1))"
            CURRENT_ITER=$((ITER_NUM+1))
            
            # Get current model path
            CURRENT_MODEL=$(grep "Current model:" "$LATEST_ITER/status.txt" | cut -d' ' -f2-)
            echo "🔸 Using model: $CURRENT_MODEL"
        else
            echo "⚠️ Iteration $ITER_NUM incomplete, checking specific stage..."
            CURRENT_ITER=$ITER_NUM
        fi
    else
        echo "⚠️ Status file not found, starting from iteration $ITER_NUM"
        CURRENT_ITER=$ITER_NUM
    fi
    
    # Check if model directory exists
    if [ -d "$LATEST_ITER/model" ]; then
        # Check if model training is complete
        if [ -f "$LATEST_ITER/model/checkpoint.txt" ]; then
            CHECKPOINT=$(cat "$LATEST_ITER/model/checkpoint.txt")
            if [ -d "$CHECKPOINT" ]; then
                echo "✅ Model training completed, using trained model: $CHECKPOINT"
                CURRENT_MODEL="$CHECKPOINT"
                
                # If LoRA, check if merged
                if [ "$TRAIN_METHOD" = "lora" ]; then
                    MERGED_MODEL_DIR="${CHECKPOINT}-merged"
                    if [ -d "$MERGED_MODEL_DIR" ]; then
                        echo "✅ LoRA model already merged: $MERGED_MODEL_DIR"
                        CURRENT_MODEL="$MERGED_MODEL_DIR"
                        # If model trained and merged, should start from next iteration
                        CURRENT_ITER=$((ITER_NUM+1))
                    else
                        echo "⚠️ LoRA model not merged, need to complete merge step"
                        # Need to merge LoRA model
                        echo "🔸 Starting LoRA model merge..."
                        swift export \
                            --adapters "$CHECKPOINT" \
                            --merge_lora true \
                            --output_dir "$MERGED_MODEL_DIR"
                            
                        if [ $? -ne 0 ]; then
                            echo "❗LoRA model merge failed, exiting"
                            exit 1
                        fi
                        
                        echo "✅ LoRA model merge completed: $MERGED_MODEL_DIR"
                        CURRENT_MODEL="$MERGED_MODEL_DIR"
                        CURRENT_ITER=$((ITER_NUM+1))
                    fi
                else
                    # Full model training completed, start from next iteration
                    CURRENT_ITER=$((ITER_NUM+1))
                fi
            else
                echo "⚠️ Model training incomplete or path doesn't exist: $CHECKPOINT"
                # Check if dataset exists, if so continue training phase
                if [ -f "$LATEST_ITER/dataset_filtered.json" ]; then
                    echo "✅ Dataset found, will continue training phase"
                    DATASET="$LATEST_ITER/dataset_filtered.json"
                    
                    # If valid checkpoint not found, try to find previous iteration's model
                    if [ "$ITER_NUM" -gt 1 ]; then
                        PREV_ITER_NUM=$((ITER_NUM-1))
                        PREV_ITER_DIR="$OUTPUT_DIR/iter_$PREV_ITER_NUM"
                        
                        if [ -f "$PREV_ITER_DIR/status.txt" ]; then
                            echo "🔸 Attempting to get model info from previous iteration..."
                            PREV_MODEL=$(grep "Current model:" "$PREV_ITER_DIR/status.txt" | cut -d' ' -f2-)
                            
                            if [ -n "$PREV_MODEL" ] && [ -d "$PREV_MODEL" ]; then
                                echo "✅ Found model from previous iteration: $PREV_MODEL"
                                CURRENT_MODEL="$PREV_MODEL"
                            fi
                        fi
                    fi
                else
                    echo "⚠️ Dataset not found, will start from sampling phase"
                fi
            fi
        else
            echo "⚠️ Model training incomplete (no checkpoint.txt)"
            # Check if dataset exists, if so continue training phase
            if [ -f "$LATEST_ITER/dataset_filtered.json" ]; then
                echo "✅ Dataset found, will continue training phase"
                DATASET="$LATEST_ITER/dataset_filtered.json"
                
                # If valid checkpoint not found, try to find previous iteration's model
                if [ "$ITER_NUM" -gt 1 ]; then
                    PREV_ITER_NUM=$((ITER_NUM-1))
                    PREV_ITER_DIR="$OUTPUT_DIR/iter_$PREV_ITER_NUM"
                    
                    if [ -f "$PREV_ITER_DIR/status.txt" ]; then
                        echo "🔸 Attempting to get model info from previous iteration..."
                        PREV_MODEL=$(grep "Current model:" "$PREV_ITER_DIR/status.txt" | cut -d' ' -f2-)
                        
                        if [ -n "$PREV_MODEL" ] && [ -d "$PREV_MODEL" ]; then
                            echo "✅ Found model from previous iteration: $PREV_MODEL"
                            CURRENT_MODEL="$PREV_MODEL"
                        fi
                    fi
                fi
            else
                echo "⚠️ Dataset not found, will start from sampling phase"
            fi
        fi
    else
        # Model directory doesn't exist, check dataset
        if [ -f "$LATEST_ITER/dataset_filtered.json" ]; then
            echo "✅ Dataset found, starting from training phase"
            DATASET="$LATEST_ITER/dataset_filtered.json"
            
            # Look for model from previous iteration
            if [ "$ITER_NUM" -gt 1 ]; then
                PREV_ITER_NUM=$((ITER_NUM-1))
                PREV_ITER_DIR="$OUTPUT_DIR/iter_$PREV_ITER_NUM"
                
                # Check if previous iteration has status file and model info
                if [ -f "$PREV_ITER_DIR/status.txt" ]; then
                    echo "🔸 Attempting to get model info from previous iteration..."
                    PREV_MODEL=$(grep "Current model:" "$PREV_ITER_DIR/status.txt" | cut -d' ' -f2-)
                    
                    if [ -n "$PREV_MODEL" ] && [ -d "$PREV_MODEL" ]; then
                        echo "✅ Found model from previous iteration: $PREV_MODEL"
                        CURRENT_MODEL="$PREV_MODEL"
                    else
                        echo "⚠️ Previous iteration's model doesn't exist or is invalid, using initial model"
                    fi
                else
                    echo "⚠️ Previous iteration has no status file, using initial model"
                fi
            else
                echo "⚠️ No previous iteration, using initial model: $CURRENT_MODEL"
            fi
        elif [ -f "$LATEST_ITER/sample_output.txt" ]; then
            # Check path in sample_output.txt
            SAMPLE_OUTPUT_PATH=$(cat "$LATEST_ITER/sample_output.txt")
            if [ -f "$SAMPLE_OUTPUT_PATH" ]; then
                echo "✅ Found dataset path from sampling output: $SAMPLE_OUTPUT_PATH"
                DATASET="$SAMPLE_OUTPUT_PATH"
            else
                echo "⚠️ Dataset from sampling output doesn't exist: $SAMPLE_OUTPUT_PATH"
                echo "⚠️ Will start from sampling phase"
            fi
        else
            echo "⚠️ Dataset not found, will start from sampling phase"
        fi
    fi
else
    # New training, create new output directory
    OUTPUT_DIR="$SAVE_DIR/${TRAIN_TYPE}_${TRAIN_METHOD}_$(date +%Y%m%d_%H%M%S)"
    if [ ! -d "$OUTPUT_DIR" ]; then
        mkdir -p "$OUTPUT_DIR" || {
            echo "❗Failed to create output directory: $OUTPUT_DIR"
            exit 1
        }
    fi
    CURRENT_ITER=1
fi

echo "🔸 Output directory: $OUTPUT_DIR"
echo "🔸 Starting iteration: $CURRENT_ITER"

# Handle initial dataset (if provided)
if [ $# -ge 5 ] && [ -z "$DATASET" ]; then
    DATASET="$5"
    
    # Check if dataset parameter is false (case insensitive)
    DATASET_LOWER=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')
    if [ "$DATASET_LOWER" = "false" ] || [ "$DATASET_LOWER" = "0" ] || [ "$DATASET_LOWER" = "no" ]; then
        echo "🔸 Dataset parameter is false, skipping initial dataset loading"
        unset DATASET
    else
        echo "🔸 Using initial dataset: $DATASET"
        
        # Check if dataset file exists
        if [ ! -f "$DATASET" ]; then
            echo "❗Dataset file does not exist: $DATASET"
            exit 1
        fi
        
        # Perform initial training
        echo "===== 🚀 Starting initial training ====="
        echo "🔸 Starting training with dataset: $DATASET"
        
        # If DPO training and using delta_reward, add extra parameters
        EXTRA_ARGS=""
        if [ "$TRAIN_TYPE" = "DPO" ] && [ "$USE_DELTA_REWARD" = "true" ]; then
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
            echo "❗Initial training phase error, exiting"
            exit 1
        fi
        
        # Update model path to new checkpoint
        CHECKPOINT=$(cat checkpoint.txt)
        echo "✅ Initial training completed, latest model checkpoint: $CHECKPOINT"
        
        # If LoRA training, need to merge model
        if [ "$TRAIN_METHOD" = "lora" ]; then
            echo "🔸 Starting initial LoRA model merge..."
            MERGED_MODEL_DIR="${CHECKPOINT}-merged"
            
            swift export \
                --adapters "$CHECKPOINT" \
                --merge_lora true \
                --output_dir "$MERGED_MODEL_DIR"
                
            if [ $? -ne 0 ]; then
                echo "❗Initial LoRA model merge failed, exiting"
                exit 1
            fi
            
            echo "✅ Initial LoRA model merge completed: $MERGED_MODEL_DIR"
            CURRENT_MODEL="$MERGED_MODEL_DIR"
        else
            CURRENT_MODEL="$CHECKPOINT"
        fi
    fi
fi

INIT_MODEL_NAME=$(basename "$CURRENT_MODEL")
MODEL_DIR=$(dirname "$CURRENT_MODEL")

# Determine if it's a VL model based on model name or directory
if [[ "$INIT_MODEL_NAME" == *"VL"* ]] || [[ "$INIT_MODEL_NAME" == *"vl"* ]] || 
   [[ "$MODEL_DIR" == *"VL"* ]] || [[ "$MODEL_DIR" == *"vl"* ]]; then
    echo "🔸 VL model detected, using multimodal training and evaluation configs..."
    CONFIG_PATH="configs/react_eval_config.yaml"
    TRAIN_CONFIG_PATH="configs/react_config.yaml"
else
    echo "🔸 Non-VL model detected, using pure language training and evaluation configs..."
    CONFIG_PATH="configs/react_llm_eval_config.yaml"
    TRAIN_CONFIG_PATH="configs/react_llm_config.yaml"
fi

# Start iteration loop
for ((i=CURRENT_ITER; i<=NUM_ITERATIONS; i++)); do
    echo "===== 🚀 Starting $TRAIN_TYPE iteration $i ====="
    
    # Create output directory for current iteration
    ITER_DIR="$OUTPUT_DIR/iter_$i"
    mkdir -p "$ITER_DIR"
    
    # Check if dataset exists, can skip to training phase (only for first iteration when resuming)
    SKIP_TO_TRAIN=false
    if [ -n "$DATASET" ] && [ -f "$DATASET" ] && [ $i -eq $CURRENT_ITER ] && [ "$RESUME" = "true" ]; then
        echo "🔸 Resuming training: Dataset found: $DATASET, skipping deployment, testing, and sampling phases of iteration $i, going directly to training phase"
        SKIP_TO_TRAIN=true
    fi
    
    if [ "$SKIP_TO_TRAIN" = "false" ]; then
        # 1. Deployment phase
        echo "🔸 Starting deployment service..."
        python -m seea.train.deploy_script --model_name "$INIT_MODEL_NAME" --model "$CURRENT_MODEL" --port 8000
        if [ $? -ne 0 ]; then
            echo "Deployment phase error, exiting loop" | tee -a "$ITER_DIR/log.txt"
            exit 1
        fi
    
        DEPLOY_PID=$(cat deploy_pid.txt)
        echo "✅ Deployment successful, PID: $DEPLOY_PID" | tee -a "$ITER_DIR/log.txt"
        sleep 5
    
        MODEL_NAME=$(basename "$CURRENT_MODEL")
        # 2. Evaluation (except for first iteration)
        if [ $i -ne 1 ]; then
            echo "🔸 Starting evaluation for iteration $i..."
            export DISPLAY=:99
            Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
            sleep 3
            
            xvfb-run -a --server-args="-screen 0 1024x768x24 -ac +extension GLX +render -noreset" \
                python -m seea.eval.evaluate \
                    --model_name "$INIT_MODEL_NAME" \
                    --base_url http://127.0.0.1:8000/v1 \
                    --output_dir "$ITER_DIR/eval_results" \
                    --config "$CONFIG_PATH"
            if [ $? -ne 0 ]; then
                echo "⚠️ Evaluation phase error, continuing with training" | tee -a "$ITER_DIR/log.txt"
            fi
            echo "✅ Evaluation for iteration $i completed" | tee -a "$ITER_DIR/log.txt"
            pkill Xvfb
        fi
    
        # 3. Sampling phase
        echo "🔸 Starting sampling..."
        MODEL_NAME=$(basename "$CURRENT_MODEL")
        
        # Ensure previous Xvfb processes are cleaned up
        pkill Xvfb || true
        sleep 2
        
        # Use random display number to avoid conflicts
        DISPLAY_NUM=$((100 + RANDOM % 900))
        export DISPLAY=:$DISPLAY_NUM
        
        # Start Xvfb with retry mechanism
        MAX_RETRIES=3
        RETRY_COUNT=0
        
        while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
            echo "🔸 Starting Xvfb (attempt $((RETRY_COUNT+1))/$MAX_RETRIES)..."
            Xvfb :$DISPLAY_NUM -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
            XVFB_PID=$!
            sleep 5
            
            # Check if Xvfb is running properly
            if kill -0 $XVFB_PID 2>/dev/null; then
                echo "✅ Xvfb started successfully, PID: $XVFB_PID"
                
                # Add different parameters based on training type
                if [ "$TRAIN_TYPE" = "DPO" ]; then
                    xvfb-run -a --server-num=$DISPLAY_NUM --server-args="-screen 0 1024x768x24 -ac +extension GLX +render -noreset" \
                        python -m main \
                            --model_name "$INIT_MODEL_NAME" \
                            --base_url http://127.0.0.1:8000/v1 \
                            --save_dir "$ITER_DIR" \
                            --enable_reflection \
                            --config "$TRAIN_CONFIG_PATH"
                else
                    xvfb-run -a --server-num=$DISPLAY_NUM --server-args="-screen 0 1024x768x24 -ac +extension GLX +render -noreset" \
                        python -m main \
                            --model_name "$INIT_MODEL_NAME" \
                            --base_url http://127.0.0.1:8000/v1 \
                            --save_dir "$ITER_DIR" \
                            --config "$TRAIN_CONFIG_PATH"
                fi
                
                SAMPLE_RESULT=$?
                
                # Clean up Xvfb process regardless of success
                if kill -0 $XVFB_PID 2>/dev/null; then
                    kill -TERM $XVFB_PID
                    sleep 5  # Increase wait time to ensure process exits
                    kill -9 $XVFB_PID 2>/dev/null || true
                fi
                pkill -f "Xvfb :$DISPLAY_NUM" || true  # Exact match for current Display
                sleep 3
                
                # If sampling successful, break the loop
                if [ $SAMPLE_RESULT -eq 0 ]; then
                    echo "✅ Sampling completed successfully"
                    break
                else
                    echo "⚠️ Sampling failed, error code: $SAMPLE_RESULT"
                fi
            else
                echo "⚠️ Xvfb startup failed"
            fi
            
            # Increment retry counter
            RETRY_COUNT=$((RETRY_COUNT+1))
            
            # If there are more retries, wait before trying again
            if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
                echo "🔄 Waiting 10 seconds before retrying..."
                sleep 10
            fi
        done
        
        # Check if all retries failed
        if [ $RETRY_COUNT -ge $MAX_RETRIES ] && [ $SAMPLE_RESULT -ne 0 ]; then
            echo "❗ Sampling phase failed after multiple attempts" | tee -a "$ITER_DIR/log.txt"
        fi
        
        if [ $? -ne 0 ]; then
            echo "Sampling phase error" | tee -a "$ITER_DIR/log.txt"
        fi
        pkill Xvfb
    
        # Choose different sampling script based on training type
        if [ "$TRAIN_TYPE" = "DPO" ]; then
            if [ "$USE_DELTA_REWARD" = "true" ]; then
                DATASET_PATH="$ITER_DIR/sample_output_dpo_reward.txt"
                echo "🔸 Using DPO dataset with delta_reward"
            else
                DATASET_PATH="$ITER_DIR/sample_output_dpo.txt"
                echo "🔸 Using regular DPO dataset"
            fi
        else
            DATASET_PATH="$ITER_DIR/sample_output.txt"
        fi
    
        # 4. Stop deployment service
        echo "🔸 Terminating deployment service..."
        kill_tree $DEPLOY_PID
        sleep 5
    
        # 5. Get and process dataset path
        if [ -f "$DATASET_PATH" ]; then
            # Read path from sample_output.txt
            DATASET=$(cat "$DATASET_PATH")
            echo "🔸 Found dataset path: $DATASET"
            
            # Check if file exists
            if [ -f "$DATASET" ]; then
                echo "✅ Dataset file confirmed: $DATASET"
            else
                echo "❗Dataset file does not exist: $DATASET"
                echo "Trying to find alternative files..."
                
                # Try to find other possible dataset files in the same directory
                DATASET_DIR=$(dirname "$DATASET")
                if [ -f "$DATASET_DIR/dataset_filtered.json" ]; then
                    DATASET="$DATASET_DIR/dataset_filtered.json"
                    echo "✅ Found alternative dataset: $DATASET"
                elif [ -f "$DATASET_DIR/dataset.jsonl" ]; then
                    DATASET="$DATASET_DIR/dataset.jsonl"
                    echo "✅ Found alternative dataset: $DATASET"
                else
                    echo "❗Cannot find valid dataset file, exiting loop" | tee -a "$ITER_DIR/log.txt"
                    exit 1
                fi
            fi
        else
            echo "❗$DATASET_PATH file not found, cannot get dataset path" | tee -a "$ITER_DIR/log.txt"
            exit 1
        fi
    fi

    # 6. Training phase
    echo "🔸 Starting $TRAIN_TYPE training with dataset: $DATASET"
    
    # Calculate learning rate for current iteration (cosine decay with warmup)
    CURRENT_LR=$(awk -v i=$i -v max=$NUM_ITERATIONS -v init=$INITIAL_LR -v final=$FINAL_LR -v warmup=$WARMUP_RATIO -v warmup_init=$WARMUP_INITIAL_LR 'BEGIN {
        # Calculate warmup steps
        warmup_steps = max * warmup
        
        if (i <= warmup_steps) {
            # Warmup phase: linear growth from warmup_init to init
            lr = warmup_init + (init - warmup_init) * (i / warmup_steps)
        } else {
            # Cosine decay phase
            pi = atan2(0, -1)
            progress = (i - warmup_steps) / (max - warmup_steps)
            lr = final + 0.5 * (init - final) * (1 + cos(pi * progress))
        }
        printf "%.10f", lr
    }')
    
    echo "🔸 Current learning rate: $CURRENT_LR" | tee -a "$ITER_DIR/log.txt"
    
    # If DPO training and using delta_reward, add extra parameters
    EXTRA_ARGS=""
    if [ "$TRAIN_TYPE" = "DPO" ] && [ "$USE_DELTA_REWARD" = "true" ]; then
        EXTRA_ARGS="--use_delta_reward"
        echo "🔸 Using DPO training with delta_reward" | tee -a "$ITER_DIR/log.txt"
    fi
    
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
        -m seea.train.train_script \
        --dataset "$DATASET" \
        --model "$CURRENT_MODEL" \
        --output_dir "$ITER_DIR/model" \
        --learning_rate $CURRENT_LR \
        --rlhf_type "$(echo "$TRAIN_TYPE" | tr '[:upper:]' '[:lower:]')" \
        --train_type "$TRAIN_METHOD" \
        $EXTRA_ARGS
    if [ $? -ne 0 ]; then
        echo "Training phase error, exiting loop" | tee -a "$ITER_DIR/log.txt"
        exit 1
    fi

    # 7. If LoRA training, need to merge model
    if [ "$TRAIN_METHOD" = "lora" ]; then
        echo "🔸 Starting LoRA model merge..."
        CHECKPOINT=$(cat "$ITER_DIR/model/checkpoint.txt")
        MERGED_MODEL_DIR="${CHECKPOINT}-merged"
        
        swift export \
            --adapters "$CHECKPOINT" \
            --merge_lora true \
            --output_dir "$MERGED_MODEL_DIR"
            
        if [ $? -ne 0 ]; then
            echo "❗LoRA model merge failed, exiting loop" | tee -a "$ITER_DIR/log.txt"
            exit 1
        fi
        
        echo "✅ LoRA model merge completed: $MERGED_MODEL_DIR" | tee -a "$ITER_DIR/log.txt"
        CURRENT_MODEL="$MERGED_MODEL_DIR"
    else
        # If full training, use checkpoint directly
        if [ -f "$ITER_DIR/model/checkpoint.txt" ]; then
            CHECKPOINT=$(cat "$ITER_DIR/model/checkpoint.txt")
        else
            CHECKPOINT=$(cat checkpoint.txt)
        fi
        CURRENT_MODEL="$CHECKPOINT"
    fi

    echo "✅ Training completed, latest model: $CURRENT_MODEL" | tee -a "$ITER_DIR/log.txt"

    # 8. Save key information for current iteration
    echo "Current model: $CURRENT_MODEL" > "$ITER_DIR/status.txt"
    echo "Dataset: $DATASET" >> "$ITER_DIR/status.txt"
    echo "Training type: $TRAIN_TYPE" >> "$ITER_DIR/status.txt"
    echo "Training method: $TRAIN_METHOD" >> "$ITER_DIR/status.txt"
    echo "Iteration number: $i" >> "$ITER_DIR/status.txt"
    
    # Clean up any remaining AI2Thor processes
    echo "🔸 Cleaning up AI2Thor processes after iteration..."
    bash ./seea/utils/kill_thor.sh
    
    echo "===== ✅ $TRAIN_TYPE iteration $i completed =====" | tee -a "$ITER_DIR/log.txt"
done

# Save final status
echo "Final model: $CURRENT_MODEL" > "$OUTPUT_DIR/final_status.txt"
echo "Training type: $TRAIN_TYPE" >> "$OUTPUT_DIR/final_status.txt"
echo "Training method: $TRAIN_METHOD" >> "$OUTPUT_DIR/final_status.txt"
if [ "$TRAIN_TYPE" = "DPO" ]; then
    echo "Used delta_reward: $USE_DELTA_REWARD" >> "$OUTPUT_DIR/final_status.txt"
fi
echo "🎉 All $TRAIN_TYPE training iterations completed! Results saved in: $OUTPUT_DIR"

# Kill all background processes
pkill -P $$
exit 0