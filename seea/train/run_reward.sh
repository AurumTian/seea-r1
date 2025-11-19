#!/bin/bash
# Script for training reward models, including sampling, evaluation, and training loops

# Add new environment variables to prevent OpenGL errors
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3
INITIAL_LR=0.000001 # 1e-6

# Usage: ./seea/train/run_reward.sh <model_path> [save_directory] [training_method(lora|full)] [initial_dataset_path]

# First, clean up any existing AI2Thor processes
echo "🔸 Cleaning up AI2Thor processes..."
bash ./seea/utils/kill_thor.sh

# Function to kill all background processes
cleanup() {
    echo "Caught termination signal, cleaning up all background processes..."
    # Create a stop file to indicate the need to exit
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

# Get overall accuracy
get_overall_accuracy() {
    local report_file=$1
    
    # Extract overall accuracy
    local overall_accuracy=$(grep "Overall Accuracy:" "$report_file" | grep -o "[0-9.]*%" | grep -o "[0-9.]*")
    
    # If the value is empty, set it to 0
    [ -z "$overall_accuracy" ] && overall_accuracy=0
    
    echo "$overall_accuracy"
}

# Check parameters
if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_path> [save_directory] [training_method(lora|full)] [initial_dataset_path]"
    echo "Example: $0 /path/to/model /path/to/save lora /path/to/initial_dataset.jsonl"
    exit 1
fi

# Get model path and check
MODEL_PATH="$1"
if [ ! -d "$MODEL_PATH" ]; then
    echo "❗Model path does not exist: $MODEL_PATH"
    exit 1
fi

# Set save directory
if [ $# -ge 2 ] && [ -n "$2" ]; then
    SAVE_DIR="$2"
else
    SAVE_DIR="reward_outputs"
fi
echo "🔸 Save directory: $SAVE_DIR"

# Set training method
if [ $# -ge 3 ] && [ -n "$3" ]; then
    TRAIN_METHOD=$(echo "$3" | tr '[:upper:]' '[:lower:]')
    if [ "$TRAIN_METHOD" != "lora" ] && [ "$TRAIN_METHOD" != "full" ]; then
        echo "❗Training method must be lora or full, defaulting to full"
        TRAIN_METHOD="full"
    fi
else
    TRAIN_METHOD="full"
fi
echo "🔸 Training method: $TRAIN_METHOD"

# Set initial dataset path
INITIAL_DATASET=""
if [ $# -ge 4 ] && [ -n "$4" ]; then
    INITIAL_DATASET="$4"
    if [ -f "$INITIAL_DATASET" ]; then
        echo "🔸 Initial dataset: $INITIAL_DATASET"
    else
        echo "❗Initial dataset file does not exist: $INITIAL_DATASET"
        INITIAL_DATASET=""
    fi
fi

# Create output directory
OUTPUT_DIR="$SAVE_DIR/reward_training_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
echo "🔸 Output directory: $OUTPUT_DIR"

# Add stop training signal file
STOP_FILE="$OUTPUT_DIR/stop_training"
echo "🔸 To stop training, please create the file: $STOP_FILE (use the command: touch $STOP_FILE)"

# Get model name and set corresponding configuration
MODEL_NAME=$(basename "$MODEL_PATH")
if [[ "$MODEL_PATH" == *"VL"* ]] || [[ "$MODEL_PATH" == *"vl"* ]]; then
    echo "🔸 Detected VL model, using multi-modal training and evaluation configuration..."
    CONFIG_PATH="configs/react_reward_config.yaml"
else
    echo "🔸 Detected non-VL model, using language-only training and evaluation configuration..."
    CONFIG_PATH="configs/react_llm_reward_config.yaml"
fi

# Set current model
CURRENT_MODEL="$MODEL_PATH"

# Function to check if Python training process was terminated externally
check_external_termination() {
    # Check reward model training process
    if [ -z "$(pgrep -f 'torchrun.*seea.train.train_script')" ] && [ -f "$ITER_DIR/training_reward_started" ] && [ ! -f "$ITER_DIR/training_reward_completed" ]; then
        echo "⚠️ Detected that the reward model training process may have been terminated externally, creating a stop signal for graceful exit..." | tee -a "$ITER_DIR/log.txt"
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

# Set accuracy threshold
ACCURACY_THRESHOLD=80.0 # Set Overall Accuracy threshold

# If there is an initial dataset, perform initial model training first
if [ -n "$INITIAL_DATASET" ]; then
    echo "===== 🚀 Starting training with initial dataset ====="
    
    # Create output directory for initial training
    INIT_DIR="$OUTPUT_DIR/initial_training"
    mkdir -p "$INIT_DIR"
    
    # Mark training start
    touch "$INIT_DIR/training_reward_started"
    
    echo "🔸 Starting reward model training with initial dataset: $INITIAL_DATASET"
    CURRENT_LR=$INITIAL_LR
    
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=6 \
        -m seea.train.train_script \
        --dataset "$INITIAL_DATASET" \
        --model "$CURRENT_MODEL" \
        --output_dir "$INIT_DIR/model_reward" \
        --learning_rate $CURRENT_LR \
        --rlhf_type "grpo" \
        --train_type "$TRAIN_METHOD" \
        --train_phase reward
    
    # Check training result
    TRAIN_RESULT=$?
    # Mark training completion
    touch "$INIT_DIR/training_reward_completed"
    
    if [ $TRAIN_RESULT -ne 0 ]; then
        echo "❗Error during initial dataset training phase, exiting" | tee -a "$INIT_DIR/log.txt"
        exit 1
    fi
    
    # Update reward model path
    if [ "$TRAIN_METHOD" = "lora" ]; then
        echo "🔸 Starting to merge reward model LoRA model..."
        REWARD_CHECKPOINT=$(cat "$INIT_DIR/model_reward/checkpoint.txt")
        MERGED_MODEL_DIR="${REWARD_CHECKPOINT}-merged"
        
        swift export \
            --adapters "$REWARD_CHECKPOINT" \
            --merge_lora true \
            --output_dir "$MERGED_MODEL_DIR"
            
        if [ $? -ne 0 ]; then
            echo "❗Reward model LoRA model merge failed, exiting" | tee -a "$INIT_DIR/log.txt"
            exit 1
        fi
        
        echo "✅ Reward model LoRA model merged successfully: $MERGED_MODEL_DIR" | tee -a "$INIT_DIR/log.txt"
        CURRENT_MODEL="$MERGED_MODEL_DIR"
    else
        CURRENT_MODEL=$(cat "$INIT_DIR/model_reward/checkpoint.txt")
    fi
    
    echo "✅ Initial dataset training completed" | tee -a "$INIT_DIR/log.txt"
    echo "Current model: $CURRENT_MODEL" | tee -a "$INIT_DIR/log.txt"
    echo "===== ✅ Initial dataset training completed ====="
fi

# Start iteration loop
ITER=1
while true; do
    # Check if training should be stopped
    if [ -f "$STOP_FILE" ]; then
        echo "🛑 Detected stop signal file ($STOP_FILE), gracefully exiting training..."
        break
    fi
    
    echo "===== 🚀 Starting iteration $ITER of reward model training ====="
    
    # Create output directory for the current iteration
    ITER_DIR="$OUTPUT_DIR/iter_$ITER"
    mkdir -p "$ITER_DIR"
    
    # 1. Deploy model
    echo "🔸 Starting model deployment service..."
    python -m seea.train.real_deploy --model_name "$MODEL_NAME" --model "$CURRENT_MODEL" --port 8000 &
    DEPLOY_PID=$!
    echo "✅ Model deployment process started, PID: $DEPLOY_PID"
    
    # Wait for the service to start
    echo "🔸 Waiting for the service to fully start..."
    sleep 60
    
    # Check if the deployment process is running
    if ! kill -0 $DEPLOY_PID 2>/dev/null; then
        echo "❗Model deployment process is not running"
        exit 1
    fi
    
    # 2. Sampling phase
    echo "🔸 Starting sampling..."
    
    # Ensure previous Xvfb processes have been cleaned up
    pkill Xvfb || true
    sleep 2
    
    # Check status
    if check_status_and_exit; then
        break
    fi
    
    # Use a random port number to avoid conflicts
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
            
            # Run sampling command
            xvfb-run -a --server-num=$DISPLAY_NUM --server-args="-screen 0 1024x768x24 -ac +extension GLX +render -noreset" \
                python -m main \
                    --model_name "$MODEL_NAME" \
                    --base_url http://127.0.0.1:8000/v1 \
                    --save_dir "$ITER_DIR" \
                    --config "$CONFIG_PATH"
            
            SAMPLE_RESULT=$?
            
            # Clean up Xvfb process regardless of success
            if kill -0 $XVFB_PID 2>/dev/null; then
                kill -TERM $XVFB_PID
                sleep 5  # Increase wait time to ensure process exits
                kill -9 $XVFB_PID 2>/dev/null || true
            fi
            pkill -f "Xvfb :$DISPLAY_NUM" || true  # Precisely match the current Display
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
        kill_tree $DEPLOY_PID
        exit 1
    fi
    
    # Clean up Xvfb processes
    pkill Xvfb || true
    
    # Get reward model dataset path
    REWARD_DATASET_PATH="$ITER_DIR/sample_output_reward.txt"
    
    # 3. Stop deployment service
    echo "🔸 Terminating deployment service..."
    kill_tree $DEPLOY_PID
    sleep 5
    
    # Check status
    if check_status_and_exit; then
        break
    fi
    
    # 4. Process dataset
    REWARD_DATASET_FOUND=false
    
    if [ -f "$REWARD_DATASET_PATH" ]; then
        REWARD_DATASET=$(cat "$REWARD_DATASET_PATH")
        echo "🔸 Found reward model dataset path: $REWARD_DATASET"
        
        # Check if the file exists
        if [ -f "$REWARD_DATASET" ]; then
            echo "✅ Reward model dataset file confirmed: $REWARD_DATASET"
            REWARD_DATASET_FOUND=true
        else
            echo "❗Reward model dataset file does not exist: $REWARD_DATASET"
            echo "⚠️ Skipping current iteration"
            ITER=$((ITER+1))
            continue
        fi
    else
        echo "⚠️ File $REWARD_DATASET_PATH not found, cannot get reward model dataset path"
        echo "⚠️ Skipping current iteration"
        ITER=$((ITER+1))
        continue
    fi
    
    # 5. Train reward model
    if [ "$REWARD_DATASET_FOUND" = "true" ]; then
        echo "🔸 Starting reward model training, dataset: $REWARD_DATASET"
        
        # Check status
        if check_status_and_exit; then
            break
        fi
        
        # Mark training start
        touch "$ITER_DIR/training_reward_started"
        
        # Calculate learning rate for the current iteration
        CURRENT_LR=$INITIAL_LR
        
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=6 \
            -m seea.train.train_script \
            --dataset "$REWARD_DATASET" \
            --model "$CURRENT_MODEL" \
            --output_dir "$ITER_DIR/model_reward" \
            --learning_rate $CURRENT_LR \
            --rlhf_type "grpo" \
            --train_type "$TRAIN_METHOD" \
            --train_phase reward
        
        # Check training result
        TRAIN_RESULT=$?
        # Mark training completion
        touch "$ITER_DIR/training_reward_completed"
        
        if [ $TRAIN_RESULT -ne 0 ]; then
            echo "❗Error during reward model training phase, exiting" | tee -a "$ITER_DIR/log.txt"
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
            
            echo "✅ Reward model LoRA model merged successfully: $MERGED_MODEL_DIR" | tee -a "$ITER_DIR/log.txt"
            CURRENT_MODEL="$MERGED_MODEL_DIR"
        else
            CURRENT_MODEL=$(cat "$ITER_DIR/model_reward/checkpoint.txt")
        fi
        
        echo "✅ Reward model training completed" | tee -a "$ITER_DIR/log.txt"
        echo "Current model: $CURRENT_MODEL" | tee -a "$ITER_DIR/log.txt"
        
        # 6. Analyze confusion matrix and check overall accuracy
        echo "🔸 Checking overall accuracy of confusion matrix..."
        
        # Directly check confusion matrix report
        ANALYSIS_DIR=$(dirname "$REWARD_DATASET")/analysis
        REPORT_PATH="$ANALYSIS_DIR/balanced_accuracy_report.txt"
        
        if [ -f "$REPORT_PATH" ]; then
            # Get overall accuracy
            OVERALL_ACCURACY=$(get_overall_accuracy "$REPORT_PATH")
            echo "🔸 Overall accuracy of the original dataset: ${OVERALL_ACCURACY}%"
            
            # Check if the threshold is met
            if (( $(echo "$OVERALL_ACCURACY >= $ACCURACY_THRESHOLD" | bc -l) )); then
                echo "✅ Reached target accuracy threshold of ${ACCURACY_THRESHOLD}%! Training completed."
                break
            else
                echo "🔸 Current accuracy ${OVERALL_ACCURACY}% has not reached the target threshold ${ACCURACY_THRESHOLD}%, continuing training."
            fi
        else
            echo "⚠️ Confusion matrix report file not found: $REPORT_PATH"
        fi
    else
        echo "⚠️ Skipping reward model training because no valid dataset was found" | tee -a "$ITER_DIR/log.txt"
    fi
    
    # 7. Save key information for the current iteration
    echo "Current model: $CURRENT_MODEL" > "$ITER_DIR/status.txt"
    echo "Training method: $TRAIN_METHOD" >> "$ITER_DIR/status.txt"
    
    # Clean up any remaining AI2Thor processes
    echo "🔸 Cleaning up AI2Thor processes after iteration..."
    bash ./seea/utils/kill_thor.sh
    
    # Check status
    if check_status_and_exit; then
        break
    fi
    
    echo "===== ✅ Iteration $ITER of reward model training completed =====" | tee -a "$ITER_DIR/log.txt"
    ITER=$((ITER+1))
done

# Save final status
echo "Current model: $CURRENT_MODEL" > "$OUTPUT_DIR/final_status.txt"
echo "Training method: $TRAIN_METHOD" >> "$OUTPUT_DIR/final_status.txt"
echo "🎉 Reward model training completed! Results saved in: $OUTPUT_DIR"

# Clean up all background processes
pkill -P $$
exit 0
