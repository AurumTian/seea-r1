#!/bin/bash

# ==============================================================================
# Script to sequentially run SFT and evaluation for multiple iteration directories.
# Each SFT training uses the same initial base model.
# ==============================================================================

# --- Default Values (can be overridden by arguments) ---
DEFAULT_LEARNING_RATE="0.000001" # 1e-6

# --- Hardcoded Configuration (modify if necessary) ---
# CUDA device settings
CUDA_DEVICES="0,1,2,3,4,5,6,7"
NPROC_PER_NODE=8

# Path to the kill_thor.sh script.
# Please ensure this path is correct relative to the directory where you run this script, or use an absolute path.
KILL_THOR_SCRIPT_PATH="./seea/utils/kill_thor.sh" # Adjust according to the actual situation

# --- Argument Parsing ---
if [ $# -lt 4 ]; then
    echo "❗Error: Insufficient arguments."
    echo "Usage: $0 <train_method(lora|full)> <initial_model_path> <iterations_parent_directory> <eval_config_path> [learning_rate]"
    echo "Example: $0 full /path/to/model /path/to/iterations_parent /path/to/eval_config.yaml [0.000002]"
    exit 1
fi

TRAIN_METHOD=$(echo "$1" | tr '[:upper:]' '[:lower:]')
INITIAL_MODEL_PATH="$2"
ITERATIONS_BASE_DIR="$3"
EVAL_CONFIG_PATH="$4"
LEARNING_RATE="${5:-$DEFAULT_LEARNING_RATE}" # Use provided LR or default

# --- Validate Arguments ---
if [ "$TRAIN_METHOD" != "lora" ] && [ "$TRAIN_METHOD" != "full" ]; then
    echo "❗Error: Training method must be lora or full. You provided: $TRAIN_METHOD"
    exit 1
fi

# Note: INITIAL_MODEL_PATH can be a HuggingFace model ID, so direct -d or -f check might be too strict.
# The training script itself will validate the model path/ID.
if [ -z "$INITIAL_MODEL_PATH" ]; then # Basic check for non-empty string
    echo "❗Error: Initial model path cannot be empty."
    exit 1
fi

if [ ! -d "$ITERATIONS_BASE_DIR" ]; then
    echo "❗Error: Parent directory for iterations does not exist: $ITERATIONS_BASE_DIR"
    exit 1
fi

if [ ! -f "$EVAL_CONFIG_PATH" ]; then
    echo "❗Error: Evaluation configuration file does not exist: $EVAL_CONFIG_PATH"
    exit 1
fi

# ==============================================================================
# Script Body
# ==============================================================================

# Add environment variables to prevent OpenGL errors
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3

# Signal handling and cleanup function
cleanup() {
    echo "Ctrl-C captured or script exiting, terminating all background processes..."
    if [ -n "$DEPLOY_PID" ] && ps -p $DEPLOY_PID > /dev/null; then kill_tree $DEPLOY_PID; rm -f deploy_pid.txt; fi
    if [ -n "$XVFB_PID" ] && ps -p $XVFB_PID > /dev/null; then kill -9 $XVFB_PID 2>/dev/null; fi
    pkill Xvfb || true
    pkill -P $$
    echo "Cleanup complete."
    exit 1
}
trap cleanup SIGINT EXIT

# Recursive process killing function
kill_tree() {
    local pid=$1
    if ! ps -p "$pid" > /dev/null; then
        echo "Process $pid does not exist, no need to terminate."
        return
    fi
    echo "Terminating process tree $pid..."
    local children=$(ps -o pid= --ppid "$pid")
    for child in $children; do
        kill_tree "$child"
    done
    kill -TERM "$pid" 2>/dev/null && sleep 1
    kill -KILL "$pid" 2>/dev/null
    echo "Terminated process $pid."
}

# First, clean up any existing AI2Thor processes
if [ -f "$KILL_THOR_SCRIPT_PATH" ]; then
    echo "🔸 Cleaning up AI2Thor processes..."
    bash "$KILL_THOR_SCRIPT_PATH"
else
    echo "⚠️ Warning: kill_thor.sh script not found at '$KILL_THOR_SCRIPT_PATH'. Skipping AI2Thor process cleanup."
fi

LOG_DIR="$ITERATIONS_BASE_DIR/sequential_sft_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
MAIN_LOG_FILE="$LOG_DIR/main_run.log"
touch "$MAIN_LOG_FILE"

echo "🔸 Starting sequential SFT training and evaluation for iter_* directories under $ITERATIONS_BASE_DIR" | tee -a "$MAIN_LOG_FILE"
echo "🔸 Script Configuration:" | tee -a "$MAIN_LOG_FILE"
echo "   Log Directory: $LOG_DIR" | tee -a "$MAIN_LOG_FILE"
echo "   Training Method: $TRAIN_METHOD" | tee -a "$MAIN_LOG_FILE"
echo "   Initial Model (for each SFT): $INITIAL_MODEL_PATH" | tee -a "$MAIN_LOG_FILE"
echo "   Iterations Parent Directory: $ITERATIONS_BASE_DIR" | tee -a "$MAIN_LOG_FILE"
echo "   Evaluation Config File: $EVAL_CONFIG_PATH" | tee -a "$MAIN_LOG_FILE"
echo "   Learning Rate: $LEARNING_RATE" | tee -a "$MAIN_LOG_FILE"
echo "   CUDA Devices: $CUDA_DEVICES ($NPROC_PER_NODE processes per node)" | tee -a "$MAIN_LOG_FILE"


# Find all iter_* directories and sort them numerically
ITER_DIRS=$(find "$ITERATIONS_BASE_DIR" -maxdepth 1 -type d -name "iter_*" | sort -V)
if [ -z "$ITER_DIRS" ]; then
    echo "❗Error: No 'iter_*' directories found in '$ITERATIONS_BASE_DIR'." | tee -a "$MAIN_LOG_FILE"
    exit 1
fi

ITER_COUNT=$(echo "$ITER_DIRS" | wc -l)
CURRENT_ITER_NUM=0

for ITER_DIR_FULL_PATH in $ITER_DIRS; do
    CURRENT_MODEL_FOR_SFT="$INITIAL_MODEL_PATH"
    
    CURRENT_ITER_NUM=$((CURRENT_ITER_NUM + 1))
    ITER_NAME=$(basename "$ITER_DIR_FULL_PATH")
    
    ITER_SPECIFIC_LOG_DIR="$LOG_DIR/$ITER_NAME"
    mkdir -p "$ITER_SPECIFIC_LOG_DIR"
    ITER_LOG_FILE="$ITER_SPECIFIC_LOG_DIR/details.log"
    touch "$ITER_LOG_FILE"

    echo "" | tee -a "$MAIN_LOG_FILE" "$ITER_LOG_FILE"
    echo "===== 🚀 Starting processing for $ITER_NAME (Iteration $CURRENT_ITER_NUM / $ITER_COUNT) =====" | tee -a "$MAIN_LOG_FILE" "$ITER_LOG_FILE"

    DATASET_PATH="$ITER_DIR_FULL_PATH/dataset_filtered.json"
    CURRENT_ITER_OUTPUT_BASE="$ITER_DIR_FULL_PATH/sft_eval_output_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$CURRENT_ITER_OUTPUT_BASE"
    
    echo "🔸 Output will be saved to: $CURRENT_ITER_OUTPUT_BASE" | tee -a "$MAIN_LOG_FILE" "$ITER_LOG_FILE"

    if [ ! -f "$DATASET_PATH" ]; then
        echo "❗ dataset_filtered.json not found in $ITER_DIR_FULL_PATH, skipping this iteration." | tee -a "$MAIN_LOG_FILE" "$ITER_LOG_FILE"
        continue
    fi

    # Initial model path check is done at the beginning.
    # If it's a HF model ID, it won't be a dir/file, sft_train_script handles it.
    # For local paths, this check would be relevant.
    # if [ ! -d "$CURRENT_MODEL_FOR_SFT" ] && [ ! -f "$CURRENT_MODEL_FOR_SFT" ]; then 
    #     echo "❗ Input model for SFT '$CURRENT_MODEL_FOR_SFT' is not a valid HuggingFace ID or local path, cannot continue training for $ITER_NAME." | tee -a "$MAIN_LOG_FILE" "$ITER_LOG_FILE"
    #     exit 1 
    # fi

    echo "🔸 SFT training will use model: $CURRENT_MODEL_FOR_SFT" | tee -a "$MAIN_LOG_FILE" "$ITER_LOG_FILE"
    echo "🔸 Using dataset: $DATASET_PATH" | tee -a "$MAIN_LOG_FILE" "$ITER_LOG_FILE"

    SFT_MODEL_OUTPUT_DIR="$CURRENT_ITER_OUTPUT_BASE/sft_model_output"
    mkdir -p "$SFT_MODEL_OUTPUT_DIR"

    # --- SFT Training Phase ---
    echo "🔸 Starting SFT training ($ITER_NAME)..." | tee -a "$MAIN_LOG_FILE" "$ITER_LOG_FILE"
    
    # Ensure PYTHONPATH includes seea modules if not installed globally
    # export PYTHONPATH=$PYTHONPATH:/path/to/seea/parent/dir 
    # (Adjust if your seea modules are not directly in python path)

    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES torchrun --nproc_per_node=$NPROC_PER_NODE \
        -m seea.train.sft_train_script \
        --dataset "$DATASET_PATH" \
        --model "$CURRENT_MODEL_FOR_SFT" \
        --output_dir "$SFT_MODEL_OUTPUT_DIR" \
        --learning_rate "$LEARNING_RATE" \
        --train_type "$TRAIN_METHOD" \
        >> "$ITER_LOG_FILE" 2>&1

    SFT_EXIT_CODE=$?
    if [ $SFT_EXIT_CODE -ne 0 ]; then
        echo "❗ SFT training phase for $ITER_NAME failed (Error code: $SFT_EXIT_CODE), see details in: $ITER_LOG_FILE. Stopping script." | tee -a "$MAIN_LOG_FILE"
        exit 1
    fi

    MODEL_AFTER_SFT=""
    if [ "$TRAIN_METHOD" = "lora" ]; then
        if [ ! -f "$SFT_MODEL_OUTPUT_DIR/checkpoint.txt" ]; then
            echo "❗ checkpoint.txt file not found in $SFT_MODEL_OUTPUT_DIR after LoRA training, cannot merge." | tee -a "$MAIN_LOG_FILE" "$ITER_LOG_FILE"
            exit 1
        fi
        CHECKPOINT_PATH_FILE_CONTENT=$(cat "$SFT_MODEL_OUTPUT_DIR/checkpoint.txt")
        
        if [[ "$CHECKPOINT_PATH_FILE_CONTENT" == /* ]]; then
            ACTUAL_CHECKPOINT_DIR="$CHECKPOINT_PATH_FILE_CONTENT"
        else 
            ACTUAL_CHECKPOINT_DIR="$SFT_MODEL_OUTPUT_DIR/$CHECKPOINT_PATH_FILE_CONTENT"
        fi
        
        if [ ! -d "$ACTUAL_CHECKPOINT_DIR" ]; then
            echo "❗ LoRA checkpoint directory '$ACTUAL_CHECKPOINT_DIR' (from $SFT_MODEL_OUTPUT_DIR/checkpoint.txt) is invalid, cannot merge." | tee -a "$MAIN_LOG_FILE" "$ITER_LOG_FILE"
            exit 1
        fi

        MERGED_MODEL_DIR="${ACTUAL_CHECKPOINT_DIR}-merged"
        
        echo "🔸 Starting LoRA adapter merge (from $ACTUAL_CHECKPOINT_DIR) ..." | tee -a "$MAIN_LOG_FILE" "$ITER_LOG_FILE"
        # Ensure swift command is available
        swift export \
            --ckpt_dir "$ACTUAL_CHECKPOINT_DIR" \
            --merge_lora true \
            --output_dir "$MERGED_MODEL_DIR" \
            >> "$ITER_LOG_FILE" 2>&1
            
        MERGE_EXIT_CODE=$?
        if [ $MERGE_EXIT_CODE -ne 0 ]; then
            echo "❗ LoRA model merge failed (Error code: $MERGE_EXIT_CODE), see details in: $ITER_LOG_FILE. Stopping script." | tee -a "$MAIN_LOG_FILE"
            exit 1
        fi
        echo "✅ LoRA model merge completed: $MERGED_MODEL_DIR" | tee -a "$MAIN_LOG_FILE" "$ITER_LOG_FILE"
        MODEL_AFTER_SFT="$MERGED_MODEL_DIR"
    else # full training
        if [ -f "$SFT_MODEL_OUTPUT_DIR/checkpoint.txt" ]; then
            CHECKPOINT_PATH_FILE_CONTENT=$(cat "$SFT_MODEL_OUTPUT_DIR/checkpoint.txt")
            if [[ "$CHECKPOINT_PATH_FILE_CONTENT" == /* ]]; then
                 ACTUAL_CHECKPOINT_DIR="$CHECKPOINT_PATH_FILE_CONTENT"
            else
                 ACTUAL_CHECKPOINT_DIR="$SFT_MODEL_OUTPUT_DIR/$CHECKPOINT_PATH_FILE_CONTENT"
            fi

            if [ -d "$ACTUAL_CHECKPOINT_DIR" ]; then
                 MODEL_AFTER_SFT="$ACTUAL_CHECKPOINT_DIR"
            else
                 echo "❗ Full training checkpoint path '$ACTUAL_CHECKPOINT_DIR' (from $SFT_MODEL_OUTPUT_DIR/checkpoint.txt) is invalid, stopping." | tee -a "$MAIN_LOG_FILE" "$ITER_LOG_FILE"
                 exit 1
            fi
        else
            echo "❗ checkpoint.txt file not found in $SFT_MODEL_OUTPUT_DIR after full training, stopping." | tee -a "$MAIN_LOG_FILE" "$ITER_LOG_FILE"
            exit 1
        fi
    fi

    if [ -z "$MODEL_AFTER_SFT" ] || [[ ! -d "$MODEL_AFTER_SFT" && ! -f "$MODEL_AFTER_SFT" ]]; then
         echo "❗ Could not determine a valid model path from training output (for evaluation), stopping." | tee -a "$MAIN_LOG_FILE" "$ITER_LOG_FILE"
         exit 1
    fi
    
    echo "✅ $ITER_NAME SFT training completed. This round of evaluation will use the newly trained model: $MODEL_AFTER_SFT" | tee -a "$MAIN_LOG_FILE" "$ITER_LOG_FILE"

    # --- Evaluation Phase ---
    echo "🔸 Starting $ITER_NAME evaluation (using model: $MODEL_AFTER_SFT)..." | tee -a "$MAIN_LOG_FILE" "$ITER_LOG_FILE"
    
    pkill Xvfb || true 
    sleep 2
    EVAL_DISPLAY_NUM=$(( ( RANDOM % 700 ) + 200 )) 
    export DISPLAY=:$EVAL_DISPLAY_NUM
    
    EVAL_PORT=8000 # Fixed evaluation deployment port to 8000
    EVAL_RESULTS_DIR="$CURRENT_ITER_OUTPUT_BASE/eval_results"
    mkdir -p "$EVAL_RESULTS_DIR"

    echo "🔸 Starting evaluation deployment service (Model: $(basename "$MODEL_AFTER_SFT"), Port: $EVAL_PORT)..." | tee -a "$MAIN_LOG_FILE" "$ITER_LOG_FILE"
    
    # Clean up old deploy_pid.txt (if it exists)
    rm -f deploy_pid.txt

    # Run deployment script in the foreground
    echo "🔸 Executing evaluation deployment script: python -m seea.train.deploy_script --model_name \"$(basename "$MODEL_AFTER_SFT")\" --model \"$MODEL_AFTER_SFT\" --port $EVAL_PORT" | tee -a "$MAIN_LOG_FILE" "$ITER_LOG_FILE"
    python -m seea.train.deploy_script --model_name "$(basename "$MODEL_AFTER_SFT")" --model "$MODEL_AFTER_SFT" --port $EVAL_PORT >> "$ITER_LOG_FILE" 2>&1
    DEPLOY_SCRIPT_EXIT_CODE=$?
    
    DEPLOY_READY=false # Default: deployment not ready
    DEPLOY_PID=""      # Initialize DEPLOY_PID

    if [ $DEPLOY_SCRIPT_EXIT_CODE -ne 0 ]; then
        echo "⚠️ Evaluation deployment script (seea.train.deploy_script) execution failed, error code: $DEPLOY_SCRIPT_EXIT_CODE. Check $ITER_LOG_FILE" | tee -a "$MAIN_LOG_FILE" "$ITER_LOG_FILE"
    else
        # Deployment script exited successfully, now check deploy_pid.txt
        if [ ! -f "deploy_pid.txt" ]; then
            echo "⚠️ Evaluation deployment script exited successfully, but deploy_pid.txt file not found. Deployment service might not have started successfully or written PID. Check $ITER_LOG_FILE" | tee -a "$MAIN_LOG_FILE" "$ITER_LOG_FILE"
        else
            DEPLOY_PID_FROM_FILE=$(cat deploy_pid.txt)
            # Validate if PID is a number and the process exists
            if ! [[ "$DEPLOY_PID_FROM_FILE" =~ ^[0-9]+$ ]]; then
                echo "⚠️ Content of deploy_pid.txt ('$DEPLOY_PID_FROM_FILE') is not a valid PID. Check $ITER_LOG_FILE" | tee -a "$MAIN_LOG_FILE" "$ITER_LOG_FILE"
            elif ! ps -p "$DEPLOY_PID_FROM_FILE" > /dev/null; then
                echo "⚠️ Process with PID ($DEPLOY_PID_FROM_FILE) from deploy_pid.txt does not currently exist. Service might have exited. Check $ITER_LOG_FILE" | tee -a "$MAIN_LOG_FILE" "$ITER_LOG_FILE"
            else
                # PID is valid and process exists
                DEPLOY_PID="$DEPLOY_PID_FROM_FILE"
                echo "✅ Evaluation deployment script indicates success. Service PID: $DEPLOY_PID (from deploy_pid.txt). Waiting 10 seconds for service to stabilize..." | tee -a "$MAIN_LOG_FILE" "$ITER_LOG_FILE"
                sleep 10 # Fixed waiting time
                
                # Perform a final port check as a reference
                if netstat -tuln | grep -q ":$EVAL_PORT"; then
                    echo "   Port $EVAL_PORT is listening." | tee -a "$ITER_LOG_FILE"
                    DEPLOY_READY=true
                else
                    echo "   ⚠️ Warning: Port $EVAL_PORT does not seem to be listening after 10 seconds. Evaluation might fail." | tee -a "$ITER_LOG_FILE"
                    # Even if the port is not listening, based on the instruction "don't wait, test directly", we still set DEPLOY_READY to true,
                    # and rely on the subsequent evaluate step to handle connection failure.
                    # If strict port checking is desired, DEPLOY_READY=true should not be set here or handled accordingly.
                    DEPLOY_READY=true # Assume deploy_script successful exit and valid PID means ready, let evaluate try
                fi
            fi
        fi
    fi
    
    if [ "$DEPLOY_READY" = "false" ]; then
        echo "⚠️ Evaluation deployment could not be confirmed. Skipping evaluation for $ITER_NAME. Please check detailed deployment logs in $ITER_LOG_FILE." | tee -a "$MAIN_LOG_FILE" "$ITER_LOG_FILE"
        # If DEPLOY_PID was read from file but process doesn't exist, DEPLOY_PID might have a value here but be invalid, kill_tree will handle it
        # Or if deploy_pid.txt was not found, DEPLOY_PID will be empty
        if [ -n "$DEPLOY_PID" ] && ps -p "$DEPLOY_PID" > /dev/null; then 
             kill_tree "$DEPLOY_PID"
        fi 
        rm -f deploy_pid.txt # Ensure cleanup
        # DEPLOY_PID="" # DEPLOY_PID might not be effectively set in this case, or its represented process no longer needs special handling
    else
        # DEPLOY_READY is true, proceed to evaluation
        EVAL_MAX_RETRIES=3
        EVAL_RETRY_COUNT=0
        EVAL_RESULT=-1
        XVFB_PID="" 

        while [ $EVAL_RETRY_COUNT -lt $EVAL_MAX_RETRIES ]; do
            pkill -f "Xvfb :$EVAL_DISPLAY_NUM" || true
            sleep 1

            echo "🔸 Starting Xvfb for evaluation (Attempt $((EVAL_RETRY_COUNT+1))/$EVAL_MAX_RETRIES) on display :$EVAL_DISPLAY_NUM..." | tee -a "$MAIN_LOG_FILE" "$ITER_LOG_FILE"
            
            Xvfb :$EVAL_DISPLAY_NUM -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
            XVFB_PID=$!
            sleep 5 

            if kill -0 $XVFB_PID 2>/dev/null; then
                echo "✅ Xvfb for evaluation started successfully, PID: $XVFB_PID" | tee -a "$MAIN_LOG_FILE" "$ITER_LOG_FILE"
                
                xvfb-run -a --server-num=$EVAL_DISPLAY_NUM --server-args="-screen 0 1024x768x24 -ac +extension GLX +render -noreset" \
                    python -m seea.eval.evaluate \
                        --model_name "$(basename "$MODEL_AFTER_SFT")" \
                        --base_url http://127.0.0.1:$EVAL_PORT/v1 \
                        --output_dir "$EVAL_RESULTS_DIR" \
                        --config "$EVAL_CONFIG_PATH" \
                        >> "$ITER_LOG_FILE" 2>&1 
                EVAL_RESULT=$?
                
                if kill -0 $XVFB_PID 2>/dev/null; then 
                    kill -TERM $XVFB_PID
                    sleep 2
                    kill -9 $XVFB_PID 2>/dev/null || true
                fi
                XVFB_PID="" 
                pkill -f "Xvfb :$EVAL_DISPLAY_NUM" || true 
                sleep 1
                
                if [ $EVAL_RESULT -eq 0 ]; then
                    echo "✅ $ITER_NAME evaluation completed successfully. Results in $EVAL_RESULTS_DIR" | tee -a "$MAIN_LOG_FILE" "$ITER_LOG_FILE"
                    break
                else
                    echo "⚠️ $ITER_NAME evaluation failed (Error code: $EVAL_RESULT), see details in: $ITER_LOG_FILE" | tee -a "$MAIN_LOG_FILE"
                fi
            else
                 echo "⚠️ Xvfb for evaluation failed to start on display :$EVAL_DISPLAY_NUM" | tee -a "$MAIN_LOG_FILE" "$ITER_LOG_FILE"
                 XVFB_PID="" 
            fi
            
            EVAL_RETRY_COUNT=$((EVAL_RETRY_COUNT+1))
            if [ $EVAL_RETRY_COUNT -lt $EVAL_MAX_RETRIES ]; then
                echo "🔄 Retrying $ITER_NAME evaluation after 5 seconds..." | tee -a "$MAIN_LOG_FILE" "$ITER_LOG_FILE"
                sleep 5
            fi
        done
        
        if [ $EVAL_RETRY_COUNT -ge $EVAL_MAX_RETRIES ] && [ $EVAL_RESULT -ne 0 ]; then
            echo "❗ Evaluation phase for $ITER_NAME still failed after multiple attempts. See details in: $ITER_LOG_FILE" | tee -a "$MAIN_LOG_FILE"
        fi

        echo "🔸 Terminating evaluation deployment service (PID: $DEPLOY_PID)..." | tee -a "$MAIN_LOG_FILE" "$ITER_LOG_FILE"
        if [ -n "$DEPLOY_PID" ] && ps -p $DEPLOY_PID > /dev/null; then kill_tree $DEPLOY_PID; rm -f deploy_pid.txt; fi
        DEPLOY_PID="" 
        sleep 5
    fi
    
    if [ -n "$XVFB_PID" ] && ps -p $XVFB_PID > /dev/null; then kill -9 $XVFB_PID 2>/dev/null; fi 
    pkill -f "Xvfb :$EVAL_DISPLAY_NUM" || true 
    pkill Xvfb || true 

    if [ -f "$KILL_THOR_SCRIPT_PATH" ]; then
        echo "🔸 Cleaning up AI2Thor processes after $ITER_NAME iteration..." | tee -a "$MAIN_LOG_FILE" "$ITER_LOG_FILE"
        bash "$KILL_THOR_SCRIPT_PATH" >> "$ITER_LOG_FILE" 2>&1
    fi
    
    echo "" >> "$MAIN_LOG_FILE"
    echo "---------------------------------------------------------------------" | tee -a "$MAIN_LOG_FILE" "$ITER_LOG_FILE"
    echo "===== ✅ $ITER_NAME processing complete =====" | tee -a "$MAIN_LOG_FILE" "$ITER_LOG_FILE"
    echo "---------------------------------------------------------------------" | tee -a "$MAIN_LOG_FILE" "$ITER_LOG_FILE"

done

echo "" | tee -a "$MAIN_LOG_FILE"
echo "🎉 All iterations processed!" | tee -a "$MAIN_LOG_FILE"
echo "Final SFT training and evaluation logs saved in: $LOG_DIR" | tee -a "$MAIN_LOG_FILE"

exit 0 