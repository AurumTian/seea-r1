#!/bin/bash
# SFT Pipeline script for seea
# Add environment variables to prevent OpenGL errors
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3
INITIAL_LR=0.000001 # 1e-6
FINAL_LR=0.0000001 # 1e-7
# WARMUP_RATIO=0.05 # SFT usually does not require warmup, kept for future use
# WARMUP_INITIAL_LR=0.0000001 # SFT usually does not require warmup

# Usage: ./seea/train/run_sft_pipeline.sh <train_method> <model_path> [save_directory] [dataset_path] [resume]

# First, clean up any existing AI2Thor processes
echo "🔸 Cleaning up AI2Thor processes..."
bash ./seea/utils/kill_thor.sh

# Signal handling and cleanup function
cleanup() {
    echo "Caught Ctrl-C, terminating all background processes..."
    pkill -P $$
    exit 1
}

# Set trap for Ctrl-C (SIGINT)
trap cleanup SIGINT

# Recursive process killing function: kills the specified PID and all its child processes
kill_tree() {
    local pid=$1
    for child in $(ps -o pid= --ppid "$pid"); do
        kill_tree "$child"
    done
    echo "Terminating process $pid"
    kill -9 "$pid" 2>/dev/null
}

# Check if resuming previous training
RESUME=false
if [ $# -ge 5 ]; then
    RESUME_ARG=$(echo "$5" | tr '[:upper:]' '[:lower:]')
    if [ "$RESUME_ARG" = "resume" ] || [ "$RESUME_ARG" = "true" ]; then
        RESUME=true
        echo "🔸 Enabling resume functionality"
    fi
fi

# Check parameters
if [ $# -lt 2 ]; then
    echo "Usage: $0 <train_method(lora|full)> <model_path> [save_directory] [dataset_path] [resume]"
    echo "Example: $0 full /path/to/model /path/to/save /path/to/dataset"
    exit 1
fi

# Get and check training method
TRAIN_METHOD=$(echo "$1" | tr '[:upper:]' '[:lower:]')
if [ "$TRAIN_METHOD" != "lora" ] && [ "$TRAIN_METHOD" != "full" ]; then
    echo "❗Training method must be lora or full"
    exit 1
fi
echo "🔸 Training method: $TRAIN_METHOD"

# Get and check model path
CLI_CURRENT_MODEL="$2" # Store command line arg for model
if [ ! -d "$CLI_CURRENT_MODEL" ]; then
    echo "❗Model path does not exist: $CLI_CURRENT_MODEL"
    exit 1
fi
CURRENT_MODEL="$CLI_CURRENT_MODEL" # Initialize CURRENT_MODEL

# Set save directory
if [ $# -ge 3 ]; then
    SAVE_DIR="$3"
else
    SAVE_DIR="outputs"
fi

# Set number of iterations
NUM_ITERATIONS=100 # Can be passed as a parameter later

# Set output directory and resume status
if [ "$RESUME" = "true" ] && [ $# -ge 3 ]; then
    if [ ! -d "$SAVE_DIR" ]; then
        echo "❗Save directory to resume from does not exist: $SAVE_DIR"
        exit 1
    fi
    OUTPUT_DIR="$SAVE_DIR"
    echo "🔸 Continuing training using specified directory: $OUTPUT_DIR"
    
    LATEST_ITER=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "iter_*" | sort -V | tail -n 1)
    if [ -z "$LATEST_ITER" ]; then
        echo "❗No iteration directory found in $OUTPUT_DIR, will start from scratch or use initial parameters."
        # This effectively means new training within the specified SAVE_DIR if it was empty of iter_*
        # Or an error if SAVE_DIR itself was expected to be an old run.
        # For SFT, we might just create a new iter_1 in this case.
        # Let's ensure OUTPUT_DIR exists, and then proceed as if it's a new run starting at iter_1
        # but within the existing OUTPUT_DIR.
        CURRENT_ITER=1
        # CURRENT_MODEL remains $CLI_CURRENT_MODEL
        # DATASET will be determined by $4 or sampling
    else
        ITER_NUM=$(basename "$LATEST_ITER" | sed 's/iter_//')
        echo "🔸 Found latest iteration: $LATEST_ITER (Iteration $ITER_NUM)"

        if [ -f "$LATEST_ITER/status.txt" ]; then
            if grep -q "Latest model" "$LATEST_ITER/status.txt"; then # Changed "最新模型" to "Latest model"
                echo "✅ Iteration $ITER_NUM completed, will start from iteration $((ITER_NUM+1))"
                CURRENT_ITER=$((ITER_NUM+1))
                CURRENT_MODEL=$(grep "Current model:" "$LATEST_ITER/status.txt" | cut -d' ' -f2-) # Changed "当前模型:"
                # For SFT, dataset is typically sampled per iteration or a global one is used.
                # If iter N is done, iter N+1 will sample new data or use global.
                # We unset DATASET here; it will be re-evaluated based on $4 or sampling.
                unset DATASET
            else
                echo "⚠️ Iteration $ITER_NUM not completed, checking specific stage..."
                CURRENT_ITER=$ITER_NUM
                
                # Try to find model from this incomplete iteration
                if [ -f "$LATEST_ITER/model/checkpoint.txt" ]; then
                    CHECKPOINT_PATH=$(cat "$LATEST_ITER/model/checkpoint.txt")
                    if [ -d "$CHECKPOINT_PATH" ]; then
                        echo "✅ Found model training checkpoint from interrupted iteration $ITER_NUM: $CHECKPOINT_PATH"
                        CURRENT_MODEL="$CHECKPOINT_PATH"
                        if [ "$TRAIN_METHOD" = "lora" ]; then
                            MERGED_MODEL_DIR="${CHECKPOINT_PATH}-merged"
                            if [ -d "$MERGED_MODEL_DIR" ]; then
                                echo "✅ LoRA model already merged: $MERGED_MODEL_DIR"
                                CURRENT_MODEL="$MERGED_MODEL_DIR"
                            else
                                echo "⚠️ LoRA model not merged, may need to merge at iteration start or retrain"
                            fi
                        fi
                    fi
                fi

                # Try to find dataset from this incomplete iteration's status.txt or sample_output.txt
                if [ -f "$LATEST_ITER/status.txt" ] && grep -q "Dataset:" "$LATEST_ITER/status.txt"; then # Changed "数据集:"
                    DATASET=$(grep "Dataset:" "$LATEST_ITER/status.txt" | cut -d' ' -f2-) # Changed "数据集:"
                    if [ -f "$DATASET" ]; then
                         echo "🔸 Resuming dataset from status.txt: $DATASET"
                    else
                         echo "⚠️ Dataset $DATASET recorded in status.txt does not exist, will try to resume from sampling file"
                         unset DATASET
                    fi
                fi
                if [ -z "$DATASET" ] && [ -f "$LATEST_ITER/sample_output.txt" ]; then
                     SFT_DATASET_PATH_IN_FILE=$(cat "$LATEST_ITER/sample_output.txt")
                     if [ -f "$SFT_DATASET_PATH_IN_FILE" ]; then
                         echo "✅ Found sampled SFT dataset for iteration $ITER_NUM: $SFT_DATASET_PATH_IN_FILE"
                         DATASET="$SFT_DATASET_PATH_IN_FILE"
                     else
                         echo "⚠️ Dataset $SFT_DATASET_PATH_IN_FILE pointed to by sample_output.txt does not exist"
                         unset DATASET
                     fi
                fi
                
                # If CURRENT_MODEL is still not valid or wasn't updated from checkpoint
                if [ ! -d "$CURRENT_MODEL" ] || [ "$CURRENT_MODEL" = "$CLI_CURRENT_MODEL" ]; then # check if it's still initial or invalid
                    if [ "$ITER_NUM" -gt 1 ]; then
                        PREV_ITER_NUM=$((ITER_NUM-1))
                        PREV_ITER_DIR="$OUTPUT_DIR/iter_$PREV_ITER_NUM"
                        if [ -f "$PREV_ITER_DIR/status.txt" ] && grep -q "Latest model" "$PREV_ITER_DIR/status.txt"; then # Changed "最新模型"
                            echo "🔸 Trying to get model from previous completed iteration iter_$PREV_ITER_NUM..."
                            CANDIDATE_MODEL=$(grep "Current model:" "$PREV_ITER_DIR/status.txt" | cut -d' ' -f2-) # Changed "当前模型:"
                            if [ -d "$CANDIDATE_MODEL" ]; then
                                CURRENT_MODEL="$CANDIDATE_MODEL"
                            fi
                        fi
                    fi
                fi
            fi
        else # status.txt not found in LATEST_ITER
            echo "⚠️ $LATEST_ITER/status.txt not found, will start from iteration $ITER_NUM"
            CURRENT_ITER=$ITER_NUM
            # Try to find dataset if sampling occurred
            if [ -f "$LATEST_ITER/sample_output.txt" ]; then
                SFT_DATASET_PATH_IN_FILE=$(cat "$LATEST_ITER/sample_output.txt")
                if [ -f "$SFT_DATASET_PATH_IN_FILE" ]; then
                     echo "✅ Found sampled SFT dataset for iteration $ITER_NUM: $SFT_DATASET_PATH_IN_FILE"
                     DATASET="$SFT_DATASET_PATH_IN_FILE"
                fi
            fi
            # Try to find model from previous iteration or stick to CLI one.
            if [ "$ITER_NUM" -gt 1 ]; then
                PREV_ITER_NUM=$((ITER_NUM-1))
                PREV_ITER_DIR="$OUTPUT_DIR/iter_$PREV_ITER_NUM"
                if [ -f "$PREV_ITER_DIR/status.txt" ] && grep -q "Latest model" "$PREV_ITER_DIR/status.txt"; then # Changed "最新模型"
                    CANDIDATE_MODEL=$(grep "Current model:" "$PREV_ITER_DIR/status.txt" | cut -d' ' -f2-) # Changed "当前模型:"
                    if [ -d "$CANDIDATE_MODEL" ]; then
                        CURRENT_MODEL="$CANDIDATE_MODEL"
                    fi
                fi
            fi
        fi
        # Fallback for CURRENT_MODEL if it's somehow invalid after all checks
        if [ ! -d "$CURRENT_MODEL" ]; then
            echo "❗ Unable to determine a valid model from resume logic, will use the initial model specified by command line: $CLI_CURRENT_MODEL"
            CURRENT_MODEL="$CLI_CURRENT_MODEL"
        fi
        echo "🔸 Resume status: Will start from iteration $CURRENT_ITER, using model $CURRENT_MODEL"
        if [ -n "$DATASET" ]; then
            echo "🔸 Resume status: Found dataset $DATASET"
        else
            echo "🔸 Resume status: Dataset will be sampled in iteration or use command line argument $4 (if provided)"
        fi
    fi
else
    # New training, create new output directory
    OUTPUT_DIR="$SAVE_DIR/SFT_${TRAIN_METHOD}_$(date +%Y%m%d_%H%M%S)"
    if [ ! -d "$OUTPUT_DIR" ]; then
        mkdir -p "$OUTPUT_DIR" || {
            echo "❗Failed to create output directory: $OUTPUT_DIR"
            exit 1
        }
    fi
    CURRENT_ITER=1
    # CURRENT_MODEL is already $CLI_CURRENT_MODEL
    unset DATASET # Ensure DATASET is clean for new run, will be set by $4 or sampling
fi

echo "🔸 Output directory: $OUTPUT_DIR"
echo "🔸 Starting from iteration $CURRENT_ITER"
echo "🔸 Using model: $CURRENT_MODEL"


# Handle dataset parameter provided by command line ($4)
# This logic applies for new runs, or can override/provide dataset for resumed runs if not found by resume logic
if [ $# -ge 4 ]; then
    CMD_ARG_DATASET="$4"
    CMD_ARG_DATASET_LOWER=$(echo "$CMD_ARG_DATASET" | tr '[:upper:]' '[:lower:]')

    if [ "$CMD_ARG_DATASET_LOWER" = "false" ] || [ "$CMD_ARG_DATASET_LOWER" = "0" ] || [ "$CMD_ARG_DATASET_LOWER" = "no" ]; then
        echo "🔸 Dataset parameter ($4) is 'false', will not use external initial dataset, relying on sampling or resumed dataset."
        # If RESUME is false, DATASET should remain unset. If RESUME is true and DATASET was found, it's kept.
        # If RESUME is true and DATASET was NOT found, and $4 is false, DATASET remains unset (rely on sampling).
        if [ "$RESUME" = "false" ]; then unset DATASET; fi

    elif [ -f "$CMD_ARG_DATASET" ]; then
        echo "🔸 Using initial dataset provided by command line: $CMD_ARG_DATASET"
        DATASET="$CMD_ARG_DATASET"
    elif [ -n "$CMD_ARG_DATASET" ]; then # Argument provided, not 'false', but not a file
        echo "⚠️ Dataset file '$CMD_ARG_DATASET' provided by command line does not exist."
        if [ "$RESUME" = "true" ] && [ -n "$DATASET" ]; then
            echo "🔸 Will continue to use dataset found from resume status: $DATASET"
        else
            echo "🔸 Will sample in iteration (if applicable)."
            unset DATASET
        fi
    # If $4 is not provided:
    #   If RESUME=false, DATASET remains unset (rely on sampling).
    #   If RESUME=true, DATASET found by resume logic is kept. If not found, it's unset (rely on sampling).
    elif [ "$RESUME" = "false" ]; then
         echo "🔸 Initial dataset parameter not provided, will sample in iteration (if applicable)."
         unset DATASET
    fi
else # Less than 4 arguments
    if [ "$RESUME" = "false" ]; then # New run and no dataset argument
        echo "🔸 Initial dataset parameter not provided, will sample in iteration (if applicable)."
        unset DATASET
    # If RESUME=true and no $4, DATASET from resume logic is authoritative.
    fi
fi

if [ -n "$DATASET" ]; then
    echo "🔸 Finally decided to use dataset: $DATASET"
else
    echo "🔸 Initial dataset not specified, will sample when needed."
fi


# Determine model type and select configuration file
INIT_MODEL_NAME=$(basename "$CURRENT_MODEL")
MODEL_DIR=$(dirname "$CURRENT_MODEL")
if [[ "$INIT_MODEL_NAME" == *"VL"* ]] || [[ "$INIT_MODEL_NAME" == *"vl"* ]] || \
   [[ "$MODEL_DIR" == *"VL"* ]] || [[ "$MODEL_DIR" == *"vl"* ]]; then
    echo "🔸 Detected VL model, using multimodal configuration..."
    CONFIG_PATH="configs/react_config.yaml" # For sampling
    EVAL_CONFIG_PATH="configs/react_eval_config.yaml" # For evaluation
else
    echo "🔸 Detected non-VL model, using language-only configuration..."
    CONFIG_PATH="configs/react_config.yaml" # For sampling (ensure this is correct for LLM SFT data gen)
    EVAL_CONFIG_PATH="configs/react_eval_config.yaml" # For evaluation (ensure this is correct for LLM eval)
    # TODO: Verify if react_llm_config.yaml should be used for non-VL as in run_pipeline.sh
    # For now, keeping SFT script's original choices, but noting potential alignment.
fi

echo "🔸 Will use sampling config: $CONFIG_PATH" | tee -a "$OUTPUT_DIR/run.log"
echo "🔸 Will use evaluation config: $EVAL_CONFIG_PATH" | tee -a "$OUTPUT_DIR/run.log"

# Start SFT iterative training
for ((i=CURRENT_ITER; i<=NUM_ITERATIONS; i++)); do
    echo "===== 🚀 Starting SFT iteration $i =====" | tee -a "$OUTPUT_DIR/run.log"
    
    ITER_DIR="$OUTPUT_DIR/iter_$i"
    mkdir -p "$ITER_DIR"
    
    # Check if sampling and deployment stages can be skipped directly
    # This happens if DATASET is already set (from CLI arg $4 or from RESUME logic for an incomplete iter)
    # AND it's the first iteration of this run (i == CURRENT_ITER).
    # For subsequent iterations (i > CURRENT_ITER), we always sample.
    SKIP_SAMPLING_DEPLOY=false
    if [ -n "$DATASET" ] && [ -f "$DATASET" ] && [ $i -eq $CURRENT_ITER ]; then
        echo "🔸 Dataset already available: $DATASET, for iteration $i, skipping deployment and sampling stages." | tee -a "$ITER_DIR/log.txt"
        SKIP_SAMPLING_DEPLOY=true
    fi

    if [ "$SKIP_SAMPLING_DEPLOY" = "false" ]; then
        # Deployment stage
        echo "🔸 Starting deployment service..." | tee -a "$ITER_DIR/log.txt"
        # Using INIT_MODEL_NAME for deploy_script, which is basename of CURRENT_MODEL at pipeline start.
        # This might need to be basename of the *current* CURRENT_MODEL if it changes significantly.
        # For SFT, CURRENT_MODEL updates iteratively. Let's use basename of current CURRENT_MODEL.
        python -m seea.train.deploy_script --model_name "$(basename "$CURRENT_MODEL")" --model "$CURRENT_MODEL" --port 8000
        if [ $? -ne 0 ]; then
            echo "Error in deployment stage, exiting loop" | tee -a "$ITER_DIR/log.txt"
            exit 1
        fi
        
        DEPLOY_PID=$(cat deploy_pid.txt)
        echo "✅ Deployment successful, PID: $DEPLOY_PID" | tee -a "$ITER_DIR/log.txt"
        sleep 5
        
        # Sampling stage
        echo "🔸 Starting sampling..." | tee -a "$ITER_DIR/log.txt"
        
        pkill Xvfb || true # Ensure previous Xvfb is cleaned
        sleep 2
        
        DISPLAY_NUM=$((100 + RANDOM % 900))
        export DISPLAY=:$DISPLAY_NUM
        
        MAX_RETRIES=3
        RETRY_COUNT=0
        SAMPLE_RESULT=-1
        
        while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
            echo "🔸 Starting Xvfb (Attempt $((RETRY_COUNT+1))/$MAX_RETRIES) on display $DISPLAY_NUM..." | tee -a "$ITER_DIR/log.txt"
            Xvfb :$DISPLAY_NUM -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
            XVFB_PID=$!
            sleep 5
            
            if kill -0 $XVFB_PID 2>/dev/null; then
                echo "✅ Xvfb started successfully, PID: $XVFB_PID" | tee -a "$ITER_DIR/log.txt"
                
                xvfb-run -a --server-num=$DISPLAY_NUM --server-args="-screen 0 1024x768x24 -ac +extension GLX +render -noreset" \
                    python -m main \
                        --model_name "$(basename "$CURRENT_MODEL")" \
                        --base_url http://127.0.0.1:8000/v1 \
                        --save_dir "$ITER_DIR" \
                        --config "$CONFIG_PATH"
                SAMPLE_RESULT=$?
                
                if kill -0 $XVFB_PID 2>/dev/null; then # Clean up current Xvfb
                    kill -TERM $XVFB_PID
                    sleep 5 
                    kill -9 $XVFB_PID 2>/dev/null || true
                fi
                pkill -f "Xvfb :$DISPLAY_NUM" || true # Precise kill
                sleep 3
                
                if [ $SAMPLE_RESULT -eq 0 ]; then
                    echo "✅ Sampling completed successfully" | tee -a "$ITER_DIR/log.txt"
                    break
                else
                    echo "⚠️ Sampling failed, error code: $SAMPLE_RESULT" | tee -a "$ITER_DIR/log.txt"
                fi
            else
                echo "⚠️ Xvfb failed to start on display $DISPLAY_NUM" | tee -a "$ITER_DIR/log.txt"
            fi
            
            RETRY_COUNT=$((RETRY_COUNT+1))
            if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
                echo "🔄 Retrying after 10 seconds..." | tee -a "$ITER_DIR/log.txt"
                sleep 10
            fi
        done
        
        if [ $RETRY_COUNT -ge $MAX_RETRIES ] && [ $SAMPLE_RESULT -ne 0 ]; then
            echo "❗ Sampling stage still failed after multiple attempts, exiting" | tee -a "$ITER_DIR/log.txt"
            kill_tree $DEPLOY_PID # stop deploy if sampling failed catastrophically
            exit 1
        fi
        
        # Stop deployment service
        echo "🔸 Terminating deployment service..." | tee -a "$ITER_DIR/log.txt"
        kill_tree $DEPLOY_PID
        sleep 5
        
        # Get and process dataset path
        SAMPLE_OUTPUT_PATH="$ITER_DIR/sample_output.txt"
        if [ -f "$SAMPLE_OUTPUT_PATH" ]; then
            DATASET_CANDIDATE=$(cat "$SAMPLE_OUTPUT_PATH")
            if [ -f "$DATASET_CANDIDATE" ]; then
                DATASET="$DATASET_CANDIDATE"
                echo "🔸 Found SFT dataset path: $DATASET" | tee -a "$ITER_DIR/log.txt"
            else
                echo "❗Dataset file recorded in sample_output.txt does not exist: $DATASET_CANDIDATE" | tee -a "$ITER_DIR/log.txt"
                # Attempt to find alternates as in run_pipeline.sh
                DATASET_DIR=$(dirname "$DATASET_CANDIDATE")
                FOUND_ALT=false
                if [ -f "$DATASET_DIR/dataset_filtered.json" ]; then
                    DATASET="$DATASET_DIR/dataset_filtered.json"
                    echo "✅ Found alternative dataset: $DATASET" | tee -a "$ITER_DIR/log.txt"
                    FOUND_ALT=true
                elif [ -f "$DATASET_DIR/dataset.jsonl" ]; then # Common SFT format
                    DATASET="$DATASET_DIR/dataset.jsonl"
                    echo "✅ Found alternative dataset: $DATASET" | tee -a "$ITER_DIR/log.txt"
                    FOUND_ALT=true
                fi
                if [ "$FOUND_ALT" = "false" ]; then
                    echo "❗Cannot find a valid SFT dataset file, exiting loop" | tee -a "$ITER_DIR/log.txt"
                    exit 1
                fi
            fi
        else
            echo "❗$SAMPLE_OUTPUT_PATH file not found, cannot get SFT dataset path, exiting loop" | tee -a "$ITER_DIR/log.txt"
            exit 1
        fi
    fi # End of SKIP_SAMPLING_DEPLOY check

    # Check if there is a valid dataset in the end
    if [ -z "$DATASET" ] || [ ! -f "$DATASET" ]; then
        echo "❗Failed to obtain a valid dataset before starting training, exiting iteration $i" | tee -a "$ITER_DIR/log.txt"
        exit 1
    fi

    # Calculate current learning rate for this iteration (linear decay)
    if [ $NUM_ITERATIONS -eq 1 ]; then
        CURRENT_LR=$INITIAL_LR
    else
        CURRENT_LR=$(awk -v iter=$i -v max_iter=$NUM_ITERATIONS -v init_lr=$INITIAL_LR -v final_lr=$FINAL_LR 'BEGIN {
            # Linear decay: iter is 1-indexed
            # lr = init_lr - (init_lr - final_lr) * (iter - 1) / (max_iter - 1)
            # To avoid division by zero if max_iter is 1, handle separately or ensure max_iter > 1 for this formula
            if (max_iter == 1) {
                lr = init_lr
            } else {
                 # Ensure iter does not exceed max_iter for calculation if loop continues beyond for some reason
                current_progress_step = (iter > max_iter) ? max_iter : iter;
                lr = init_lr - (init_lr - final_lr) * (current_progress_step - 1) / (max_iter - 1)
            }
            if (lr < final_lr && init_lr >=final_lr) { lr = final_lr } # clamp to final_lr
            if (lr > init_lr && init_lr <=final_lr) { lr = final_lr } 
            printf "%.10f", lr
        }')
    fi
    
    echo "🔸 Current learning rate (iteration $i): $CURRENT_LR" | tee -a "$ITER_DIR/log.txt"
    
    # Run SFT training
    echo "🔸 Starting SFT training, dataset: $DATASET" | tee -a "$ITER_DIR/log.txt"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
        -m seea.train.sft_train_script \
        --dataset "$DATASET" \
        --model "$CURRENT_MODEL" \
        --output_dir "$ITER_DIR/model" \
        --learning_rate $CURRENT_LR \
        --train_type "$TRAIN_METHOD"
    
    if [ $? -ne 0 ]; then
        echo "Error in SFT training stage, exiting loop" | tee -a "$ITER_DIR/log.txt"
        exit 1
    fi

    # If LoRA training, merge the model
    if [ "$TRAIN_METHOD" = "lora" ]; then
        echo "🔸 Starting to merge LoRA model..." | tee -a "$ITER_DIR/log.txt"
        if [ ! -f "$ITER_DIR/model/checkpoint.txt" ]; then
            echo "❗checkpoint.txt file not found after LoRA training, cannot merge" | tee -a "$ITER_DIR/log.txt"
            exit 1
        fi
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
        
        echo "✅ LoRA model merged successfully: $MERGED_MODEL_DIR" | tee -a "$ITER_DIR/log.txt"
        CURRENT_MODEL="$MERGED_MODEL_DIR"
    else
        # If full training, use the checkpoint directly
        if [ -f "$ITER_DIR/model/checkpoint.txt" ]; then
            CHECKPOINT=$(cat "$ITER_DIR/model/checkpoint.txt")
            if [ -d "$CHECKPOINT" ]; then
                 CURRENT_MODEL="$CHECKPOINT"
            else
                 echo "❗ Full training checkpoint path ($CHECKPOINT) is invalid, exiting" | tee -a "$ITER_DIR/log.txt"
                 exit 1
            fi
        elif [ -f checkpoint.txt ]; then # Fallback to root checkpoint.txt, though sft_train_script should place it in output_dir
            CHECKPOINT=$(cat checkpoint.txt)
            if [ -d "$CHECKPOINT" ]; then
                 CURRENT_MODEL="$CHECKPOINT"
            else
                 echo "❗ Full training root checkpoint.txt path ($CHECKPOINT) is invalid, exiting" | tee -a "$ITER_DIR/log.txt"
                 exit 1
            fi
        else
            echo "❗checkpoint.txt file not found after full training, exiting" | tee -a "$ITER_DIR/log.txt"
            exit 1
        fi
    fi

    echo "✅ Training completed, latest model: $CURRENT_MODEL" | tee -a "$ITER_DIR/log.txt"

    # Save key information for the current iteration
    echo "Current model: $CURRENT_MODEL" > "$ITER_DIR/status.txt" # Changed "当前模型:"
    echo "Dataset: $DATASET" >> "$ITER_DIR/status.txt" # Changed "数据集:"
    echo "Training method: $TRAIN_METHOD" >> "$ITER_DIR/status.txt" # Changed "训练方式:"
    echo "Iteration number: $i" >> "$ITER_DIR/status.txt" # Changed "迭代号:"
    echo "Learning rate: $CURRENT_LR" >> "$ITER_DIR/status.txt" # Changed "学习率:"
    echo "Latest model" >> "$ITER_DIR/status.txt" # Mark as completed for this iteration. Changed "最新模型"
    
    # Clean up any remaining AI2Thor processes
    echo "🔸 Cleaning up AI2Thor processes after iteration..." | tee -a "$ITER_DIR/log.txt"
    bash ./seea/utils/kill_thor.sh
    
    echo "===== ✅ SFT iteration $i completed =====" | tee -a "$ITER_DIR/log.txt"
    
    # Perform evaluation
    echo "🔸 Starting evaluation for iteration $i..." | tee -a "$ITER_DIR/log.txt"
    
    pkill Xvfb || true # Ensure previous Xvfb is cleaned
    sleep 2
    EVAL_DISPLAY_NUM=$((200 + RANDOM % 800)) # Use a different range for eval display
    export DISPLAY=:$EVAL_DISPLAY_NUM
    
    # Deploy new model for evaluation (use different port to avoid conflict with main deployment if it's still running)
    # For SFT, main deploy was already stopped. So we can reuse port or use a new one.
    EVAL_PORT=8001
    echo "🔸 Starting evaluation deployment service (Model: $(basename "$CURRENT_MODEL"), Port: $EVAL_PORT)..." | tee -a "$ITER_DIR/log.txt"
    python -m seea.train.deploy_script --model_name "$(basename "$CURRENT_MODEL")" --model "$CURRENT_MODEL" --port $EVAL_PORT
    if [ $? -ne 0 ]; then
        echo "⚠️ Error in evaluation deployment stage, skipping evaluation" | tee -a "$ITER_DIR/log.txt"
    else
        EVAL_DEPLOY_PID=$(cat deploy_pid.txt)
        echo "✅ Evaluation deployment successful, PID: $EVAL_DEPLOY_PID, URL: http://127.0.0.1:$EVAL_PORT/v1" | tee -a "$ITER_DIR/log.txt"
        sleep 5
        
        EVAL_MAX_RETRIES=3
        EVAL_RETRY_COUNT=0
        EVAL_RESULT=-1

        while [ $EVAL_RETRY_COUNT -lt $EVAL_MAX_RETRIES ]; do
            echo "🔸 Starting Xvfb for evaluation (Attempt $((EVAL_RETRY_COUNT+1))/$EVAL_MAX_RETRIES) on display $EVAL_DISPLAY_NUM..." | tee -a "$ITER_DIR/log.txt"
            Xvfb :$EVAL_DISPLAY_NUM -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
            EVAL_XVFB_PID=$!
            sleep 5

            if kill -0 $EVAL_XVFB_PID 2>/dev/null; then
                echo "✅ Xvfb for evaluation started successfully, PID: $EVAL_XVFB_PID" | tee -a "$ITER_DIR/log.txt"
                
                xvfb-run -a --server-num=$EVAL_DISPLAY_NUM --server-args="-screen 0 1024x768x24 -ac +extension GLX +render -noreset" \
                    python -m seea.eval.evaluate \
                        --model_name "$(basename "$CURRENT_MODEL")" \
                        --base_url http://127.0.0.1:$EVAL_PORT/v1 \
                        --output_dir "$ITER_DIR/eval_results" \
                        --config "$EVAL_CONFIG_PATH"
                EVAL_RESULT=$?
                
                if kill -0 $EVAL_XVFB_PID 2>/dev/null; then
                    kill -TERM $EVAL_XVFB_PID
                    sleep 5
                    kill -9 $EVAL_XVFB_PID 2>/dev/null || true
                fi
                pkill -f "Xvfb :$EVAL_DISPLAY_NUM" || true
                sleep 3
                
                if [ $EVAL_RESULT -eq 0 ]; then
                    echo "✅ Evaluation completed successfully" | tee -a "$ITER_DIR/log.txt"
                    break
                else
                    echo "⚠️ Evaluation failed, error code: $EVAL_RESULT" | tee -a "$ITER_DIR/log.txt"
                fi
            else
                 echo "⚠️ Xvfb for evaluation failed to start on display $EVAL_DISPLAY_NUM" | tee -a "$ITER_DIR/log.txt"
            fi

            EVAL_RETRY_COUNT=$((EVAL_RETRY_COUNT+1))
            if [ $EVAL_RETRY_COUNT -lt $EVAL_MAX_RETRIES ]; then
                echo "🔄 Retrying evaluation after 10 seconds..." | tee -a "$ITER_DIR/log.txt"
                sleep 10
            fi
        done
        
        if [ $EVAL_RETRY_COUNT -ge $EVAL_MAX_RETRIES ] && [ $EVAL_RESULT -ne 0 ]; then
            echo "❗ Evaluation stage still failed after multiple attempts" | tee -a "$ITER_DIR/log.txt"
        fi

        # Stop evaluation deployment
        echo "🔸 Terminating evaluation deployment service..." | tee -a "$ITER_DIR/log.txt"
        kill_tree $EVAL_DEPLOY_PID
        sleep 5
    fi
    
    # If any Xvfb for eval is lingering, try to kill it one last time
    pkill Xvfb || true
    
    # After each iteration, reset DATASET so that next iteration performs sampling
    # unless a global DATASET was provided by $4 and we want to reuse it (current logic re-samples).
    # If we want to use $4 for ALL iterations if provided, this unset needs to be conditional.
    # For typical SFT iterative refinement, sampling new data each iter (or after first iter) is common.
    unset DATASET
done

# Save final status
echo "Final model: $CURRENT_MODEL" > "$OUTPUT_DIR/final_status.txt" # Changed "最终模型:"
echo "Training method: $TRAIN_METHOD" >> "$OUTPUT_DIR/final_status.txt" # Changed "训练方式:"
echo "🎉 All SFT training iterations completed! Results saved in: $OUTPUT_DIR" | tee -a "$OUTPUT_DIR/run.log"

# Terminate all background processes
pkill -P $$
exit 0 