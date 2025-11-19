#!/bin/bash

# Cleanup function: kill all background processes
cleanup() {
    echo "Ctrl-C captured, cleaning up all processes..."
    if [ ! -z "$DEPLOY_PID" ]; then
        echo "Terminating deployment process $DEPLOY_PID..."
        kill -9 $DEPLOY_PID
    fi
    pkill -P $$
    pkill Xvfb
    pkill -f "thor-201909061227-Linux"
    exit 1
}

# Set trap to capture Ctrl-C (SIGINT)
trap cleanup SIGINT

# Recursive kill process function: kill specified PID and all its child processes
kill_tree() {
    local pid=$1
    for child in $(ps -o pid= --ppid "$pid"); do
        kill_tree "$child"
    done
    echo "Killing process $pid"
    kill -9 "$pid" 2>/dev/null
}

# Define array of model paths to be evaluated
MODEL_PATHS=(
    "/media/users/name/models/Qwen/Qwen2.5-7B-Instruct")

# Create main results directory
MAIN_OUTPUT_DIR="eval_results_multiple_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$MAIN_OUTPUT_DIR"
echo "🔸 Main output directory: $MAIN_OUTPUT_DIR"

# Record start time
START_TIME=$(date +%s)

# Loop to evaluate each model
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    echo "===== 🚀 Starting evaluation for model: $(basename "$MODEL_PATH") ====="
    
    if [ ! -d "$MODEL_PATH" ]; then
        echo "❗Model path does not exist: $MODEL_PATH, skipping this model"
        continue
    fi
    
    # Set output directory for the current model
    MODEL_NAME=$(basename "$MODEL_PATH")
    OUTPUT_DIR="$MAIN_OUTPUT_DIR/$MODEL_NAME"
    mkdir -p "$OUTPUT_DIR"
    echo "🔸 Current model output directory: $OUTPUT_DIR"
    
    # Determine if it's a VL model based on the model name
    if [[ "$MODEL_NAME" == *"VL"* ]] || [[ "$MODEL_NAME" == *"vl"* ]] || [[ "$MODEL_NAME" == *"Vl"* ]]; then
        echo "🔸 Detected VL model, using multimodal evaluation config..."
        CONFIG_PATH="configs/react_eval_config.yaml"
    else
        echo "🔸 Detected non-VL model, using language-only evaluation config..."
        CONFIG_PATH="configs/react_llm_eval_config.yaml"
    fi
    
    # 1. Deploy model
    echo "===== 🚀 Starting model deployment ====="
    echo "🔸 Starting deployment service..."
    python -m seea.train.real_deploy --model_name "$MODEL_NAME" --model "$MODEL_PATH" --port 8000 &
    DEPLOY_PID=$!
    
    # Wait for service to start
    echo "🔸 Waiting for service to start..."
    MAX_RETRIES=60
    RETRY_COUNT=0
    while ! curl -s "http://localhost:8000/v1/models" > /dev/null; do
        if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
            echo "❗Service startup timed out, skipping this model"
            if [ ! -z "$DEPLOY_PID" ]; then
                kill_tree $DEPLOY_PID
            fi
            continue 2  # Skip to the next model
        fi
        echo "Waiting for service to start... $RETRY_COUNT"
        sleep 5
        RETRY_COUNT=$((RETRY_COUNT + 1))
    done
    
    # Get actual available model name
    AVAILABLE_MODELS=$(curl -s "http://localhost:8000/v1/models" | grep -o '"id":"[^"]*"' | cut -d'"' -f4)
    if [ -z "$AVAILABLE_MODELS" ]; then
        echo "❗Unable to get available model list, using default name $MODEL_NAME"
        ACTUAL_MODEL_NAME="$MODEL_NAME"
    else
        echo "🔸 Available models: $AVAILABLE_MODELS"
        ACTUAL_MODEL_NAME=$(echo "$AVAILABLE_MODELS" | head -n1)
        echo "🔸 Using model: $ACTUAL_MODEL_NAME"
    fi
    
    echo "✅ Service started"
    
    # 2. Run evaluation
    echo "===== 🚀 Starting evaluation ====="
    
    # Ensure previous Xvfb processes are cleaned up
    pkill Xvfb || true
    sleep 2
    
    # Use random port number to avoid conflicts
    DISPLAY_NUM=$((100 + RANDOM % 900))
    export DISPLAY=:$DISPLAY_NUM
    
    # Start Xvfb and add retry mechanism
    MAX_RETRIES=3
    RETRY_COUNT=0
    EVAL_STATUS=1  # Default to failure status
    
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        echo "🔸 Starting Xvfb (Attempt $((RETRY_COUNT+1))/$MAX_RETRIES)..."
        Xvfb :$DISPLAY_NUM -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
        XVFB_PID=$!
        sleep 5
        
        # Check if Xvfb is running normally
        if kill -0 $XVFB_PID 2>/dev/null; then
            echo "✅ Xvfb started successfully, PID: $XVFB_PID"
            
            # Clean up any existing AI2Thor processes
            pkill -f "thor-201909061227-Linux" || true
            sleep 2
            
            # Run evaluation
            echo "🔸 Running evaluation..."
            xvfb-run -a --server-num=$DISPLAY_NUM --server-args="-screen 0 1024x768x24 -ac +extension GLX +render -noreset" \
                python -m seea.eval.evaluate \
                    --model_name "$ACTUAL_MODEL_NAME" \
                    --base_url http://127.0.0.1:8000/v1 \
                    --config "$CONFIG_PATH" \
                    --output_dir "$OUTPUT_DIR"
            
            EVAL_STATUS=$?
            
            # Clean up Xvfb process regardless of success or failure
            kill $XVFB_PID || true
            pkill Xvfb || true
            sleep 2
            
            # If evaluation is successful, break the loop
            if [ $EVAL_STATUS -eq 0 ]; then
                echo "✅ Evaluation completed successfully"
                break
            else
                echo "⚠️ Evaluation failed, error code: $EVAL_STATUS"
            fi
        else
            echo "⚠️ Xvfb startup failed"
        fi
        
        # Increment retry count
        RETRY_COUNT=$((RETRY_COUNT+1))
        
        # If there are still retries left, wait for a while before retrying
        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            echo "🔄 Retrying in 10 seconds..."
            sleep 10
        fi
    done
    
    # Check if all retries failed
    if [ $RETRY_COUNT -ge $MAX_RETRIES ] && [ $EVAL_STATUS -ne 0 ]; then
        echo "❗ Evaluation phase still failed after multiple attempts" | tee -a "$OUTPUT_DIR/log.txt"
    fi
    
    # 3. Clean up processes
    echo "🔸 Cleaning up processes..."
    # Clean up AI2Thor processes
    pkill -f "thor-201909061227-Linux" || true
    
    # Stop deployment service
    if [ ! -z "$DEPLOY_PID" ]; then
        echo "Terminating deployment process $DEPLOY_PID..."
        kill_tree $DEPLOY_PID
    fi
    
    if [ $EVAL_STATUS -eq 0 ]; then
        echo "✅ Model $MODEL_NAME evaluation completed! Results saved in: $OUTPUT_DIR"
    else
        echo "❗Error during model $MODEL_NAME evaluation"
    fi
    
    # Save current model evaluation status
    echo "Evaluation time: $(date)" > "$OUTPUT_DIR/status.txt"
    echo "Model path: $MODEL_PATH" >> "$OUTPUT_DIR/status.txt"
    echo "Model name: $MODEL_NAME" >> "$OUTPUT_DIR/status.txt"
    echo "Actual model name used: $ACTUAL_MODEL_NAME" >> "$OUTPUT_DIR/status.txt"
    echo "Evaluation status: $EVAL_STATUS" >> "$OUTPUT_DIR/status.txt"
    echo "Config file: $CONFIG_PATH" >> "$OUTPUT_DIR/status.txt"
    
    # Wait for a while to ensure all processes are cleaned up
    sleep 10
done

# Calculate total time taken
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(( (TOTAL_TIME % 3600) / 60 ))
SECONDS=$((TOTAL_TIME % 60))

# Generate summary report
echo "===== 📊 Evaluation Summary =====" > "$MAIN_OUTPUT_DIR/summary.txt"
echo "Start time: $(date -d @$START_TIME)" >> "$MAIN_OUTPUT_DIR/summary.txt"
echo "End time: $(date -d @$END_TIME)" >> "$MAIN_OUTPUT_DIR/summary.txt"
echo "Total time taken: ${HOURS}h ${MINUTES}m ${SECONDS}s" >> "$MAIN_OUTPUT_DIR/summary.txt"
echo "" >> "$MAIN_OUTPUT_DIR/summary.txt"
echo "Evaluated models list:" >> "$MAIN_OUTPUT_DIR/summary.txt"

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    MODEL_NAME=$(basename "$MODEL_PATH")
    if [ -f "$MAIN_OUTPUT_DIR/$MODEL_NAME/status.txt" ]; then
        EVAL_STATUS=$(grep "Evaluation status" "$MAIN_OUTPUT_DIR/$MODEL_NAME/status.txt" | cut -d: -f2 | tr -d ' ')
        ACTUAL_MODEL=$(grep "Actual model name used" "$MAIN_OUTPUT_DIR/$MODEL_NAME/status.txt" | cut -d: -f2 | tr -d ' ')
        CONFIG_FILE=$(grep "Config file" "$MAIN_OUTPUT_DIR/$MODEL_NAME/status.txt" | cut -d: -f2 | tr -d ' ')
        if [ "$EVAL_STATUS" == "0" ]; then
            STATUS="✅ Success"
        else
            STATUS="❌ Failed"
        fi
        echo "- $MODEL_NAME (Used: $ACTUAL_MODEL, Config: $CONFIG_FILE): $STATUS" >> "$MAIN_OUTPUT_DIR/summary.txt"
    else
        STATUS="❓ Unknown"
        echo "- $MODEL_NAME: $STATUS" >> "$MAIN_OUTPUT_DIR/summary.txt"
    fi
done

echo "===== 🎉 All model evaluations completed ====="
echo "Summary report saved in: $MAIN_OUTPUT_DIR/summary.txt"
echo "Total time taken: ${HOURS}h ${MINUTES}m ${SECONDS}s"

# Clean up all background processes
pkill -P $$
pkill -f "thor-201909061227-Linux"
exit 0