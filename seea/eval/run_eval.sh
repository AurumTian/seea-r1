#!/bin/bash

# Cleanup function: kill all background processes
cleanup() {
    echo "Ctrl-C captured, cleaning up all processes..."
    if [ ! -z "$DEPLOY_PID" ]; then
        echo "Terminating deployment process $DEPLOY_PID..."
        kill_tree $DEPLOY_PID
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

# Check parameters
if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_path>"
    exit 1
fi

MODEL_PATH="$1"
if [ ! -d "$MODEL_PATH" ]; then
    echo "❗Model path does not exist: $MODEL_PATH"
    exit 1
fi

# Set output directory
OUTPUT_DIR="eval_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
echo "🔸 Output directory: $OUTPUT_DIR"

# Get model name
MODEL_NAME=$(basename "$MODEL_PATH")

# Determine if it's a VL model based on the model name
if [[ "$MODEL_NAME" == *"VL"* ]] || [[ "$MODEL_NAME" == *"vl"* ]] || [[ "$MODEL_NAME" == *"Vl"* ]]; then
    echo "🔸 Detected VL model, using multimodal evaluation config..."
    CONFIG_PATH="configs/eval_config.yaml"
else
    echo "🔸 Detected non-VL model, using language-only evaluation config..."
    CONFIG_PATH="configs/eval_config_llm.yaml"
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
        echo "❗Service startup timed out"
        cleanup
        exit 1
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
                --split dev \
                --num_games 140 \
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
    echo "✅ Evaluation completed! Results saved in: $OUTPUT_DIR"
else
    echo "❗Error during evaluation process"
    exit 1
fi

# Save final status
echo "Evaluation time: $(date)" > "$OUTPUT_DIR/final_status.txt"
echo "Model path: $MODEL_PATH" >> "$OUTPUT_DIR/final_status.txt"
echo "Model name: $MODEL_NAME" >> "$OUTPUT_DIR/final_status.txt"
echo "Actual model name used: $ACTUAL_MODEL_NAME" >> "$OUTPUT_DIR/final_status.txt"
echo "Config file: $CONFIG_PATH" >> "$OUTPUT_DIR/final_status.txt"
echo "Evaluation status: $EVAL_STATUS" >> "$OUTPUT_DIR/final_status.txt"

# Clean up all background processes
pkill -P $$
pkill -f "thor-201909061227-Linux"
exit 0