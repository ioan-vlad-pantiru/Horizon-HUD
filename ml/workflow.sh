#!/bin/bash
# Horizon-HUD ML Training Workflow
# Complete pipeline from dataset to deployed TFLite model

set -e

# Configuration
DATASET_ROOT="${DATASET_ROOT:-data/bdd100k}"
DATASET_LABELS_ROOT="${DATASET_LABELS_ROOT:-}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-baseline_ssd_mobilenet_v2_320_int8}"
CONFIG_DIR="ml/config"
MODELS_DIR="models/experiments"

LABELS_ARGS=()
if [ -n "$DATASET_LABELS_ROOT" ]; then
    LABELS_ARGS=(--labels-root "$DATASET_LABELS_ROOT")
fi

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Horizon-HUD ML Training Workflow${NC}"
echo -e "${BLUE}========================================${NC}"

# Step 1: Create labelmap
echo -e "\n${GREEN}Step 1: Creating labelmap...${NC}"
python ml/utils/create_labelmap.py --output labels/horizon_labelmap.txt

# Step 2: Prepare experiment config
echo -e "\n${GREEN}Step 2: Preparing experiment configuration...${NC}"
EXPERIMENT_CONFIG="${MODELS_DIR}/${EXPERIMENT_NAME}/experiment_config.yaml"
if [ ! -f "$EXPERIMENT_CONFIG" ]; then
    echo -e "${YELLOW}Creating experiment config from template...${NC}"
    mkdir -p "${MODELS_DIR}/${EXPERIMENT_NAME}"
    cp "${CONFIG_DIR}/experiment_config.yaml" "$EXPERIMENT_CONFIG"
    echo -e "${YELLOW}Please edit ${EXPERIMENT_CONFIG} with your settings${NC}"
    read -p "Press enter after editing the config..."
fi

# Step 3: Train model
echo -e "\n${GREEN}Step 3: Training model...${NC}"
read -p "Start training? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python ml/training/train.py \
        --config "$EXPERIMENT_CONFIG" \
        --dataset-root "$DATASET_ROOT" \
        "${LABELS_ARGS[@]}"
fi

# Step 4: Evaluate model
echo -e "\n${GREEN}Step 4: Evaluating model...${NC}"
BEST_MODEL="${MODELS_DIR}/${EXPERIMENT_NAME}/checkpoints/best_model.keras"
if [ -f "$BEST_MODEL" ]; then
    python ml/evaluation/evaluate.py \
        --model "$BEST_MODEL" \
        --dataset-root "$DATASET_ROOT" \
        "${LABELS_ARGS[@]}" \
        --split test \
        --config "${CONFIG_DIR}/model_config.yaml" \
        --output "${MODELS_DIR}/${EXPERIMENT_NAME}/evaluation_results.json"
else
    echo -e "${YELLOW}Best model not found. Skipping evaluation.${NC}"
fi

# Step 5: Convert to TFLite
echo -e "\n${GREEN}Step 5: Converting to TFLite...${NC}"
if [ -f "$BEST_MODEL" ]; then
    TFLITE_OUTPUT="${MODELS_DIR}/${EXPERIMENT_NAME}/detector.tflite"
    python ml/conversion/convert_to_tflite.py \
        --model "$BEST_MODEL" \
        --output "$TFLITE_OUTPUT" \
        --quantization int8 \
        --dataset-root "$DATASET_ROOT" \
        "${LABELS_ARGS[@]}" \
        --config "${CONFIG_DIR}/model_config.yaml"
    
    echo -e "\n${GREEN}TFLite model saved to: ${TFLITE_OUTPUT}${NC}"
else
    echo -e "${YELLOW}Best model not found. Skipping conversion.${NC}"
fi

# Step 6: Benchmark (if on Raspberry Pi)
echo -e "\n${GREEN}Step 6: Benchmarking (optional)...${NC}"
if [ -f "${MODELS_DIR}/${EXPERIMENT_NAME}/detector.tflite" ]; then
    read -p "Run benchmark on this machine? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python ml/evaluation/benchmark_pi.py \
            --model "${MODELS_DIR}/${EXPERIMENT_NAME}/detector.tflite" \
            --input-size 320 320 \
            --num-runs 1000 \
            --num-threads 4 \
            --output "${MODELS_DIR}/${EXPERIMENT_NAME}/benchmark_results.json"
    fi
else
    echo -e "${YELLOW}TFLite model not found. Skipping benchmark.${NC}"
fi

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Workflow complete!${NC}"
echo -e "${BLUE}========================================${NC}"
