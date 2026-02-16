#!/bin/bash
# GR00T N1.6 Fine-tuning Script for A100/H100

export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="$HOME/.cache"
unset HF_DATASETS_CACHE 2>/dev/null || true
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"

set -e

HF_DATASET="${HF_DATASET:-ganatrask/reachy2_100}"
LOCAL_DATASET_DIR="${LOCAL_DATASET_DIR:-./datasets/reachy2_100}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/groot_output/reachy2}"
MODALITY_CONFIG="${MODALITY_CONFIG:-./configs/reachy2_modality_config.py}"
NUM_GPUS="${NUM_GPUS:-1}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-reachy2-groot-finetune}"

BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_STEPS="${MAX_STEPS:-30000}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
SAVE_STEPS="${SAVE_STEPS:-3000}"
NUM_WORKERS="${NUM_WORKERS:-10}"

TOTAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

echo "=============================================="
echo "GR00T N1.6 Fine-tuning"
echo "=============================================="
echo "Dataset: $HF_DATASET"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS, Batch: $TOTAL_BATCH_SIZE, Steps: $MAX_STEPS"
echo ""

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ ! -d "gr00t" ]; then
    echo "Error: Run from Isaac-GR00T directory"
    exit 1
fi

mkdir -p $OUTPUT_DIR

if [ ! -f "$LOCAL_DATASET_DIR/meta/info.json" ]; then
    echo "Downloading dataset..."
    huggingface-cli download "$HF_DATASET" --repo-type dataset --local-dir "$LOCAL_DATASET_DIR"
fi

echo ""
echo "Starting training..."

CMD="python gr00t/experiment/launch_finetune.py \
    --base_model_path nvidia/GR00T-N1.6-3B \
    --dataset_path $LOCAL_DATASET_DIR \
    --embodiment_tag REACHY2 \
    --modality_config_path $MODALITY_CONFIG \
    --num_gpus $NUM_GPUS \
    --global_batch_size $TOTAL_BATCH_SIZE \
    --max_steps $MAX_STEPS \
    --save_steps $SAVE_STEPS \
    --save_total_limit 5 \
    --learning_rate $LEARNING_RATE \
    --output_dir $OUTPUT_DIR \
    --dataloader_num_workers $NUM_WORKERS \
    --num_shards_per_epoch 10000 \
    --tune_projector \
    --tune_diffusion_model"

if [ "$USE_WANDB" = "true" ]; then
    CMD="$CMD --use_wandb"
fi

eval $CMD

echo ""
echo "=============================================="
echo "Training complete! Checkpoints: $OUTPUT_DIR"
echo "=============================================="
