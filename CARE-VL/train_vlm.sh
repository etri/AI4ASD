#!/bin/bash
# ============================================================================
# CARE-VL: Stage 1 - VLM Fine-tuning Script
# Fine-tunes LLaVA-OneVision on SIIC instruction-tuning dataset
# ============================================================================

export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0  # Change to your network interface (e.g., ens8f0)
export NCCL_DEBUG=INFO

# ============================================================================
# Model Configuration
# ============================================================================
LLM_VERSION="Qwen/Qwen2-7B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"

VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

# Base model checkpoint (LLaVA-OneVision pretrained)
PREV_STAGE_CHECKPOINT='lmms-lab/llava-onevision-qwen2-7b-ov'

# ============================================================================
# Training Configuration
# ============================================================================
PROMPT_VERSION="qwen_1_5"
RUN_NAME="care-vl-siglip-Qwen2-7B-SIIC"
OUTPUT_DIR="checkpoints/${RUN_NAME}"

# Dataset config (update paths in the YAML file)
DATA_PATH="configs/onevision_SIIC.yaml"

# Video data root directory (update to your data path)
VIDEO_FOLDER="/path/to/SIIC_data/"

# ============================================================================
# Distributed Training Settings
# ============================================================================
NUM_GPUS=8          # Number of GPUs
NNODES=1            # Number of nodes (1 for single-node training)
RANK=0              # Node rank (0 for single-node)
ADDR="localhost"    # Master node address
PORT=29500          # Master node port

echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "RUN_NAME: ${RUN_NAME}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"

# ============================================================================
# Launch Training
# ============================================================================
ACCELERATE_CPU_AFFINITY=1 torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --nnodes="${NNODES}" \
    --node_rank="${RANK}" \
    --master_addr="${ADDR}" \
    --master_port="${PORT}" \
    llava/train/train_mem.py \
    --deepspeed configs/zero3.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path $DATA_PATH \
    --video_folder $VIDEO_FOLDER \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 16
