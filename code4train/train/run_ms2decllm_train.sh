#!/bin/bash

# NCCL IB environment variables
export NCCL_IB_HCA=mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export OMP_NUM_THREADS=1

PROJECT_NAME="ms2decllm"
PARENT_SAVE_DIR="output_models6.7/"
PARENT_TENSORBOARD_DIR="tensorboard6.7/"
PARENT_CONFIG_FILE="configs6.7/"
PRETRAINED_MODEL_PATH="/data3/liupei/llm4decompile/llm4dec-6.7b"

mkdir -p $PARENT_SAVE_DIR $PARENT_TENSORBOARD_DIR $PARENT_CONFIG_FILE

declare -a dataset="/data3/liupei/NDSS2026/TrainData/4PreparedData4Train/prompt1/arrow"

TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
FULL_PROJECT_NAME="${PROJECT_NAME}-${TIMESTAMP}"
SAVE_DIR="${PARENT_SAVE_DIR}${FULL_PROJECT_NAME}"
TENSORBOARD_DIR="${PARENT_TENSORBOARD_DIR}${FULL_PROJECT_NAME}"
CONFIG_FILE="${PARENT_CONFIG_FILE}${FULL_PROJECT_NAME}.json"

touch $CONFIG_FILE

NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=ALL

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
colossalai run --nproc_per_node 8 --hostfile hostfile --master_port 53368 train.py \
    --pretrained $PRETRAINED_MODEL_PATH \
    --dataset ${dataset[@]} \
    --plugin "zero2" \
    --save_interval 125 \
    --save_dir $SAVE_DIR \
    --tensorboard_dir $TENSORBOARD_DIR \
    --config_file $CONFIG_FILE \
    --num_epochs 1 \
    --micro_batch_size 12 \
    --accumulation_steps 8 \
    --lr 2e-5 \
    --mixed_precision "bf16" \
    --grad_clip 1.0 \
    --weight_decay 0.01 \
    --warmup_steps 100 \
    --use_grad_checkpoint \
    --padding_mode "longest" \
    --max_length 4096 \
    --use_flash_attn \
    --pad_token "eos"
