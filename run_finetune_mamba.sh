#!/usr/bin/env bash
set -euo pipefail

GPUS="${GPUS:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29502}"
FINETUNE="${FINETUNE:-}"
EVAL_ONLY="${EVAL_ONLY:-0}"
USE_SWANLAB="${USE_SWANLAB:-0}"

FINETUNE_ARGS=()
if [[ -n "${FINETUNE}" ]]; then
    FINETUNE_ARGS=(--finetune "${FINETUNE}")
fi

EVAL_ARGS=()
if [[ "${EVAL_ONLY}" == "1" ]]; then
    EVAL_ARGS=(--eval --all_mAP)
fi

TRACKING_ARGS=()
if [[ "${USE_SWANLAB}" == "1" ]]; then
    TRACKING_ARGS=(--wandb)
fi

OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
CUDA_VISIBLE_DEVICES="${GPUS}" \
torchrun \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --master_port "${MASTER_PORT}" \
    main_finetune_mamba.py \
    --batch_size "${BATCH_SIZE:-4}" \
    --epochs "${EPOCHS:-50}" \
    --warmup_epochs "${WARMUP_EPOCHS:-0}" \
    --model "${MODEL:-Mamba_raddet}" \
    --blr "${BLR:-1e-3}" \
    --min_lr "${MIN_LR:-1e-6}" \
    --weight_decay "${WEIGHT_DECAY:-0.10}" \
    --drop_path_rate "${DROP_PATH_RATE:-0.00}" \
    --data_path "${DATA_PATH:-not_used}" \
    --RADDet_config_path "${RADDET_CONFIG:-./models/RADDet_finetune/config.json}" \
    --output_dir "${OUTPUT_DIR:-./ft_output_dir}" \
    --device "${DEVICE:-cuda:0}" \
    --seed "${SEED:-41}" \
    --distributed \
    --pin_mem \
    --num_workers "${NUM_WORKERS:-2}" \
    --attn_type "${ATTN_TYPE:-Normal}" \
    --checkpoint_period "${CHECKPOINT_PERIOD:-1}" \
    --data_type "${DATA_TYPE:-RADDet}" \
    --comment "${COMMENT:-Mamba_raddet}" \
    "${EVAL_ARGS[@]}" \
    "${TRACKING_ARGS[@]}" \
    "${FINETUNE_ARGS[@]}"
