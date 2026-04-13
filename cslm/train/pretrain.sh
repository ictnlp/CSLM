#!/bin/bash

DATA_ROOT=${}
DATA_PATH=${DATA_ROOT}/
MODEL_DIR=Meta-Llama-3.1-8B-Instruct

CACHE_DIR=
mkdir -p ${CACHE_DIR}/tokenized/train
mkdir -p ${CACHE_DIR}/tokenized/valid

OUT_DIR=
mkdir -p ${OUT_DIR}

epoch=1

hostfile=
export MASTER_ADDR=
export MASTER_PORT=
export WORLD_SIZE=
export NODE_RANK=


deepspeed --num_nodes 3 --num_gpus 8 --hostfile ${hostfile} \
    --master_addr ${} --master_port ${} \
    pretrain.py \
    --deepspeed ds_z2_config.json \
    --model_name_or_path ${MODEL_DIR} \
    --data_path ${DATA_PATH} \
    --cache_dir ${CACHE_DIR} \
    --preprocessing_num_workers 8 \
    --model_max_length 1024 \
    --val_set_size 2000 \
    --bf16 True \
    --do_train \
    --train_on_inputs False \
    --output_dir ${OUT_DIR} \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 6 \
    --num_train_epochs ${epoch} \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --learning_rate 6e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --log_level debug \
    --logging_steps 100
