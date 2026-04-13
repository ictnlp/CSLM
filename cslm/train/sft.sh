#!/bin/bash

DATA_ROOT=${}
DATA_PATH=${DATA_ROOT}/

# MODEL_DIR: path to pretrained model
MODEL_DIR=

CACHE_DIR=
mkdir -p ${CACHE_DIR}/tokenized/train
mkdir -p ${CACHE_DIR}/tokenized/valid

OUT_DIR=
mkdir -p ${OUT_DIR}

epoch=1

deepspeed --num_gpus 8 \
    sft.py \
    --deepspeed ds/ds_z2_config.json \
    --model_name_or_path ${MODEL_DIR} \
    --data_path ${DATA_PATH} \
    --cache_dir ${CACHE_DIR} \
    --preprocessing_num_workers 8 \
    --model_max_length 4096 \
    --val_set_size 1000 \
    --bf16 True \
    --do_train \
    --train_on_inputs False \
    --output_dir ${OUT_DIR} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs ${epoch} \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --log_level debug \
    --logging_steps 100
