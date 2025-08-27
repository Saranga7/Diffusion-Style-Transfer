#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$pwd"


CUDA_VISIBLE_DEVICES=1 python src/trainer.py \
--dataset_path "data/rayonismDataset" \
--center_crop false \
--lora_r 4 \
--lora_alpha 16 \
--lora_dropout 0.1 \
--train_strength 1.0 \
--learning_rate 5e-6 \
--batch_size 4 \
--grad_accumulation 2 \
--max_epochs 5 \
--save_name "lora_5epochs"

