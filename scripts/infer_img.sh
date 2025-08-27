#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$pwd"

LORA_MODEL="lora_100epochs"

CUDA_VISIBLE_DEVICES=1 python src/transfer_style.py \
--lora_path "lora_ckpt/$LORA_MODEL" \
--image_path "google_images/stunning-alpine-scenery-breathtaking-dolomites-600nw-2454497339.jpg.webp" \
--strength 0.67 \
--prompt "A Rayonism artwork style phto" \
--infer_steps 30 \
--save_path "styled_image_$LORA_MODEL.jpg" \
--seed 777