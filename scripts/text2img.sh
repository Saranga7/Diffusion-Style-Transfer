#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$pwd"

CUDA_VISIBLE_DEVICES=1 python src/text2img.py \
--prompt "A picture of the Eiffel Tower in Rayonism artwork style" \
--lora_path "lora_ckpt/lora_200epochs" \
--infer_steps 50 \
--strength 0.3 \
--save_path "T2I_output.jpg"