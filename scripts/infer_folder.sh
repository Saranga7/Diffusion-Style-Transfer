#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$pwd"

# models to test
MODELS=("lora_5epochs" "lora_100epochs" "lora_200epochs")
BASE_IMAGE_FOLDER="google_images/"
OUT_FOLDER="out"
PROMPT="A Rayonism style artwork"


# hyperparameter grid 
STRENGTHS=(0.3 0.5 0.7 0.9)
INF_STEPS=(10 30 50)

if [ ! -d "$OUT_FOLDER" ]; then
    mkdir -p "$OUT_FOLDER"
fi

for LORA_MODEL in "${MODELS[@]}"; do
    for STRENGTH in "${STRENGTHS[@]}"; do
        for INF_STEP in "${INF_STEPS[@]}"; do

        echo "----------------------------------------------------------------"
        echo "MODEL: $LORA_MODEL | STRENGTH: $STRENGTH | STEPS: $INF_STEP"
        echo "Saving to: $OUT_FOLDER/$LORA_MODEL/${BASE_IMAGE_FOLDER}/strength_${STRENGTH}"
        echo "----------------------------------------------------------------"


        CUDA_VISIBLE_DEVICES=1 python src/transfer_style_folder.py \
        --lora_path "lora_ckpt/$LORA_MODEL" \
        --folder_path "$BASE_IMAGE_FOLDER" \
        --strength $STRENGTH \
        --prompt "$PROMPT" \
        --infer_steps $INF_STEP \
        --save_path "$OUT_FOLDER/$LORA_MODEL/${BASE_IMAGE_FOLDER}/strength_${STRENGTH}_steps_${INF_STEP}" \
        --seed 777

        echo "Finished: ${LORA_MODEL} s=${STRENGTH} steps=${INF_STEP}"

        done
    done
done


echo "All runs finished. Results at: $OUT_ROOT"

