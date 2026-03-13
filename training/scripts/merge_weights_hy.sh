#!/bin/bash

source ~/venv/bin/activate

BASE_MODEL_PATH="/home/e/e0985887/CS4248-MT-proj/Hunyuan-MT-7B"

domains=("literary" "speech" "news" "social")

# Base path for finding finetuned adapter checkpoints
ADAPTER_BASE_PATH="/home/e/e0985887/CS4248-MT-proj/finetune/weights"

# Base path for saving the new merged models
MERGED_BASE_PATH="/home/e/e0985887/CS4248-MT-proj"


for domain in "${domains[@]}"; do

    # Capitalise first letter of domain for naming consistency
    domain_name="$(tr '[:lower:]' '[:upper:]' <<< ${domain:0:1})${domain:1}"
    
    ADAPTER_MODEL_PATH="${ADAPTER_BASE_PATH}/hf_train_output_${domain}/checkpoint-200"
    MERGED_OUTPUT_PATH="${MERGED_BASE_PATH}/Hunyuan-MT-7B-Finetuned-${domain_name}"

    python merge_lora_weight.py \
        --base_model_path ${BASE_MODEL_PATH} \
        --adapter_model_path ${ADAPTER_MODEL_PATH} \
        --output_path ${MERGED_OUTPUT_PATH} \
        --save_dtype bf16

done