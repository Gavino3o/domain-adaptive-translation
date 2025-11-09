#!/bin/bash

source ~/venv/bin/activate

BASE_MODEL_PATH="/home/e/e0985887/CS4248-MT-proj/Hunyuan-MT-7B"
ADAPTER_MODEL_PATH="/home/e/e0985887/CS4248-MT-proj/Hunyuan-MT/finetune/hf_train_output/checkpoint-200"
MERGED_OUTPUT_PATH="/home/e/e0985887/CS4248-MT-proj/Hunyuan-MT-7B-Finetuned-Literature"

echo "Base Model:    ${BASE_MODEL_PATH}"
echo "Adapter:       ${ADAPTER_MODEL_PATH}"
echo "Output to:     ${MERGED_OUTPUT_PATH}"

python merge_lora_weight.py \
    --base_model_path ${BASE_MODEL_PATH} \
    --adapter_model_path ${ADAPTER_MODEL_PATH} \
    --output_path ${MERGED_OUTPUT_PATH} \
    --save_dtype bf16


