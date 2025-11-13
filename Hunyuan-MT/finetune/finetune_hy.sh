#!/bin/bash

# Modify the paths as needed
model_path="/home/e/e0985887/CS4248-MT-proj/Hunyuan-MT-7B"
model_size="7B"
tokenizer_path=${model_path}
ds_config_file=ds_zero3_offload_no_auto.json

domains=("literary" "speech" "news" "social")

batch_time=$(date "+%Y.%m.%d-%H.%M.%S")

for domain in "${domains[@]}"; do

    echo "============================================================"
    echo "STARTING: Fine-tuning for domain: ${domain}"
    echo "============================================================"

    train_data_file="../../finetune/data/train_dataset_${domain}_hy.jsonl"
    output_path="../../finetune/weights/hf_train_output_${domain}"
    
    mkdir -p ${output_path}
    
    log_file="${output_path}/log_${batch_time}_${domain}.txt"

    echo "Paths configured for this run:"
    echo "Model path: ${model_path}"
    echo "Training data: ${train_data_file}"
    echo "DS Config: ${ds_config_file}"
    echo "Output path: ${output_path}"
    echo "Log file: ${log_file}"

    deepspeed finetune.py \
        --do_train \
        --use_lora \
        --model_size ${model_size} \
        --model_name_or_path ${model_path} \
        --tokenizer_name_or_path ${tokenizer_path} \
        --train_data_file ${train_data_file} \
        --deepspeed ${ds_config_file} \
        --output_dir ${output_path} \
        --overwrite_output_dir \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --gradient_checkpointing \
        --lr_scheduler_type cosine_with_min_lr \
        --logging_steps 1 \
        --max_steps 200 \
        --save_steps 100 \
        --learning_rate 1e-5 \
        --min_lr 1e-6 \
        --warmup_ratio 0.01 \
        --save_strategy steps \
        --save_safetensors True \
        --model_max_length 4096 \
        --max_seq_length 4096 \
        --bf16 | tee ${log_file}
    
    echo "------------------------------------------------------------"
    echo "COMPLETED: Fine-tuning for domain: ${domain}"
    echo "Outputs saved to: ${output_path}"
    echo "------------------------------------------------------------"
    echo ""

done

echo "============================================================"
echo "All fine-tuning jobs for all domains are complete."
echo "============================================================"