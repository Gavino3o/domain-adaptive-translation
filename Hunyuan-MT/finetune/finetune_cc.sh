#!/bin/bash

model_path="/home/e/e0985887/CS4248-MT-proj/Hunyuan-MT-7B"

model_size="7B"
tokenizer_path=${model_path}

train_data_file=train_dataset_literary_final.jsonl

ds_config_file=ds_zero3_offload_no_auto.json
output_path=./hf_train_output

mkdir -p ${output_path}

current_time=$(date "+%Y.%m.%d-%H.%M.%S")
log_file=${output_path}/"log_${current_time}.txt"

echo "Paths configured:"
echo "Model path: ${model_path}"
echo "Training data: ${train_data_file}"
echo "DS Config: ${ds_config_file}"
echo "Output path: ${output_path}"

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


