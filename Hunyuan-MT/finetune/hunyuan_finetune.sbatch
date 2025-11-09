#!/bin/bash
#SBATCH --job-name=hunyuan-finetune
#SBATCH --partition=gpu     
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e0985887@u.nus.edu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

cd ~
source ~/venv/bin/activate

cd ~/CS4248-MT-proj/Hunyuan-MT/finetune
bash finetune_cc.sh
