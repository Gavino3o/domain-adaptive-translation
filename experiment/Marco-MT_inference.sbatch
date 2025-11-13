#!/bin/bash
#SBATCH --job-name=Marco-MT_inference
#SBATCH --partition=gpu     
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=user@mail.com
#SBATCH --gres=gpu:a100-40:1
#SBATCH --mem=40G
#SBATCH --time=24:00:00

echo "=========================================================="
echo "Job Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node List: $SLURM_JOB_NODELIST"
echo "=========================================================="

cd ~/inference
source ~/venv/bin/activate

python Marco-MT.py

echo "=========================================================="
echo "Job Finished: $(date)"
echo "=========================================================="
