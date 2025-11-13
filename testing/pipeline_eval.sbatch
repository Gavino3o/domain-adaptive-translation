#!/bin/bash
#SBATCH --job-name=hy-eval-pipe
#SBATCH --partition=gpu     
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e0985887@u.nus.edu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -e

echo "Activating virtual environment..."
cd ~
source ~/venv/bin/activate

# --- 1. File Definitions ---
export SOURCE_FILE="/home/e/e0985887/CS4248-MT-proj/testing/generaltest2022.zh-en.src.zh"
export REFERENCE_FILE="/home/e/e0985887/CS4248-MT-proj/testing/generaltest2022.zh-en.ref.A.en"

export OUTPUT_DIR="/home/e/e0985887/CS4248-MT-proj/testing/outputs"

export BASE_OUTPUT="$OUTPUT_DIR/wmttest2022.base.en"

export PIPELINE_OUTPUT="$OUTPUT_DIR/wmttest2022.pipeline.en"

export PIPELINE_SCORE_FILE="$OUTPUT_DIR/pipeline_model_score.txt"
export COMP_PIPELINE_FILE="$OUTPUT_DIR/comparison_stats_base_vs_pipeline.txt"


cd ~/CS4248-MT-proj/testing

echo "=============================================="
echo "STEP 1: SCORING Pipeline output"
echo "=============================================="

if [ -f "$PIPELINE_SCORE_FILE" ]; then
    echo "Score file $PIPELINE_SCORE_FILE exists. Skipping."
else
    if [ ! -f "$PIPELINE_OUTPUT" ]; then
        echo "ERROR: Pipeline output file not found: $PIPELINE_OUTPUT"
        exit 1
    fi
    echo "Scoring $PIPELINE_OUTPUT..."
    python score.py $SOURCE_FILE $PIPELINE_OUTPUT $REFERENCE_FILE > $PIPELINE_SCORE_FILE
fi


echo "=============================================="
echo "STEP 2: STATISTICAL COMPARISON (Base vs. Pipeline)"
echo "=============================================="

echo "--- Comparing Base vs. Pipeline ---"
if [ -f "$COMP_PIPELINE_FILE" ]; then
    echo "Comparison file $COMP_PIPELINE_FILE exists. Skipping."
else
    # --- Fail-Safe: Check that required files exist before comparing ---
    if [ ! -f "$BASE_OUTPUT" ]; then
        echo "ERROR: Base output file not found: $BASE_OUTPUT"
        echo "Please run the main evaluate.sbatch script first to generate it."
        exit 1
    fi
    if [ ! -f "$PIPELINE_OUTPUT" ]; then
        echo "ERROR: Pipeline output file not found: $PIPELINE_OUTPUT"
        exit 1
    fi
    
    echo "Running comet-compare: Base vs. Pipeline..."
    comet-compare -s $SOURCE_FILE -t $BASE_OUTPUT $PIPELINE_OUTPUT -r $REFERENCE_FILE > $COMP_PIPELINE_FILE
fi


# --- FINAL OUTPUT ---

echo "Done."