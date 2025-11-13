#!/bin/bash
#SBATCH --job-name=hy-eval
#SBATCH --partition=gpu     
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e0985887@u.nus.edu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -e

echo "Activating virtual environment..."

cd ~
source ~/venv/bin/activate

# --- Model & File Definitions ---
export BASE_MODEL="/home/e/e0985887/CS4248-MT-proj/Hunyuan-MT-7B"
export FINETUNED_MODEL_LIT="/home/e/e0985887/CS4248-MT-proj/Hunyuan-MT-7B-Finetuned-Literature"
export FINETUNED_MODEL_SPEECH="/home/e/e0985887/CS4248-MT-proj/Hunyuan-MT-7B-Finetuned-Speech"
export FINETUNED_MODEL_NEWS="/home/e/e0985887/CS4248-MT-proj/Hunyuan-MT-7B-Finetuned-News"
export FINETUNED_MODEL_SOCIAL="/home/e/e0985887/CS4248-MT-proj/Hunyuan-MT-7B-Finetuned-Social"

export SOURCE_FILE="/home/e/e0985887/CS4248-MT-proj/testing/generaltest2022.zh-en.src.zh"
export REFERENCE_FILE="/home/e/e0985887/CS4248-MT-proj/testing/generaltest2022.zh-en.ref.A.en"

# --- Centralized Output Directory Definition ---
export OUTPUT_DIR="/home/e/e0985887/CS4248-MT-proj/testing/outputs"

# --- Output file paths ---
export BASE_OUTPUT="$OUTPUT_DIR/wmttest2022.base.en"
export FINETUNED_OUTPUT_LIT="$OUTPUT_DIR/wmttest2022.finetuned.lit.en"
export FINETUNED_OUTPUT_SPEECH="$OUTPUT_DIR/wmttest2022.finetuned.speech.en"
export FINETUNED_OUTPUT_NEWS="$OUTPUT_DIR/wmttest2022.finetuned.news.en"
export FINETUNED_OUTPUT_SOCIAL="$OUTPUT_DIR/wmttest2022.finetuned.social.en"

# --- Score file paths ---
export BASE_SCORE_FILE="$OUTPUT_DIR/base_model_score.txt"
export LIT_SCORE_FILE="$OUTPUT_DIR/finetuned_lit_score.txt"
export SPEECH_SCORE_FILE="$OUTPUT_DIR/finetuned_speech_score.txt"
export NEWS_SCORE_FILE="$OUTPUT_DIR/finetuned_news_score.txt"
export SOCIAL_SCORE_FILE="$OUTPUT_DIR/finetuned_social_score.txt"

# --- Comparison file paths ---
export COMP_LIT_FILE="$OUTPUT_DIR/comparison_stats_base_vs_lit.txt"
export COMP_SPEECH_FILE="$OUTPUT_DIR/comparison_stats_base_vs_speech.txt"
export COMP_NEWS_FILE="$OUTPUT_DIR/comparison_stats_base_vs_news.txt"
export COMP_SOCIAL_FILE="$OUTPUT_DIR/comparison_stats_base_vs_social.txt"


echo "=============================================="
echo "SETUP: Ensuring output directory exists..."
echo "=============================================="
mkdir -p $OUTPUT_DIR
echo "Output directory $OUTPUT_DIR is ready."

cd ~/CS4248-MT-proj/testing


echo "=============================================="
echo "STEP 1: GENERATING with BASE model"
echo "=============================================="
if [ -f "$BASE_OUTPUT" ];
then
    echo "Output file $BASE_OUTPUT already exists. Skipping translation."
else
    echo "Generating translations for BASE model..."
    python translate_cli.py \
        --model_name_or_path $BASE_MODEL \
        --source_file $SOURCE_FILE \
        --output_file $BASE_OUTPUT
    echo "BASE model translation complete."
fi


echo "=============================================="
echo "STEP 2a: GENERATING with FINETUNED (Literary)"
echo "=============================================="
if [ -f "$FINETUNED_OUTPUT_LIT" ];
then
    echo "Output file $FINETUNED_OUTPUT_LIT already exists. Skipping translation."
else
    echo "Generating translations for LITERARY model..."
    python translate_cli.py \
        --model_name_or_path $FINETUNED_MODEL_LIT \
        --source_file $SOURCE_FILE \
        --output_file $FINETUNED_OUTPUT_LIT
    echo "LITERARY model translation complete."
fi


echo "=============================================="
echo "STEP 2b: GENERATING with FINETUNED (Speech)"
echo "=============================================="
if [ -f "$FINETUNED_OUTPUT_SPEECH" ];
then
    echo "Output file $FINETUNED_OUTPUT_SPEECH already exists. Skipping translation."
else
    echo "Generating translations for SPEECH model..."
    python translate_cli.py \
        --model_name_or_path $FINETUNED_MODEL_SPEECH \
        --source_file $SOURCE_FILE \
        --output_file $FINETUNED_OUTPUT_SPEECH
    echo "SPEECH model translation complete."
fi

echo "=============================================="
echo "STEP 2c: GENERATING with FINETUNED (News)"
echo "=============================================="
if [ -f "$FINETUNED_OUTPUT_NEWS" ];
then
    echo "Output file $FINETUNED_OUTPUT_NEWS already exists. Skipping translation."
else
    echo "Generating translations for NEWS model..."
    python translate_cli.py \
        --model_name_or_path $FINETUNED_MODEL_NEWS \
        --source_file $SOURCE_FILE \
        --output_file $FINETUNED_OUTPUT_NEWS
    echo "NEWS model translation complete."
fi

echo "=============================================="
echo "STEP 2d: GENERATING with FINETUNED (Social)"
echo "=============================================="
if [ -f "$FINETUNED_OUTPUT_SOCIAL" ];
then
    echo "Output file $FINETUNED_OUTPUT_SOCIAL already exists. Skipping translation."
else
    echo "Generating translations for SOCIAL model..."
    python translate_cli.py \
        --model_name_or_path $FINETUNED_MODEL_SOCIAL \
        --source_file $SOURCE_FILE \
        --output_file $FINETUNED_OUTPUT_SOCIAL
    echo "SOCIAL model translation complete."
fi


echo "=============================================="
echo "STEP 3: SCORING outputs"
echo "=============================================="

echo "--- Scoring Base Model ---"
if [ -f "$BASE_SCORE_FILE" ];
then
    echo "Score file $BASE_SCORE_FILE exists. Skipping."
else
    python score.py $SOURCE_FILE $BASE_OUTPUT $REFERENCE_FILE > $BASE_SCORE_FILE
fi

echo "--- Scoring Finetuned (Literary) ---"
if [ -f "$LIT_SCORE_FILE" ];
then
    echo "Score file $LIT_SCORE_FILE exists. Skipping."
else
    python score.py $SOURCE_FILE $FINETUNED_OUTPUT_LIT $REFERENCE_FILE > $LIT_SCORE_FILE
fi

echo "--- Scoring Finetuned (Speech) ---"
if [ -f "$SPEECH_SCORE_FILE" ];
then
    echo "Score file $SPEECH_SCORE_FILE exists. Skipping."
else
    python score.py $SOURCE_FILE $FINETUNED_OUTPUT_SPEECH $REFERENCE_FILE > $SPEECH_SCORE_FILE
fi

echo "--- Scoring Finetuned (News) ---"
if [ -f "$NEWS_SCORE_FILE" ];
then
    echo "Score file $NEWS_SCORE_FILE exists. Skipping."
else
    python score.py $SOURCE_FILE $FINETUNED_OUTPUT_NEWS $REFERENCE_FILE > $NEWS_SCORE_FILE
fi

echo "--- Scoring Finetuned (Social) ---"
if [ -f "$SOCIAL_SCORE_FILE" ];
then
    echo "Score file $SOCIAL_SCORE_FILE exists. Skipping."
else
    python score.py $SOURCE_FILE $FINETUNED_OUTPUT_SOCIAL $REFERENCE_FILE > $SOCIAL_SCORE_FILE
fi


echo "=============================================="
echo "STEP 4: STATISTICAL COMPARISONS (vs. Base)"
echo "=============================================="

echo "--- Comparing Base vs. Literary ---"
if [ -f "$COMP_LIT_FILE" ];
then
    echo "Comparison file $COMP_LIT_FILE exists. Skipping."
else
    comet-compare -s $SOURCE_FILE -t $BASE_OUTPUT $FINETUNED_OUTPUT_LIT -r $REFERENCE_FILE > $COMP_LIT_FILE
fi

echo "--- Comparing Base vs. Speech ---"
if [ -f "$COMP_SPEECH_FILE" ];
then
    echo "Comparison file $COMP_SPEECH_FILE exists. Skipping."
else
    comet-compare -s $SOURCE_FILE -t $BASE_OUTPUT $FINETUNED_OUTPUT_SPEECH -r $REFERENCE_FILE > $COMP_SPEECH_FILE
fi

echo "--- Comparing Base vs. News ---"
if [ -f "$COMP_NEWS_FILE" ];
then
    echo "Comparison file $COMP_NEWS_FILE exists. Skipping."
else
    comet-compare -s $SOURCE_FILE -t $BASE_OUTPUT $FINETUNED_OUTPUT_NEWS -r $REFERENCE_FILE > $COMP_NEWS_FILE
fi

echo "--- Comparing Base vs. Social ---"
if [ -f "$COMP_SOCIAL_FILE" ];
then
    echo "Comparison file $COMP_SOCIAL_FILE exists. Skipping."
else
    comet-compare -s $SOURCE_FILE -t $BASE_OUTPUT $FINETUNED_OUTPUT_SOCIAL -r $REFERENCE_FILE > $COMP_SOCIAL_FILE
fi



echo "=============================================="
echo "EVALUATION COMPLETE. Final scores:"
echo "=============================================="

echo "--- BASE MODEL SCORE (Individual) ---"
cat $BASE_SCORE_FILE

echo "--- FINETUNED (LITERARY) SCORE (Individual) ---"
cat $LIT_SCORE_FILE

echo "--- FINETUNED (SPEECH) SCORE (Individual) ---"
cat $SPEECH_SCORE_FILE

echo "--- FINETUNED (NEWS) SCORE (Individual) ---"
cat $NEWS_SCORE_FILE

echo "--- FINETUNED (SOCIAL) SCORE (Individual) ---"
cat $SOCIAL_SCORE_FILE

echo "--- STATISTICAL COMPARISON: BASE vs. LITERARY ---"
echo "(see $COMP_LIT_FILE for details)"
cat $COMP_LIT_FILE

echo "--- STATISTICAL COMPARISON: BASE vs. SPEECH ---"
echo "(see $COMP_SPEECH_FILE for details)"
cat $COMP_SPEECH_FILE

echo "--- STATISTICAL COMPARISON: BASE vs. NEWS ---"
echo "(see $COMP_NEWS_FILE for details)"
cat $COMP_NEWS_FILE


echo "--- STATISTICAL COMPARISON: BASE vs. SOCIAL ---"
echo "(see $COMP_SOCIAL_FILE for details)"
cat $COMP_SOCIAL_FILE

echo "Successfully completed all evaluation steps for Hunyuan-MT models."