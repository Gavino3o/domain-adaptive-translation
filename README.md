# Domain-Adaptive Chinese-English Machine Translation

A comprehensive framework for domain-adaptive neural machine translation using **Hunyuan MT** models with fine-tuned adapters for specialised domains (news, social media, speech, literary) and quality evaluation using **COMET metrics**.

## Project Overview

This project implements a two-stage translation pipeline:

1. **Stage 1: Domain-Specific Translation** - Translate text using multiple domain-specific fine-tuned adapters (news, social, speech, literary)
2. **Stage 2: Ensemble Refinement** - Use the Chimera ensemble model to select the highest-quality translation from Stage 1 outputs

Suitable for tasks requiring high-quality translations customised to specific domains.

## ✨ Key Features

- **Domain-Adaptive Translation** - Fine-tuned LoRA adapters for 4 specialized domains
- **Ensemble Selection** - Chimera model selects best translation from multiple domains
- **COMET Evaluation** - Automated quality scoring with reference-based metrics
- **Flexible Configuration** - Environment variables and centralised settings
- **Batch Processing** - Efficient translation with customizable batch sizes

## 📋 Prerequisites

- **Python** 3.10+
- **GPU** with CUDA support (NVIDIA recommended for reasonable inference speed)
- **Disk Space** ~30GB for models (Hunyuan-MT-7B + Chimera + adapters)
- **VRAM** ~20GB for inference (for fp16/bf16 models)

**Optional:**
- SLURM cluster access for distributed training
- HuggingFace account (for downloading models from Hub)

## Quick Start

### 1. Clone & Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd domain-adaptive-translation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or: .\venv\Scripts\Activate.ps1  # Windows

# Install dependencies
pip install -r requirements.txt

# Download PyTorch (CUDA-enabled, example for CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Download Models

```bash
# Option A: Download to local folders (preferred)
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='tencent/Hunyuan-MT-7B', local_dir='./Hunyuan-MT-7B')
snapshot_download(repo_id='tencent/Hunyuan-MT-Chimera-7B', local_dir='./Hunyuan-MT-Chimera-7B')
"

# Option B: Or use Hub IDs directly (auto-downloads on first run)
# Set environment variables:
export HUNYUAN_7B_PATH="tencent/Hunyuan-MT-7B"
export HUNYUAN_CHIMERA_PATH="tencent/Hunyuan-MT-Chimera-7B"
```

### 3. Run Your First Translation

```bash
# Baseline translation (single model)
python scripts/baseline_translate.py data/raw/test1.zh output.en

# Domain-adaptive pipeline (4 domains + Chimera selection)
python scripts/pipeline_translate.py data/raw/test1.zh output_pipeline.en

# Evaluate quality (requires reference)
python scripts/evaluate_translation.py data/raw/test1.zh output.en data/reference/test1.en
```

**Done!** Check `output.en` and `output_pipeline.en` for translations.

## Project Structure

```
domain-adaptive-translation/
├── config/                        # Configuration management
│   └── settings.py               # Centralized settings & paths
│
├── src/                          # Reusable Python modules
│   ├── models/
│   │   └── base.py               # Hunyuan model wrapper
│   ├── data/
│   │   └── loaders.py            # File I/O utilities
│   ├── evaluation/
│   │   └── comet_scorer.py       # COMET wrapper
│   └── utils/
│
├── scripts/                      # Executable scripts
│   ├── baseline_translate.py     # Single-model translation
│   ├── pipeline_translate.py     # Two-stage domain-adaptive pipeline
│   ├── evaluate_translation.py   # COMET scoring
│   ├── refine_translations.py    # Chimera refinement only
│   └── download_models.py        # Model downloader utility
│
├── data/                         # Data directory
│   ├── raw/                      # Test/validation datasets
│   ├── reference/                # Reference translations
│   └── processed/                # Preprocessed data (for future use)
│
├── models/                       # Model storage
│   ├── base/                     # Base Hunyuan models
│   ├── finetuned/                # Domain-specific adapters
│   │   ├── news/checkpoint-200
│   │   ├── social/checkpoint-17502
│   │   ├── speech/checkpoint-200
│   │   └── literary/checkpoint-200
│   └── checkpoints/              # Other model checkpoints
│
├── training/                     # Training infrastructure
│   ├── scripts/                  # Fine-tuning scripts
│   ├── configs/                  # DeepSpeed configs
│   ├── data/                     # Training datasets
│   └── outputs/                  # Training checkpoints
│
├── testing/                      # Evaluation outputs
│   ├── outputs/                  # Translation & scoring results
│   └── stats/                    # Comparison statistics
│
├── notebooks/                    # Jupyter notebooks
├── docs/                         # Documentation
└── README.md                     # This file
```

## Usage Guide

### Basic Translation (Baseline Model)

Translate text using the base Hunyuan-MT-7B model:

```bash
python scripts/baseline_translate.py <input_file> <output_file>

# Examples:
python scripts/baseline_translate.py data/raw/test1.zh baseline_output.en
python scripts/baseline_translate.py data/raw/wmttest2022.zh wmttest2022_base.en
```

**Input:** One source sentence per line (UTF-8 encoded)  
**Output:** One translation per line (same line count as input)

### Domain-Adaptive Pipeline

Two-stage translation for higher quality:

```bash
python scripts/pipeline_translate.py <input_file> <output_file>

# Example:
python scripts/pipeline_translate.py data/raw/test1.zh pipeline_output.en
```

**Stage 1:** Translates with 4 domain adapters (news, social, speech, literary)  
**Stage 2:** Chimera model selects the best translation from Stage 1  
**Output:** Intermediate translations in `pipeline_output_<basename>/`

### Evaluate Translation Quality

Score translations using COMET (requires reference):

```bash
python scripts/evaluate_translation.py <source> <mt> <reference> [output_file]

# Example:
python scripts/evaluate_translation.py \
  data/raw/test1.zh \
  pipeline_output.en \
  data/reference/test1.en \
  scores.txt
```

**Metric:** COMET score (0-1 scale, higher is better)

### Refine Existing Translations

Use Chimera model to refine pre-existing translations:

```bash
python scripts/refine_translations.py <source_file> <output_file>

# Example:
python scripts/refine_translations.py data/raw/test1.zh refined_output.en
```

Requires Stage 1 translations in `testing/outputs/` folder.

### Download Models

Pre-download evaluation models:

```bash
python scripts/download_models.py
```

Downloads COMET model to HuggingFace cache. Base models should be downloaded separately.

## ⚙️ Configuration

### Environment Variables

Set model paths via environment (useful for cluster deployments):

```bash
export HUNYUAN_7B_PATH="tencent/Hunyuan-MT-7B"              # or local path
export HUNYUAN_CHIMERA_PATH="tencent/Hunyuan-MT-Chimera-7B" # or local path
export BATCH_SIZE=2                                         # inference batch size
```

### Configure in Code

Edit `config/settings.py` to customize:

```python
# Model paths
MODELS = {
    "base": {"path": "Hunyuan-MT-7B"},
    "chimera": {"path": "Hunyuan-MT-Chimera-7B"}
}

# Inference parameters
INFERENCE_CONFIG = {
    "batch_size": 2,
    "max_new_tokens": 128,
    "temperature": 0.7,
    "top_k": 20,
    "top_p": 0.6
}
```

## Advanced Usage

### Fine-tune a Domain Adapter

Located in `training/` directory:

```bash
cd training/scripts

# View training script requirements
cat train_domain_adapter.py

# Example fine-tuning command
python train_domain_adapter.py \
  --model_name_or_path tencent/Hunyuan-MT-Chimera-7B \
  --train_file ../data/train_dataset_news.jsonl \
  --output_dir ../outputs/news \
  --num_train_epochs 3
```

For SLURM cluster:
```bash
sbatch hunyuan_finetune.sbatch
sbatch hunyuan_merge_lora.sbatch  # Merge LoRA weights
```

### Merge LoRA Weights

Combine adapter weights with base model:

```bash
cd training/scripts
python merge_lora_weight.py --adapters_path ../outputs/news --output_dir ../merged/news
```

### Batch Processing with Custom Scripts

Use the modular `src/` modules in your own scripts:

```python
from src.models.base import HunyuanTranslator
from src.data.loaders import load_file, save_file

# Load translations
translator = HunyuanTranslator("Hunyuan-MT-7B")
texts = load_file("data/raw/input.zh")

# Translate in batches
translations = translator.translate_batch(texts, batch_size=4)

# Save results
save_file("output.en", translations)
```

## Understanding Domain Adapters

This project includes fine-tuned LoRA adapters for 4 domains:

| Domain | Checkpoint | Typical Use Cases |
|--------|------------|------------------|
| **News** | checkpoint-200 | News articles, journalistic writing |
| **Social** | checkpoint-17502 | Social media, casual text |
| **Literary** | checkpoint-200 | Fiction, creative writing, literature |
| **Speech** | checkpoint-200 | Transcribed speech, spoken language |

**How the pipeline works:**
1. Input text is translated with all 4 adapters in parallel
2. Chimera model evaluates all 4 candidates
3. Best translation is selected based on Chimera's judgment

## Troubleshooting

### Memory Issues (Out of Memory)

**Reduce batch size:**
```bash
export BATCH_SIZE=1
python scripts/baseline_translate.py ...
```

**Use quantized model:**
```bash
export HUNYUAN_7B_PATH="tencent/Hunyuan-MT-7B-fp8"
```

**Use smaller model:**
Not available for Hunyuan; consider alternative models

### Slow Inference

- Ensure GPU is being used: Check `nvidia-smi` during inference
- GPU memory usage < 50% = CPU bottleneck (model too large for GPU)
- For CPU inference: expect 5-10x slower speed

### Model Download Issues

If Hugging Face Hub is slow/blocked:
1. Download models manually from [Hugging Face](https://huggingface.co/tencent)
2. Place in project root or set `HUNYUAN_7B_PATH` to local path

### Import Errors

Ensure project root is in Python path:
```bash
cd /path/to/domain-adaptive-translation
python scripts/...  # Run from project root
```

## References & Links

**Models:**
- [Hunyuan-MT on Hugging Face](https://huggingface.co/collections/tencent/hunyuan-mt-68b42f76d473f82798882597)
- [Hunyuan-MT Technical Report](https://arxiv.org/pdf/2509.05209)

**Evaluation:**
- [COMET Documentation](https://github.com/Unbabel/COMET)
- [COMET Paper (Rei et al., 2020)](https://aclanthology.org/2020.emnlp-main.213/)

**Fine-tuning:**
- [LoRA: Low-Rank Adaptation](https://huggingface.co/docs/peft/en/conceptual_guides/adapter)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)

## Citation

If you use this project, please cite:

```bibtex
@article{hunyuan2024,
  title={Hunyuan-MT: Advancing Neural Machine Translation with Domain Adaptation},
  author={Tencent},
  year={2024},
  note={https://github.com/Tencent-Hunyuan/Hunyuan-MT}
}
```

## License

See `LICENSE` file (or follow the Hunyuan-MT model license from Tencent).

## Contributing

Found a bug or have a feature request? Open an issue or submit a pull request!

---

**Last Updated:** March 2026  
**Maintainer:** Your Name/Team
- The models require ~15-20GB total disk space

## Troubleshooting

### PowerShell Execution Policy Error
If you get an error activating the virtual environment:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### CUDA Out of Memory
- Use the FP8 quantized model (`Hunyuan-MT-7B-fp8`)
- Reduce batch size in translation script
- Close other GPU-intensive applications

### Missing Dependencies
Make sure all dependencies are installed in the virtual environment:
```bash
pip list
```

## License

This project uses models with the following licenses:
- Hunyuan models: See individual model LICENSE files
- COMET: Apache 2.0

## References

- [Hunyuan MT Models](https://huggingface.co/Tencent)
- [COMET Metric](https://github.com/Unbabel/COMET)
