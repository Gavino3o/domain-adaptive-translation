# Chinese-English Machine Translation Project

This project uses Hunyuan MT models for Chinese-to-English translation and COMET for translation quality evaluation.

## Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (recommended)
- ~15GB disk space for models
- Git LFS (for large model files if storing separately)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd CS4248-MT-proj
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 4. Install dependencies

This repository includes a minimal `requirements.txt`. The recommended (and simplest) way to install the Python dependencies is:

```bash
python -m pip install -r requirements.txt
```

Platform notes:
- If you have an NVIDIA GPU and want CUDA-accelerated PyTorch on Linux, install PyTorch following the official instructions at https://pytorch.org/get-started/locally/ (select your CUDA version). Example (Linux / CUDA 11.8):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

- On macOS (Apple Silicon or Intel) there is no CUDA; follow the PyTorch "macOS" instructions on the PyTorch site (they provide the correct wheel / install method for CPU/Metal).

- If you prefer to install pieces manually, the main packages used are:

```bash
pip install transformers unbabel-comet huggingface-hub
```

### 5. Download Hunyuan models

The Hunyuan models are large and not included in the repo. You can either use the Hugging Face Hub directly (preferred) or download snapshots to local folders. The model name used in the translation scripts (for example `translate.py`) is `Hunyuan-MT-7B` by default; that may be either a local folder (e.g. `./Hunyuan-MT-7B`) or the Hub id `tencent/Hunyuan-MT-7B`.

Option A — Hub (recommended): the transformer loader will accept a Hub id or a local path. Example in Python (using the Hugging Face Hub API):

```python
from huggingface_hub import snapshot_download

# Download a model snapshot into a local directory
snapshot_download(repo_id="tencent/Hunyuan-MT-7B", local_dir="./Hunyuan-MT-7B")
```

Option B — Manual: visit the model pages and follow the model license/usage instructions, then place the extracted model folder in the project root (or set `MODEL_NAME` in the scripts to the hub id).

Common model ids / folder names used in this repo:
- `tencent/Hunyuan-MT-7B`  (local folder `Hunyuan-MT-7B`)
- `tencent/Hunyuan-MT-7B-fp8`  (local folder `Hunyuan-MT-7B-fp8`)
- `tencent/Hunyuan-MT-Chimera-7B` (local folder `Hunyuan-MT-Chimera-7B`)

If you set the script variable (e.g. `MODEL_NAME` in `translate.py`) to the Hugging Face id, the model will be loaded from the Hub (if you have network access and credentials if required).

### 6. Download COMET Model

The COMET scoring model will be downloaded automatically on first use, or you can download it manually:

```bash
python download_comet_models.py
```

This will download the `wmt22-comet-da` model to your HuggingFace cache.

## Usage

## Quick start

After cloning and creating a virtual environment, run:

```bash
# Create and activate a venv (macOS / Linux)
python -m venv venv
source venv/bin/activate

# Install requirements
python -m pip install -r requirements.txt

# (optional) download the COMET scoring model once
python download_comet_models.py
```

Translate (single script):

```bash
# The translate script reads the source file and writes translations.
python translate.py
```

By default `translate.py` sets `MODEL_NAME = "Hunyuan-MT-7B"`. Edit that variable if you want to point to a different local folder or Hub id (for example `tencent/Hunyuan-MT-7B`).

Translate using the pipeline helper (domain-aware checkpoints):

```bash
python translate_pipeline.py <input_file>
# Example:
python translate_pipeline.py test1.zh
```

Score translations using COMET:

```bash
python score.py <source_file> <mt_file> <reference_file>
# Example:
python score.py test1.zh test1.en wmttest2022.AnnA.en
```

Other helper scripts:
- `download_comet_models.py` — download reference-based COMET model to cache and print paths
- `refine_only.py` — small helper (see header for usage)

Notes:
- Scripts assume one sentence per line for source/mt/reference files.
- If you run on CPU (no GPU) expect much slower inference. For macOS, install PyTorch as recommended on the PyTorch website (select macOS and CPU/Metal).

### Scoring Translations

Evaluate translation quality using COMET:

```bash
python score.py <source_file> <mt_file> <reference_file>
```

**Example:**
```bash
python score.py test1.zh test1.en wmttest2022.AnnA.en
```

**Arguments:**
- `source_file`: Chinese source text (one sentence per line)
- `mt_file`: Machine-translated English text (one sentence per line)
- `reference_file`: Human reference English translation (one sentence per line)

**Output:**
- Individual COMET scores for each sentence (0-1 scale, higher is better)
- Average COMET score across all sentences

## Project structure (high level)

```
.
├── translate.py              # Translation script (edit MODEL_NAME / input/output)
├── translate_pipeline.py     # Domain-aware pipeline that picks finetuned checkpoints
├── refine_only.py            # Small helper script (usage in header)
├── score.py                  # COMET scoring script
├── download_comet_models.py  # Helper to download COMET models
├── requirements.txt          # Python dependencies
├── test1.zh/test1.en         # Small samples used in repo
├── mt/                       # Training/test corpora (large files)
├── finetune/                 # Finetuning adapters/checkpoints and helper data
├── Hunyuan-MT/               # Hunyuan-MT helper code and finetune scripts
└── testing/                  # Evaluation pipelines and outputs
```

## Models

Translation models used in this repo (examples):
- `tencent/Hunyuan-MT-7B` — full precision 7B translation model
- `tencent/Hunyuan-MT-7B-fp8` — quantized FP8 variant (smaller, cheaper GPU memory)
- `tencent/Hunyuan-MT-Chimera-7B` — ensemble Chimera model

Evaluation model:
- `Unbabel/wmt22-comet-da` — reference-based COMET scoring model (downloaded by `download_comet_models.py` or automatically by `score.py`).

COMET returns per-sentence scores (higher is better). See `score.py` for usage and options.

## Notes

- First run will take longer due to model loading
- GPU is highly recommended for translation
- COMET scoring works on both CPU and GPU
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
