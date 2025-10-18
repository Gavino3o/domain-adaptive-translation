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
cd Project
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

### 4. Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install unbabel-comet
pip install huggingface-hub
```

### 5. Download Hunyuan Models

The Hunyuan models are too large for GitHub. Download them manually:

**Option A: Using Hugging Face Hub**

```python
from huggingface_hub import snapshot_download

# Download Hunyuan-MT-7B
snapshot_download(
    repo_id="Tencent/Hunyuan-MT-7B",
    local_dir="./Hunyuan-MT-7B"
)

# Download Hunyuan-MT-7B-fp8 (quantized, smaller)
snapshot_download(
    repo_id="Tencent/Hunyuan-MT-7B-fp8",
    local_dir="./Hunyuan-MT-7B-fp8"
)

# Download Hunyuan-MT-Chimera-7B
snapshot_download(
    repo_id="Tencent/Hunyuan-MT-Chimera-7B",
    local_dir="./Hunyuan-MT-Chimera-7B"
)
```

**Option B: Manual Download**

Visit these URLs and download manually:
- [Hunyuan-MT-7B](https://huggingface.co/Tencent/Hunyuan-MT-7B)
- [Hunyuan-MT-7B-fp8](https://huggingface.co/Tencent/Hunyuan-MT-7B-fp8)
- [Hunyuan-MT-Chimera-7B](https://huggingface.co/Tencent/Hunyuan-MT-Chimera-7B)

Place the downloaded models in the project root directory.

### 6. Download COMET Model

The COMET scoring model will be downloaded automatically on first use, or you can download it manually:

```bash
python download_comet_models.py
```

This will download the `wmt22-comet-da` model to your HuggingFace cache.

## Usage

### Translation

Translate Chinese text to English:

```bash
python translate.py
```

Edit `translate.py` to specify your input/output files and model paths.

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

## Project Structure

```
.
├── translate.py              # Translation script
├── score.py                  # COMET scoring script
├── download_comet_models.py  # Helper to download COMET models
├── test1.zh                  # Sample Chinese source
├── test1.en                  # Sample English MT output
├── wmttest2022.AnnA.en      # Sample reference translation
├── mt/                       # Training/test data directory
│   ├── train.zh-en.zh
│   ├── train.zh-en.en
│   ├── tatoeba.zh
│   ├── tatoeba.en
│   ├── wmttest2022.zh
│   └── wmttest2022.AnnA.en
├── Hunyuan-MT-7B/           # Model directory (gitignored)
├── Hunyuan-MT-7B-fp8/       # Quantized model (gitignored)
├── Hunyuan-MT-Chimera-7B/   # Chimera model (gitignored)
└── venv/                     # Virtual environment (gitignored)
```

## Models

### Translation Models
- **Hunyuan-MT-7B**: Full precision model (~14GB)
- **Hunyuan-MT-7B-fp8**: FP8 quantized model (~7GB, faster inference)
- **Hunyuan-MT-Chimera-7B**: Chimera variant

### Evaluation Model
- **wmt22-comet-da**: COMET reference-based evaluation model
  - Scores range from 0 to 1 (higher is better)
  - Compares MT output against human reference translations
  - Uses source text as additional context

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
