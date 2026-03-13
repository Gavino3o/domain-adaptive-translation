import os
from pathlib import Path
from typing import Dict

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Model paths
MODELS = {
    "base": {
        "name": "Hunyuan-MT-7B",
        "path": os.getenv("HUNYUAN_7B_PATH", "Hunyuan-MT-7B")
    },
    "chimera": {
        "name": "Hunyuan-MT-Chimera-7B",
        "path": os.getenv("HUNYUAN_CHIMERA_PATH", "Hunyuan-MT-Chimera-7B")
    }
}

# Domain adapter paths (symlinked to actual fine-tuned weights)
DOMAIN_ADAPTERS = {
    "news": PROJECT_ROOT / "models/finetuned/news/checkpoint-200",
    "social": PROJECT_ROOT / "models/finetuned/social/checkpoint-17502",
    "speech": PROJECT_ROOT / "models/finetuned/speech/checkpoint-200",
    "literary": PROJECT_ROOT / "models/finetuned/literary/checkpoint-200"
}

# Data paths
DATA_PATHS = {
    "raw": PROJECT_ROOT / "data/raw",
    "processed": PROJECT_ROOT / "data/processed",
    "reference": PROJECT_ROOT / "data/reference"
}

# Output paths
OUTPUT_PATHS = {
    "translations": PROJECT_ROOT / "outputs/translations",
    "scores": PROJECT_ROOT / "outputs/scores",
    "evaluations": PROJECT_ROOT / "testing/outputs"
}

# Model parameters
INFERENCE_CONFIG = {
    "batch_size": int(os.getenv("BATCH_SIZE", "2")),
    "max_new_tokens": 128,
    "do_sample": False,
    "temperature": 0.7,
    "top_k": 20,
    "top_p": 0.6,
    "repetition_penalty": 1.05
}

# Create output directories
for path in OUTPUT_PATHS.values():
    path.mkdir(parents=True, exist_ok=True)