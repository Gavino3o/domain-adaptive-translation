"""
Download required models for translation and evaluation.

Downloads:
  - COMET model (for evaluation)
  
Base models (Hunyuan) should be downloaded separately or set via environment variables.

Usage:
    python scripts/download_models.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.comet_scorer import CometEvaluator


def main():
    """Download required models."""
    
    print("="*60)
    print("DOWNLOADING REQUIRED MODELS")
    print("="*60 + "\n")
    
    # Download COMET evaluator model
    print("Downloading COMET model (wmt22-comet-da)...")
    try:
        evaluator = CometEvaluator(model_name="wmt22-comet-da")
        print("✓ COMET model downloaded successfully!\n")
    except Exception as e:
        print(f"✗ Error downloading COMET model: {e}\n")
        sys.exit(1)
    
    print("="*60)
    print("INFORMATION: Base Models")
    print("="*60)
    print("""
Base Hunyuan MT models should be downloaded separately:

Option 1: Via Hugging Face Hub (recommended)
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id="tencent/Hunyuan-MT-7B", local_dir="./Hunyuan-MT-7B")
    snapshot_download(repo_id="tencent/Hunyuan-MT-Chimera-7B", local_dir="./Hunyuan-MT-Chimera-7B")

Option 2: Set environment variables
    export HUNYUAN_7B_PATH="tencent/Hunyuan-MT-7B"
    export HUNYUAN_CHIMERA_PATH="tencent/Hunyuan-MT-Chimera-7B"

Models will be auto-downloaded from Hub on first use if environment
variables point to Hub IDs instead of local paths.
    """)
    
    print("="*60)
    print("✓ All required models are ready!")
    print("="*60)


if __name__ == "__main__":
    main()
