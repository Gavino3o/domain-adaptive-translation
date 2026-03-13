"""
Translate Chinese to English using base Hunyuan MT model.

Usage:
    python scripts/baseline_translate.py data/raw/test.zh output.en
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import MODELS, INFERENCE_CONFIG
from src.models.base import HunyuanTranslator
from src.data.loaders import load_file, save_file


def main(input_file: str, output_file: str):
    """Translate file using baseline model."""
    
    print(f"Loading source sentences from {input_file}...")
    source_sentences = load_file(input_file)
    print(f"Loaded {len(source_sentences)} sentences")
    
    print(f"\nInitializing model: {MODELS['base']['name']}")
    translator = HunyuanTranslator(MODELS['base']['path'])
    
    print(f"\nTranslating with batch size: {INFERENCE_CONFIG['batch_size']}")
    translations = translator.translate_batch(
        source_sentences,
        batch_size=INFERENCE_CONFIG['batch_size']
    )
    
    print(f"\nSaving translations to {output_file}")
    save_file(output_file, translations)
    print("✓ Done!")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/baseline_translate.py <input_file> <output_file>")
        print("Example: python scripts/baseline_translate.py data/raw/test.zh output.en")
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2])