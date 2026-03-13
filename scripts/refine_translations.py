"""
Refine existing translations using Chimera model.

Assumes Stage 1 translations already exist in testing/outputs/

Usage:
    python scripts/refine_translations.py data/raw/test.zh output.en
"""

import sys
import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import MODELS
from src.data.loaders import load_file, save_file


def load_model(model_name: str):
    """Load model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return tokenizer, model


def main(source_file: str, output_file: str):
    """Refine translations using Chimera model."""
    
    # Expected domain translation files
    testing_outputs_dir = PROJECT_ROOT / "testing/outputs"
    domain_files = {
        "literary": testing_outputs_dir / "wmttest2022.finetuned.lit.en",
        "social": testing_outputs_dir / "wmttest2022.finetuned.social.en",
        "speech": testing_outputs_dir / "wmttest2022.finetuned.speech.en",
        "base": testing_outputs_dir / "wmttest2022.base.en"
    }
    
    print(f"Loading source sentences from {source_file}...")
    source_sentences = load_file(source_file)
    print(f"Loaded {len(source_sentences)} sentences\n")
    
    # Load domain translations
    print("Loading domain-specific translations...")
    domain_translations = {}
    for domain, filepath in domain_files.items():
        if os.path.exists(filepath):
            domain_translations[domain] = load_file(str(filepath))
            print(f"  ✓ {domain}: {filepath}")
        else:
            print(f"  ✗ Missing: {filepath}")
    
    if len(domain_translations) < 2:
        print("\n✗ Error: Not enough domain translations found!")
        sys.exit(1)
    
    # Load Chimera model
    print(f"\nLoading Chimera model: {MODELS['chimera']['path']}")
    chimera_tokenizer, chimera_model = load_model(MODELS['chimera']['path'])
    print("✓ Chimera model loaded\n")
    
    # Refine translations
    print("="*60)
    print("REFINING TRANSLATIONS WITH CHIMERA")
    print("="*60 + "\n")
    
    final_translations = []
    
    for i, source in tqdm(enumerate(source_sentences), total=len(source_sentences), desc="Refining"):
        # Get translations from all available domains
        candidates = {
            domain: domain_translations[domain][i]
            for domain in domain_translations.keys()
            if i < len(domain_translations[domain])
        }
        
        if not candidates:
            print(f"✗ No translations found for sentence {i+1}")
            final_translations.append("")
            continue
        
        # Create prompt for Chimera to select best translation
        candidates_text = "\n".join(
            f"{j+1}. ({domain}): {trans}"
            for j, (domain, trans) in enumerate(candidates.items())
        )
        
        prompt = f"""你是翻译质量评估专家。给定中文原文和多个翻译版本，请选择最好的翻译。

原文: {source}

翻译候选:
{candidates_text}

请只输出最好的翻译（不要序号或解释）:"""
        
        messages = [{"role": "user", "content": prompt}]
        
        tokenized = chimera_tokenizer.apply_chat_template(
            [messages],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True
        ).to(chimera_model.device)
        
        with torch.no_grad():
            outputs = chimera_model.generate(
                tokenized,
                max_new_tokens=128,
                do_sample=False,
                eos_token_id=chimera_tokenizer.eos_token_id,
                pad_token_id=chimera_tokenizer.pad_token_id
            )
        
        prompt_length = len(tokenized[0])
        translation_tokens = outputs[0][prompt_length:]
        refinement = chimera_tokenizer.decode(translation_tokens, skip_special_tokens=True)
        final_translations.append(refinement.replace('\n', ' ').strip())
    
    # Save output
    print(f"\nSaving refined translations to {output_file}")
    save_file(output_file, final_translations)
    print("✓ Done!")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/refine_translations.py <source_file> <output_file>")
        print("Example: python scripts/refine_translations.py data/raw/test.zh output.en")
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2])
