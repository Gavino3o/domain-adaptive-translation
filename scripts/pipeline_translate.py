"""
Two-stage pipeline translation using domain-specific adapters.

Stage 1: Translate with 4 domain-specific models
Stage 2: Use Chimera model to refine translations

Usage:
    python scripts/pipeline_translate.py data/raw/test.zh output.en
"""

import sys
import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import MODELS, DOMAIN_ADAPTERS, INFERENCE_CONFIG
from src.data.loaders import load_file, save_file


def load_base_model(model_name: str):
    """Load base tokenizer and model."""
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


def translate_with_adapter(text: str, base_model, base_tokenizer, adapter_path: str):
    """Translate text using base model + domain adapter."""
    if not os.path.exists(adapter_path):
        return text  # Fallback to original if adapter not found
    
    # Load adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    messages = [
        {"role": "user", "content": f"把下面的文本翻译成English，不要额外解释。\n\n{text}"}
    ]
    
    tokenized = base_tokenizer.apply_chat_template(
        [messages],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            tokenized,
            max_new_tokens=128,
            do_sample=False,
            eos_token_id=base_tokenizer.eos_token_id,
            pad_token_id=base_tokenizer.pad_token_id
        )
    
    prompt_length = len(tokenized[0])
    translation_tokens = outputs[0][prompt_length:]
    translation = base_tokenizer.decode(translation_tokens, skip_special_tokens=True)
    
    # Unload adapter to save memory
    model = model.merge_and_unload()
    
    return translation.replace('\n', ' ').strip()


def translate_stage1(source_sentences, base_model, base_tokenizer, output_dir):
    """Stage 1: Translate with all 4 domain-specific adapters."""
    print("\n" + "="*60)
    print("STAGE 1: Translating with 4 domain-specific adapters")
    print("="*60)
    
    domain_outputs = {}
    for domain, adapter_path in DOMAIN_ADAPTERS.items():
        print(f"\n[{domain.upper()}] Loading adapter: {adapter_path}")
        domain_translations = []
        
        for sentence in tqdm(source_sentences, desc=f"Translating with {domain} adapter"):
            translation = translate_with_adapter(
                sentence, base_model, base_tokenizer, str(adapter_path)
            )
            domain_translations.append(translation)
        
        # Save domain-specific output
        output_file = os.path.join(output_dir, f"stage1_{domain}.en")
        save_file(output_file, domain_translations)
        domain_outputs[domain] = domain_translations
        print(f"✓ Saved: {output_file}")
    
    return domain_outputs


def translate_stage2(source_sentences, domain_outputs, base_model, base_tokenizer):
    """Stage 2: Use Chimera model to select best translation from 4 domains."""
    print("\n" + "="*60)
    print("STAGE 2: Using Chimera to refine translations")
    print("="*60)
    
    chimera_tokenizer, chimera_model = load_base_model(MODELS['chimera']['path'])
    
    final_translations = []
    
    for i, source in tqdm(enumerate(source_sentences), total=len(source_sentences), desc="Refining"):
        # Get translations from all domains
        candidates = {domain: outputs[i] for domain, outputs in domain_outputs.items()}
        
        # Create prompt for Chimera to select best translation
        candidates_text = "\n".join(
            f"{j+1}. ({domain}): {trans}"
            for j, (domain, trans) in enumerate(candidates.items())
        )
        
        prompt = f"""你是翻译质量评估专家。给定中文原文和4个不同领域的翻译版本，请选择最好的翻译。

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
    
    return final_translations


def main(input_file: str, output_file: str):
    """Run two-stage translation pipeline."""
    
    print(f"Loading source sentences from {input_file}...")
    source_sentences = load_file(input_file)
    print(f"Loaded {len(source_sentences)} sentences\n")
    
    # Create output directory for intermediate files
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_dir = f"pipeline_output_{base_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load base models
    print(f"Loading base model: {MODELS['base']['path']}")
    base_tokenizer, base_model = load_base_model(MODELS['base']['path'])
    print("✓ Base model loaded\n")
    
    # Stage 1: Translate with domain adapters
    domain_outputs = translate_stage1(source_sentences, base_model, base_tokenizer, output_dir)
    
    # Stage 2: Refine with Chimera
    final_translations = translate_stage2(source_sentences, domain_outputs, base_model, base_tokenizer)
    
    # Save final output
    print(f"\nSaving final translations to {output_file}")
    save_file(output_file, final_translations)
    print("✓ Done!")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/pipeline_translate.py <input_file> <output_file>")
        print("Example: python scripts/pipeline_translate.py data/raw/test.zh output.en")
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2])
