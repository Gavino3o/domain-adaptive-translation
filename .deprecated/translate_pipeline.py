import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time
from tqdm import tqdm
import os
import sys
import gc

# --- Configuration ---
BASE_MODEL_NAME = "Hunyuan-MT-Chimera-7B"
CHIMERA_MODEL_NAME = "Hunyuan-MT-Chimera-7B"

# Domain-specific adapter paths
ADAPTER_PATHS = {
    "news": "finetune/weights/hf_train_output_news/checkpoint-200",
    "social": "finetune/weights/hf_train_output_social/checkpoint-17502",
    "speech": "finetune/weights/hf_train_output_speech/checkpoint-200",
    "literary": "finetune/weights/hf_train_output_literary/checkpoint-200"
}

# Get input file from command line argument
if len(sys.argv) < 2:
    print("Usage: python translate_pipeline.py <input_file>")
    print("Example: python translate_pipeline.py test2.zh")
    sys.exit(1)

SOURCE_FILE_PATH = sys.argv[1]

# Generate output filenames based on input
base_name = os.path.splitext(os.path.basename(SOURCE_FILE_PATH))[0]
OUTPUT_DIR = f"pipeline_output_{base_name}"
FINAL_OUTPUT_FILE = f"{base_name}.pipeline.en"

print(f"Input file: {SOURCE_FILE_PATH}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Final output: {FINAL_OUTPUT_FILE}")

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Load Base Tokenizer ---
print("Loading base model tokenizer...")
base_tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_NAME,
    padding_side='left'
)
if base_tokenizer.pad_token is None:
    base_tokenizer.pad_token = base_tokenizer.eos_token
print("Tokenizer loaded.")

# --- 2. Translation Function ---
def translate_with_model(text, model, tokenizer):
    """Translate text using a specific model."""
    messages = [
        {"role": "user", "content": f"把下面的文本翻译成English，不要额外解释。\n\n{text}"}
    ]
    
    tokenized = tokenizer.apply_chat_template(
        [messages],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True
    ).to(model.device)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    with torch.no_grad():
        outputs = model.generate(
            tokenized,
            max_new_tokens=128,
            num_beams=1,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    
    prompt_length = len(tokenized[0])
    translation_tokens = outputs[0][prompt_length:]
    translation = tokenizer.decode(translation_tokens, skip_special_tokens=True)
    
    # Remove internal newlines
    translation = translation.replace('\n', ' ').replace('\r', ' ')
    return translation.strip()

# --- 3. Read Source Sentences ---
print(f"Reading source sentences from {SOURCE_FILE_PATH}...")
with open(SOURCE_FILE_PATH, 'r', encoding='utf-8') as f:
    source_sentences = [line.strip() for line in f if line.strip()]
print(f"Loaded {len(source_sentences)} sentences.")

# --- STAGE 1: Translate with all 4 domain models ---
print("\n" + "="*60)
print("STAGE 1: Translating with 4 domain-specific models")
print("="*60)

domain_output_files = {}
for domain, adapter_path in ADAPTER_PATHS.items():
    print(f"\n--- Processing {domain.upper()} domain ---")
    
    try:
        # Load model
        print(f"Loading {domain} model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(
            base_model,
            adapter_path
        )
        model.config.pad_token_id = base_tokenizer.pad_token_id
        model.eval()
        print(f"{domain.capitalize()} model loaded.")
        
        # Translate all sentences
        output_file = os.path.join(OUTPUT_DIR, f"{domain}.en")
        domain_output_files[domain] = output_file
        
        translations = []
        print(f"Translating {len(source_sentences)} sentences with {domain} model...")
        for sentence in tqdm(source_sentences, desc=f"{domain.capitalize()}"):
            translation = translate_with_model(sentence, model, base_tokenizer)
            translations.append(translation)
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            for translation in translations:
                f.write(translation + '\n')
        print(f"Saved {domain} translations to {output_file}")
        
        # Clean up model to free memory
        del model
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        time.sleep(2)
        print(f"{domain.capitalize()} model unloaded.")
        
    except Exception as e:
        print(f"⚠️  ERROR loading or running {domain} model: {e}")
        print(f"Skipping {domain} domain and continuing with others...")
        # Clean up if model was partially loaded
        try:
            if 'model' in locals():
                del model
            if 'base_model' in locals():
                del base_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            time.sleep(1)
        except:
            pass
        continue

print("\n" + "="*60)
print("STAGE 1 COMPLETE: All 4 domain translations saved")
print("="*60)

# Clean up before Stage 2
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
time.sleep(2)

# --- STAGE 2: Refine with Chimera ---
print("\n" + "="*60)
print("STAGE 2: Refining translations with Chimera")
print("="*60)

# Load Chimera model
print("Loading Chimera model for refinement...")
chimera_tokenizer = AutoTokenizer.from_pretrained(
    CHIMERA_MODEL_NAME,
    padding_side='left'
)
if chimera_tokenizer.pad_token is None:
    chimera_tokenizer.pad_token = chimera_tokenizer.eos_token

chimera_model = AutoModelForCausalLM.from_pretrained(
    CHIMERA_MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
chimera_model.config.pad_token_id = chimera_tokenizer.pad_token_id
chimera_model.eval()
print("Chimera model loaded.")

# Load all translation files (only those that were successfully created)
print("\nLoading domain translations...")
domain_translations = {}
for domain in ADAPTER_PATHS.keys():
    if domain in domain_output_files and os.path.exists(domain_output_files[domain]):
        with open(domain_output_files[domain], 'r', encoding='utf-8') as f:
            domain_translations[domain] = [line.strip() for line in f]
        print(f"Loaded {len(domain_translations[domain])} {domain} translations")
    else:
        print(f"⚠️  Skipping {domain} - no translation file found")

if len(domain_translations) == 0:
    print("\n❌ ERROR: No domain translations were successful. Cannot proceed with refinement.")
    sys.exit(1)

print(f"\n✓ Successfully loaded {len(domain_translations)} domain translation files")

# Refinement function
def refine_with_chimera(source_text, translations):
    """Refine multiple translations using Chimera model."""
    # Simplified prompt - just list translations without backticks to save tokens
    translation_list = "\n".join([f"{i+1}. {trans}" for i, trans in enumerate(translations)])
    
    prompt = f"""Given these English translations of "{source_text}", select or combine them into one best translation:

{translation_list}

Best translation:"""

    messages = [{"role": "user", "content": prompt}]
    
    tokenized = chimera_tokenizer.apply_chat_template(
        [messages],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True
    ).to(chimera_model.device)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    with torch.no_grad():
        outputs = chimera_model.generate(
            tokenized,
            max_new_tokens=64,  # Reduced from 128
            num_beams=1,
            do_sample=False,
            eos_token_id=chimera_tokenizer.eos_token_id,
            pad_token_id=chimera_tokenizer.pad_token_id
        )
    
    prompt_length = len(tokenized[0])
    translation_tokens = outputs[0][prompt_length:]
    refined_translation = chimera_tokenizer.decode(translation_tokens, skip_special_tokens=True)
    
    # Remove internal newlines
    refined_translation = refined_translation.replace('\n', ' ').replace('\r', ' ')
    return refined_translation.strip()

# Refine all sentences
print(f"\nRefining {len(source_sentences)} sentences with Chimera...")
final_translations = []
available_domains = list(domain_translations.keys())
print(f"Using translations from: {', '.join(available_domains)}")

for i in tqdm(range(len(source_sentences)), desc="Refining"):
    source_text = source_sentences[i]
    
    # Get all available translations for this sentence
    translations = []
    for domain in available_domains:
        translations.append(domain_translations[domain][i])
    
    # Refine
    refined = refine_with_chimera(source_text, translations)
    final_translations.append(refined)

# Save final output
print(f"\nSaving final refined translations to {FINAL_OUTPUT_FILE}...")
with open(FINAL_OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for translation in final_translations:
        f.write(translation + '\n')

print("\n" + "="*60)
print("PIPELINE COMPLETE!")
print("="*60)
print(f"\nOutput files:")
print(f"  Domain translations:")
for domain, filepath in domain_output_files.items():
    print(f"    - {domain}: {filepath}")
print(f"  Final refined: {FINAL_OUTPUT_FILE}")
print(f"\nTotal sentences processed: {len(source_sentences)}")
