"""
Stage 2 Only: Refine existing translations with Chimera
Assumes Stage 1 translation files already exist
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
import sys

# --- Configuration ---
CHIMERA_MODEL_NAME = "Hunyuan-MT-Chimera-7B"
TESTING_OUTPUTS_DIR = "testing/outputs"

# Get input file from command line argument
if len(sys.argv) < 2:
    print("Usage: python refine_only.py <source_file>")
    print("Example: python refine_only.py mt/wmttest2022.zh")
    sys.exit(1)

SOURCE_FILE_PATH = sys.argv[1]

# Generate paths based on input
base_name = os.path.splitext(os.path.basename(SOURCE_FILE_PATH))[0]
FINAL_OUTPUT_FILE = f"{base_name}.pipeline.en"

# Domain translation files from testing/outputs folder
DOMAIN_FILES = {
    "literary": os.path.join(TESTING_OUTPUTS_DIR, "wmttest2022.finetuned.lit.en"),
    "social": os.path.join(TESTING_OUTPUTS_DIR, "wmttest2022.finetuned.social.en"),
    "speech": os.path.join(TESTING_OUTPUTS_DIR, "wmttest2022.finetuned.speech.en"),
    "base": os.path.join(TESTING_OUTPUTS_DIR, "wmttest2022.base.en")
}

print(f"Source file: {SOURCE_FILE_PATH}")
print(f"Translation directory: {TESTING_OUTPUTS_DIR}")
print(f"Final output: {FINAL_OUTPUT_FILE}\n")

# --- Load source sentences ---
print(f"Reading source sentences from {SOURCE_FILE_PATH}...")
with open(SOURCE_FILE_PATH, 'r', encoding='utf-8') as f:
    source_sentences = [line.strip() for line in f if line.strip()]
print(f"Loaded {len(source_sentences)} sentences.")

# --- Load Chimera model ---
print("\nLoading Chimera model for refinement...")
chimera_tokenizer = AutoTokenizer.from_pretrained(
    CHIMERA_MODEL_NAME,
    padding_side='left'
)
if chimera_tokenizer.pad_token is None:
    chimera_tokenizer.pad_token = chimera_tokenizer.eos_token

chimera_model = AutoModelForCausalLM.from_pretrained(
    CHIMERA_MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    max_memory={0: "6GB", "cpu": "20GB"}
)
chimera_model.config.pad_token_id = chimera_tokenizer.pad_token_id
chimera_model.eval()
print("Chimera model loaded.")

# --- Load domain translations ---
print("\nLoading domain translations...")
domain_translations = {}
for domain, filepath in DOMAIN_FILES.items():
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            domain_translations[domain] = [line.strip() for line in f]
        print(f"✓ Loaded {len(domain_translations[domain])} {domain} translations")
    else:
        print(f"✗ {domain} translation file not found: {filepath}")

if not domain_translations:
    print("\nError: No translation files found!")
    sys.exit(1)

# Get list of available domains
available_domains = list(domain_translations.keys())
print(f"\nUsing translations from: {', '.join(available_domains)}")

# --- Refinement function ---
def refine_with_chimera(source_text, translations):
    """Refine multiple translations using Chimera model."""
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
            max_new_tokens=64,
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

# --- Refine all sentences ---
print(f"\nRefining {len(source_sentences)} sentences with Chimera...")
final_translations = []

for i in tqdm(range(len(source_sentences)), desc="Refining"):
    source_text = source_sentences[i]
    
    # Get all available translations for this sentence
    translations = []
    for domain in available_domains:
        if i < len(domain_translations[domain]):
            translations.append(domain_translations[domain][i])
    
    # Refine
    refined = refine_with_chimera(source_text, translations)
    final_translations.append(refined)

# --- Save final output ---
print(f"\nSaving final refined translations to {FINAL_OUTPUT_FILE}...")
with open(FINAL_OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for translation in final_translations:
        f.write(translation + '\n')

print("\n" + "="*60)
print("REFINEMENT COMPLETE!")
print("="*60)
print(f"Final output: {FINAL_OUTPUT_FILE}")
print(f"Total sentences refined: {len(final_translations)}")
