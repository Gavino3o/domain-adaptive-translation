import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from tqdm import tqdm

# --- Configuration ---
MODEL_NAME = "Hunyuan-MT-7B"
SOURCE_FILE_PATH = "test.zh"
OUTPUT_FILE_PATH = "test.en"
BATCH_SIZE = 1 # Adjust based on your GPU memory. Lower if you get OutOfMemory errors.

# --- 1. Load Model and Tokenizer ---
print("Loading model and tokenizer...")
# Set padding_side to 'left' to fix the warning and ensure correct batch generation for decoder-only models.
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    padding_side='left'
)
# The model does not have a default pad token, so we set it to the end-of-sentence token.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
# Inform the model of the new pad token id.
model.config.pad_token_id = tokenizer.pad_token_id

model.eval()
print("Model loaded successfully.")

# --- 2. Load and Batch the Source Sentences ---
print(f"Reading source sentences from {SOURCE_FILE_PATH}...")
with open(SOURCE_FILE_PATH, 'r', encoding='utf-8') as f:
    source_sentences = [line.strip() for line in f if line.strip()]

batches = [source_sentences[i:i + BATCH_SIZE] for i in range(0, len(source_sentences), BATCH_SIZE)]
print(f"Created {len(batches)} batches with a size of {BATCH_SIZE}.")

# --- 3. Generate Translations (Using Official Method) ---
all_translations = []
start_time = time.time()

print("Starting translation...")
# Wrap the main loop with tqdm for a progress bar
for batch_sentences in tqdm(batches, desc="Translating Batches"):
    
    # Create the 'messages' structure for each sentence in the batch
    # This is the core of the official chat/instruction format
    batch_messages = []
    for sentence in batch_sentences:
        messages = [
            {"role": "user", "content": f"把下面的文本翻译成English，不要额外解释。\n\n{sentence}"}
        ]
        batch_messages.append(messages)

    # Tokenize the batch of conversations using the chat template
    # `padding=True` is crucial for handling batches of different lengths
    tokenized_chats = tokenizer.apply_chat_template(
        batch_messages,
        tokenize=True,
        add_generation_prompt=True, # This adds the '<|im_start|>assistant' tokens
        return_tensors="pt",
        padding=True 
    ).to(model.device)

    # Generate translations using the recommended inference parameters
    outputs = model.generate(
        tokenized_chats,
        max_new_tokens=128,
        # --- Official Recommended Parameters ---
        do_sample=True,      # Enable sampling
        top_k=20,
        top_p=0.6,
        temperature=0.7,
        repetition_penalty=1.05,
        # -------------------------------------
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Decode the outputs, making sure to skip the prompt part
    # We find the length of the original prompt tokens and slice the output
    prompt_lengths = [len(chat) for chat in tokenized_chats]
    decoded_outputs = []
    for j in range(len(outputs)):
        translation_tokens = outputs[j][prompt_lengths[j]:]
        translation_text = tokenizer.decode(translation_tokens, skip_special_tokens=True)
        decoded_outputs.append(translation_text)

    all_translations.extend(decoded_outputs)
    
    # We no longer need the manual print statement here as tqdm handles it
    # print(f"  - Translated batch {i+1}/{len(batches)}")

end_time = time.time()
print(f"Translation finished in {end_time - start_time:.2f} seconds.")

# --- 4. Save the Translations ---
print(f"Saving translations to {OUTPUT_FILE_PATH}...")
with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
    for translation in all_translations:
        f.write(translation.strip() + '\n')

print("Done!")