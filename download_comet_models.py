from comet import download_model

# Download the reference-based model
print("Downloading reference-based model...")
model_path = download_model("Unbabel/wmt22-comet-da")
print(f"✓ Reference-based model downloaded to: {model_path}")

print("\n" + "="*60)
print("Copy these paths into your score.py file:")
print(f"model_path = r'{model_path}'")
print("="*60)
