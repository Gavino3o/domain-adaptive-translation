from comet import download_model, load_from_checkpoint
import sys

# --- Load data from files ---
def load_file(filepath):
    """Load lines from a file, stripping whitespace."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

# Check command line arguments
if len(sys.argv) != 4:
    print("Usage: python score.py <src_file> <mt_file> <ref_file>")
    print("Example: python score.py test.zh test.en reference.en")
    sys.exit(1)

src_file = sys.argv[1]
mt_file = sys.argv[2]
ref_file = sys.argv[3]

print(f"Loading files...")
print(f"  Source: {src_file}")
print(f"  MT: {mt_file}")
print(f"  Reference: {ref_file}")

# Load lines from files
src_lines = load_file(src_file)
mt_lines = load_file(mt_file)
ref_lines = load_file(ref_file)

# Check that all files have the same number of lines
if not (len(src_lines) == len(mt_lines) == len(ref_lines)):
    print(f"\nError: Files have different number of lines!")
    print(f"  Source: {len(src_lines)} lines")
    print(f"  MT: {len(mt_lines)} lines")
    print(f"  Reference: {len(ref_lines)} lines")
    sys.exit(1)

print(f"  Loaded {len(src_lines)} translation pairs\n")

# --- Prepare your data ---
# It needs to be a list of dictionaries
data = [
    {
        "src": src,
        "mt": mt,
        "ref": ref
    }
    for src, mt, ref in zip(src_lines, mt_lines, ref_lines)
]

# --- 1. Reference-Based Scoring (Recommended for your project) ---
# Option 1: Use download_model (will use cache if already downloaded)
model_path = download_model("Unbabel/wmt22-comet-da")
model = load_from_checkpoint(model_path)

# Calculate scores
output = model.predict(data, batch_size=8, gpus=1)
ref_based_scores = output.scores

print("\nReference-Based COMET Scores:")
print("="*60)
for i, score in enumerate(ref_based_scores):
    print(f"  Line {i+1}: {score:.4f}")

# Calculate and print average score
avg_score = sum(ref_based_scores) / len(ref_based_scores)
print("="*60)
print(f"Average Score: {avg_score:.4f}")
print(f"Total Lines: {len(ref_based_scores)}")
print("="*60)


# --- 2. Reference-Free Scoring (For quick checks) ---
# This section is commented out since the reference-free model needs authentication
# You can use just the reference-based scoring above for your project
"""
# Option A: Download once (run this first time only)
# model_path_qe = download_model("Unbabel/wmt22-cometkiwi-da")
# print(f"QE Model downloaded to: {model_path_qe}")

# Option B: Load from local path (use after downloading)
# Replace with the actual path printed from Option A
model_path_qe = "path/to/downloaded/qe/model"  # Update this path
model_qe = load_from_checkpoint(model_path_qe)

# Calculate scores (notice 'ref' is not used)
ref_free_scores, _ = model_qe.predict(data, batch_size=8, gpus=1)
print("Reference-Free (QE) Scores:", ref_free_scores)
"""