"""
Evaluate translation quality using COMET metric.

Usage:
    python scripts/evaluate_translation.py data/raw/test.zh output.en data/reference/test.en
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders import load_file, validate_files
from src.evaluation.comet_scorer import CometEvaluator


def main(src_file: str, mt_file: str, ref_file: str, output_file: str = None):
    """Evaluate translation quality."""
    
    print("Loading files...")
    print(f"  Source: {src_file}")
    print(f"  MT: {mt_file}")
    print(f"  Reference: {ref_file}")
    
    if not validate_files(src_file, mt_file, ref_file):
        sys.exit(1)
    
    src_lines = load_file(src_file)
    mt_lines = load_file(mt_file)
    ref_lines = load_file(ref_file)
    
    print(f"\nLoaded {len(src_lines)} translation pairs\n")
    
    print("Initializing COMET evaluator...")
    evaluator = CometEvaluator()
    
    print("Scoring translations...")
    results = evaluator.score(src_lines, mt_lines, ref_lines)
    
    print(f"\n{'='*60}")
    print(f"Mean COMET Score: {results['mean_score']:.4f}")
    print(f"{'='*60}")
    
    # Optional: save results
    if output_file:
        with open(output_file, 'w') as f:
            f.write(f"Mean COMET Score: {results['mean_score']:.4f}\n")
            for i, score in enumerate(results['scores'], 1):
                f.write(f"Sentence {i}: {score:.4f}\n")
        print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python scripts/evaluate_translation.py <src> <mt> <ref> [output_file]")
        print("Example: python scripts/evaluate_translation.py data/raw/test.zh output.en data/reference/test.en results.txt")
        sys.exit(1)
    
    src = sys.argv[1]
    mt = sys.argv[2]
    ref = sys.argv[3]
    out = sys.argv[4] if len(sys.argv) > 4 else None
    
    main(src, mt, ref, out)