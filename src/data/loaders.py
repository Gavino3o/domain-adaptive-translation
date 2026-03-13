from typing import List

def load_file(filepath: str) -> List[str]:
    """Load lines from a file, stripping whitespace."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def save_file(filepath: str, lines: List[str]) -> None:
    """Save lines to a file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')

def validate_files(*filepaths: str) -> bool:
    """Check that all files have the same number of lines."""
    line_counts = [len(load_file(f)) for f in filepaths]
    if len(set(line_counts)) > 1:
        print("Error: Files have different number of lines!")
        for f, count in zip(filepaths, line_counts):
            print(f"  {f}: {count} lines")
        return False
    return True