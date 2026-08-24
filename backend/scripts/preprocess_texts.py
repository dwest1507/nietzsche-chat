"""
Preprocess raw Nietzsche texts by removing Project Gutenberg boilerplate.

Run from anywhere:
    python backend/scripts/preprocess_texts.py

Reads text files from content/nietzsche/raw/ and extracts only the core
content between the START and END markers, saving cleaned versions to
content/nietzsche/preprocessed/.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = REPO_ROOT / "content" / "nietzsche" / "raw"
PREPROCESSED_DIR = REPO_ROOT / "content" / "nietzsche" / "preprocessed"


def extract_content(text: str) -> str:
    """Extract content between Project Gutenberg markers."""
    lines = text.split("\n")

    start_idx = None
    end_idx = None

    # Find the START marker
    for i, line in enumerate(lines):
        lowered = line.lower()
        if "start of the project gutenberg" in lowered or "start of project gutenberg" in lowered:
            start_idx = i + 1  # Start from the line after the marker
            break

    # Find the END marker
    for i, line in enumerate(lines):
        lowered = line.lower()
        if "end of the project gutenberg" in lowered or "end of project gutenberg" in lowered:
            end_idx = i  # Stop before this line
            break

    if start_idx is None:
        raise ValueError("START marker not found")

    if end_idx is None:
        raise ValueError("END marker not found")

    cleaned_content = "\n".join(lines[start_idx:end_idx]).strip()

    # Remove extra newlines after numbers and roman numerals while preserving them
    cleaned_content = re.sub(r"(?<=\d)\s+", r" ", cleaned_content)
    cleaned_content = re.sub(r"(?<=\d\.)\s+", r" ", cleaned_content)
    cleaned_content = re.sub(r"(?<=[IVX])\n\s+", r" ", cleaned_content)
    cleaned_content = re.sub(r"(?<=[IVX]\.)\n\s+", r" ", cleaned_content)

    # Remove underscores (Gutenberg italics markup)
    cleaned_content = cleaned_content.replace("_", "")

    return cleaned_content


def preprocess_file(input_path: Path, output_path: Path) -> tuple[bool, str | None]:
    """Process a single file, extracting clean content."""
    raw_text = input_path.read_text(encoding="utf-8")
    try:
        output_path.write_text(extract_content(raw_text), encoding="utf-8")
        return True, None
    except Exception as e:  # noqa: BLE001 — report per-file errors in the summary
        return False, str(e)


def main() -> None:
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(RAW_DIR.glob("*.txt"))
    if not raw_files:
        print(f"No .txt files found in {RAW_DIR}")
        return

    print(f"Found {len(raw_files)} files to preprocess\n")

    success_count = 0
    errors: list[tuple[str, str]] = []

    for raw_file in raw_files:
        print(f"Processing: {raw_file.name}...", end=" ")
        success, error_msg = preprocess_file(raw_file, PREPROCESSED_DIR / raw_file.name)
        if success:
            print("[OK]")
            success_count += 1
        else:
            print(f"[ERROR]: {error_msg}")
            errors.append((raw_file.name, error_msg or "unknown error"))

    print(f"\n{'=' * 60}")
    print("Processing complete!")
    print(f"Successfully processed: {success_count} files")
    print(f"Errors: {len(errors)} files")
    for filename, error_msg in errors:
        print(f"  - {filename}: {error_msg}")


if __name__ == "__main__":
    main()
