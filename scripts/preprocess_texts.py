"""
Preprocess raw Nietzsche texts by removing Project Gutenberg boilerplate.

This script reads text files from content/nietzsche/raw/ and extracts only
the core content between the START and END markers, saving cleaned versions
to content/nietzsche/preprocessed/.
"""

import os
import re
from pathlib import Path


def extract_content(text):
    """
    Extract content between Project Gutenberg markers.
    
    Args:
        text (str): Full text content from raw file
        
    Returns:
        str: Cleaned content without Project Gutenberg boilerplate
    """
    lines = text.split('\n')
    
    start_idx = None
    end_idx = None
    
    # Find the START marker
    for i, line in enumerate(lines):
        if 'start of the project gutenberg' in line.lower() or 'start of project gutenberg' in line.lower():
            start_idx = i + 1  # Start from the line after the marker
            break
    
    # Find the END marker
    for i, line in enumerate(lines):
        if 'end of the project gutenberg' in line.lower() or 'end of project gutenberg' in line.lower():
            end_idx = i  # Stop before this line
            break
    
    if start_idx is None:
        raise ValueError("START marker not found")
    
    if end_idx is None:
        raise ValueError("END marker not found")
    
    # Extract content between markers
    content_lines = lines[start_idx:end_idx]
    
    # Join back into text, preserving original formatting
    cleaned_content = '\n'.join(content_lines)
    
    # Strip leading/trailing whitespace from the entire content
    cleaned_content = cleaned_content.strip()

    # Remove extra newlines after numbers and roman numerals while preserving them
    cleaned_content = re.sub(r'(?<=\d)\s+', r' ', cleaned_content)
    cleaned_content = re.sub(r'(?<=\d\.)\s+', r' ', cleaned_content)
    cleaned_content = re.sub(r'(?<=[IVX])\n\s+', r' ', cleaned_content) 
    cleaned_content = re.sub(r'(?<=[IVX]\.)\n\s+', r' ', cleaned_content)

    # Remove underscores
    cleaned_content = cleaned_content.replace('_', '')
    
    return cleaned_content


def preprocess_file(input_path, output_path):
    """
    Process a single file, extracting clean content.
    
    Args:
        input_path (Path): Path to raw text file
        output_path (Path): Path to save preprocessed text
    """
    # Read the raw file
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    # Extract clean content
    try:
        cleaned_text = extract_content(raw_text)
        
        # Save to output file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        
        return True, None
    except Exception as e:
        return False, str(e)


def main():
    """Main preprocessing function."""
    # Set up paths
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / 'content' / 'nietzsche' / 'raw'
    preprocessed_dir = project_root / 'content' / 'nietzsche' / 'preprocessed'
    
    # Ensure preprocessed directory exists
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all .txt files from raw directory
    raw_files = sorted(raw_dir.glob('*.txt'))
    
    if not raw_files:
        print(f"No .txt files found in {raw_dir}")
        return
    
    print(f"Found {len(raw_files)} files to preprocess\n")
    
    # Process each file
    success_count = 0
    error_count = 0
    errors = []
    
    for raw_file in raw_files:
        output_file = preprocessed_dir / raw_file.name
        
        print(f"Processing: {raw_file.name}...", end=' ')
        
        success, error_msg = preprocess_file(raw_file, output_file)
        
        if success:
            print("[OK]")
            success_count += 1
        else:
            print(f"[ERROR]: {error_msg}")
            error_count += 1
            errors.append((raw_file.name, error_msg))
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Successfully processed: {success_count} files")
    print(f"Errors: {error_count} files")
    
    if errors:
        print(f"\nFiles with errors:")
        for filename, error_msg in errors:
            print(f"  - {filename}: {error_msg}")


if __name__ == '__main__':
    main()

