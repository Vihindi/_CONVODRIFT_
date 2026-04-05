#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, List

def strip_evaluation_keys(data: Dict[str, Any], keys: List[str]):
    """
    Recursively removes specified evaluation keys from a data object 
    to prepare it for a fresh evaluation round.
    """
    # Remove from top level
    for key in keys:
        data.pop(key, None)
    
    # Remove from nested 'data' if present
    if "data" in data and isinstance(data["data"], dict):
        for key in keys:
            data["data"].pop(key, None)

def process_dataset(source_file: str, dest_file: str):
    """
    Reads a labeled JSONL dataset, strips existing evaluation metrics/labels, 
    and saves the cleaned records to a new 'pending evaluation' destination.

    Args:
        source_file (str): Path to the source JSONL file.
        dest_file (str): Path to the output JSONL file.
    """
    keys_to_strip = [
        "ratings",
        "refined_drift_label",
        "refined_direction_label",
        "feedbacks",
        "evaluation_status",
        "timestamp"
    ]
    
    # Ensure destination directory exists
    dest_dir = os.path.dirname(dest_file)
    if dest_dir and not os.path.exists(dest_dir):
        try:
            os.makedirs(dest_dir, exist_ok=True)
            print(f"Created directory: {dest_dir}")
        except OSError as e:
            print(f"Error creating directory {dest_dir}: {e}")
            return

    if not os.path.exists(source_file):
        print(f"Error: Source file not found: {source_file}")
        return

    print(f"Preparing evaluation data...\nSource: {source_file}\nTarget: {dest_file}")

    processed_count = 0
    try:
        with open(source_file, 'r', encoding='utf-8') as infile, \
             open(dest_file, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    strip_evaluation_keys(data, keys_to_strip)
                    
                    outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                    processed_count += 1
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON Error on line {line_num}: {e}")
        
    except IOError as e:
        print(f"IO Error during processing: {e}")

    print(f"Successfully processed {processed_count} records.")

def main():
    """
    Utility to strip ground truth or prior LLM evaluation labels from a dataset,
    preparing it for a new evaluation pass.
    """
    parser = argparse.ArgumentParser(description="Clean datasets for re-evaluation by stripping metrics.")
    parser.add_argument("--source", required=True, help="Path to source labeled JSONL file")
    parser.add_argument("--dest", required=True, help="Path to destination pending JSONL file")
    args = parser.parse_args()

    process_dataset(args.source, args.dest)

if __name__ == "__main__":
    main()
