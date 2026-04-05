#!/usr/bin/env python3
import argparse
import glob
import json
import os
from typing import Any, Dict

def create_convo_index(input_target: str, output_dir: str):
    """
    Creates a conversation index mapping from a target JSONL file or directory.
    The index maps conversation_id to the full conversation data object.

    Args:
        input_target (str): Path to a JSONL file or directory containing JSONL files.
        output_dir (str): Directory where the generated JSON index files will be saved.
    """
    # Identify files to process
    if os.path.isfile(input_target):
        files_to_process = [input_target]
    elif os.path.isdir(input_target):
        files_to_process = glob.glob(os.path.join(input_target, "*.jsonl"))
    else:
        print(f"Error: Invalid input path: {input_target}")
        return

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for file_path in files_to_process:
        file_name = os.path.basename(file_path)
        print(f"Indexing conversations in: {file_name}")
        
        base_name = os.path.splitext(file_name)[0]
        output_file = os.path.join(output_dir, f"{base_name}.json")
        convo_map: Dict[str, Any] = {}

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        # Extract the data object and its unique identifier
                        target = entry.get("data", entry)
                        conv_id = target.get("conversation_id") or target.get("convo_ID")
                        
                        if conv_id:
                            convo_map[str(conv_id)] = target
                    except json.JSONDecodeError:
                        print(f"  Warning: JSON Decode Error on line {line_num} of {file_name}")
            
            # Persist index to JSON
            with open(output_file, 'w', encoding='utf-8') as out_f:
                json.dump(convo_map, out_f, indent=2, ensure_ascii=False)
            
            print(f"  Successfully created index: {output_file} ({len(convo_map)} records)")
            
        except Exception as e:
            print(f"  Critical error processing {file_name}: {e}")

def main():
    """
    Utility to crawl JSONL datasets and create an ID-to-content index file.
    This index is used by the JudgeLM evaluation pipeline for fast lookup 
    of few-shot conversation examples.
    """
    parser = argparse.ArgumentParser(description="Create conversation ID index for JudgeLM evaluation.")
    parser.add_argument("--input", required=True, help="Input JSONL file or directory")
    parser.add_argument("--output", required=True, help="Output directory for JSON indexes")
    args = parser.parse_args()

    create_convo_index(args.input, args.output)

if __name__ == "__main__":
    main()
