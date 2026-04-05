#!/usr/bin/env python3
import argparse
import glob
import json
import os
from typing import Any, Dict, List

def create_domain_index(input_target: str, output_dir: str):
    """
    Creates a domain-to-conversation mapping index.
    The index maps each domain (e.g., 'Health', 'Finance') to a list of its 
    associated conversation IDs.

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
        print(f"Indexing domains in: {file_name}")
        
        base_name = os.path.splitext(file_name)[0]
        output_file = os.path.join(output_dir, f"{base_name}.json")
        domain_map: Dict[str, List[str]] = {}

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        # Extract domain and conversation ID from the record
                        target = entry.get("data", entry)
                        domain = target.get("domain") or target.get("conversation_genre")
                        conv_id = target.get("conversation_id") or target.get("convo_ID")
                        
                        if domain and conv_id:
                            domain_str = str(domain)
                            if domain_str not in domain_map:
                                domain_map[domain_str] = []
                            domain_map[domain_str].append(str(conv_id))
                    except json.JSONDecodeError:
                        print(f"  Warning: JSON Decode Error on line {line_num} of {file_name}")
            
            # Persist domain mapping to JSON
            with open(output_file, 'w', encoding='utf-8') as out_f:
                json.dump(domain_map, out_f, indent=2, ensure_ascii=False)
            
            print(f"  Successfully created domain index: {output_file} ({len(domain_map)} domains)")
            
        except Exception as e:
            print(f"  Critical error processing {file_name}: {e}")

def main():
    """
    Utility to crawl JSONL datasets and create a mapping of conversation IDs 
    grouped by their respective domains. This index supports domain-specific 
    sampling for the JudgeLM evaluation pipeline.
    """
    parser = argparse.ArgumentParser(description="Create domain-to-ID mapping index for JudgeLM evaluation.")
    parser.add_argument("--input", required=True, help="Input JSONL file or directory")
    parser.add_argument("--output", required=True, help="Output directory for domain JSON indexes")
    args = parser.parse_args()

    create_domain_index(args.input, args.output)

if __name__ == "__main__":
    main()
