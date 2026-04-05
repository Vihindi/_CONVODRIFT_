#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, List

def merge_rater_feedback(main_path: str, rating_path: str, output_path: str):
    """
    Merges feedback data from a rater's JSON file into the main JSONL dataset.

    Args:
        main_path (str): Path to the source JSONL dataset.
        rating_path (str): Path to the JSON file containing rater feedback/responses.
        output_path (str): Destination path for the merged JSONL output.
    """
    if not os.path.exists(main_path):
        print(f"  Error: Main dataset not found: {main_path}")
        return
    if not os.path.exists(rating_path):
        print(f"  Error: Rating feedback file not found: {rating_path}")
        return

    # Load the rater's feedback responses
    try:
        with open(rating_path, 'r', encoding='utf-8') as f:
            rating_data = json.load(f)
        feedback_lookup = rating_data.get("responses", {})
    except (json.JSONDecodeError, IOError) as e:
        print(f"  Error loading feedback from {rating_path}: {e}")
        return

    processed_count = 0
    try:
        with open(main_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    entry = json.loads(line)
                    # Support nested 'data' wrapper
                    target = entry.get("data", entry)
                    conv_id = target.get("conversation_id")
                    
                    if conv_id and conv_id in feedback_lookup:
                        # Extract and merge feedback/rationale
                        rater_entry = feedback_lookup[conv_id]
                        target["feedbacks"] = rater_entry.get("feedbacks", {})
                    
                    outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    processed_count += 1
                    
                except json.JSONDecodeError:
                    print(f"  Warning: Skipping invalid JSON on line {line_num}")
        
        print(f"  Merged {processed_count} records. Saved to: {output_path}")
        
    except IOError as e:
        print(f"  IO Error during merge: {e}")

def main():
    """
    Utility to merge human-validated feedback and rationales back into the 
    main evaluation datasets for specific raters.
    """
    parser = argparse.ArgumentParser(description="Merge rater feedback into main JSONL datasets.")
    parser.add_argument("--root", default=".", help="Project root directory containing 'dataset/'")
    args = parser.parse_args()

    # Base directories relative to root
    base_data_dir = os.path.join(args.root, "dataset")
    pre_schema_main = os.path.join(base_data_dir, "pre_schema", "main")
    pre_schema_rating = os.path.join(base_data_dir, "pre_schema", "rating")
    output_dir = os.path.join(base_data_dir, "evaluated_data")

    os.makedirs(output_dir, exist_ok=True)

    # Known raters in the project
    raters = ["chandi", "dilanka", "gayani", "pamoda"]

    for rater in raters:
        print(f"Processing rater: {rater}")
        
        main_file = f"dataset_{rater}_final.jsonl"
        rating_file = f"refined_direction_label_{rater}_fixed.json"
        
        main_path = os.path.join(pre_schema_main, main_file)
        rating_path = os.path.join(pre_schema_rating, rating_file)
        output_path = os.path.join(output_dir, main_file)
        
        merge_rater_feedback(main_path, rating_path, output_path)

if __name__ == "__main__":
    main()
