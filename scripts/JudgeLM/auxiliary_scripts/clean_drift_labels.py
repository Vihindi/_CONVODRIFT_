#!/usr/bin/env python3
import argparse
import glob
import json
import os
from typing import Any, Dict, List, Tuple, Optional

def clean_drift_list(drift_label: Any) -> Tuple[bool, str]:
    """
    Validates and cleans a drift label list.
    
    The expected length of a drift label list is 6. If the length is 7 and 
    index 0 is None, the extra None is removed.
    
    Args:
        drift_label: The drift label list to validate/clean.

    Returns:
        tuple: (is_modified, reason_if_error)
            is_modified (bool): True if the list was corrected.
            reason_if_error (str): Description of any validation failure.
    """
    if drift_label is None:
        return False, "refined_drift_label is missing/null"
    if not isinstance(drift_label, list):
        return False, "refined_drift_label is not a list"
    
    modified = False
    if len(drift_label) == 7:
        if drift_label[0] is None:
            drift_label.pop(0)
            modified = True
        else:
            return False, f"Length is 7 but index 0 is not null (Value: {drift_label[0]})"
    
    if len(drift_label) != 6:
        return modified, f"Unexpected length: {len(drift_label)} (Expected 6)"
        
    return modified, ""

def process_evaluated_data(data_dir: str, issues_list: List[str]):
    """
    Scans and corrects drift labels in JSONL evaluated data files.

    Args:
        data_dir (str): Path to the directory containing .jsonl evaluated files.
        issues_list (list): List to accumulate error/audit messages.
    """
    print(f"Scanning evaluated data in: {data_dir}")
    jsonl_files = glob.glob(os.path.join(data_dir, "*.jsonl"))
    
    for file_path in jsonl_files:
        file_name = os.path.basename(file_path)
        modified_lines = []
        file_modified = False
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                
                # Support both nested 'data' wrappers and flat structures
                target = entry.get("data", entry)
                convo_id = target.get("conversation_id") or target.get("convo_ID", "unknown")
                drift_label = target.get("refined_drift_label")
                
                modified, issue = clean_drift_list(drift_label)
                if issue:
                    issues_list.append(f"File: {file_name} | CID: {convo_id} | {issue}")
                
                if modified:
                    file_modified = True
                    modified_lines.append(json.dumps(entry, ensure_ascii=False) + "\n")
                else:
                    modified_lines.append(line)
                    
            except json.JSONDecodeError:
                issues_list.append(f"File: {file_name} | JSON Decode Error")
                modified_lines.append(line)
        
        if file_modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(modified_lines)
            print(f"  [FIXED] {file_name}")

def process_convo_index(index_dir: str, issues_list: List[str]):
    """
    Scans and corrects drift labels in conversation index JSON files.

    Args:
        index_dir (str): Path to the directory containing conversation index .json files.
        issues_list (list): List to accumulate error/audit messages.
    """
    print(f"Scanning conversation index in: {index_dir}")
    json_files = glob.glob(os.path.join(index_dir, "*.json"))
    
    for file_path in json_files:
        file_name = os.path.basename(file_path)
        file_modified = False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for convo_id, content in data.items():
                drift_label = content.get("refined_drift_label")
                modified, issue = clean_drift_list(drift_label)
                
                if issue:
                    issues_list.append(f"File: {file_name} | CID: {convo_id} | {issue}")
                if modified:
                    file_modified = True
            
            if file_modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"  [FIXED] {file_name}")
                
        except (json.JSONDecodeError, IOError) as e:
             issues_list.append(f"File: {file_name} | Critical Error: {e}")

def main():
    """
    Utility to audit and fix off-by-one errors in refined drift label lists
    across the JudgeLM dataset. 
    
    This script is used to ensure all refined_drift_label lists have exactly 
    6 entries (matching the conversation pairs).
    """
    parser = argparse.ArgumentParser(description="Audit and fix drift label list lengths.")
    parser.add_argument("--root", default=".", help="Project root directory containing 'dataset/'")
    args = parser.parse_args()

    # Resolve paths relative to the provided root
    evaluated_dir = os.path.join(args.root, "dataset", "evaluated_data")
    index_dir = os.path.join(args.root, "dataset", "indexed_data", "convo_index")
    log_file = os.path.join(args.root, "drift_label_audit.txt")

    issues = []
    
    if os.path.exists(evaluated_dir):
        process_evaluated_data(evaluated_dir, issues)
    else:
        print(f"Warning: Evaluated data directory not found at {evaluated_dir}")

    if os.path.exists(index_dir):
        process_convo_index(index_dir, issues)
    else:
        print(f"Warning: Conversation index directory not found at {index_dir}")
    
    with open(log_file, 'w', encoding='utf-8') as f:
        if not issues:
            f.write("Status: Clean. No label length issues detected.\n")
        else:
            f.write(f"Audit Results: {len(issues)} issues found\n" + "="*30 + "\n")
            for issue in issues:
                f.write(issue + "\n")
    
    print(f"Audit complete. Results summarized in {log_file}")

if __name__ == "__main__":
    main()
