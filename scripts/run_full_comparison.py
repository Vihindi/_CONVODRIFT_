#!/usr/bin/env python3
import os
import json
import csv
import argparse
from typing import Dict, Any, List
from metrics_reporter_module import compute_metrics_for_model
from stability_multimodel import run_stability_experiment

def get_persona_files(base_dir: str) -> Dict[str, str]:
    """
    Identifies available persona-labeled dataset files in a specific directory.
    
    Args:
        base_dir (str): Directory where the persona files are stored.

    Returns:
        dict: Mapping of {persona_id: file_path}.
    """
    persona_files = {}
    for pid in ["A", "B", "C", "D", "E"]:
        path = os.path.join(base_dir, f"labeled_dataset_Persona_{pid}.jsonl")
        if os.path.exists(path):
            persona_files[pid] = path
    return persona_files

def main():
    """
    Executes the full comparison pipeline across different LLM preference models.
    Compiles variance metrics (entropy, sensitivity) and stability scores into a single report.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input dataset path for stability testing")
    parser.add_argument("--results_dir", required=True, help="Base directory containing model results")
    parser.add_argument("--out_dir", required=True, help="Output destination for comparison reports")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Note: Models should match the subdirectory names in --results_dir
    models = {
        "GPT-5": os.path.join(args.results_dir, "gpt-5-2025-08-07"),
        "GPT-5-Mini": os.path.join(args.results_dir, "gpt-5-mini-2025-08-07"),
        "Claude-Haiku": os.path.join(args.results_dir, "claude-haiku-4-5-20251001"),
        "Gemini-Flash": os.path.join(args.results_dir, "gemini-2.5-flash")
    }
    
    # 1. Variance Metrics (Entropy & Sensitivity)
    print("\n--- Phase 1: Calculating Variance Metrics (Entropy, Sensitivity) ---")
    final_metrics = {} # {model_friendly_name: {metrics}}

    for name, directory in models.items():
        print(f"Analyzing {name} outputs...")
        files = get_persona_files(directory)
        
        if not files:
            print(f"  Warning: No persona files found for {name} in {directory}.")
            final_metrics[name] = {"sensitivity": 0.0, "entropy_avg": 0.0, "formal_rate_avg": 0.0}
            continue
            
        res = compute_metrics_for_model(files)
        
        # Aggregate statistics from individual personas
        entropies = [p["style_entropy"] for p in res["personas"].values() if "style_entropy" in p]
        formal_rates = [p["formal_rate"] for p in res["personas"].values() if "formal_rate" in p]
        
        avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0
        avg_formal = sum(formal_rates) / len(formal_rates) if formal_rates else 0.0
        sensitivity = res["overall"].get("sensitivity", 0.0)
        
        final_metrics[name] = {
            "sensitivity": sensitivity,
            "entropy_avg": avg_entropy,
            "formal_rate_avg": avg_formal
        }
        print(f"  > Summary: Sensitivity={sensitivity:.4f}, Entropy={avg_entropy:.4f}")

    # 2. Stability Check (New Inference Samples)
    print("\n--- Phase 2: Running Stability Experiment (5 iterations) ---")
    stability_res = run_stability_experiment(args.input, args.out_dir, sample_size=20)
    
    # Map stability results back to friendly names
    # Note: Keys in stability_res must match the internal model IDs used in stability_multimodel
    key_map = {
        "gpt-5-2025-08-07": "GPT-5",
        "gpt-5-mini-2025-08-07": "GPT-5-Mini",
        "claude-haiku-4-5-20251001": "Claude-Haiku",
        "gemini-2.5-flash": "Gemini-Flash"
    }
    
    for raw_id, score in stability_res.items():
        name = key_map.get(raw_id, raw_id)
        if name in final_metrics:
            final_metrics[name]["stability"] = score
        else:
            final_metrics[name] = {"stability": score}

    # 3. Compile Final Report
    print("\n--- Phase 3: Compiling Final Comparison Report ---")
    report_path = os.path.join(args.out_dir, "model_comparison_report.csv")
    headers = [
        "Model", 
        "Stability (Consistency)", 
        "Sensitivity (Pairwise Disagreement)", 
        "Avg Style Entropy", 
        "Avg Formal Rate"
    ]
    
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for name, m in final_metrics.items():
            row = [
                name,
                f"{m.get('stability', 0.0):.4f}",
                f"{m.get('sensitivity', 0.0):.4f}",
                f"{m.get('entropy_avg', 0.0):.4f}",
                f"{m.get('formal_rate_avg', 0.0):.4f}"
            ]
            writer.writerow(row)
            print(f"  {name}: Stability={row[1]}, Sensitivity={row[2]}")

    print(f"\nSuccessfully generated comparison report: {report_path}")

if __name__ == "__main__":
    main()
