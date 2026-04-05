#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import os
import random
import requests
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional

# Ensure dependencies from sibling script are accessible
try:
    from run_multimodel_generation import (
        call_openai, call_anthropic, load_system_prompt, 
        format_user_content, retry_with_backoff
    )
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from run_multimodel_generation import (
        call_openai, call_anthropic, load_system_prompt, 
        format_user_content, retry_with_backoff
    )

def call_gemini(api_key: str, model_id: str, system_prompt: str, user_content: str) -> Optional[str]:
    """Sends a generation request to Google Gemini API."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": user_content}]}],
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 50}
    }

    def _req():
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        candidates = data.get('candidates', [])
        if candidates:
            parts = candidates[0].get('content', {}).get('parts', [])
            if parts:
                return parts[0].get('text', '').strip()
        return None

    return retry_with_backoff(_req)

def run_single_stability_check(func, entry: Dict[str, Any], system_prompt: str, runs: int = 5) -> List[Optional[str]]:
    """Runs a single prompt multiple times and collects the labels to measure consistency."""
    user_content = format_user_content(entry)
    predictions = []
    
    for _ in range(runs):
        pred = func(system_prompt, user_content)
        label = None
        if pred:
            # Heuristic parsing
            if '1' in pred and '2' not in pred: label = '1'
            elif '2' in pred and '1' not in pred: label = '2'
            elif pred.strip() == '1': label = '1'
            elif pred.strip() == '2': label = '2'
        predictions.append(label)
        time.sleep(0.5) 
        
    return predictions

def run_stability_experiment(input_path: str, output_dir: str, sample_size: int = 20) -> Dict[str, float]:
    """
    Measures the response stability (consistency) of multiple LLM models across a sampled dataset.
    
    Args:
        input_path (str): Path to the source JSONL dataset.
        output_dir (str): Directory to save detailed stability results.
        sample_size (int): Number of random samples to test per persona.

    Returns:
        dict: Mapping of model_id to its average stability score [0.0, 1.0].
    """
    # Note: Keys and funcs should be configured based on available environment variables
    # For extraction purposes, we use placeholder keys or expect them to be set
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    
    models_to_test = {
        "gpt-5-2025-08-07": lambda sp, uc: call_openai("gpt-5-2025-08-07", sp, uc),
        "gpt-5-mini-2025-08-07": lambda sp, uc: call_openai("gpt-5-mini-2025-08-07", sp, uc),
        "claude-haiku-4-5-20251001": lambda sp, uc: call_anthropic("claude-haiku-4-5-20251001", sp, uc)
    }
    if gemini_key:
        models_to_test["gemini-2.5-flash"] = lambda sp, uc: call_gemini(gemini_key, "gemini-2.5-flash", sp, uc)

    dataset = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    
    random.seed(42)
    sampled_indices = random.sample(range(len(dataset)), min(sample_size, len(dataset)))
    sampled_dataset = [dataset[i] for i in sampled_indices]
    
    results_summary = {} 
    # Personas A-E
    personas = [{"id": f"Persona_{c}", "file": f"persona_{c}.txt"} for c in "ABCDE"]

    for model_id, func in models_to_test.items():
        print(f"\n--- Testing Stability: {model_id} ---")
        model_records = []
        
        for persona in personas:
            pid = persona['id']
            # Prompts dir is assumed to be sibling to scripts
            prompts_dir = os.path.join(os.path.dirname(os.path.dirname(input_path)), "persona_prompts")
            try:
                sys_prompt = load_system_prompt(prompts_dir, persona['file'])
            except FileNotFoundError:
                continue
            
            for entry in sampled_dataset:
                preds = run_single_stability_check(func, entry, sys_prompt, runs=5)
                
                valid_preds = [p for p in preds if p]
                score = 0.0
                if valid_preds:
                     counts = Counter(valid_preds)
                     majority_count = counts.most_common(1)[0][1]
                     score = majority_count / len(valid_preds) 
                
                model_records.append({
                    "convo_ID": entry.get("convo_ID") or entry.get("conversation_id"),
                    "persona": pid,
                    "predictions": preds,
                    "stability_score": score
                })
        
        if model_records:
            avg_stability = sum(r['stability_score'] for r in model_records) / len(model_records)
        else:
            avg_stability = 0.0
        
        results_summary[model_id] = avg_stability
        print(f"  > Overall Stability: {avg_stability:.2%}")
        
        # Save individual model results
        out_file = os.path.join(output_dir, f"stability_{model_id}.json")
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(model_records, f, indent=2, ensure_ascii=False)

    return results_summary

def main():
    """CLI entry point for running stability checks standalone."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL dataset")
    parser.add_argument("--output_dir", required=True, help="Output directory for results")
    parser.add_argument("--sample_size", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    run_stability_experiment(args.input, args.output_dir, args.sample_size)

if __name__ == "__main__":
    main()
