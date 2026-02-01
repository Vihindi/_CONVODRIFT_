
import json
import time
import os
import random
import concurrent.futures
from collections import Counter
import requests
try:
    from run_multimodel_generation import call_openai, call_anthropic, load_system_prompt, format_user_content, PERSONAS, retry_with_backoff
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from run_multimodel_generation import call_openai, call_anthropic, load_system_prompt, format_user_content, PERSONAS, retry_with_backoff

# Gemini Configuration
GEMINI_API_KEY = ""
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

def call_gemini(system_prompt, user_content):
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": user_content}]}],
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1000}
    }

    def _req():
        response = requests.post(GEMINI_URL, headers=headers, json=payload, timeout=30)
        if response.status_code != 200:
             raise Exception(f"Gemini Error {response.status_code}: {response.text}")
        data = response.json()
        if 'candidates' in data and data['candidates']:
            parts = data['candidates'][0].get('content', {}).get('parts', [])
            if parts:
                return parts[0]['text'].strip()
        return None

    return retry_with_backoff(_req)

# Define Models to Test
MODELS_TO_TEST = {
    "gemini-2.5-flash": { "func": call_gemini },
    "claude-haiku-4-5-20251001": { "func": lambda sp, uc: call_anthropic("claude-haiku-4-5-20251001", sp, uc) },
    "gpt-5-mini-2025-08-07": { "func": lambda sp, uc: call_openai("gpt-5-mini-2025-08-07", sp, uc) },
    "gpt-5-2025-08-07": { "func": lambda sp, uc: call_openai("gpt-5-2025-08-07", sp, uc) }
}

def run_single_stability_check(func, entry, system_prompt, runs=5):
    user_content = format_user_content(entry)
    predictions = []
    
    for i in range(runs):
        pred = func(system_prompt, user_content)
        lbl = None
        if pred:
            if '1' in pred and '2' not in pred: lbl = '1'
            elif '2' in pred and '1' not in pred: lbl = '2'
            elif pred.strip() == '1': lbl = '1'
            elif pred.strip() == '2': lbl = '2'
        predictions.append(lbl)
        time.sleep(0.5) 
        
    return predictions

def run_stability_experiment(input_path, output_dir, sample_size=20):
    dataset = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    
    random.seed(42)
    real_sample_size = min(sample_size, len(dataset))
    sampled_dataset = random.sample(dataset, real_sample_size)
    print(f"Running stability check on {real_sample_size} samples (5 iters) for {list(MODELS_TO_TEST.keys())}")

    results_summary = {} 

    for model_name, config in MODELS_TO_TEST.items():
        print(f"\n--- Testing Stability: {model_name} ---")
        model_results = []
        func = config["func"]
        
        for persona in PERSONAS:
            pid = persona['id']
            # print(f"  Persona {pid}...", end="\r")
            
            # Note: This relies on person prompts loading correctly from run_multimodel_generation paths
            sys_prompt = load_system_prompt(persona['file'])
            
            persona_scores = []
            
            for idx, entry in enumerate(sampled_dataset):
                preds = run_single_stability_check(func, entry, sys_prompt, runs=5)
                
                valid = [p for p in preds if p]
                score = 0.0
                if valid:
                     counts = Counter(valid)
                     majority = counts.most_common(1)[0][1]
                     score = majority / len(valid) 
                
                persona_scores.append({
                    "convo_ID": entry.get("convo_ID"),
                    "persona": pid,
                    "predictions": preds,
                    "stability_score": score
                })
            
            if persona_scores:
                avg = sum(p['stability_score'] for p in persona_scores) / len(persona_scores)
            else:
                avg = 0
            model_results.extend(persona_scores)

        if model_results:
            model_avg = sum(p['stability_score'] for p in model_results) / len(model_results)
        else:
            model_avg = 0
        
        results_summary[model_name] = model_avg
        print(f"  > Overall Stability: {model_avg:.2%}")
        
        out_file = os.path.join(output_dir, f"stability_{model_name}.json")
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(model_results, f, indent=2)

    return results_summary
