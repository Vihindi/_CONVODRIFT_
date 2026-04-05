#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import os
import requests
import sys
import time
from typing import Any, Dict, List, Optional
from openai import OpenAI
from dotenv import load_dotenv

try:
    load_dotenv()
except ImportError:
    pass

# Configuration Constants
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()

# Initialize OpenAI client if key is available
openai_client = None
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

def load_system_prompt(prompt_dir: str, filename: str) -> str:
    """Loads a persona system prompt from a text file."""
    path = os.path.join(prompt_dir, filename)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def format_user_content(entry: Dict[str, Any]) -> str:
    """Formats a dataset entry into a prompt for the preference judge."""
    return (
        f"Prompt: {entry.get('prompt', '')}\n\n"
        f"Response A:\n{entry.get('response_A', '')}\n\n"
        f"Response B:\n{entry.get('response_B', '')}\n\n"
        "Evaluate the responses based on your persona. "
        "Which response do you prefer? Respond with '1' for Response A or '2' for Response B."
    )

def retry_with_backoff(func, retries: int = 5, initial_backoff: int = 5):
    """Executes a function with exponential backoff on failure."""
    for i in range(retries):
        try:
            return func()
        except Exception as e:
            error_msg = str(e).lower()
            # Check for Rate Limit errors (429) or common overload messages
            if "429" in error_msg or "too many requests" in error_msg or "rate limit" in error_msg:
                sleep_time = initial_backoff * (2 ** i)
                print(f"    Rate limit encountered. Retrying in {sleep_time}s...", file=sys.stderr)
                time.sleep(sleep_time)
            elif "500" in error_msg or "server error" in error_msg:
                time.sleep(2)
            else:
                print(f"    API Error: {e}", file=sys.stderr)
                time.sleep(1)
    return None

def call_openai(model_id: str, system_prompt: str, user_content: str) -> Optional[str]:
    """Sends a chat completion request to OpenAI."""
    if not openai_client:
        return None
    
    def _req():
        response = openai_client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()

    return retry_with_backoff(_req)

def call_anthropic(model_id: str, system_prompt: str, user_content: str) -> Optional[str]:
    """Sends a message request to Anthropic."""
    if not ANTHROPIC_API_KEY:
        return None
        
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    payload = {
        "model": model_id, 
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": 50,
        "temperature": 0.0
    }
    
    def _req():
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()['content'][0]['text'].strip()

    return retry_with_backoff(_req)

def process_entry(model_info: Dict[str, str], persona_id: str, 
                  system_prompt: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    """Handles the full generation cycle for a single entry and model."""
    user_content = format_user_content(entry)
    provider = model_info['provider']
    mid = model_info['model_id']
    
    prediction = None
    if provider == 'openai':
        prediction = call_openai(mid, system_prompt, user_content)
    elif provider == 'anthropic':
        prediction = call_anthropic(mid, system_prompt, user_content)
        
    label = None
    if prediction:
        # Heuristic parsing for labels
        if '1' in prediction and '2' not in prediction: label = 1
        elif '2' in prediction and '1' not in prediction: label = 2
        elif prediction.strip() == '1': label = 1
        elif prediction.strip() == '2': label = 2
    
    new_entry = entry.copy()
    new_entry['persona_ID'] = persona_id
    new_entry['preference_label'] = label
    return new_entry

def main():
    """Main execution loop for multi-model preference label generation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input dataset path (JSONL)")
    parser.add_argument("--prompts_dir", required=True, help="Directory for persona prompts")
    parser.add_argument("--output_base", required=True, help="Base directory for results")
    parser.add_argument("--model_id", default="gpt-4o", help="Model identifier")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    parser.add_argument("--workers", type=int, default=5, help="Concurrent workers")
    args = parser.parse_args()

    # Load dataset
    print(f"Loading data from {args.input}...")
    dataset = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))

    model_info = {"provider": args.provider, "model_id": args.model_id}
    model_name = args.model_id.replace("/", "_")
    
    # Process standard personas A-E
    personas = [f"Persona_{c}" for c in "ABCDE"]
    for pid in personas:
        print(f"\nProcessing {model_name} for {pid}...")
        
        prompt_file = f"{pid.lower()}.txt"
        try:
            system_prompt = load_system_prompt(args.prompts_dir, prompt_file)
        except FileNotFoundError:
            print(f"  Warning: Prompt file {prompt_file} not found. Skipping.")
            continue
            
        output_dir = os.path.join(args.output_base, model_name)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"labeled_dataset_{pid}.jsonl")
        
        with open(output_path, 'w', encoding='utf-8') as outfile:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(process_entry, model_info, pid, system_prompt, entry): entry 
                           for entry in dataset}
                
                for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                    try:
                        res = future.result()
                        outfile.write(json.dumps(res, ensure_ascii=False) + '\n')
                        if i % 10 == 0:
                            print(f"  Progress: {i}/{len(dataset)}...", end="\r")
                    except Exception as e:
                        print(f"  Error processing entry: {e}", file=sys.stderr)
        
        print(f"\n  Results saved to {output_path}")

if __name__ == "__main__":
    main()
