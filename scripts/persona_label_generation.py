import argparse
import concurrent.futures
import json
import os
import requests
import sys
import threading
import time
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configuration Constants
API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL_NAME = "claude-3-5-haiku-20241022" 
API_URL = "https://api.anthropic.com/v1/messages"

def load_system_prompt(prompt_dir: str, filename: str) -> str:
    """Loads a persona system prompt from a text file."""
    path = os.path.join(prompt_dir, filename)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def get_claude_preference(system_prompt: str, user_content: str) -> Optional[str]:
    """
    Submits a preference evaluation request to the Anthropic API.
    
    Args:
        system_prompt (str): The persona-defining system prompt.
        user_content (str): The prompt and pair of responses to evaluate.

    Returns:
        str: The raw model response or None if the request fails.
    """
    if not API_KEY:
        return None

    headers = {
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    payload = {
        "model": MODEL_NAME,
        "max_tokens": 50,
        "temperature": 0.0,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_content}]
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                content = data.get('content', [])
                if content and isinstance(content, list):
                    return content[0].get('text', '').strip()
                return None
            
            # Error handling with exponential backoff
            if response.status_code in (429, 529):
                wait_time = 10 if response.status_code == 429 else 5
                time.sleep(wait_time * (attempt + 1))
            else:
                time.sleep(2)
        except requests.exceptions.RequestException as e:
            print(f"Request failed (Attempt {attempt+1}): {e}", file=sys.stderr)
            time.sleep(2)
            
    return None

def format_user_content(entry: Dict[str, Any]) -> str:
    """Formats a dataset entry into a prompt for the preference judge."""
    return (
        f"Prompt: {entry.get('prompt', '')}\n\n"
        f"Response A:\n{entry.get('response_A', '')}\n\n"
        f"Response B:\n{entry.get('response_B', '')}\n\n"
        "Evaluate the responses based on your persona. "
        "Which response do you prefer? Respond with '1' for Response A or '2' for Response B."
    )

def process_entry(entry: Dict[str, Any], system_prompt: str, persona_id: str) -> Dict[str, Any]:
    """
    Processes a single dataset entry: generates a prompt, gets a prediction, and parses the result.
    """
    user_content = format_user_content(entry)
    prediction = get_claude_preference(system_prompt, user_content)
    
    # Heuristic parsing for non-standard model responses
    label = None
    if prediction:
        prediction = prediction.strip()
        if prediction in ('1', '2'):
            label = int(prediction)
        elif len(prediction) < 20: # Attempt to find a digit if response is short
            if '1' in prediction and '2' not in prediction:
                label = 1
            elif '2' in prediction and '1' not in prediction:
                label = 2
    
    new_entry = entry.copy()
    new_entry['persona_ID'] = persona_id
    new_entry['preference_label'] = label
    return new_entry

def main():
    """Main execution loop for persona-based label generation."""
    parser = argparse.ArgumentParser(description="Generate persona labels using Claude.")
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument("--prompts_dir", required=True, help="Directory containing persona .txt files")
    parser.add_argument("--persona_id", required=True, help="The ID of the persona to process")
    parser.add_argument("--persona_file", required=True, help="The filename of the persona prompt")
    parser.add_argument("--output", required=True, help="Path to save the labeled output")
    parser.add_argument("--test", action="store_true", help="Process only the first 5 records")
    parser.add_argument("--workers", type=int, default=5, help="Number of concurrent threads")
    args = parser.parse_args()
    
    if not API_KEY:
        print("CRITICAL: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
        return

    # Load dataset
    dataset = []
    print(f"Loading dataset from {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    
    if args.test:
        dataset = dataset[:5]

    try:
        system_prompt = load_system_prompt(args.prompts_dir, args.persona_file)
    except FileNotFoundError:
        print(f"Error: Persona file {args.persona_file} not found in {args.prompts_dir}", file=sys.stderr)
        return

    print(f"Processing persona {args.persona_id} with {args.workers} workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_entry, entry, system_prompt, args.persona_id): entry for entry in dataset}
        
        with open(args.output, 'w', encoding='utf-8') as outfile:
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                try:
                    result = future.result()
                    outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
                    if i % 10 == 0:
                        print(f"Progress: {i}/{len(dataset)} entries processed.")
                except Exception as e:
                    print(f"Error processing entry: {e}", file=sys.stderr)

    print(f"Successfully processed {args.persona_id}. Results saved to {args.output}")

if __name__ == "__main__":
    main()
