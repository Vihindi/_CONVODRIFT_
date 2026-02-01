import json
import requests
import time
import os
import argparse
import concurrent.futures
import threading
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass # Assume env vars are already set if dotenv is missing

# Configuration
# Ensure ANTHROPIC_API_KEY is set in your environment variables (or .env file)
API_KEY = os.getenv("ANTHROPIC_API_KEY") 
MODEL_NAME = "claude-haiku-4-5-20251001"
API_URL = "https://api.anthropic.com/v1/messages"
INPUT_FILE = "D:\\IIT\\4th year\\fyp\\dataset research\\RLHF_datset\\Main_dataset_eval\\Final_rlhf_dataset_flipped_cleaned.jsonl"
PERSONA_PROMPTS_DIR = "D:\\IIT\\4th year\\fyp\\dataset research\\RLHF_datset\\persona_prompts"

PERSONAS = [
 #   {"id": "Persona_A", "file": "persona_A.txt"},
  #  {"id": "Persona_B", "file": "persona_B.txt"},
   # {"id": "Persona_C", "file": "persona_C.txt"},
    #{"id": "Persona_D", "file": "persona_D.txt"},
    {"id": "Persona_E", "file": "persona_E.txt"},
]

def load_system_prompt(filename):
    path = os.path.join(PERSONA_PROMPTS_DIR, filename)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def get_claude_preference(system_prompt, user_content):
    if not API_KEY:
        # Warning printed in main()
        return None

    headers = {
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    payload = {
        "model": MODEL_NAME,
        "max_tokens": 1000,
        "temperature": 0.0,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_content}
        ]
    }

    max_retries = 3
    try:
        # Rename log file to differentiate from Gemini
        with open("debug_claude.log", "a", encoding="utf-8") as debug_f:
            for attempt in range(max_retries):
                try:
                    debug_f.write(f"\n--- Attempt {attempt+1} ---\n")
                    
                    response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
                    
                    debug_f.write(f"Response Status: {response.status_code}\n")
                    debug_f.flush()
                    
                    if response.status_code != 200:
                        print(f"API Error (Attempt {attempt+1}/{max_retries}): Status {response.status_code}")
                        if response.status_code == 429:
                            time.sleep(10) # Rate limit
                        elif response.status_code == 529: # Overloaded
                            time.sleep(5)
                        else:
                            time.sleep(2)
                        continue

                    data = response.json()
                    
                    if 'content' in data and isinstance(data['content'], list):
                        text_content = ""
                        for block in data['content']:
                            if block.get('type') == 'text':
                                text_content += block.get('text', '')
                        return text_content.strip()
                    
                    print(f"Unexpected response format: {data}")
                    return None

                except requests.exceptions.RequestException as e:
                    debug_f.write(f"Request Exception: {e}\n")
                    print(f"Request Exception (Attempt {attempt+1}/{max_retries}): {e}")
                    time.sleep(2)
    except Exception as e:
        print(f"Logging failed: {e}")
            
    return None

def format_user_content(entry):
    return (
        f"Prompt: {entry['prompt']}\n\n"
        f"Response A:\n{entry['response_A']}\n\n"
        f"Response B:\n{entry['response_B']}\n\n"
        "Evaluate the responses based on your persona. "
        "Which response do you prefer? Respond with '1' for Response A or '2' for Response B."
    )

# Thread-safe file writing
write_lock = threading.Lock()

def process_entry(entry, system_prompt, persona_id):
    user_content = format_user_content(entry)
    prediction = get_claude_preference(system_prompt, user_content)
    
    # Relaxed parsing logic
    if prediction and prediction not in ['1', '2']:
        if len(prediction) < 10:
             if '1' in prediction and '2' not in prediction: prediction = '1'
             elif '2' in prediction and '1' not in prediction: prediction = '2'
    
    new_entry = entry.copy()
    new_entry['persona_ID'] = persona_id
    
    try:
        new_entry['preference_label'] = int(prediction) if prediction in ['1', '2'] else None
    except:
        new_entry['preference_label'] = None
            
    return new_entry

def main():
    parser = argparse.ArgumentParser(description="Generate persona-based preference labels using Claude 3.5 Haiku.")
    parser.add_argument("--test", action="store_true", help="Run on first 5 rows only for testing.")
    args = parser.parse_args()
    
    if not API_KEY:
         print("CRITICAL WARNING: ANTHROPIC_API_KEY environment variable is not set. API calls will fail.")

    # Load dataset
    print(f"Loading dataset from {INPUT_FILE}...")
    dataset = []
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: Input file {INPUT_FILE} not found. Please verify the path.")
        return
    
    if args.test:
        dataset = dataset[:5]
        print("Running in TEST mode (processing first 5 rows only).")

    # Process each persona
    for persona in PERSONAS:
        persona_id = persona['id']
        print(f"\nProcessing Persona: {persona_id}...")
        
        try:
            system_prompt = load_system_prompt(persona['file'])
        except FileNotFoundError:
             print(f"Warning: Persona prompt file {persona['file']} not found. Skipping.")
             continue

        output_filename = f"D:\\IIT\\4th year\\fyp\\dataset research\\RLHF_datset\\Main_dataset_eval\\main_labeled_personas\\labeled_Main_final_dataset_{persona_id}.jsonl"
        output_path = os.path.join(os.path.dirname(INPUT_FILE), output_filename)
        
        # Use ThreadPoolExecutor
        # Max wrokers set to 10 for higher RPM
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_entry = {executor.submit(process_entry, entry, system_prompt, persona_id): entry for entry in dataset}
            
            processed_count = 0
            with open(output_path, 'w', encoding='utf-8') as outfile:
                for future in concurrent.futures.as_completed(future_to_entry):
                    try:
                        result = future.result()
                        outfile.write(json.dumps(result) + '\n')
                        outfile.flush()
                        
                        processed_count += 1
                        if processed_count % 10 == 0:
                            print(f"Processed {processed_count}/{len(dataset)} entries for {persona_id}")
                    except Exception as exc:
                        print(f"Generated an exception: {exc}")

        print(f"Finished processing {persona_id}. Saved to {output_filename}")

if __name__ == "__main__":
    main()
