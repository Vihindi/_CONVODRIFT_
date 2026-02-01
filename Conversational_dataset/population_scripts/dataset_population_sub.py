
import os, sys, json, re
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
from prompts.quotes_wishes_prompt import PROMPT

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    sys.stderr.write("ERROR: OPENAI_API_KEY missing. Create a .env with OPENAI_API_KEY=...\n")
    sys.exit(1)

from openai import OpenAI
client = OpenAI(api_key=api_key)


MODEL_NAME = "gpt-4.1-mini-2025-04-14"
TOTAL_SAMPLES = 1500
OUTFILE = "data\\quotes_wishes_data\\new_model_quotes_wishes_data_05.jsonl"


def build_generation_prompt(domain: str, scenario: str) -> str:
    return PROMPT.format(domain=domain, scenario=scenario)

def get_output_text(resp) -> str:
    text = getattr(resp, "output_text", None)
    print("output text", text)
    if text:
        return text
    try:
        return resp.output[0].content[0].text
    except Exception:
        return str(resp)

def coerce_json(text: str) -> Dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE | re.DOTALL)
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{"); end = text.rfind("}")
    print("start and end", start, end)
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end+1])
    raise ValueError("No valid JSON object found")

def validate_conversation(obj: Dict[str, Any]) -> Dict[str, Any]:
    info = {"valid_json": True, "six_pairs": False, "first_drift_false": False, "min_two_drifts": False}
    try:
        pairs = obj.get("pairs", [])
        info["six_pairs"] = isinstance(pairs, list) and len(pairs) == 6
        if info["six_pairs"]:
            info["first_drift_false"] = (pairs[0].get("drift") is False)
            drifts = sum(1 for p in pairs[1:] if p.get("drift") is True)
            info["min_two_drifts"] = drifts >= 2
    except Exception:
        info["valid_json"] = False
    info["is_valid"] = all([info["valid_json"], info["six_pairs"], info["first_drift_false"], info["min_two_drifts"]])
    return info
