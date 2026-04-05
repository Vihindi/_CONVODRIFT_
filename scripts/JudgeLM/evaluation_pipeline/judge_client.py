import json
import time
from typing import Any, Dict, Optional
import requests
from loguru import logger
from .config import MODEL_NAME
from .credentials import GOOGLE_API_KEY

class JudgeClient:
    """
    Client for interacting with the Google Gemini API to perform 
    conversation evaluations.
    """
    def __init__(self):
        self.api_key = GOOGLE_API_KEY
        # Primary endpoint for Gemini generation (supports system instructions)
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={self.api_key}"
        self.headers = {"Content-Type": "application/json"}

    def evaluate(self, system_prompt: str, user_message: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """
        Submits a prompt to the LLM judge and returns the parsed JSON evaluation.

        Args:
            system_prompt (str): Instructions defining the judge's persona and rules.
            user_message (str): The specific conversation and domain context to evaluate.
            max_retries (int): Number of connection or API-level retries.

        Returns:
            dict: The parsed JSON evaluation or None if all retries fail.
        """
        payload = {
            "system_instruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [
                {"parts": [{"text": user_message}]}
            ],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.0 # Force deterministic output for evaluation
            }
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(self.url, headers=self.headers, json=payload, timeout=60)
                
                if response.status_code == 429:
                    # Rate limit hit
                    wait_time = 10 * (attempt + 1)
                    logger.warning(f"Rate limited (429). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                result = response.json()
                
                # Gemini path: candidates -> content -> parts -> text
                candidates = result.get("candidates", [])
                if not candidates:
                    logger.error(f"Gemini returned no candidates: {result}")
                    continue

                content_parts = candidates[0].get("content", {}).get("parts", [])
                if not content_parts or "text" not in content_parts[0]:
                    logger.error(f"Unexpected candidate structure: {candidates[0]}")
                    continue

                raw_text = content_parts[0]["text"]
                return self._parse_json_from_text(raw_text)
                
            except Exception as e:
                logger.error(f"API Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                else:
                    return None
        return None

    def _parse_json_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extracts and parses JSON content from the raw LLM string response, 
        stripping any markdown artifacts if present.
        """
        clean_text = text.strip()
        
        # Handle cases where LLM wraps output in markdown code blocks despite payload settings
        if clean_text.startswith("```"):
            lines = clean_text.splitlines()
            # Remove first line if it's a code block start (e.g., ```json)
            if lines[0].startswith("```"):
                lines = lines[1:]
            # Remove trailing code block line
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            clean_text = "\n".join(lines).strip()
            
        try:
            return json.loads(clean_text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON Decode Error (start of text): {clean_text[:200]}")
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")
