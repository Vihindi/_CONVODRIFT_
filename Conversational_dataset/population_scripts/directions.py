import os, json, time, re, sys
from typing import Optional, Tuple, List, Dict, Any
import requests
from dotenv import load_dotenv

# ---------------- Config ----------------
INPUT_FILE  = r"updated_only_directions\\Dilanka_updated_only.jsonl"
OUTPUT_FILE = r"updated_only_directions\\curated_Dilanka_updated_only.jsonl"
MODEL       = "gpt-5-nano-2025-08-07"
OPENAI_BASE = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
TIMEOUT_S   = 30
MAX_RETRIES = 4
BACKOFF     = 1.8

# ------------- Setup ----------
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}" if API_KEY else "",
           "Content-Type": "application/json"}

SYSTEM_PROMPT = (
    "You are a strict classifier. You will be given ONLY a user prompt.\n"
    "Determine whether the prompt requests a TONE SHIFT (a 'drift').\n"
    "If there is NO tone-shift request (explicit or implicit), output '0'.\n"
    "If there IS a tone-shift request, classify the DIRECTION:\n"
    " - '1' if the request moves from more formal to more casual.\n"
    " - '2' if the request moves from more casual to more formal.\n"
    "Return ONLY a single digit: '0', '1', or '2'."
)

# ---------- Parsing helpers ----------
DIGIT_RE = re.compile(r"\b([012])\b")

def extract_digit(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = DIGIT_RE.search(text.strip())
    return int(m.group(1)) if m else None

# ---------- API callers ----------
def call_responses(prompt_text: str) -> Optional[int]:
    """Primary: Responses API. No extra params (use model defaults)."""
    url = f"{OPENAI_BASE}/responses"
    payload = {
        "model": MODEL,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text}
        ]
    }
    for attempt in range(1, MAX_RETRIES+1):
        try:
            print(f"[DEBUG] Responses API call (attempt {attempt})…")
            r = requests.post(url, headers=HEADERS, json=payload, timeout=TIMEOUT_S)
            if r.status_code == 200:
                data = r.json()

                # Preferred short-circuit
                if isinstance(data.get("output_text"), str):
                    text = data["output_text"].strip()
                    print(f"[DEBUG] Responses output_text: {text!r}")
                    return extract_digit(text)

                # Fallback: parse blocks
                out = data.get("output", [])
                if out and isinstance(out, list):
                    content_blocks = out[0].get("content", [])
                    if content_blocks and isinstance(content_blocks, list):
                        # find first text-like block
                        for blk in content_blocks:
                            txt = blk.get("text")
                            if isinstance(txt, str) and txt.strip():
                                print(f"[DEBUG] Responses block text: {txt!r}")
                                return extract_digit(txt)

                # As absolute last resort, print raw for diagnosis
                raw = json.dumps(data, ensure_ascii=False)
                print(f"[DEBUG] Responses raw (no text found): {raw[:500]}...")
                return None

            # transient?
            if r.status_code in (429, 500, 502, 503, 504):
                sleep_s = BACKOFF**(attempt-1)
                print(f"[WARN] Responses {r.status_code}. Retry in {sleep_s:.1f}s")
                time.sleep(sleep_s)
                continue

            print(f"[ERROR] Responses {r.status_code}: {r.text[:400]}")
            return None

        except requests.RequestException as e:
            sleep_s = BACKOFF**(attempt-1)
            print(f"[WARN] Responses request error: {e}. Retry in {sleep_s:.1f}s")
            time.sleep(sleep_s)
    return None

def call_chat_completions(prompt_text: str) -> Optional[int]:
    """Fallback: Chat Completions with correct params this model tolerates."""
    url = f"{OPENAI_BASE}/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text}
        ]
        # do NOT send temperature / max_tokens here (let defaults apply)
    }
    for attempt in range(1, MAX_RETRIES+1):
        try:
            print(f"[DEBUG] Chat Completions call (attempt {attempt})…")
            r = requests.post(url, headers=HEADERS, json=payload, timeout=TIMEOUT_S)
            if r.status_code == 200:
                data = r.json()
                msg = (data.get("choices",[{}])[0].get("message") or {}).get("content")
                msg = (msg or "").strip()
                print(f"[DEBUG] Chat content: {msg!r}")
                return extract_digit(msg)

            if r.status_code in (429, 500, 502, 503, 504):
                sleep_s = BACKOFF**(attempt-1)
                print(f"[WARN] Chat {r.status_code}. Retry in {sleep_s:.1f}s")
                time.sleep(sleep_s)
                continue

            print(f"[ERROR] Chat {r.status_code}: {r.text[:400]}")
            return None

        except requests.RequestException as e:
            sleep_s = BACKOFF**(attempt-1)
            print(f"[WARN] Chat request error: {e}. Retry in {sleep_s:.1f}s")
            time.sleep(sleep_s)
    return None

def classify_direction(prompt_text: str) -> int:
    """
    Try Responses → Chat Completions.
    If both fail or return empty, force 0 (no drift) so you never get None.
    """
    res = call_responses(prompt_text)
    if res in (0,1,2):
        return res
    print("[DEBUG] Falling back to Chat Completions…")
    res = call_chat_completions(prompt_text)
    if res in (0,1,2):
        return res
    print("[DEBUG] Both endpoints returned empty/invalid → default to 0")
    return 0

# ---------- JSON helpers ----------
def get_pairs_ref(row: Dict[str, Any]) -> Tuple[Optional[List[Dict[str, Any]]], str, Optional[Dict[str, Any]]]:
    if isinstance(row.get("pairs"), list):
        return row["pairs"], "row.pairs", row
    data = row.get("data")
    if isinstance(data, dict) and isinstance(data.get("pairs"), list):
        return data["pairs"], "row.data.pairs", data
    return None, "(not found)", None

def truthy(v) -> bool:
    if isinstance(v, bool): return v
    if isinstance(v, str): return v.strip().lower() in {"true","1","yes","y"}
    return bool(v)

# ---------- Main ----------
def process_file(in_path: str, out_path: str):
    if not API_KEY:
        print("[ERROR] OPENAI_API_KEY missing.")
        sys.exit(1)

    total_rows=updated_rows=rows_without_pairs=0
    total_pairs_seen=updated_pairs=0

    print(f"[INFO] Starting: {in_path}")
    with open(in_path, "r", encoding="utf-8") as infile, \
         open(out_path, "w", encoding="utf-8") as outfile:

        try:
            total_lines = sum(1 for _ in open(in_path, "r", encoding="utf-8"))
        except Exception:
            total_lines = None
        infile.seek(0)

        for line_num, line in enumerate(infile, start=1):
            s = line.strip()
            if not s: continue
            try:
                row = json.loads(s)
            except json.JSONDecodeError as e:
                print(f"[WARN] Bad JSON at line {line_num}: {e}")
                continue

            total_rows += 1
            pct = f" ({int(line_num/total_lines*100)}%)" if total_lines else ""
            conv_id = row.get("conversation_id") or (row.get("data") or {}).get("conversation_id")
            print(f"\n[INFO] Row {line_num}{pct} (conversation_id={conv_id})")

            pairs, path_str, parent = get_pairs_ref(row)
            if pairs is None:
                rows_without_pairs += 1
                print(f"[WARN] No pairs at row {line_num}. Checked row.pairs and row.data.pairs.")
                outfile.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            print(f"[DEBUG] Pairs at {path_str}: {len(pairs)} items")
            row_changed = False

            for idx, pair in enumerate(pairs, start=1):
                if not isinstance(pair, dict):
                    print(f"[WARN] Pair {idx} not an object → skip")
                    continue
                total_pairs_seen += 1

                # Only classify labeled drifts to save calls (your original requirement)
                if not truthy(pair.get("drift")):
                    print(f"[DEBUG] Pair {idx}: drift is not true → skip")
                    continue

                prompt_text = pair.get("prompt") or ""
                if not prompt_text.strip():
                    print(f"[DEBUG] Pair {idx}: empty prompt → skip")
                    continue

                print(f"[INFO] -> Classifying Pair {idx}. Prompt[:100]: {prompt_text[:100]!r}")
                direction = classify_direction(prompt_text)
                pair["direction"] = direction   # always set 0/1/2
                print(f"[RESULT] Pair {idx}: direction={direction}")
                row_changed = True
                updated_pairs += 1

            if row_changed:
                updated_rows += 1

            outfile.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n[SUMMARY] Done.")
    print(f"Rows processed: {total_rows}, Rows updated: {updated_rows}")
    print(f"Rows with NO pairs: {rows_without_pairs}")
    print(f"Pairs seen: {total_pairs_seen}, Pairs updated (direction set): {updated_pairs}")
    print(f"[INFO] Wrote: {out_path}")

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] Cannot find {INPUT_FILE} in {os.getcwd()}")
        sys.exit(1)
    process_file(INPUT_FILE, OUTPUT_FILE)
