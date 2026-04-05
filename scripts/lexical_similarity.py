#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

# Regex for basic tokenization
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

def tokenize(text: str) -> List[str]:
    """
    Splits text into lowercase tokens using alphanumeric characters.

    Args:
        text (str): The input text.

    Returns:
        list[str]: A list of lowercase tokens.
    """
    if not text:
        return []
    return TOKEN_RE.findall(text.lower())

def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """
    Generates n-grams from a list of tokens.

    Args:
        tokens (list[str]): The input tokens.
        n (int): The size of each n-gram.

    Returns:
        list[tuple[str]]: A list of n-gram tuples.
    """
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def jaccard(a: set, b: set) -> float:
    """
    Computes the Jaccard similarity coefficient between two sets.

    Args:
        a (set): First set.
        b (set): Second set.

    Returns:
        float: Jaccard similarity [0.0, 1.0]. Returns 1.0 if both sets are empty.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0

def lcs_length(a: List[str], b: List[str]) -> int:
    """
    Computes the length of the Longest Common Subsequence using 2-row DP.

    Args:
        a (list[str]): First sequence of tokens.
        b (list[str]): Second sequence of tokens.

    Returns:
        int: LCS length.
    """
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    cur  = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur[0] = 0
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev, cur = cur, prev
    return prev[len(b)]

def rouge_l_f1(a_text: str, b_text: str) -> float:
    """
    Computes the ROUGE-L F1 score based on token-level LCS.

    Args:
        a_text (str): First text string.
        b_text (str): Second text string.

    Returns:
        float: ROUGE-L F1 score [0.0, 1.0].
    """
    a = tokenize(a_text)
    b = tokenize(b_text)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    lcs = lcs_length(a, b)
    precision = lcs / len(b)
    recall = lcs / len(a)
    return (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

def unwrap_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unwraps nested record data if present within a 'data' key.

    Args:
        record (dict): The potential wrapper record.

    Returns:
        dict: The inner data dictionary or the original record.
    """
    if isinstance(record, dict) and isinstance(record.get("data"), dict):
        return record["data"]
    return record

def extract_conversation(record: Dict[str, Any]) -> Tuple[str, List[str], List[Optional[bool]]]:
    """
    Extracts conversation ID, assistant responses, and drift edges from a record.

    Args:
        record (dict): The original JSONL record.

    Returns:
        tuple: (conv_id, list of response texts, list of drift labels for transitions).
    """
    rec = unwrap_record(record)
    conv_id = str(rec.get("conversation_id") or rec.get("id") or rec.get("convo_id") or "")
    pairs = rec.get("pairs")

    if not isinstance(pairs, list) or len(pairs) < 2:
        return conv_id, [], []

    texts: List[str] = []
    drift_turn: List[Optional[bool]] = []

    for p in pairs:
        if not isinstance(p, dict):
            continue
        resp = p.get("response") or p.get("text")
        texts.append(str(resp) if resp is not None else "")

        drift = p.get("drift")
        if isinstance(drift, bool) or drift is None:
            drift_turn.append(drift)
        else:
            # Handle integer representations of booleans
            drift_turn.append(bool(drift) if drift in (0, 1) else None)

    if len(texts) < 2:
        return conv_id, [], []

    # Map drift labels to the transitions (edge i is transition between response i and i+1)
    drift_edges = [drift_turn[i + 1] if i + 1 < len(drift_turn) else None for i in range(len(texts) - 1)]

    return conv_id, texts, drift_edges

def summarize(vals: List[float]) -> Dict[str, float]:
    """
    Computes basic descriptive statistics for a list of values.
    """
    if not vals:
        return {"n": 0, "mean": math.nan, "std": math.nan}
    return {"n": len(vals), "mean": mean(vals), "std": pstdev(vals) if len(vals) > 1 else 0.0}

def cliffs_delta(a: List[float], b: List[float]) -> float:
    """
    Computes Cliff's delta effect size between two groups.
    """
    if not a or not b:
        return math.nan
    gt, lt = 0, 0
    for x in a:
        for y in b:
            if x > y: gt += 1
            elif x < y: lt += 1
    denom = len(a) * len(b)
    return (gt - lt) / denom if denom else math.nan

def cliffs_magnitude(delta: float) -> str:
    """
    Interprets the magnitude of Cliff's delta using standard thresholds.
    """
    if math.isnan(delta): return "NA"
    ad = abs(delta)
    if ad < 0.147: return "negligible"
    if ad < 0.33: return "small"
    if ad < 0.474: return "medium"
    return "large"

def _rankdata(values: List[float]) -> List[float]:
    """
    Internal helper for ranking data with tie handling.
    """
    sorted_idx = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    r = 1
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[sorted_idx[j + 1]] == values[sorted_idx[i]]:
            j += 1
        avg_rank = (r + (r + (j - i))) / 2.0
        for k in range(i, j + 1):
            ranks[sorted_idx[k]] = avg_rank
        r += (j - i + 1)
        i = j + 1
    return ranks

def mann_whitney_u(a: List[float], b: List[float]) -> Dict[str, float]:
    """
    Computes two-sided Mann-Whitney U test using normal approximation with tie correction.
    """
    if not a or not b:
        return {"U": math.nan, "z": math.nan, "p": math.nan}

    n1, n2 = len(a), len(b)
    all_vals = a + b
    ranks = _rankdata(all_vals)
    r1 = sum(ranks[:n1])
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u = min(u1, u2)

    counts = {}
    for v in all_vals:
        counts[v] = counts.get(v, 0) + 1
    tie_sum = sum(c**3 - c for c in counts.values() if c > 1)

    mu = n1 * n2 / 2.0
    denom = (n1 + n2) * (n1 + n2 - 1)
    if denom == 0:
        return {"U": u, "z": math.nan, "p": math.nan}

    sigma_sq = (n1 * n2 / 12.0) * ((n1 + n2 + 1) - (tie_sum / denom))
    sigma = math.sqrt(sigma_sq) if sigma_sq > 0 else 0.0

    if sigma == 0:
        return {"U": u, "z": math.nan, "p": 1.0}

    z = (u - mu + 0.5) / sigma
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return {"U": u, "z": z, "p": p}

def main():
    """
    Main logic to compute adjacent-turn lexical similarities and perform statistical analysis.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to JSONL source file")
    ap.add_argument("--out_csv", required=True, help="Path for output CSV results")
    ap.add_argument("--use_bigrams", action="store_true", help="Include bigram Jaccard similarity")
    ap.add_argument("--max_lines", type=int, default=0, help="Limit lines for debugging (0=all)")
    args = ap.parse_args()

    # Collectors for global and segmented pools
    pools = {
        "rouge": {True: [], False: []},
        "j1": {True: [], False: []},
        "j2": {True: [], False: []}
    }
    
    per_edge = {
        "rouge": {True: defaultdict(list), False: defaultdict(list)},
        "j1":    {True: defaultdict(list), False: defaultdict(list)},
        "j2":    {True: defaultdict(list), False: defaultdict(list)},
    }

    rows = []
    total_lines = 0
    kept_conversations = 0

    with open(args.input, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            if args.max_lines and line_idx > args.max_lines:
                break
            total_lines += 1
            line = line.strip()
            if not line: continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            conv_id, texts, drift_edges = extract_conversation(record)
            if len(texts) < 2: continue

            kept_conversations += 1
            segment_id = 0

            for i in range(len(texts) - 1):
                a, b = texts[i], texts[i + 1]
                drift = drift_edges[i]

                rl = rouge_l_f1(a, b)
                tok_a, tok_b = tokenize(a), tokenize(b)
                j1 = jaccard(set(tok_a), set(tok_b))
                j2 = jaccard(set(ngrams(tok_a, 2)), set(ngrams(tok_b, 2))) if args.use_bigrams else math.nan

                rows.append({
                    "conversation_id": conv_id, "edge_index": i, "segment_id": segment_id,
                    "drift": drift, "edge_type": "boundary" if drift is True else ("within" if drift is False else "unknown"),
                    "rouge_l_f1": rl, "jaccard_unigram": j1, "jaccard_bigram": j2,
                    "prev_tokens": len(tok_a), "next_tokens": len(tok_b),
                })

                if drift in (True, False):
                    pools["rouge"][drift].append(rl)
                    pools["j1"][drift].append(j1)
                    per_edge["rouge"][drift][i].append(rl)
                    per_edge["j1"][drift][i].append(j1)
                    if args.use_bigrams:
                        pools["j2"][drift].append(j2)
                        per_edge["j2"][drift][i].append(j2)
                    if drift is True: segment_id += 1

    # Save to CSV
    fieldnames = [
        "conversation_id","edge_index","segment_id","drift","edge_type",
        "rouge_l_f1","jaccard_unigram","jaccard_bigram","prev_tokens","next_tokens"
    ]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Reporting
    print("\n=== Global Similarity Summary ===")
    print(f"Total lines: {total_lines}, Kept conversations: {kept_conversations}")
    print(f"ROUGE-L (drift=False): {summarize(pools['rouge'][False])}")
    print(f"ROUGE-L (drift=True):  {summarize(pools['rouge'][True])}")

    print("\n=== Significance and Effect Size ===")
    for metric_name, m_key in [("ROUGE-L", "rouge"), ("Jaccard-1", "j1"), ("Jaccard-2", "j2")]:
        if m_key == "j2" and not args.use_bigrams: continue
        a, b = pools[m_key][False], pools[m_key][True]
        mw = mann_whitney_u(a, b)
        cd = cliffs_delta(a, b)
        print(f"\n{metric_name}:")
        print(f"  MW p-value: {mw['p']:.6g}, Cliff's d: {cd:.3f} ({cliffs_magnitude(cd)})")

    print("\n=== Per-Edge-Index Analysis (0..4) ===")
    max_idx = max((r["edge_index"] for r in rows), default=0)
    for m_key in ["rouge", "j1", "j2"]:
        if m_key == "j2" and not args.use_bigrams: continue
        print(f"\nMetric: {m_key.upper()}")
        for i in range(min(max_idx + 1, 5)):
            v_f, v_t = per_edge[m_key][False][i], per_edge[m_key][True][i]
            s_f, s_t = summarize(v_f), summarize(v_t)
            print(f"  Edge {i}: False {s_f} | True {s_t}")
            if len(v_f) >= 5 and len(v_t) >= 5:
                mw = mann_whitney_u(v_f, v_t)
                cd = cliffs_delta(v_f, v_t)
                print(f"         MW p={mw['p']:.6g}, Cliff d={cd:.3f}")

if __name__ == "__main__":
    main()
