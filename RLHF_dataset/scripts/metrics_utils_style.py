# metrics_utils_style.py
"""
Metrics that operate on *style preference* (formal vs casual), not positional A/B labels.

- `direction` encodes which side is more formal:
    direction == 1 : A is more formal, B is more casual  (shift toward casual)
    direction == 2 : A is more casual, B is more formal  (shift toward formal)
    direction == 0 : no clear direction (we drop these from style-based metrics)

- `preference_label` encodes which response was chosen:
    '1' or 'A' => chose response_A
    '2' or 'B' => chose response_B
"""

import math
from collections import Counter
from itertools import combinations


def _norm_dir(d):
    if d is None:
        return None
    if isinstance(d, str):
        d = d.strip()
        if d.isdigit():
            return int(d)
        return None
    if isinstance(d, (int, float)):
        return int(d)
    return None


def _norm_pref(x):
    if x is None:
        return None
    if isinstance(x, str):
        x = x.strip()
        if x in {"A", "B", "1", "2"}:
            return "1" if x == "A" else ("2" if x == "B" else x)
        if x.isdigit():
            return x
    if isinstance(x, (int, float)):
        return str(int(x))
    return None


def pref_to_style(direction, preference_label):
    """
    Map (direction, chosen side) -> preferred style: 'formal' or 'casual'.
    Returns None if direction is 0/unknown.
    """
    d = _norm_dir(direction)
    p = _norm_pref(preference_label)
    if d == 1:
        # A formal, B casual
        return "formal" if p == "1" else "casual"
    if d == 2:
        # A casual, B formal
        return "formal" if p == "2" else "casual"
    return None


def shannon_entropy(counts: Counter) -> float:
    n = sum(counts.values())
    if n == 0:
        return 0.0
    H = 0.0
    for c in counts.values():
        if c == 0:
            continue
        p = c / n
        H -= p * math.log2(p)
    return H


def style_entropy_from_rows(rows) -> dict:
    """
    Compute style entropy and formal-rate from a list of JSONL rows for one persona.
    Expects each row has: direction, preference_label.
    """
    style_counts = Counter()
    dropped = 0
    for r in rows:
        style = pref_to_style(r.get("direction"), r.get("preference_label"))
        if style is None:
            dropped += 1
            continue
        style_counts[style] += 1

    used = sum(style_counts.values())
    formal_rate = (style_counts["formal"] / used) if used > 0 else 0.0
    return {
        "style_counts": dict(style_counts),
        "used": used,
        "dropped": dropped,
        "formal_rate": formal_rate,
        "style_entropy": shannon_entropy(style_counts),
    }


def persona_sensitivity_pairwise(per_sample_persona_preds: dict) -> float:
    """
    Pairwise disagreement averaged per sample, then averaged across samples.

    per_sample_persona_preds: {sample_id: {persona_id: pred}}
    pred can be 'formal'/'casual' (recommended) OR '1'/'2'.
    """
    per_sample = []
    for sid, pmap in per_sample_persona_preds.items():
        personas = list(pmap.keys())
        if len(personas) < 2:
            continue
        dis = []
        for a, b in combinations(personas, 2):
            dis.append(1.0 if pmap[a] != pmap[b] else 0.0)
        per_sample.append(sum(dis) / len(dis))
    return (sum(per_sample) / len(per_sample)) if per_sample else 0.0
