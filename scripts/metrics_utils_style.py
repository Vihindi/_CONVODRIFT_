import math
from collections import Counter
from itertools import combinations
from typing import Any, Dict, List, Optional, Union

def _normalize_direction(direction: Any) -> Optional[int]:
    """
    Normalizes direction labels into integers (1, 2, or 0).
    
    Mapping:
        1 : A is more formal, B is more casual (shift toward casual)
        2 : A is more casual, B is more formal (shift toward formal)
        0 : No clear direction
    """
    if direction is None:
        return None
    if isinstance(direction, str):
        direction = direction.strip()
        if direction.isdigit():
            return int(direction)
        return None
    if isinstance(direction, (int, float)):
        return int(direction)
    return None

def _normalize_preference(preference: Any) -> Optional[str]:
    """
    Normalizes preference labels into standard "1" (Response A) or "2" (Response B).
    """
    if preference is None:
        return None
    if isinstance(preference, str):
        p = preference.strip()
        if p in {"A", "1"}:
            return "1"
        if p in {"B", "2"}:
            return "2"
        if p.isdigit():
            return p
    if isinstance(preference, (int, float)):
        return str(int(preference))
    return None

def pref_to_style(direction: Any, preference_label: Any) -> Optional[str]:
    """
    Maps a (direction, chosen_response) pair to a preferred style: 'formal' or 'casual'.

    Args:
        direction: The direction label (1, 2, or 0).
        preference_label: The label indicating which response was chosen (A or B).

    Returns:
        str: 'formal', 'casual', or None if the direction is unknown/invalid.
    """
    d = _normalize_direction(direction)
    p = _normalize_preference(preference_label)
    
    if d == 1:
        # A is formal, B is casual
        return "formal" if p == "1" else "casual"
    if d == 2:
        # A is casual, B is formal
        return "formal" if p == "2" else "casual"
    return None

def shannon_entropy(counts: Counter) -> float:
    """
    Computes Shannon entropy (base 2) for a given frequency distribution.

    Args:
        counts (Counter): Frequency of each category.

    Returns:
        float: Calculated entropy.
    """
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    
    entropy_val = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy_val -= p * math.log2(p)
    return entropy_val

def style_entropy_from_rows(rows: List[Dict[str, Any]]) -> Dict[str, Union[Dict[str, int], int, float]]:
    """
    Computes style entropy and formalization rate from a list of record rows.

    Args:
        rows (list): List of dictionaries, each containing 'direction' and 'preference_label'.

    Returns:
        dict: Summary statistics including style_counts, used/dropped counts, and entropy.
    """
    style_counts = Counter()
    dropped_count = 0
    
    for row in rows:
        style = pref_to_style(row.get("direction"), row.get("preference_label"))
        if style is None:
            dropped_count += 1
            continue
        style_counts[style] += 1

    total_used = sum(style_counts.values())
    formal_rate = (style_counts["formal"] / total_used) if total_used > 0 else 0.0
    
    return {
        "style_counts": dict(style_counts),
        "used": total_used,
        "dropped": dropped_count,
        "formal_rate": formal_rate,
        "style_entropy": shannon_entropy(style_counts),
    }

def persona_sensitivity_pairwise(per_sample_persona_preds: Dict[str, Dict[str, str]]) -> float:
    """
    Calculates the average pairwise disagreement across personas for each sample.

    Args:
        per_sample_persona_preds (dict): Mapping of {sample_id: {persona_id: prediction}}.
            Predictions should be consistent labels (e.g., 'formal'/'casual').

    Returns:
        float: Mean pairwise disagreement [0.0, 1.0].
    """
    disagreements_per_sample = []
    
    for sample_id, persona_map in per_sample_persona_preds.items():
        personas = list(persona_map.keys())
        if len(personas) < 2:
            continue
            
        pairs = list(combinations(personas, 2))
        num_disagreements = sum(1.0 for p1, p2 in pairs if persona_map[p1] != persona_map[p2])
        disagreements_per_sample.append(num_disagreements / len(pairs))
    
    if not disagreements_per_sample:
        return 0.0
    return sum(disagreements_per_sample) / len(disagreements_per_sample)
