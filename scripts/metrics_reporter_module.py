import json
from collections import defaultdict
from typing import Dict, List, Any
from metrics_utils_style import style_entropy_from_rows, pref_to_style, persona_sensitivity_pairwise

def compute_metrics_for_model(persona_files: Dict[str, str]) -> Dict[str, Any]:
    """
    Computes style metrics (entropy and sensitivity) for a given model's persona outputs.

    Args:
        persona_files (dict): Mapping of {persona_id: file_path}.

    Returns:
        dict: A dictionary containing 'personas' stats and 'overall' sensitivity.
    """
    persona_results = {}
    per_sample_preds = defaultdict(dict) # {convo_id: {persona_id: style}}

    for pid, path in persona_files.items():
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        
        # Calculate entropy and formal rate for this persona
        stats = style_entropy_from_rows(rows)
        persona_results[pid] = stats

        # Collect predictions for sensitivity calculation
        for r in rows:
            # Note: We assume the record has 'convo_ID' (or 'id') and 'direction'/'preference_label'
            cid = r.get("convo_ID") or r.get("conversation_id") or r.get("id")
            style = pref_to_style(r.get("direction"), r.get("preference_label"))
            if cid is not None and style is not None:
                per_sample_preds[str(cid)][pid] = style

    # Calculate overall sensitivity across all personas
    sensitivity = persona_sensitivity_pairwise(dict(per_sample_preds))

    return {
        "personas": persona_results,
        "overall": {
            "sensitivity": sensitivity
        }
    }
