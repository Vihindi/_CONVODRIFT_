# experiment_runner_style_report.py
"""
Compute persona variance metrics using *existing persona-labeled JSONL files*.

Outputs:
- Style entropy per persona (formal vs casual)
- Formal rate per persona
- Persona sensitivity (pairwise disagreement) on style labels, per sample
"""

import json
from collections import defaultdict
from metrics_utils_style import pref_to_style, style_entropy_from_rows, persona_sensitivity_pairwise


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    persona_files = {
    "A": "D:\\IIT\\4th year\\fyp\\dataset research\\RLHF_datset\\Main_dataset_eval\\main_processed_dataset\\labeled_sample_final_dataset_Persona_A.jsonl",
    "B": "D:\\IIT\\4th year\\fyp\\dataset research\\RLHF_datset\\Main_dataset_eval\\main_processed_dataset\\labeled_sample_final_dataset_Persona_B.jsonl",
    "C": "D:\\IIT\\4th year\\fyp\\dataset research\\RLHF_datset\\Main_dataset_eval\\main_processed_dataset\\labeled_sample_final_dataset_Persona_C.jsonl",
    "D": "D:\\IIT\\4th year\\fyp\\dataset research\\RLHF_datset\\Main_dataset_eval\\main_processed_dataset\\labeled_sample_final_dataset_Persona_D.jsonl",
    "E": "D:\\IIT\\4th year\\fyp\\dataset research\\RLHF_datset\\Main_dataset_eval\\main_processed_dataset\\labeled_sample_final_dataset_Persona_E.jsonl",
    }

    # 1) Per-persona style entropy
    persona_rows = {}
    for pid, path in persona_files.items():
        rows = load_jsonl(path)
        persona_rows[pid] = rows

    print("\\n--- Style entropy per persona (formal vs casual) ---")
    for pid, rows in persona_rows.items():
        stats = style_entropy_from_rows(rows)
        print(f"Persona {pid}: used={stats['used']} dropped={stats['dropped']} "
              f"formal_rate={stats['formal_rate']:.3f} style_entropy={stats['style_entropy']:.4f} "
              f"counts={stats['style_counts']}")

    # 2) Persona sensitivity on style (per sample)
    per_sample = defaultdict(dict)  # {convo_ID: {persona: style}}
    for pid, rows in persona_rows.items():
        for r in rows:
            sid = r.get("convo_ID")
            style = pref_to_style(r.get("direction"), r.get("preference_label"))
            if sid is None or style is None:
                continue
            per_sample[sid][pid] = style

    sens = persona_sensitivity_pairwise(per_sample)
    print("\\n--- Persona sensitivity (pairwise disagreement on style) ---")
    print(f"Sensitivity: {sens:.4f}")


if __name__ == "__main__":
    main()
