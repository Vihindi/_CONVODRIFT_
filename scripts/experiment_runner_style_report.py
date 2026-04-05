import json
import os
from collections import defaultdict
from metrics_utils_style import pref_to_style, style_entropy_from_rows, persona_sensitivity_pairwise

def load_jsonl(path):
    """
    Loads records from a JSONL file.

    Args:
        path (str): The absolute path to the JSONL file.

    Returns:
        list[dict]: A list of dictionary objects representing each line.
    """
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def main():
    """
    Main entry point for computing style entropy and persona sensitivity 
    across existing persona-labeled datasets.
    """
    # Define file paths for each persona's labeled dataset
    base_data_path = r"D:\IIT\4th year\fyp\dataset research\RLHF_datset\Main_dataset_eval\main_processed_dataset"
    persona_files = {
        "A": os.path.join(base_data_path, "labeled_sample_final_dataset_Persona_A.jsonl"),
        "B": os.path.join(base_data_path, "labeled_sample_final_dataset_Persona_B.jsonl"),
        "C": os.path.join(base_data_path, "labeled_sample_final_dataset_Persona_C.jsonl"),
        "D": os.path.join(base_data_path, "labeled_sample_final_dataset_Persona_D.jsonl"),
        "E": os.path.join(base_data_path, "labeled_sample_final_dataset_Persona_E.jsonl"),
    }

    persona_rows = {}
    for pid, path in persona_files.items():
        rows = load_jsonl(path)
        persona_rows[pid] = rows

    print("\n--- Style entropy per persona (formal vs casual) ---")
    for pid, rows in persona_rows.items():
        if not rows:
            print(f"Persona {pid}: No data found.")
            continue
        stats = style_entropy_from_rows(rows)
        print(f"Persona {pid}: used={stats['used']} dropped={stats['dropped']} "
              f"formal_rate={stats['formal_rate']:.3f} style_entropy={stats['style_entropy']:.4f} "
              f"counts={stats['style_counts']}")

    # Aggregating style preferences per sample (conversation ID)
    per_sample = defaultdict(dict)  # {convo_ID: {persona: style}}
    for pid, rows in persona_rows.items():
        for r in rows:
            sid = r.get("convo_ID")
            style = pref_to_style(r.get("direction"), r.get("preference_label"))
            if sid is not None and style is not None:
                per_sample[sid][pid] = style

    if per_sample:
        sens = persona_sensitivity_pairwise(per_sample)
        print("\n--- Persona sensitivity (pairwise disagreement on style) ---")
        print(f"Sensitivity: {sens:.4f}")
    else:
        print("\n--- Persona sensitivity ---")
        print("No valid samples for sensitivity calculation.")

if __name__ == "__main__":
    main()
