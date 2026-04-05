import json
import os
import pandas as pd
import simpledorff

# Configuration: Mapping of raters to their respective JSONL file paths
FILES = {
    "Chandi": r"",
    "Pamoda": r"",
    "Gayani": r""
}

# Limit processing to a specific number of records
ROW_LIMIT = 175

def normalize_drift(val):
    """
    Normalizes drift label values into a consistent string format ("True" or "False").

    Args:
        val: The raw drift value (bool, str, or None).

    Returns:
        str: Normalized "True", "False", or None if the input is None.
    """
    if val is None:
        return None
    if isinstance(val, bool):
        return "True" if val else "False"
    if isinstance(val, str):
        v = val.lower().strip()
        if v == "true":
            return "True"
        if v == "false":
            return "False"
    return str(val)

def normalize_direction(val, drift_val):
    """
    Normalizes direction label values and handles "NA" cases based on drift.

    Args:
        val: The raw direction value.
        drift_val (str): The normalized drift value for the same pair.

    Returns:
        str: Normalized integer string or "NA".
    """
    # If drift is False, the direction is inherently not applicable
    if drift_val == "False":
        return "NA"
    
    if val is None or val == "":
        return "NA"
    
    # Coerce to integer string by taking everything before a decimal point
    return str(val).split(".")[0]

def extract_annotations():
    """
    Parses the configured JSONL files and extracts drift and direction annotations.

    Returns:
        tuple: (list of drift annotations, list of direction annotations)
    """
    data_drift = []
    data_direction = []

    for rater, filepath in FILES.items():
        if not filepath or not os.path.exists(filepath):
            if filepath:
                print(f"WARNING: File not found for {rater}: {filepath}")
            continue
            
        print(f"Processing {rater} labels...")
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # Process only up to the defined row limit
        subset = lines[:ROW_LIMIT]
        
        for line_idx, line in enumerate(subset):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            data = obj.get("data", {})
            conversation_id = data.get("conversation_id", f"unknown_{line_idx}")
            pairs = data.get("pairs", [])
            
            refined_drift_list = data.get("refined_drift_label", [])
            refined_direction_list = data.get("refined_direction_label", [])
            
            for p_idx, pair in enumerate(pairs):
                unit_id = f"{conversation_id}_{p_idx}"
                
                # --- DRIFT EXTRACTION ---
                # Prioritize refined labels if available
                ref_idx = p_idx + 1
                refined_drift_val = None
                if ref_idx < len(refined_drift_list):
                    r_val = refined_drift_list[ref_idx]
                    if r_val is not None and str(r_val).strip() != "":
                        refined_drift_val = r_val
                
                final_drift_val = refined_drift_val if refined_drift_val is not None else pair.get("drift")
                norm_drift = normalize_drift(final_drift_val)
                
                if norm_drift is not None:
                    data_drift.append({
                        "unit_id": unit_id,
                        "annotator_id": rater,
                        "annotation": norm_drift
                    })
                
                # --- DIRECTION EXTRACTION ---
                # Prioritize refined labels if available
                refined_dir_val = None
                if ref_idx < len(refined_direction_list):
                    r_val = refined_direction_list[ref_idx]
                    if r_val is not None and str(r_val).strip() != "":
                        refined_dir_val = r_val
                
                final_dir_val = refined_dir_val if refined_dir_val is not None else pair.get("direction")
                norm_dir = normalize_direction(final_dir_val, norm_drift)
                
                if norm_dir is not None:
                    data_direction.append({
                        "unit_id": unit_id,
                        "annotator_id": rater,
                        "annotation": norm_dir
                    })
    
    return data_drift, data_direction

def main():
    """
    Main entry point for calculating and saving Krippendorff's Alpha metrics.
    """
    data_drift, data_direction = extract_annotations()

    output_results = []

    # Calculate Alpha for Drift
    df_d = pd.DataFrame(data_drift)
    if not df_d.empty:
        alpha_d = simpledorff.calculate_krippendorffs_alpha_for_df(
            df_d,
            experiment_col="unit_id",
            annotator_col="annotator_id",
            class_col="annotation"
        )
        msg_d = f"Drift Alpha: {alpha_d}"
    else:
        msg_d = "No drift data extracted."
    
    print(msg_d)
    output_results.append(msg_d)

    # Calculate Alpha for Direction
    df_dir = pd.DataFrame(data_direction)
    if not df_dir.empty:
        try:
            alpha_dir = simpledorff.calculate_krippendorffs_alpha_for_df(
                df_dir,
                experiment_col="unit_id",
                annotator_col="annotator_id",
                class_col="annotation"
            )
            msg_dir = f"Direction Alpha: {alpha_dir}"
        except Exception as e:
            msg_dir = f"Error calculating direction alpha: {e}"
    else:
        msg_dir = "No direction data extracted."
    
    print(msg_dir)
    output_results.append(msg_dir)

    # Save results to file
    with open("final_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output_results) + "\n")

if __name__ == "__main__":
    main()
