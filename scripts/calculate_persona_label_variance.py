import json
import os
import numpy as np
import glob
from collections import defaultdict

# Constants
RESULTS_DIR = r"d:/IIT/4th year/fyp/dataset research/RLHF_datset/llm_experiments/results"
PERSONAS = ['A', 'B', 'C', 'D', 'E']

def load_persona_data(model_dir):
    """
    Loads preference labels for each persona from JSONL files in a model directory.

    Args:
        model_dir (str): Path to the directory containing model-specific persona files.

    Returns:
        dict: A mapping of persona names to another dictionary of {conversation_id: preference_label}.
    """
    persona_data = defaultdict(dict)
    
    for persona in PERSONAS:
        filename = f"labeled_dataset_Persona_{persona}.jsonl"
        filepath = os.path.join(model_dir, filename)
        
        if not os.path.exists(filepath):
            continue
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        cid = record.get('convo_ID')
                        pref = record.get('preference_label')
                        
                        if cid and pref is not None:
                            try:
                                val = float(pref)
                                if val in [1.0, 2.0]:
                                    persona_data[persona][cid] = val
                            except ValueError:
                                pass 
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            
    return persona_data

def compute_fleiss_kappa(ratings_matrix):
    """
    Computes Fleiss' Kappa for a given ratings matrix.

    Args:
        ratings_matrix (np.ndarray): An (N items x n raters) matrix of categories.

    Returns:
        float: The calculated Fleiss' Kappa score. Returns 1.0 if chance agreement is perfect.
    """
    if ratings_matrix.size == 0:
        return 0.0

    N, n = ratings_matrix.shape
    categories = [1.0, 2.0]
    k = len(categories)
    
    # Calculate frequency of each category per item
    M = np.zeros((N, k))
    for i in range(N):
        for j in range(n):
            val = ratings_matrix[i, j]
            if val == 1.0:
                M[i, 0] += 1
            elif val == 2.0:
                M[i, 1] += 1
                
    # Calculate extent of agreement (P_i)
    P_i = (np.sum(M**2, axis=1) - n) / (n * (n - 1))
    P_bar = np.mean(P_i)
    
    # Calculate expected agreement by chance (P_e)
    p_j = np.sum(M, axis=0) / (N * n)
    P_e = np.sum(p_j**2)
    
    if P_e >= 1.0:
        return 1.0
        
    kappa = (P_bar - P_e) / (1 - P_e)
    return kappa

def calculate_pairwise_kappa(persona_data):
    """
    Constructs a pairwise Fleiss' Kappa agreement matrix between all personas.

    Args:
        persona_data (dict): Mapping of persona names to {convo_id: label} mappings.

    Returns:
        np.ndarray: A 5x5 matrix containing pairwise agreement scores.
    """
    n_personas = len(PERSONAS)
    matrix = np.zeros((n_personas, n_personas))
    
    for i, p1 in enumerate(PERSONAS):
        for j, p2 in enumerate(PERSONAS):
            if i == j:
                matrix[i, j] = 1.0
                continue
                
            data1 = persona_data.get(p1, {})
            data2 = persona_data.get(p2, {})
            common_ids = sorted(list(set(data1.keys()) & set(data2.keys())))
            
            if not common_ids:
                matrix[i, j] = 0.0
                continue
                
            ratings = [[data1[cid], data2[cid]] for cid in common_ids]
            ratings_np = np.array(ratings)
            matrix[i, j] = compute_fleiss_kappa(ratings_np)
                
    return matrix

def main():
    """
    Main entry point for scanning Claude model results and generating a pairwise Kappa report.
    """
    print(f"Scanning results in: {RESULTS_DIR} for 'claude' models...")
    
    model_dirs = [d for d in glob.glob(os.path.join(RESULTS_DIR, "*")) if os.path.isdir(d)]
    claude_models = [d for d in model_dirs if "claude" in os.path.basename(d).lower()]
    
    if not claude_models:
        print("No 'claude' model directories found.")
        return

    report_path = os.path.join(RESULTS_DIR, "claude_pairwise_kappa_report.txt")
    
    with open(report_path, "w", encoding="utf-8") as f:
        for model_path in claude_models:
            model_name = os.path.basename(model_path)
            print(f"\nProcessing Model: {model_name}")
            f.write(f"Model: {model_name}\n" + "-" * 30 + "\n")
            
            data = load_persona_data(model_path)
            if not data:
                print("No data found.")
                f.write("No data found.\n\n")
                continue
                
            matrix = calculate_pairwise_kappa(data)
            
            # Write matrix header
            header = "      " + " ".join([f"{p:>8}" for p in PERSONAS])
            print(header)
            f.write(header + "\n")
            
            # Write matrix rows
            for i, row in enumerate(matrix):
                row_str = f"{PERSONAS[i]:<4} |" + " ".join([f"{val:>8.4f}" for val in row])
                print(row_str)
                f.write(row_str + "\n")
            
            f.write("\n")
            
        footer = ("\nNote: Values represent Pairwise Fleiss' Kappa.\n"
                  "1.0 = Perfect Agreement, 0.0 = Agreement by Chance, < 0 = Disagreement.")
        print(footer)
        f.write(footer + "\n")

    print(f"\nReport saved to: {report_path}")

if __name__ == "__main__":
    main()
