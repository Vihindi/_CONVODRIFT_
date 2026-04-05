
import json

def test_logic():
    # Mock data based on user example and file inspection
    original_pairs = [
        {"drift": False, "direction": None}, # Pair 0
        {"drift": False, "direction": None}, # Pair 1
        {"drift": False, "direction": None}, # Pair 2
        {"drift": True, "direction": 1},     # Pair 3
        {"drift": False, "direction": None}, # Pair 4
        {"drift": True, "direction": 1}      # Pair 5
    ]
    

    refined_drift_source = [
        None,   # Index 0
        "",     # Index 1 (Pair 0) - Orig: False. Match.
        "true", # Index 2 (Pair 1) - Orig: False. Mismatch.
        "false",# Index 3 (Pair 2) - Orig: False. Match.
        "false",# Index 4 (Pair 3) - Orig: True. Mismatch.
        "",     # Index 5 (Pair 4) - Orig: False. Match.
        ""      # Index 6 (Pair 5) - Orig: True. Mismatch? If ""=False, then Mismatch.
    ]
    

    refined_direction_source = [
        None,
        "",
        "1",
        "",
        "0", # Explicit 0
        "",
        ""
    ]
    
    print("Testing Logic...")
    
    new_refined_drift = []
    
    # Handle Drift
    for i in range(1, 7):
        pair_idx = i - 1
        if pair_idx >= len(original_pairs):
            break
            
        orig_drift_bool = original_pairs[pair_idx]["drift"]
        refined_val = refined_drift_source[i]
        
        # Normalize Refined to Boolean for comparison
        # "" -> False, "false" -> False, "true" -> True
        if refined_val is None:
            refined_bool = False # Treat None as False/Empty
        else:
            r_str = str(refined_val).lower().strip()
            if r_str == "true":
                refined_bool = True
            else:
                refined_bool = False
                
        # Compare
        if orig_drift_bool == refined_bool:
            # Agree -> Keep empty
            new_val = ""
        else:

            if refined_val is None:
                new_val = ""
            else:
                new_val = refined_val
                
        new_refined_drift.append(new_val)
        
        print(f"Pair {pair_idx}: Orig={orig_drift_bool}, RefinedSource='{refined_val}' -> RefinedBool={refined_bool}. Match={orig_drift_bool==refined_bool}. Output='{new_val}'")

    print("\nResult Drift List (Length 6):", new_refined_drift)
    
if __name__ == "__main__":
    test_logic()
