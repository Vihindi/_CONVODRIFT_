import argparse
import json
import csv
import random
from collections import defaultdict
import numpy as np

# Constant: Evaluation questions Q1 through Q8
QUESTIONS = [f"Q{i}" for i in range(1, 9)]

def extract_conversation_id(obj):
    """
    Extracts the conversation ID from a record object, supporting multiple schemas.

    Args:
        obj (dict): The JSON record object.

    Returns:
        str: The conversation ID if found, otherwise None.
    """
    if isinstance(obj, dict):
        if "conversation_id" in obj:
            return obj["conversation_id"]
        if "data" in obj and isinstance(obj["data"], dict) and "conversation_id" in obj["data"]:
            return obj["data"]["conversation_id"]
    return None

def extract_ratings(obj):
    """
    Extracts ratings from a record object, supporting various nested structures.

    Args:
        obj (dict): The JSON record object.

    Returns:
        dict: The ratings dictionary if found, otherwise None.
    """
    if not isinstance(obj, dict):
        return None

    if "ratings" in obj and isinstance(obj["ratings"], dict):
        return obj["ratings"]

    data = obj.get("data")
    if isinstance(data, dict):
        if "ratings" in data and isinstance(data["ratings"], dict):
            return data["ratings"]
        validation = data.get("validation")
        if isinstance(validation, dict) and "ratings" in validation and isinstance(validation["ratings"], dict):
            return validation["ratings"]

    return None

def load_jsonl(path, verbose=False, name="A"):
    """
    Loads ratings from a JSONL file and extracts valid Q1-Q8 integer ratings.

    Args:
        path (str): Path to the JSONL file.
        verbose (bool): Whether to print parsing statistics.
        name (str): Label for the annotator (used in verbose output).

    Returns:
        dict: A mapping of conversation_id to a dictionary of {question: rating}.
    """
    out = {}
    stats = {"lines": 0, "with_cid": 0, "with_ratings": 0, "saved": 0}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stats["lines"] += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            cid = extract_conversation_id(obj)
            if cid is None:
                continue
            stats["with_cid"] += 1

            ratings = extract_ratings(obj)
            if ratings is None:
                continue
            stats["with_ratings"] += 1

            clean = {}
            for q in QUESTIONS:
                if q in ratings and ratings[q] is not None:
                    try:
                        clean[q] = int(ratings[q])
                    except (ValueError, TypeError):
                        pass

            if clean:
                out[cid] = clean
                stats["saved"] += 1

    if verbose:
        print(f"[{name}] lines={stats['lines']} with_cid={stats['with_cid']} "
              f"with_ratings={stats['with_ratings']} saved={stats['saved']}")
        ex = list(out.keys())[:3]
        if ex:
            print(f"[{name}] example cids: {ex}")
            print(f"[{name}] example ratings for {ex[0]}: {out[ex[0]]}")
        else:
            print(f"[{name}] WARNING: no usable items parsed.")

    return out

def krippendorff_alpha_ordinal(ratings_matrix, k_min=1, k_max=5):
    """
    Calculates Krippendorff's alpha for ordinal data.

    Args:
        ratings_matrix (list[list]): Matrix of shape (n_items, n_annotators).
        k_min (int): Minimum possible rating.
        k_max (int): Maximum possible rating.

    Returns:
        float: The calculated alpha coefficient, or None if insufficient overlap.
    """
    def delta(a, b):
        return ((a - b) ** 2) / ((k_max - k_min) ** 2)

    # Observed Disagreement (Do)
    do_num = 0.0
    do_den = 0.0
    all_vals = []

    for row in ratings_matrix:
        vals = [v for v in row if v is not None]
        all_vals.extend(vals)
        m = len(vals)
        if m < 2:
            continue
        for i in range(m):
            for j in range(i + 1, m):
                do_num += delta(vals[i], vals[j])
        do_den += (m * (m - 1)) / 2.0

    if do_den == 0:
        return None

    do = do_num / do_den

    # Expected Disagreement (De)
    if len(all_vals) < 2:
        return None

    freq = defaultdict(int)
    for v in all_vals:
        freq[v] += 1

    de_num = 0.0
    de_den = 0.0
    keys = sorted(freq.keys())
    for i in range(len(keys)):
        for j in range(len(keys)):
            if i != j:
                a, b = keys[i], keys[j]
                de_num += freq[a] * freq[b] * delta(a, b)
                de_den += freq[a] * freq[b]

    if de_den == 0:
        return None

    de = de_num / de_den
    return 1.0 - (do / de) if de != 0 else 1.0

def bootstrap_ci(items, compute_fn, iters=1000, seed=42):
    """
    Computes a 95% bootstrap confidence interval for a given metric.

    Args:
        items (list): The original data items to resample.
        compute_fn (callable): Function to compute the metric on a sample.
        iters (int): Number of bootstrap iterations.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (lower bound, upper bound).
    """
    rng = random.Random(seed)
    n = len(items)
    if n == 0:
        return (None, None)
    
    boot_vals = []
    for _ in range(iters):
        sample = [items[rng.randrange(n)] for _ in range(n)]
        v = compute_fn(sample)
        if v is not None:
            boot_vals.append(v)
            
    if not boot_vals:
        return (None, None)
    
    boot_vals.sort()
    lo = boot_vals[int(0.025 * len(boot_vals))]
    hi = boot_vals[int(0.975 * len(boot_vals)) - 1]
    return (lo, hi)

def overall_weighted_alpha(results):
    """
    Computes the weighted average of alpha and its CI bounds across questions.

    Args:
        results (list[dict]): List of result dictionaries per question.

    Returns:
        dict: Weighted alpha, and weighted CI bounds if available.
    """
    total_items = 0
    weighted_sum = 0.0
    ci_low_sum = 0.0
    ci_high_sum = 0.0
    ci_available = True

    for r in results:
        alpha = r.get("alpha_ordinal")
        n = r.get("n_items_used", 1)

        if alpha is None or alpha == "":
            continue

        total_items += n
        weighted_sum += alpha * n

        if "ci_low" in r and "ci_high" in r and r["ci_low"] != "" and r["ci_high"] != "":
            ci_low_sum += r["ci_low"] * n
            ci_high_sum += r["ci_high"] * n
        else:
            ci_available = False

    if total_items == 0:
        return {"weighted_alpha": None, "ci_low": None, "ci_high": None}

    weighted_alpha = weighted_sum / total_items
    ci_low = ci_low_sum / total_items if ci_available else None
    ci_high = ci_high_sum / total_items if ci_available else None

    return {"weighted_alpha": weighted_alpha, "ci_low": ci_low, "ci_high": ci_high}

def main():
    """
    Main logic for computing question-wise Krippendorff's alpha across multiple annotators.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="Annotator A JSONL file")
    ap.add_argument("--b", required=True, help="Annotator B JSONL file")
    ap.add_argument("--c", required=True, help="Annotator C JSONL file")
    ap.add_argument("--out", default="krippendorff_questionwise.csv", help="Output CSV path")
    ap.add_argument("--bootstrap", type=int, default=0, help="Number of bootstrap iterations (0 to disable)")
    ap.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = ap.parse_args()

    A = load_jsonl(args.a, verbose=args.verbose, name="A")
    B = load_jsonl(args.b, verbose=args.verbose, name="B")
    C = load_jsonl(args.c, verbose=args.verbose, name="C")

    common_ids = sorted(set(A.keys()) & set(B.keys()) & set(C.keys()))
    if args.verbose:
        print(f"[INFO] common_ids across A,B,C: {len(common_ids)}")

    if not common_ids:
        print("\n[ERROR] No overlapping conversation_ids found across the three files.")
        return

    results = []
    for q in QUESTIONS:
        rows = []
        for cid in common_ids:
            # Gather scores; at least 2 annotators must have provided a rating
            row = [A[cid].get(q), B[cid].get(q), C[cid].get(q)]
            if sum(v is not None for v in row) >= 2:
                rows.append(row)

        n_items_used = len(rows)
        alpha = krippendorff_alpha_ordinal(rows)

        ci_low, ci_high = (None, None)
        if args.bootstrap and alpha is not None and n_items_used > 1:
            ci_low, hi = bootstrap_ci(rows, lambda s: krippendorff_alpha_ordinal(s), iters=args.bootstrap)
            ci_high = hi # workaround for variable name

        results.append({
            "question": q,
            "n_common_ids": len(common_ids),
            "n_items_used": n_items_used,
            "alpha_ordinal": alpha if alpha is not None else "",
            "ci_low": ci_low if ci_low is not None else "",
            "ci_high": ci_high if ci_high is not None else "",
        })

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote question-wise Krippendorff’s alpha (ordinal) -> {args.out}")
    print("\n=== Overall weighted average alpha ===")
    print(overall_weighted_alpha(results))

    if args.verbose:
        for r in results:
            print(r)

if __name__ == "__main__":
    main()
