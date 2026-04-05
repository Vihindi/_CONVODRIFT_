#!/usr/bin/env python3
from __future__ import annotations
import argparse
import copy
import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

def _is_jsonl(path: str) -> bool:
    """Checks if a file path points to a JSONL file."""
    return path.lower().endswith(".jsonl")

def load_records(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Loads records from a JSONL file or a JSON array, indexed by conversation ID.

    Args:
        path (str): Path to the input file.

    Returns:
        dict: A mapping of conversation_id to the record object.
    """
    recs: Dict[str, Dict[str, Any]] = {}

    def extract_cid(obj: Dict[str, Any]) -> str:
        """Helper to extract conversation ID from various nested structures."""
        if "conversation_id" in obj and obj["conversation_id"]:
            return str(obj["conversation_id"])
        if isinstance(obj.get("data"), dict) and obj["data"].get("conversation_id"):
            return str(obj["data"]["conversation_id"])
        raise ValueError("Missing conversation_id")

    with open(path, "r", encoding="utf-8") as f:
        if _is_jsonl(path):
            for ln, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                try:
                    recs[extract_cid(obj)] = obj
                except ValueError as e:
                    raise ValueError(f"{path}:{ln} {e}")
        else:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"{path} must be a JSON array or JSONL file")
            for i, obj in enumerate(data):
                if not isinstance(obj, dict):
                    raise ValueError(f"{path}[{i}] is not an object")
                try:
                    recs[extract_cid(obj)] = obj
                except ValueError as e:
                    raise ValueError(f"{path}[{i}] {e}")

    return recs

def get_list_field(rec: Dict[str, Any], key: str) -> List[Any]:
    """Retrieves a list field from a record, ensuring it is indeed a list."""
    v = rec.get(key)
    if v is None:
        return []
    if not isinstance(v, list):
        raise ValueError(f"Field '{key}' must be a list; got {type(v)}")
    return v

def normalize_bool(x: Any) -> Optional[bool]:
    """Normalizes various truthy/falsy inputs into a boolean or None."""
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        if x == 0: return False
        if x == 1: return True
        return None
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "t", "yes", "y", "1"}: return True
        if s in {"false", "f", "no", "n", "0"}: return False
    return None

def normalize_direction_raw(x: Any) -> Optional[str]:
    """Normalizes direction labels into standard strings."""
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return str(int(x)) if int(x) == x else str(x)
    if isinstance(x, str):
        s = x.strip()
        return s if s else None
    return None

def try_parse_int(s: Optional[str]) -> Optional[int]:
    """Tries to parse a string into an integer, returning None on failure."""
    if s is None:
        return None
    try:
        return int(str(s).strip())
    except (ValueError, TypeError):
        return None

def _vote_counts(values: List[Any]) -> Dict[str, int]:
    """Returns a frequency count of non-null values."""
    cleaned = [v for v in values if v is not None]
    c = Counter(cleaned)
    return {str(k): int(v) for k, v in c.items()}

def median_numeric(values: List[Optional[int]]) -> Tuple[Optional[int], bool, Dict[str, int]]:
    """
    Computes a median for numeric labels among 3 annotators.
    Returns (winner_value, has_conflict_flag, counts_metadata).
    """
    cleaned = [v for v in values if v is not None]
    counts = _vote_counts(values)

    # Require at least 2 non-null values
    if len(cleaned) < 2:
        return None, True, counts

    # If exactly 2 values disagree, it's a tie/conflict
    if len(cleaned) == 2 and cleaned[0] != cleaned[1]:
        return None, True, counts

    # With 3 values, median is the middle one after sorting
    cleaned_sorted = sorted(cleaned)
    mid_val = cleaned_sorted[len(cleaned_sorted) // 2]
    return mid_val, False, counts

def mode_majority(values: List[Any]) -> Tuple[Optional[Any], bool, Dict[str, int]]:
    """
    Computes a simple majority (mode) for labels.
    """
    cleaned = [v for v in values if v is not None]
    counts = _vote_counts(values)

    if len(cleaned) < 2:
        return None, True, counts

    c = Counter(cleaned)
    top_val, top_n = c.most_common(1)[0]
    # Majority requires at least 2 votes out of 3
    if top_n >= 2:
        return top_val, False, counts

    return None, True, counts

def aggregate_drift(values_bool: List[Optional[bool]], method: str) -> Tuple[Optional[bool], bool, Dict[str, int]]:
    """
    Aggregates binary drift labels using either median or majority.
    """
    if method not in {"median", "majority"}:
        raise ValueError("drift_agg must be 'median' or 'majority'")

    if method == "majority":
        return mode_majority(values_bool)

    # Median over {0, 1}
    mapped = [(1 if v else 0) if v is not None else None for v in values_bool]
    med, conflict, counts = median_numeric(mapped)
    if med is None:
        return None, True, counts
    return (med == 1), conflict, counts

def aggregate_direction(values_dir: List[Optional[str]], method: str) -> Tuple[Optional[str], bool, Dict[str, int]]:
    """
    Aggregates direction labels. Fallbacks to majority if median is not numerically applicable.
    """
    if method not in {"median", "majority"}:
        raise ValueError("direction_agg must be 'median' or 'majority'")

    if method == "majority":
        winner, conflict, counts = mode_majority(values_dir)
        return (str(winner) if winner is not None else None), conflict, counts

    raw_non_null = [v for v in values_dir if v is not None]
    parsed = [try_parse_int(v) for v in values_dir]
    parsed_non_null = [v for v in parsed if v is not None]

    # Fallback to majority if labels aren't numeric
    if len(raw_non_null) >= 2 and len(parsed_non_null) < len(raw_non_null):
        winner, conflict, counts = mode_majority(values_dir)
        return (str(winner) if winner is not None else None), conflict, counts

    med, conflict, counts = median_numeric(parsed)
    return (str(med) if med is not None else None), conflict, counts

def main():
    """
    Main logic for merging refined annotations from three raters using median/majority voting.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="Annotator A file (jsonl or json array)")
    ap.add_argument("--b", required=True, help="Annotator B file (jsonl or json array)")
    ap.add_argument("--c", required=True, help="Annotator C file (jsonl or json array)")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--max_turns", type=int, default=None, help="Force number of turns")
    ap.add_argument("--drift_agg", choices=["median", "majority"], default="median")
    ap.add_argument("--direction_agg", choices=["median", "majority"], default="median")
    ap.add_argument("--keep_annotator_labels", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    A, B, C = load_records(args.a), load_records(args.b), load_records(args.c)
    common_ids = sorted(set(A.keys()) & set(B.keys()) & set(C.keys()))
    if not common_ids:
        raise RuntimeError("No overlapping conversation IDs found across the three files.")

    merged_path = os.path.join(args.out_dir, "merged_median.jsonl")
    drift_conf_turns, dir_conf_turns = [], []
    drift_conf_ids, dir_conf_ids = set(), set()

    with open(merged_path, "w", encoding="utf-8") as out_f:
        for cid in common_ids:
            ra, rb, rc = A[cid], B[cid], C[cid]

            drift_a = [normalize_bool(x) for x in get_list_field(ra, "refined_drift_label")]
            drift_b = [normalize_bool(x) for x in get_list_field(rb, "refined_drift_label")]
            drift_c = [normalize_bool(x) for x in get_list_field(rc, "refined_drift_label")]

            dir_a = [normalize_direction_raw(x) for x in get_list_field(ra, "refined_direction_label")]
            dir_b = [normalize_direction_raw(x) for x in get_list_field(rb, "refined_direction_label")]
            dir_c = [normalize_direction_raw(x) for x in get_list_field(rc, "refined_direction_label")]

            n_turns = args.max_turns or max(len(drift_a), len(drift_b), len(drift_c), 
                                           len(dir_a), len(dir_b), len(dir_c))

            final_drift, final_dir = [], []
            for t in range(n_turns):
                votes_drift = [
                    drift_a[t] if t < len(drift_a) else None,
                    drift_b[t] if t < len(drift_b) else None,
                    drift_c[t] if t < len(drift_c) else None,
                ]

                d_winner, d_conflict, d_counts = aggregate_drift(votes_drift, args.drift_agg)
                final_drift.append(d_winner)
                if d_conflict:
                    drift_conf_ids.add(cid)
                    drift_conf_turns.append({"conversation_id": cid, "turn": t, "votes": d_counts})

                # Direction is only aggregated if the finalized drift is True
                if d_winner is True:
                    votes_dir = [
                        dir_a[t] if t < len(dir_a) else None,
                        dir_b[t] if t < len(dir_b) else None,
                        dir_c[t] if t < len(dir_c) else None,
                    ]
                    dir_winner, dir_conflict, dir_counts = aggregate_direction(votes_dir, args.direction_agg)
                    final_dir.append(dir_winner)
                    if dir_conflict:
                        dir_conf_ids.add(cid)
                        dir_conf_turns.append({"conversation_id": cid, "turn": t, "votes": dir_counts})
                else:
                    final_dir.append(None)

            # Build merged record (using record A as base)
            merged = copy.deepcopy(ra)
            # Ensure finalized labels are added back (commented out in original, but keeping logic clean)
            # merged["final_refined_drift"] = final_drift
            # merged["final_refined_direction"] = final_dir

            if args.keep_annotator_labels:
                merged["annotators_refined"] = {
                    p: {"drift": get_list_field(r, "refined_drift_label"), 
                        "direction": get_list_field(r, "refined_direction_label")}
                    for p, r in [("A", ra), ("B", rb), ("C", rc)]
                }

            out_f.write(json.dumps(merged, ensure_ascii=False) + "\n")

    # Save conflict data
    def save_ids(ids, filename):
        with open(os.path.join(args.out_dir, filename), "w", encoding="utf-8") as f:
            for i in sorted(ids): f.write(i + "\n")

    def save_json(data, filename):
        with open(os.path.join(args.out_dir, filename), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    save_ids(drift_conf_ids, "conflicts_drift_ids.txt")
    save_ids(dir_conf_ids, "conflicts_direction_ids.txt")
    save_json(drift_conf_turns, "conflicts_drift_turns.json")
    save_json(dir_conf_turns, "conflicts_direction_turns.json")

    print(f"Done. Merged output at {merged_path}")

if __name__ == "__main__":
    main()
