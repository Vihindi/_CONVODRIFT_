#!/usr/bin/env python3
import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Tuple

def unwrap_record(obj: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """
    Unwraps a record if it is nested within a 'data' key.
    
    Returns:
        tuple: (unwrapped_dict, was_wrapped_bool)
    """
    if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], dict):
        return obj["data"], True
    return obj, False

def rewrap_record(conv: Dict[str, Any], wrapped: bool) -> Dict[str, Any]:
    """Rewraps a record into the 'status':'ok', 'data':{...} format if required."""
    return {"status": "ok", "data": conv} if wrapped else conv

def read_jsonl(path: str) -> List[Tuple[Dict[str, Any], bool]]:
    """Reads a JSONL file and returns a list of (record, was_wrapped) tuples."""
    items_list: List[Tuple[Dict[str, Any], bool]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"ERROR: {path}:{ln} invalid JSON: {e}", file=sys.stderr)
                sys.exit(1)
            items_list.append(unwrap_record(raw))
    return items_list

def write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    """Writes a list of dictionaries to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def index_by_id(items: List[Tuple[Dict[str, Any], bool]], id_field: str) -> Dict[str, Tuple[Dict[str, Any], bool]]:
    """Indexes a list of records by a specific ID field."""
    out_map: Dict[str, Tuple[Dict[str, Any], bool]] = {}
    for obj, wrapped in items:
        if id_field not in obj:
            print(f"ERROR: Missing ID field '{id_field}'.", file=sys.stderr)
            sys.exit(1)
        cid = obj[id_field]
        if cid in out_map:
            print(f"ERROR: Duplicate ID {cid} found.", file=sys.stderr)
            sys.exit(1)
        out_map[cid] = (obj, wrapped)
    return out_map


def to_bool_or_none(x: Any) -> Optional[bool]:
    """Coerces various input types into a boolean or None."""
    if x is None or (isinstance(x, str) and not x.strip()):
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        if int(x) == 1: return True
        if int(x) == 0: return False
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "1"): return True
        if s in ("false", "0"): return False
    return None

def to_dir_or_none(x: Any) -> Optional[int]:
    """Coerces input into a direction integer (0, 1, or 2) or None."""
    if x is None or (isinstance(x, str) and not x.strip()):
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        v = int(x)
        return v if v in (0, 1, 2) else None
    if isinstance(x, str):
        s = x.strip()
        if s in ("0", "1", "2"):
            return int(s)
    return None


def align_list(lst: List[Any], target_len: int) -> List[Any]:
    """Ensures a list matches a target length by truncating or padding with None."""
    if len(lst) >= target_len:
        return lst[:target_len]
    return lst + [None] * (target_len - len(lst))

def compute_consensus(a_list: List[Any], b_list: List[Any], norm_func) -> List[Any]:
    """Generic consensus logic requiring exact matches between two annotators."""
    target_len = max(len(a_list), len(b_list))
    a = align_list(a_list, target_len)
    b = align_list(b_list, target_len)
    
    result = []
    for i in range(target_len):
        va, vb = norm_func(a[i]), norm_func(b[i])
        result.append(va if (va is not None and va == vb) else None)
    return result

def parse_rating(x: Any) -> Optional[float]:
    """Parses a generic input into a float rating."""
    if x is None or (isinstance(x, str) and not x.strip()):
        return None
    try:
        return float(x)
    except (ValueError, TypeError):
        return None

def compute_final_ratings(
    ra: Dict[str, Any], rb: Dict[str, Any], 
    final_drift: List[Any], final_dir: List[Any], 
    round_vals: bool
) -> Dict[str, Any]:
    """
    Computes final ratings (Q1-Q8) based on consensus labels and averaged scores.
    """
    out = {}
    out["Q1"] = 5 - sum(1 for v in final_drift if v is not None)
    out["Q2"] = 5 - sum(1 for v in final_dir if v is not None)

    for q in ["Q3", "Q4", "Q5", "Q6", "Q7", "Q8"]:
        va, vb = parse_rating(ra.get(q)), parse_rating(rb.get(q))
        if va is None and vb is None:
            continue
        
        m = (va + vb) / 2.0 if (va is not None and vb is not None) else (va or vb)
        out[q] = int(round(m)) if round_vals else (int(m) if m.is_integer() else m)
    return out


# -------------------- MAIN --------------------

def main():
    """
    Main entry point for merging two JSONL annotation files based on overlap and consensus.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="Annotator A JSONL source")
    ap.add_argument("--b", required=True, help="Annotator B JSONL source")
    ap.add_argument("--out", required=True, help="Output destination for merged JSONL")
    ap.add_argument("--id_field", default="conversation_id")
    ap.add_argument("--round_ratings", action="store_true", help="Round Q3-Q8 to the nearest integer")
    args = ap.parse_args()

    # Load data from both sources
    map_a = index_by_id(read_jsonl(args.a), args.id_field)
    map_b = index_by_id(read_jsonl(args.b), args.id_field)

    all_ids = sorted(set(map_a.keys()) & set(map_b.keys()))
    output_records = []

    for cid in all_ids:
        a_obj, a_wrapped = map_a[cid]
        b_obj, _ = map_b[cid]

        # Consensus for refined labels
        final_rd = compute_consensus(a_obj.get("refined_drift_label", []), 
                                    b_obj.get("refined_drift_label", []), to_bool_or_none)
        final_dir = compute_consensus(a_obj.get("refined_direction_label", []), 
                                     b_obj.get("refined_direction_label", []), to_dir_or_none)

        # Merge ratings
        final_ratings = compute_final_ratings(a_obj.get("ratings", {}), b_obj.get("ratings", {}), 
                                              final_rd, final_dir, args.round_ratings)

        # Build output record using A as the base metadata/text source
        merged = dict(a_obj)
        merged["refined_drift_label"] = final_rd
        merged["refined_direction_label"] = final_dir
        merged["ratings"] = final_ratings

        output_records.append(rewrap_record(merged, a_wrapped))

    write_jsonl(args.out, output_records)
    print(f"Succefully merged {len(output_records)} overlapping records into {args.out}")

if __name__ == "__main__":
    main()
