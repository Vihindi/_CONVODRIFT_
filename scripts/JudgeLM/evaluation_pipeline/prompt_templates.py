import json
from typing import Any, Dict, List, Optional, Tuple

# --- Scoring Guidelines & Evaluation Persona ---
SCORING_CRITERIA_TEXT = """
ROLE: You are an expert conversation evaluator specializing in style drift and linguistic naturalness.

EVALUATION CRITERIA (Q1-Q8):

Q1: Tone Consistency & Drift Annotation Accuracy
- Drift Definition: Detects if the prompt explicitly or implicitly requests a shift in tone (Casual <-> Formal).
- IF prompt requests tone shift -> drift=true.
- IF prompt requests ONLY content/structure updates -> drift=false.
- Scoring: Start at 5. Subtract 1 point for each incorrect drift label (index 0 MUST be false). Min 1.

Q2: Direction Label Correctness (Response-based, where drift=true)
- Direction Definition: Compares CURRENT response tone against PREVIOUS response tone.
- 1 = Shift towards Casual, 2 = Shift towards Formal, 0 = No clear style change.
- Scoring: Start at 5. Subtract 1 point for each missing/incorrect direction label on drift=true pairs. Min 1.

Q3: Realistic Drift Distribution
- Are drifts occurred at natural points in the dialogue? Unnatural flips or clustered endings reduce quality.
- Scoring: Start at 5. Subtract points for clearly unnatural flows.

Q4: Training Reliability (Overall Quality)
- Rounded average of (Q1 + Q2), adjusted by ±1 for extreme qualitative edge cases.

Q5: Fluency & Clarity
- General linguistic quality. Subtract points for repetitive or awkward phrasing.

Q6: Coherence Across Conversation
- Do responses logically flow and maintain context? Subtract points for contradictions.

Q7: Task Relevance (Prompt-Following)
- Did the response satisfy the prompt instructions (length, structure, tone)?

Q8: Domain-Specific Naturalness
- Do responses fit the genre (e.g., Business Email vs. Tweet)? Subtract points for genre-mismatched artifacts.

CRITICAL RULES:
- The first pair (index 0) is the INITIAL state. Drift and direction must be ignored/skipped.
- Provide feedback ONLY if a score is below 5 or a label correction is made.
"""

SYSTEM_PROMPT = f"""
{SCORING_CRITERIA_TEXT}

OUTPUT FORMAT:
Return ONLY a valid JSON object with the following schema:
{{
  "ratings": {{
    "Q1": <int>, "Q2": <int>, "Q3": <int>, "Q4": <int>,
    "Q5": <int>, "Q6": <int>, "Q7": <int>, "Q8": <int>
  }},
  "refined_drift_label": [null, <bool>, ...], // null if original is correct, true/false if correction needed
  "refined_direction_label": [null, "<0|1|2>", ...], // null if correct, "0"|"1"|"2" if correction needed
  "feedbacks": {{
    "Q1": "...", "Q2": "...", "Q3": "...", "Q4": "...",
    "Q5": "...", "Q6": "...", "Q7": "...", "Q8": "..."
  }}
}}
"""

def generate_sparse_labels(pairs: List[Dict[str, Any]], 
                           refined_drift: List[Optional[bool]], 
                           refined_direction: List[Optional[str]]) -> Tuple[List[Optional[bool]], List[Optional[str]]]:
    """
    Creates sparse lists representing human corrections. 
    Labels are 'null' if the original matches the human validator's judgment.
    """
    sparse_drift = []
    sparse_direction = []
    
    for i, pair in enumerate(pairs):
        # Drift Comparison
        orig_drift = pair.get("drift")
        human_drift = refined_drift[i] if i < len(refined_drift) else None
        
        if human_drift is None or str(human_drift).lower() == str(orig_drift).lower():
             sparse_drift.append(None)
        else:
             sparse_drift.append(human_drift)
             
        # Direction Comparison
        orig_dir = str(pair.get("direction", "")).lower()
        human_dir = str(refined_direction[i] if i < len(refined_direction) else "").lower()
        
        # Normalize "null" and None
        norm_orig = None if orig_dir in ["null", "none", ""] else orig_dir
        norm_human = None if human_dir in ["null", "none", ""] else human_dir
        
        if norm_human == norm_orig:
            sparse_direction.append(None)
        else:
            sparse_direction.append(norm_human)
            
    return sparse_drift, sparse_direction

def format_example_conversation(data: Dict[str, Any]) -> str:
    """
    Formats a human-validated conversation to show corrections in-line for LLM learning.
    """
    pairs = data.get("pairs", [])
    refined_drift = data.get("refined_drift_label", [])
    refined_direction = data.get("refined_direction_label", [])
    
    annotated_pairs = []
    for i, pair in enumerate(pairs):
        p_obj = {
            "prompt": pair.get("prompt"),
            "response": pair.get("response")
        }
        
        # In-line Drift annotation
        orig_d = pair.get("drift")
        ref_d = refined_drift[i] if i < len(refined_drift) else None
        if ref_d is not None and str(ref_d).lower() != str(orig_d).lower():
             p_obj["drift"] = f"[CORRECTION] Original: {orig_d}, Human: {ref_d}"
        else:
             p_obj["drift"] = orig_d

        # In-line Direction annotation
        orig_dir = pair.get("direction")
        ref_dir = refined_direction[i] if i < len(refined_direction) else None
        if ref_dir is not None and str(ref_dir).lower() != str(orig_dir).lower():
             p_obj["direction"] = f"[CORRECTION] Original: {orig_dir}, Human: {ref_dir}"
        else:
             p_obj["direction"] = orig_dir
             
        annotated_pairs.append(p_obj)
        
    return json.dumps(annotated_pairs, indent=2)

def construct_user_message(target_convo: Dict[str, Any], examples: List[Dict[str, Any]]) -> str:
    """
    Constructs the final user message consisting of few-shot examples 
    and the target conversation for evaluation.
    """
    msg_parts = []
    
    # Section: Few-Shot Examples
    msg_parts.append("--- EXAMPLES OF HUMAN EVALUATIONS ---")
    msg_parts.append("Study how human experts corrected drift and direction labels based on the criteria.\n")
    
    for i, ex in enumerate(examples, 1):
        msg_parts.append(f"EXAMPLE {i}:")
        msg_parts.append(f"Conversation Data:\n{format_example_conversation(ex)}")
        
        # Calculate sparse representation for the example's ground truth
        sp_drift, sp_dir = generate_sparse_labels(
            ex.get("pairs", []),
            ex.get("refined_drift_label", []),
            ex.get("refined_direction_label", [])
        )
        
        ground_truth = {
            "ratings": ex.get("ratings", {}),
            "refined_drift_label": sp_drift,
            "refined_direction_label": sp_dir,
            "feedbacks": ex.get("feedbacks", {})
        }
        msg_parts.append(f"Expert Output JSON:\n{json.dumps(ground_truth, indent=2)}\n")
        
    # Section: Target Conversation
    msg_parts.append("--- TARGET CONVERSATION FOR EVALUATION ---")
    msg_parts.append("Evaluate the following conversation according to the instructions in Section 1.")
    target_json = json.dumps(target_convo.get("pairs", []), indent=2)
    msg_parts.append(f"{target_json}\n")
    
    msg_parts.append("Provide your evaluation in the specified JSON format.")
    
    return "\n".join(msg_parts)

