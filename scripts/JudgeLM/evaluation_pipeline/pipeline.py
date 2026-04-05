#!/usr/bin/env python3
import time
import sys
from loguru import logger
from .config import (
    MAX_CONSECUTIVE_ERRORS, 
    ERROR_LOG_FILE, 
    EVALUATION_MODE,
    TARGET_IDS,
    ID_RANGE,
    DEBUG_MODE
)
from .data_manager import data_manager, get_pending_conversations, get_evaluated_ids
from .judge_client import JudgeClient
from .prompt_templates import SYSTEM_PROMPT, construct_user_message

def log_error_to_file(msg: str):
    """
    Appends an error message with a timestamp to the configured error log file.
    
    Args:
        msg (str): The error message to log.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(ERROR_LOG_FILE, 'a', encoding='utf-8', errors='replace') as f:
            f.write(f"[{timestamp}] {msg}\n")
    except IOError as e:
        logger.error(f"Failed to write to error log file {ERROR_LOG_FILE}: {e}")

def main():
    """
    Main entry point for the LLM evaluation pipeline.
    Iterates through pending conversations, applies filtering, constructs prompts 
    using few-shot examples, and saves LLM evaluation results.
    """
    logger.info("Initializing JudgeLM Evaluation Pipeline...")
    
    # Initialize data manager and load indexes
    try:
        data_manager._load_indexes()
    except Exception as e:
        logger.critical(f"Critical failure: Could not load metadata indexes. {e}")
        sys.exit(1)

    client = JudgeClient()
    evaluated_ids = get_evaluated_ids()
    logger.info(f"Loaded {len(evaluated_ids)} already evaluated conversation IDs.")
    
    consecutive_errors = 0
    processed_count = 0
    
    # Main Processing Loop
    for convo in get_pending_conversations():
        # Extract core conversation metadata
        convo_data = convo.get("data", convo) if isinstance(convo, dict) else {}
        cid = str(convo_data.get("conversation_id") or convo_data.get("convo_ID", ""))

        if not cid:
            logger.warning("Skipping entry with missing conversation ID.")
            continue

        # --- Scope Filtering ---
        if EVALUATION_MODE == "list":
            if cid not in TARGET_IDS:
                continue
        elif EVALUATION_MODE == "range":
            start_id, end_id = ID_RANGE
            # Lexicographical comparison (requires consistent zero-padding)
            if not (start_id <= cid <= end_id):
                continue
        # Deduplication check
        if cid in evaluated_ids:
            logger.debug(f"Skipping {cid} (already evaluated).")
            continue

        logger.info(f"Evaluating conversation: {cid}")
        
        domain = convo_data.get("domain", "General")
        try:
            # 1. Retrieve domain-specific examples for few-shot prompting
            examples = data_manager.get_domain_examples(domain, cid)
            
            # 2. Construct the final user prompt
            user_msg = construct_user_message(convo_data, examples)
            
            if DEBUG_MODE:
                logger.debug(f"\nPROMPT FOR {cid}:\n{user_msg}\n")

            # 3. Call the LLM judge via the client
            result = client.evaluate(SYSTEM_PROMPT, user_msg)
            
            if result:
                # 4. Consolidate evaluation result with original data
                output_entry = {
                    "evaluation_status": "success",
                    "timestamp": time.time(),
                    "conversation_id": cid,
                    "domain": domain,
                    "metrics": {
                        "ratings": result.get("ratings"),
                        "refined_drift": result.get("refined_drift_label") or result.get("drift"),
                        "refined_direction": result.get("refined_direction_label") or result.get("direction"),
                        "rationale": result.get("feedbacks")
                    },
                    "original_data": convo_data
                }
                
                # 5. Persistent storage
                data_manager.save_result(output_entry)
                evaluated_ids.add(cid)
                consecutive_errors = 0
                processed_count += 1
                logger.success(f"Saved evaluation for {cid}")
            else:
                raise ValueError("LLM judge returned an empty response or failed to parse.")

        except Exception as e:
            consecutive_errors += 1
            error_msg = f"Failed to evaluate {cid}: {str(e)}"
            logger.error(error_msg)
            log_error_to_file(error_msg)
            
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                logger.critical(f"Stopped pipeline: {consecutive_errors} consecutive failures.")
                break
        
        # Add slight delay to respect API rate limits (can be adjusted)
        time.sleep(0.5)

    logger.info(f"Pipeline execution complete. Total conversations evaluated: {processed_count}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
