import glob
import json
import os
import random
from typing import Any, Dict, Iterator, List, Optional, Set
from loguru import logger
from .config import (
    INPUT_FILE, 
    OUTPUT_FILE, 
    DOMAIN_INDEX_DIR, 
    CONVO_INDEX_DIR,
    NUM_EXAMPLES_PER_PROMPT
)

def get_pending_conversations() -> Iterator[Dict[str, Any]]:
    """
    Yields conversation objects from the pending evaluation file.
    
    Yields:
        dict: A conversation record containing 'conversation_id', 'prompt', etc.
    """
    if not os.path.exists(INPUT_FILE):
        logger.error(f"Input file not found: {INPUT_FILE}")
        return
        
    logger.info(f"Loading pending conversations from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON line {ln} in {INPUT_FILE}")

def get_evaluated_ids() -> Set[str]:
    """
    Returns a set of conversation IDs that have already been evaluated and saved to the output file.
    
    Returns:
        set: A unique set of evaluated conversation IDs.
    """
    evaluated_ids = set()
    if not os.path.exists(OUTPUT_FILE):
        return evaluated_ids

    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            try:
                data = json.loads(line)
                # Handle both wrapped {"data": {...}} and flat record formats
                target = data.get("data", data) if isinstance(data, dict) else data
                cid = target.get("conversation_id") or target.get("convo_ID")
                if cid:
                    evaluated_ids.add(str(cid))
            except json.JSONDecodeError:
                continue
    return evaluated_ids

class DataManager:
    """
    Handles the loading of conversation indexes and retrieval of domain-specific 
    examples for few-shot prompting.
    """
    def __init__(self):
        self._domain_cache: Dict[str, List[str]] = {} # domain -> [list of conversation_ids]
        self._convo_cache: Dict[str, Dict[str, Any]] = {} # conversation_id -> record
        self._indexes_loaded = False
        
    def _load_indexes(self):
        """Loads domain and conversation indexes from the local filesystem."""
        if self._indexes_loaded:
            return
        
        logger.info("Initializing metadata indexes...")
        
        # 1. Load Domain Index (mapping of domain name to list of eligible IDs)
        domain_files = glob.glob(os.path.join(DOMAIN_INDEX_DIR, "*.json"))
        for fpath in domain_files:
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    dmap = json.load(f)
                    for domain, ids in dmap.items():
                        if domain not in self._domain_cache:
                            self._domain_cache[domain] = []
                        self._domain_cache[domain].extend(ids)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load domain index {fpath}: {e}")

        # 2. Pre-load conversation data for potential few-shot examples
        convo_files = glob.glob(os.path.join(CONVO_INDEX_DIR, "*.json"))
        for fpath in convo_files:
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._convo_cache.update(data)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load convo index {fpath}: {e}")

        self._indexes_loaded = True

    def get_domain_examples(self, domain: str, exclude_id: str, n: int = NUM_EXAMPLES_PER_PROMPT) -> List[Dict[str, Any]]:
        """
        Retrieves random examples from the same domain to be used as few-shot demonstrations.

        Args:
            domain (str): The domain name (e.g., 'Health', 'Finance').
            exclude_id (str): The ID of the conversation currently being evaluated.
            n (int): Number of examples to retrieve.

        Returns:
            list: A list of conversation records.
        """
        self._load_indexes()
        
        candidates = self._domain_cache.get(domain, [])
        valid_candidates = [cid for cid in candidates if str(cid) != str(exclude_id)]
        
        if not valid_candidates:
            logger.debug(f"No secondary domain examples found for domain: {domain}")
            return []
            
        selected_ids = random.sample(valid_candidates, min(len(valid_candidates), n))
        examples = []
        for cid in selected_ids:
            convo_data = self.get_convo_by_id(cid)
            if convo_data:
                examples.append(convo_data)
        
        return examples

    def get_convo_by_id(self, conv_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a full conversation record by its ID.

        Args:
            conv_id (str): The conversation identifier.

        Returns:
            dict: The conversation record or None if not found.
        """
        self._load_indexes()
        return self._convo_cache.get(str(conv_id))

    def save_result(self, result: Dict[str, Any]):
        """
        Appends an evaluation result to the output JSONL file.

        Args:
            result (dict): The evaluated record to save.
        """
        try:
            with open(OUTPUT_FILE, 'a', encoding='utf-8', errors='ignore') as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        except IOError as e:
            logger.error(f"Failed to append result for {result.get('conversation_id')}: {e}")

# Global singleton instance for use across the pipeline
data_manager = DataManager()
