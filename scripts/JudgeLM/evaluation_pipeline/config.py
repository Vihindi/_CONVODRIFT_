"""
Configuration settings for the JudgeLM Evaluation Pipeline.
This file defines global paths, API settings, and evaluation parameters.
"""

import os

# Base directory for the JudgeLM module
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Data & Directory Paths ---
# Source file containing conversation records pending evaluation
INPUT_FILE = os.path.join(BASE_DIR, "dataset", "pending_evaluation", "final_curated_dataset_merged.jsonl")

# Destination for the LLM-evaluated dataset
OUTPUT_FILE = os.path.join(BASE_DIR, "dataset", "evaluated_data", "LLM_evaluated_final_curated_dataset_merged.jsonl")

# Directories for indexed metadata
DOMAIN_INDEX_DIR = os.path.join(BASE_DIR, "dataset", "indexed_data", "domain_index")
CONVO_INDEX_DIR = os.path.join(BASE_DIR, "dataset", "indexed_data", "convo_index")

# Evaluation error logging
ERROR_LOG_FILE = os.path.join(BASE_DIR, "evaluation_pipeline", "error_log.txt")

# --- API & Evaluation Parameters ---
# Primary LLM model used for judging
MODEL_NAME = "gemini-3-flash-preview"

# Fault tolerance: stop after N consecutive API failures
MAX_CONSECUTIVE_ERRORS = 5

# Number of few-shot examples to include in each prompt
NUM_EXAMPLES_PER_PROMPT = 3

# --- Evaluation Scope Control ---
# Modes: 
#   "all": Process the entire input file.
#   "list": Only process IDs specified in TARGET_IDS.
#   "range": Process a lexicographical range (inclusive) from ID_RANGE.
EVALUATION_MODE = "all"

# Specific IDs for "list" mode
TARGET_IDS = ["conv_0001"]

# Lower and upper bounds for "range" mode (inclusive)
# Note: Ensure IDs are zero-padded for consistent string comparison (e.g. "conv_0001")
ID_RANGE = ("conv_0002", "conv_0005")

# --- Development & Debugging ---
# If True, the full formatted prompt will be printed to stdout before each API call
DEBUG_MODE = True
