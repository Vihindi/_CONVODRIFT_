# CONVODRIFT: A Multi-Turn Conversational Dataset for Modeling Stylistic Tone Evolution

**ConvoDrift** is a dataset designed to model progressive conversational tone drift under fixed semantic intent. It addresses the limitation of existing NLP benchmarks that often treat style as a static property or focus on single-turn transfers. ConvoDrift explicitly models how stylistic tone evolves across multi-turn interactions.

This repository contains the codebase and datasets described in the paper:  
*CONVODRIFT: A Multi-Turn Conversational Dataset for Modeling Stylistic Tone Evolution*

## Overview

The dataset consists of two core components:

1.  **ConvoDrift-Conversation:** A multi-turn conversational dataset capturing progressive stylistic tone drift.
2.  **ConvoDrift-RLHF:** A persona-conditioned pairwise preference dataset constructed from the conversational trajectories to enable personalized and pluralistic alignment.

## Dataset Structure

The repository is organized into the following main directories:

### 1. Conversational Dataset (`Conversational_dataset/`)

This directory contains the core multi-turn conversations.

*   **Location:** `Conversational_dataset/conversational_data/`
*   **Genres:** The data is stratified across 5 communication genres:
    *   `business_emails/`
    *   `casual_emails/`
    *   `linkedin_posts/`
    *   `quotes_wishes/`
    *   `tweets/`
*   **Format:** JSONL
*   **Structure:** Each conversation consists of 6 prompt-response pairs.
    *   **Drift Label (`Drift`):** Binary (`True`/`False`). Indicates if the user prompt explicity requested a tone change.
    *   **Direction Label (`Direction`):** Ternary label indicating the direction of stylistic shift:
        *   `0`: No stylistic change.
        *   `1`: Shift towards **Casual / Expressive** tone.
        *   `2`: Shift towards **Formal / Precise** tone.

### 2. RLHF Dataset (`RLHF_dataset/`)

This directory contains preference data for Reinforcement Learning from Human Feedback (RLHF), conditioned on specific personas.

*   **Location:** `RLHF_dataset/main_labeled_personas/`
*   **Files:**
    *   `labeled_Main_final_dataset_Persona_A.jsonl`
    *   `labeled_Main_final_dataset_Persona_B.jsonl`
    *   `labeled_Main_final_dataset_Persona_C.jsonl`
    *   `labeled_Main_final_dataset_Persona_D.jsonl`
    *   `labeled_Main_final_dataset_Persona_E.jsonl`

#### Personas
The RLHF dataset uses 5 distinct personas to model pluralistic preferences:

| Persona | Description |
| :--- | :--- |
| **Task-First** | Focuses on efficiency, clarity, and structured reasoning. Prioritizes actionable outcomes. |
| **Relationship-First** | Emphasizes warmth, empathy, and cooperative language. |
| **Authority-First** | Communicates with confidence, decisiveness, and assertive tone. |
| **Calm & Careful** | Uses measured, precise language to minimize risk and ensure safety. |
| **Expressive & Energetic** | Engages with enthusiasm, vivid language, and excitement. |

## Methodology

### Data Generation
*   **Conversation Generation:** Used **GPT-4.1-mini** to generate consistent, quality multi-turn conversations where prompts progressively request stylistic refinements while preserving semantic intent.
*   **Drift/Direction Labeling:** Used **GPT-5-nano** to annotate explicit drift and direction labels.

### Evaluation
The dataset has undergone comprehensive evaluation including:
*   **Human Validation:** Using three annotators to validate drift annotations and conversation quality.
*   **LLM-as-a-Judge:** Using models like Gemini and Claude to validate label accuracy.
*   **Automatic Metrics:** Semantic similarity (SBERT) and lexical similarity (ROUGE-L, Jaccard) to ensure meaning is preserved while style changes.
