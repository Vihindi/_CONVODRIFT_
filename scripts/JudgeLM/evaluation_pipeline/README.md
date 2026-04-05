# LLM Evaluation Pipeline

This pipeline automatically evaluates conversations using Claude 3.5 Sonnet.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **API Key**:
    - Rename `credentials_example.py` to `credentials.py` (if not already done).
    - Open `credentials.py` and paste your Anthropic/Claude API key.

3.  **Configuration**:
    - Open `config.py`.
    - Set `EVALUATION_MODE`:
        - `"all"`: Evaluates all pending conversations.
        - `"list"`: Evaluates specific IDs in `TARGET_IDS`.
        - `"range"`: Evaluates a range of IDs in `ID_RANGE`.
    - Set `DEBUG_MODE = True` to see prompts in the console.

## Running the Validator

Run the pipeline from the project root or the `evaluation_pipeline` directory:

```bash
# From project root (recommended)
python -m evaluation_pipeline.pipeline
```

## Output

- **Evaluated Data**: Saved to `dataset/evaluated_data/LLM_evaluated_data.jsonl`.
- **Logs**: Errors are logged to `evaluation_pipeline/error_log.txt`.

## Troubleshooting

- **Missing API Key**: Ensure `credentials.py` exists and has a valid key.
- **Import Error**: Make sure you run as a module (`python -m evaluation_pipeline.pipeline`) from the parent folder (ProjectX/JudgeLM).
