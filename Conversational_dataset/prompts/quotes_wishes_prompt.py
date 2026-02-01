PROMPT = """
You are a Quotes & Wishes Writing Assistant.

Generate only 1 conversation about a QUOTE or WISH in the sub-domain: "{domain}".

The FIRST prompt describes {scenario}. Think out of the box and think about a scenario related to this and write prompt.
- The first prompt must clearly explain the scenario in natural language, as if a user is asking an AI assistant to help create a quote or wish. 
- The first response must be a complete quote or wish for that scenario.
- Avoid placeholders like [topic], [audience], [event] — always substitute with realistic details.
STRICT REQUIREMENTS:
- MUST output exactly 6 prompt–response pairs.
- When generating the responses for the given prompt ensure that you are maintaing the context without dropping any important details mentioned in a previosu prompt.
- Pair 1:
  - "prompt": the request to write the quote or wish for the scenario.
  - "response": the full quote or wish for that scenario.
  - "drift": false
- Pairs 2..6:
  - "prompt": organically written feedback or revision requests for fine-tuning the original. Phrase them naturally and vary how they are asked.
  - "response": rewrite the FULL quote or wish accordingly.
  - "drift": true ONLY if the tone changes noticeably (e.g., formal ↔ casual, lighthearted ↔ heartfelt). For all other refinements, "drift": false.
- The FIRST pair’s "drift" must always be false.
- Among pairs 2–6, at least 2 and at most 3 must have drift=true. Others false. They should be distributed naturally, not in a strict sequence.
- Do not output examples of fine-tuning types. Only follow the user prompts.
- Output ONLY a single JSON object (no prose, no code fences).

FORMAT:
{{
  "conversation_id": "conv_XXXX",
  "domain": "{domain}",
  "pairs": [
    {{
      "prompt": "string",
      "response": "string",
      "drift": false
    }},
    {{
      "prompt": "string",
      "response": "string",
      "drift": true | false
    }},
    {{
      "prompt": "string",
      "response": "string",
      "drift": true | false
    }},
    {{
      "prompt": "string",
      "response": "string",
      "drift": true | false
    }},
    {{
      "prompt": "string",
      "response": "string",
      "drift": true | false
    }},
    {{
      "prompt": "string",
      "response": "string",
      "drift": true | false
    }}
  ]
}}
""".strip()