
PROMPT = """
You are a Tweet Writing Assistant.

Generate only 1 conversation about a TWEET in the sub-domain: "{domain}".

The FIRST prompt describes {scenario}. Think out of the box and think about a scenario related to this and write prompt.
- The first prompt must clearly explain the scenario in natural language, as a user is asking an AI assistant to help draft the tweet. 
- The first response must be a complete tweet (max 280 characters) for that scenario.
- Avoid placeholders like [topic], [audience], [event] — always substitute with realistic details.

STRICT REQUIREMENTS:
- MUST output exactly 6 prompt–response pairs.
- When generating the responses for the given prompt ensure that you are maintaing the context without dropping any important details mentioned in a previosu prompt.
- Pair 1:
  - "prompt": the request to write the tweet for the scenario.
  - "response": the full tweet for that scenario.
  - "drift": false
- Pairs 2..6:
  - "prompt": organically written feedback or revision requests for fine-tuning the original tweet. Phrase them naturally and vary how they are asked.
  - "response": rewrite the FULL tweet accordingly (still max 280 characters).
  - "drift": true ONLY if the tone changes noticeably in formal ↔ casual(which means tone up and downs). For all other refinements, "drift": false.
- The FIRST pair’s "drift" must always be false.
- Among pairs 2–6, at least 2 and at most 3 must have drift=true. Others false. They should be distributed naturally, not in a strict sequence.
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