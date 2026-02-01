PROMPT = """
You are a Casual Email Writing Assistant.

Generate only 1 conversation about a CASUAL EMAIL in the sub-domain: "{domain}".

The FIRST prompt describes {scenario}. Think out of the box and think about a scenario related to this and write prompt.
- When asking the first prompt, phrase it naturally as if a human user is asking an AI assistant to help draft an email message. 
- Keep the style personal, warm, and informal (not corporate). 
- Avoid placeholders like [name], [date] — always substitute with realistic details (friend’s name, city, hobbies, etc.).
- The conversation always should be shown as if between a human user and an AI assistant. So when writing the prompt, make sure it sounds like a human.

STRICT REQUIREMENTS:
- MUST output exactly 6 prompt–response pairs.
- When generating the responses for the given prompt ensure that you are maintaing the context without dropping any important details mentioned in a previosu prompt.
- Pair 1:
  - "prompt": the scenario request to write the email, phrased naturally.
  - "response": a COMPLETE, friendly email for that scenario.
  - "drift": false
- Pairs 2..6:
  - "prompt": organically written feedback or revision requests. Vary how they’re phrased, sound casual.
  - "response": rewrite the FULL email accordingly.
  - "drift": true ONLY if the tone noticeably changes in tone up and down. (e.g., from lighthearted to more heartfelt, or from casual to slightly tone up like semi formal or vice versa. ). Otherwise false.
- The FIRST pair's "drift" must be false. Among pairs 2–6, 2–3 must have drift true, the others false. They should be distributed organically, not in a pattern.
- Keep the content strictly casual/friendly, with no formal business language.
- Output ONLY a single JSON object (no code fences, no prose).
- When stating DO NOT call Hi AI, or anyother greeting. just ask for the email directly.

FORMAT (output a single JSON object and nothing else): YOU MUST FOLLOW THIS EXACT FORMAT:
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