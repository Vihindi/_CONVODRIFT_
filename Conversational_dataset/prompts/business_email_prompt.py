
PROMPT="""
You are an Email Writing Assistant.

Generate only 1 conversation about a BUSINESS EMAIL in the sub-domain: "{domain}".


The FIRST prompt describes {scenario}. Think out of the box and think about a scenario related to this and write prompt.
- When asking the first prompt, phrase it naturally as if a human user is requesting help from an AI assistant to draft the email by explaining the context and purpose of the email and any key details to include.
- Avoid using placeholders like [company], [lastname], [date] — always substitute with natural details..
- The conversation always should be as a conversation between a human user and an AI assistant. so when provding the prompt, make sure it sounds like a human.

STRICT REQUIREMENTS:
- It is a MUST to Output exactly 6 prompt–response pairs. NOT 5 or 7 It MUST be 6 pairs.
- When generating the responses for the given prompt ensure that you are maintaing the context without dropping any important details mentioned in a previosu prompt.
- Pair 1:
  - "prompt": the scenario request to write an email, phrased naturally as if a user is asking an AI to draft it.
  - "response": a COMPLETE, professional email for that scenario.
  - "drift": false
- Pairs 2..6:
  - "prompt": organically written feedback or revision requests. Vary phrasing naturally. Do NOT repeat templates.
  - "response": rewrite the FULL email according to the new prompt.
  - "drift": true ONLY if the tone actually changes from formal to casual or vice versa; otherwise false.
- Keep content strictly business-appropriate and unbiased.
- It is a MUST that the FIRST pair's "drift" is false, and Maximum 3 and Minimum 2 of the subsequent pairs MUST have "drift" as true. Others should be none tone changes. Keep in mind that this none tone changes and tone changes should be distributed organically across pairs 2 to 6.
- Ensure you output **only a single JSON object** (no code fences, no prose, no comments).
- Double checkwhether you have generated exactly 6 pairs. If not, regenerate.
You MUST always be stick with the following output format. DO NOT deviate from this exact JSON schema and structure:
FORMAT (output a single JSON object and nothing else):
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