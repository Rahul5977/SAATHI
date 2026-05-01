import json

# 1. PERSONAS COMPATIBLE WITH C5
PERSONAS = {
    "P3": "P3 — UPSC/PSC aspirant (22–30, M/F), Multi-year prep, coaching institute life, family investment, dropper stigma.",
    "P4": "P4 — IT/software professional (24–35, M/F), Tier-1 or Tier-2 city, MNC or startup, EMI pressure, WFH isolation.",
    "P7": "P7 — Farmer / rural man (30–55), Kisan identity, debt, crop failure, moneylender, no help-seeking norm.",
    "P8": "P8 — First-gen urban migrant (20–35, M), Metro isolation, remittance burden, no community, language gap.",
    "P9": "P9 — Middle-aged man (40–55), breadwinner, Business owner or salaried, family head, health neglect.",
    "P12": "P12 — Single mother / divorced woman (28–45), Stigma, economic precarity, children's wellbeing, community exclusion."
}

# 2. TRIGGER EVENTS (Work/Financial Triggers)
TRIGGER_EVENTS = [
    "Received termination email this morning",
    "Boss humiliated me in team meeting",
    "Bank EMI bounce notification",
    "Crop destroyed; couldn't repay loan installment",
    "Client cancelled contract; business nearly closed",
    "Medical bill arrived; savings wiped",
    "Salary cut announced without notice"
]

# 3. CATEGORY 5 STRESSORS & COPING MAPPING
STRESSORS = [
    {"SUB": "EMI default risk", "DESC": "Home/vehicle loan; unable to pay; family fear", "COP": "Duty-Based Coping"},
    {"SUB": "Crop failure / kisan debt", "DESC": "Monsoon dependence; moneylender pressure; no crop insurance", "COP": "Somatization"},
    {"SUB": "Medical expense spiral", "DESC": "Hospital bills draining savings; selling assets", "COP": "Sequential Coping"},
    {"SUB": "Family remittance trap", "DESC": "Sending most salary home; nothing left for self", "COP": "Relational Preservation"},
    {"SUB": "Stock/crypto loss", "DESC": "Urban middle class; invested savings; shame of loss", "COP": "Sequential Coping"}
]

# 4. ELABORATE PROMPT GENERATOR
def generate_c5_prompts(stressor_list, persona_dict, trigger_list):
    prompts = []
    for s in stressor_list:
        for p_id, p_text in persona_dict.items():
            # Pairing Trigger to Stressor (cycling through the list)
            trigger = trigger_list[stressor_list.index(s) % len(trigger_list)]
            
            detailed_prompt = f"""
You are an expert in Indian socio-cultural psychology and mental health dataset curation.
Generate a series of realistic, emotionally rich situations faced by Indian help-seekers.
Each situation must clearly embed the assigned cultural coping mechanism so that it can later be used for CoT reasoning.

TARGET DIMENSIONS FOR THIS BATCH:
- Indian Context Category: Financial Debt & Economic Distress
- Core Stressor: {s['SUB']} ({s['DESC']})
- Persona: {p_text}
- Immediate Trigger Event: {trigger}
- Target Coping Mechanism: {s['COP']}

COPING MECHANISM DEFINITIONS (STRICT ADHERENCE):
1. Sequential (Emotion-First): Seeker is flooded with emotion and cannot think about solutions; problem is secondary.
2. Somatization (Idioms of Distress): Pain is expressed through physical symptoms (sir dard, thakan, chest tightness, neend nahi) instead of mental health terms.
3. Relational Preservation: Conflict or financial pressure is suppressed due to duty, guilt, or fear of damaging 'izzat'.
4. Duty-Based (Structural Endurance): Seeker is exhausted by obligations and feels they 'cannot afford to stop'.

RULES:
- Generate EXACTLY 5 highly distinct situations.
- Each situation must be 2-4 sentences long, in FIRST PERSON and HINGLISH format.
- Embed authentic markers: EMI bounce, sahukaar, kisan, chit fund, savings wiped, remittance, log kya kahenge, etc.
- Do NOT write Western-centric situations.
- Output MUST be a valid JSON array of 5 objects. No conversational text outside the array.

JSON SCHEMA PER OBJECT:
{{
  "situation id": "SXXX",
  "text": "string (Hinglish situation)",
  "context_category": "Financial Debt & Economic Distress",
  "coping_mechanism": "{s['COP']}",
  "emotion seed": "dominant emotion (e.g. panic, shame, helplessness, exhaustion)",
  "intensity_seed": "integer 1-5",
  "cultural markers": ["list", "of", "markers"]
}}
"""
            prompts.append(detailed_prompt.strip())
    return prompts

# 5. EXECUTION
all_c5_prompts = generate_c5_prompts(STRESSORS, PERSONAS, TRIGGER_EVENTS)

# Example: Output the logic volume and the first combination 
print(f"Total C5 prompts generated: {len(all_c5_prompts)} (5 stressors x 6 personas)")
print(f"Total Situations this will yield: {len(all_c5_prompts) * 5}\n")

print("--- PROMPT EXAMPLE: EMI Default Risk x P3 (UPSC Aspirant) ---")
print(all_c5_prompts[29])