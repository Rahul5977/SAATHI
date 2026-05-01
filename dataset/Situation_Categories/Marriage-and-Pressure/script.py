import json

# 1. PERSONAS COMPATIBLE WITH C3
PERSONAS = {
    "P2": "P2 — Female college student (18–22), Safety concerns, restricted mobility, marriage pressure alongside studies.",
    "P5": "P5 — Married woman (25–40), homemaker, Joint family, saas conflict, career gap, child pressure, lost identity.",
    "P6": "P6 — Married woman (25–40), working, Dual burden — career + ghar, guilt, suppressed ambition.",
    "P11": "P11 — LGBTQ young adult (20–30), Hidden identity, family fear, urban but isolated.",
    "P12": "P12 — Single mother / divorced woman (28–45), Stigma, economic precarity, children's wellbeing, community exclusion."
}

# 2. TRIGGER EVENTS (Family/Marriage Triggers)
TRIGGER_EVENTS = [
    "Argument last night that ended without resolution",
    "Father announced a decision without asking me",
    "Saas compared me to her daughter again today",
    "Rishta party visited and rejected",
    "Brother got larger share of property",
    "In-laws announced they're moving in permanently",
    "Heard parents discussing my 'settling down' on phone"
]

# 3. CATEGORY 3 STRESSORS & COPING MAPPING
STRESSORS = [
    {"SUB": "Rishta rejection spiral", "DESC": "Multiple rejections; self-worth erosion; log kya kahenge", "COP": "Sequential Coping"},
    {"SUB": "Love marriage opposition", "DESC": "Family rejecting partner; caste/religion mismatch", "COP": "Relational Preservation"},
    {"SUB": "Post-wedding adjustment shock", "DESC": "Leaving natal home; new family expectations", "COP": "Duty-Based Coping"},
    {"SUB": "No child pressure", "DESC": "Sasural/society pressure after marriage; medical issue", "COP": "Somatization"},
    {"SUB": "Divorce shame", "DESC": "Returning home; community judgment; starting over", "COP": "Sequential Coping"},
    {"SUB": "Age pressure (unmarried)", "DESC": "25+ anxiety for women; 30+ for men; 'expired' feeling", "COP": "Relational Preservation"}
]

# 4. ELABORATE PROMPT GENERATOR
def generate_c3_prompts(stressor_list, persona_dict, trigger_list):
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
- Indian Context Category: Marriage & Rishta Pressure
- Core Stressor: {s['SUB']} ({s['DESC']})
- Persona: {p_text}
- Immediate Trigger Event: {trigger}
- Target Coping Mechanism: {s['COP']}

COPING MECHANISM DEFINITIONS (STRICT ADHERENCE):
1. Sequential (Emotion-First): Seeker is flooded with emotion and cannot think about solutions; problem is secondary.
2. Somatization (Idioms of Distress): Pain is expressed through physical symptoms (sir dard, thakan, chest tightness) instead of mental health terms.
3. Relational Preservation: Conflict with authority is suppressed due to duty, guilt, or fear of damaging 'izzat'.
4. Duty-Based (Structural Endurance): Seeker is exhausted by obligations and feels they 'cannot afford to stop'.

RULES:
- Generate EXACTLY 5 highly distinct situations.
- Each situation must be 2-4 sentences long, in FIRST PERSON and HINGLISH format.
- Embed authentic markers: kundali, caste mismatch, rishta, dowry, log kya kahenge, sasural, etc.
- Do NOT write Western-centric situations (e.g., "going on a bad Tinder date").
- Output MUST be a valid JSON array of 5 objects. No conversational text outside the array.

JSON SCHEMA PER OBJECT:
{{
  "situation id": "SXXX",
  "text": "string (Hinglish situation)",
  "context_category": "Marriage & Rishta Pressure",
  "coping_mechanism": "{s['COP']}",
  "emotion seed": "dominant emotion (e.g. shame, despair, anxiety)",
  "intensity_seed": "integer 1-5",
  "cultural markers": ["list", "of", "markers"]
}}
"""
            prompts.append(detailed_prompt.strip())
    return prompts

# 5. EXECUTION
all_prompts = generate_c3_prompts(STRESSORS, PERSONAS, TRIGGER_EVENTS)

# Example: Output the logic volume and the first combination 
print(f"Total C3 prompts generated: {len(all_prompts)} (6 stressors x 5 personas)")
print(f"Total Situations this will yield: {len(all_prompts) * 5}\n")

print("--- PROMPT EXAMPLE: Rishta Rejection x Restricted Female Student ---")
print(all_prompts[29])