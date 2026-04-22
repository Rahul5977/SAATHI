import json

# ==========================================
# 1. PERSONAS COMPATIBLE WITH C7
# ==========================================
PERSONAS = {
    "P5": "P5 — Married woman (25–40), homemaker, Joint family, saas conflict, career gap, child pressure, lost identity.",
    "P7": "P7 — Farmer / rural man (30–55), Kisan identity, debt, crop failure, moneylender, no help-seeking norm.",
    "P9": "P9 — Middle-aged man (40–55), breadwinner, Business owner or salaried, family head, health neglect, 'kuch nahi hone dunga' mask.",
    "P10": "P10 — Elderly parent / in-law (60+), Dependent, ignored, generational gap, loneliness, health.",
    "P12": "P12 — Single mother / divorced woman (28–45), Stigma, economic precarity, children's wellbeing, community exclusion."
}

# ==========================================
# 2. TRIGGER EVENTS (Health / Identity / Isolation)
# ==========================================
TRIGGER_EVENTS = [
    "Doctor dismissed my symptoms as 'stress' again",
    "Couldn't attend a friend's wedding — too exhausted from caregiving",
    "Partner said 'you've changed since coming to this city'",
    "Overheard colleague joking about my accent",
    "Told a friend I might be gay; they went quiet",
    "Husband asked 'why are you always tired'",
    "Realized I haven't spoken to anyone in 3 days"
]

# ==========================================
# 3. CATEGORY 7 STRESSORS & COPING MAPPING
# ==========================================
# Coping mechanisms logically assigned based on the socio-cultural context of the stressor
STRESSORS = [
    {"SUB": "Elderly parent caregiver", "DESC": "Dementia/bedridden parent; no support system; burnout", "COP": "Duty-Based Coping"},
    {"SUB": "Own mental health stigma", "DESC": "Afraid to say 'depression'; hiding medication from family", "COP": "Relational Preservation"},
    {"SUB": "Child with disability", "DESC": "School rejections; family blame; isolation", "COP": "Sequential Coping"},
    {"SUB": "Chronic pain with no diagnosis", "DESC": "Dismissed as 'natak'; somatization invalidated by doctor", "COP": "Somatization"}
]

# ==========================================
# 4. ELABORATE PROMPT GENERATOR
# ==========================================
def generate_c7_prompts(stressor_list, persona_dict, trigger_list):
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
- Indian Context Category: Health & Chronic Illness (Self/Family)
- Core Stressor: {s['SUB']} ({s['DESC']})
- Persona: {p_text}
- Immediate Trigger Event: {trigger}
- Target Coping Mechanism: {s['COP']}

COPING MECHANISM DEFINITIONS (STRICT ADHERENCE):
1. Sequential (Emotion-First): Seeker is flooded with emotion and cannot think about solutions; problem is secondary.
2. Somatization (Idioms of Distress): Pain is expressed through physical symptoms (sir dard, thakan, chest tightness, body ache) instead of mental health terms.
3. Relational Preservation: Conflict or authentic struggle is suppressed due to duty, guilt, or fear of damaging family harmony/'izzat'.
4. Duty-Based (Structural Endurance): Seeker is exhausted by obligations (caregiving, providing) and feels they 'cannot afford to stop'.

RULES:
- Generate EXACTLY 5 highly distinct situations.
- Each situation must be 2-4 sentences long, in FIRST PERSON and HINGLISH format.
- Embed authentic markers: natak, bimari, log kya kahenge, hospital ka kharcha, saas-bahu, kisan, etc.
- Do NOT write Western-centric situations.
- Output MUST be a valid JSON array of 5 objects. No conversational text outside the array.

JSON SCHEMA PER OBJECT:
{{
  "situation id": "SXXX",
  "text": "string (Hinglish situation)",
  "context_category": "Health & Chronic Illness",
  "coping_mechanism": "{s['COP']}",
  "emotion seed": "dominant emotion (e.g. exhaustion, despair, shame, invalidation)",
  "intensity_seed": "integer 1-5",
  "cultural markers": ["list", "of", "markers"]
}}
"""
            prompts.append(detailed_prompt.strip())
    return prompts

# ==========================================
# 5. EXECUTION
# ==========================================
all_c7_prompts = generate_c7_prompts(STRESSORS, PERSONAS, TRIGGER_EVENTS)

# Example: Output the logic volume and the first combination 
print(f"Total C7 prompts generated: {len(all_c7_prompts)} (4 stressors x 5 personas)")
print(f"Total Situations this will yield: {len(all_c7_prompts) * 5}\n")

print("--- PROMPT EXAMPLE: Elderly parent caregiver x P5 (Married woman, homemaker) ---")
print(all_c7_prompts[19])