import json

# ==========================================
# 1. PERSONAS COMPATIBLE WITH C6
# ==========================================
PERSONAS = {
    "P2": "P2 — Female college student (18–22), Safety concerns, restricted mobility, marriage pressure alongside studies.",
    "P4": "P4 — IT/software professional (24–35, M/F), Tier-1 or Tier-2 city, MNC or startup, EMI pressure, WFH isolation.",
    "P5": "P5 — Married woman (25–40), homemaker, Joint family, saas conflict, career gap, child pressure, lost identity.",
    "P6": "P6 — Married woman (25–40), working, Dual burden — career + ghar, guilt, suppressed ambition.",
    "P11": "P11 — LGBTQ young adult (20–30), Hidden identity, family fear, urban but isolated.",
    "P12": "P12 — Single mother / divorced woman (28–45), Stigma, economic precarity, children's wellbeing, community exclusion."
}

# ==========================================
# 2. TRIGGER EVENTS (Health/Identity/Migration)
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
# 3. CATEGORY 6 STRESSORS & COPING MAPPING
# ==========================================
STRESSORS = [
    {"SUB": "Bahu role vs career", "DESC": "Expected to quit job after marriage; ambition suppressed", "COP": "Relational Preservation"},
    {"SUB": "LGBTQ identity hiding", "DESC": "Section 377 legacy fear; family dishonour anxiety", "COP": "Relational Preservation"},
    {"SUB": "Male crying/seeking help shame", "DESC": "'Mard nahi rota'; suppression of emotional need", "COP": "Somatization"},
    {"SUB": "Single mother stigma", "DESC": "Divorce or widowhood; community exclusion; children's schooling", "COP": "Duty-Based Coping"}
]

# ==========================================
# 4. ELABORATE PROMPT GENERATOR
# ==========================================
def generate_c6_prompts(stressor_list, persona_dict, trigger_list):
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
- Indian Context Category: Gender Role & Identity Conflict
- Core Stressor: {s['SUB']} ({s['DESC']})
- Persona: {p_text}
- Immediate Trigger Event: {trigger}
- Target Coping Mechanism: {s['COP']}

COPING MECHANISM DEFINITIONS (STRICT ADHERENCE):
1. Sequential (Emotion-First): Seeker is flooded with emotion and cannot think about solutions; problem is secondary.
2. Somatization (Idioms of Distress): Pain is expressed through physical symptoms (sir dard, thakan, chest tightness) instead of mental health terms.
3. Relational Preservation: Conflict or authentic identity is suppressed due to duty, guilt, or fear of damaging 'izzat'.
4. Duty-Based (Structural Endurance): Seeker is exhausted by obligations and feels they 'cannot afford to stop'.

RULES:
- Generate EXACTLY 5 highly distinct situations.
- Each situation must be 2-4 sentences long, in FIRST PERSON and HINGLISH format.
- Embed authentic markers: mard nahi rota, bahu, log kya kahenge, izzat, samaj, divorce stigma, etc.
- Do NOT write Western-centric situations.
- Output MUST be a valid JSON array of 5 objects. No conversational text outside the array.

JSON SCHEMA PER OBJECT:
{{
  "situation id": "SXXX",
  "text": "string (Hinglish situation)",
  "context_category": "Gender Role & Identity Conflict",
  "coping_mechanism": "{s['COP']}",
  "emotion seed": "dominant emotion (e.g. shame, suffocation, exhaustion, fear)",
  "intensity_seed": "integer 1-5",
  "cultural markers": ["list", "of", "markers"]
}}
"""
            prompts.append(detailed_prompt.strip())
    return prompts

# ==========================================
# 5. EXECUTION
# ==========================================
all_c6_prompts = generate_c6_prompts(STRESSORS, PERSONAS, TRIGGER_EVENTS)

# Example: Output the logic volume and the first combination 
print(f"Total C6 prompts generated: {len(all_c6_prompts)} (4 stressors x 6 personas)")
print(f"Total Situations this will yield: {len(all_c6_prompts) * 5}\n")

print("--- PROMPT EXAMPLE: Bahu role vs career x P2 (Female College Student) ---")
print(all_c6_prompts[23])