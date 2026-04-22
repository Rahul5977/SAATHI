import json

# ==========================================
# 1. PERSONAS COMPATIBLE WITH C8
# ==========================================
PERSONAS = {
    "P1": "P1 — Male college student (18–22), Hostel, first-gen or aspirational family, JEE/NEET dropout or placement pressure, Hindi belt or metro.",
    "P2": "P2 — Female college student (18–22), Safety concerns, restricted mobility, marriage pressure alongside studies.",
    "P8": "P8 — First-gen urban migrant (20–35, M), Metro isolation, remittance burden, no community, language gap.",
    "P11": "P11 — LGBTQ young adult (20–30), Hidden identity, family fear, urban but isolated."
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
# 3. CATEGORY 8 STRESSORS & COPING MAPPING
# ==========================================
# Coping mechanisms logically assigned based on the socio-cultural context of the displacement
STRESSORS = [
    {"SUB": "Rural to metro migration", "DESC": "Alone in big city; language barrier; no community", "COP": "Sequential Coping"},
    {"SUB": "NRI return shock", "DESC": "Returned to India; misfit in both cultures; alienation", "COP": "Relational Preservation"},
    {"SUB": "Disaster-induced displacement", "DESC": "Flood/fire loss; government apathy; starting over", "COP": "Duty-Based Coping"}
]

# ==========================================
# 4. ELABORATE PROMPT GENERATOR
# ==========================================
def generate_c8_prompts(stressor_list, persona_dict, trigger_list):
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
- Indian Context Category: Migration, Displacement & Acculturation
- Core Stressor: {s['SUB']} ({s['DESC']})
- Persona: {p_text}
- Immediate Trigger Event: {trigger}
- Target Coping Mechanism: {s['COP']}

COPING MECHANISM DEFINITIONS (STRICT ADHERENCE):
1. Sequential (Emotion-First): Seeker is flooded with emotion and cannot think about solutions; problem is secondary.
2. Somatization (Idioms of Distress): Pain is expressed through physical symptoms (sir dard, thakan, chest tightness) instead of mental health terms.
3. Relational Preservation: Conflict or authentic struggle is suppressed due to duty, guilt, or fear of damaging family harmony/'izzat'.
4. Duty-Based (Structural Endurance): Seeker is exhausted by obligations (rebuilding, adjusting) and feels they 'cannot afford to stop'.

RULES:
- Generate EXACTLY 5 highly distinct situations.
- Each situation must be 2-4 sentences long, in FIRST PERSON and HINGLISH format.
- Embed authentic markers: gaon, shehar, NRI, biradari, bhasha, akelepan, log kya kahenge, etc.
- Do NOT write Western-centric situations.
- Output MUST be a valid JSON array of 5 objects. No conversational text outside the array.

JSON SCHEMA PER OBJECT:
{{
  "situation id": "SXXX",
  "text": "string (Hinglish situation)",
  "context_category": "Migration, Displacement & Acculturation",
  "coping_mechanism": "{s['COP']}",
  "emotion seed": "dominant emotion (e.g. isolation, alienation, exhaustion, grief)",
  "intensity_seed": "integer 1-5",
  "cultural markers": ["list", "of", "markers"]
}}
"""
            prompts.append(detailed_prompt.strip())
    return prompts

# ==========================================
# 5. EXECUTION
# ==========================================
all_c8_prompts = generate_c8_prompts(STRESSORS, PERSONAS, TRIGGER_EVENTS)

# Example: Output the logic volume and the first combination 
print(f"Total C8 prompts generated: {len(all_c8_prompts)} (3 stressors x 4 personas)")
print(f"Total Situations this will yield: {len(all_c8_prompts) * 5}\n")

print("--- PROMPT EXAMPLE: Rural to metro migration x P8 (First-gen urban migrant) ---")
print(all_c8_prompts[11]) # Index 2 matches P8 for the first stressor