"""Academic-pressure situation seeds (C1) for dataset generation."""

import json
PERSONAS = {
    "P1": "P1 — Male college student (18–22), Hostel, first-gen or aspirational family, JEE/NEET dropout or placement pressure, Hindi belt or metro.",
    "P2": "P2 — Female college student (18–22), Safety concerns, restricted mobility, marriage pressure alongside studies.",
    "P3": "P3 — UPSC/PSC aspirant (22–30, M/F), Multi-year prep, coaching institute life, family investment, dropper stigma."
}

# 2. ACADEMIC TRIGGER EVENTS (C1 Only)
TRIGGER_EVENTS = [
    "Result card shown to parents and silence followed",
    "Did not get shortlisted in campus placement drive",
    "Rank declared; far from cutoff",
    "Asked to drop another year",
    "Classmate got placed; I didn't",
    "Father stopped talking after results"
]

# 3. CATEGORY 1 STRESSORS & BASE COPING
# Note: 'Somatization' is explicitly excluded here per your compatibility rule (C1 is too cognitive).
STRESSORS = [
    {"SUB": "12th Board result failure", "DESC": "Marks below family threshold; kuch nahi hoga mindset", "BASE_COP": "Sequential Coping"},
    {"SUB": "JEE/NEET repeat attempt", "DESC": "Dropper shame; 2nd/3rd attempt identity crisis", "BASE_COP": "Duty-Based Coping"},
    {"SUB": "Placement season anxiety", "DESC": "Final year; on-campus offer pressure; peer comparison", "BASE_COP": "Sequential Coping"},
    {"SUB": "UPSC repeated failure", "DESC": "Years of sacrifice; family investment; shame of giving up", "BASE_COP": "Duty-Based Coping"},
    {"SUB": "Forced branch/course", "DESC": "Engineering/medicine forced by parents; no interest", "BASE_COP": "Relational Preservation"},
    {"SUB": "Low CGPA shame", "DESC": "Back papers; attendance shortage; hostel pressure", "BASE_COP": "Sequential Coping"}
]

# 4. COMPATIBILITY LOGIC ENGINE
def resolve_coping_mechanism(persona_id, base_coping):
    """
    Adjusts the coping mechanism based on the Persona x Coping compatibility table.
    """
    # Rule: P1 rarely uses Relational Preservation
    if persona_id == "P1" and base_coping == "Relational Preservation":
        return "Sequential Coping"  # Fallback to emotion-flood
        
    # Rule: P2 rarely uses Duty-Based
    if persona_id == "P2" and base_coping == "Duty-Based Coping":
        return "Sequential Coping"  # Fallback to emotion-flood
        
    return base_coping

def get_regional_markers(persona_id):
    """
    Injects regional constraints based on Persona to guide the AI's vocabulary.
    """
    if persona_id in ["P1", "P3"]:
        return "Hindi Belt markers (Mukherjee Nagar, Kota, izzat, sarkari babu, joint family pressure)"
    elif persona_id == "P2":
        return "South India / Metro markers (NRI aspirations, restricted mobility, 'log kya kahenge')"
    return "Urban/Metro markers (anonymity, severe peer pressure)"

# 5. ELABORATE PROMPT GENERATOR
def generate_c1_prompts(stressor_list, persona_dict, trigger_list):
    prompts = []
    for s in stressor_list:
        for p_id, p_text in persona_dict.items():
            
            # Apply compatibility rules to get the correct coping mechanism
            final_coping = resolve_coping_mechanism(p_id, s['BASE_COP'])
            regional_instructions = get_regional_markers(p_id)
            trigger = trigger_list[stressor_list.index(s) % len(trigger_list)]
            
            detailed_prompt = f"""
You are an expert in Indian socio-cultural psychology and mental health dataset curation.
Generate a series of realistic, emotionally rich situations faced by Indian help-seekers.

TARGET DIMENSIONS FOR THIS BATCH:
- Indian Context Category: Academic & Competitive Achievement
- Core Stressor: {s['SUB']} ({s['DESC']})
- Persona: {p_text}
- Immediate Trigger Event: {trigger}
- Target Coping Mechanism: {final_coping}

COPING MECHANISM DEFINITIONS (STRICT ADHERENCE):
1. Sequential (Emotion-First): Seeker is flooded with emotion and cannot think about solutions; problem is secondary.
2. Relational Preservation: Conflict or resentment regarding academics is suppressed due to guilt or fear of damaging family 'izzat'.
3. Duty-Based (Structural Endurance): Seeker is exhausted by academic obligations/prep and feels they 'cannot afford to stop'.
*Note: Somatization is intentionally excluded for this cognitive/achievement category.*

RULES:
- Generate EXACTLY 5 highly distinct situations.
- Each situation must be 2-4 sentences long, in FIRST PERSON and HINGLISH format.
- Integrate these cultural/regional elements seamlessly: {regional_instructions}.
- Embed authentic markers: dropper, rank, cutoff, back paper, Sharma ji ka beta, etc.
- Do NOT write Western-centric situations.
- Output MUST be a valid JSON array of 5 objects. No conversational text outside the array.

JSON SCHEMA PER OBJECT:
{{
  "situation id": "SXXX",
  "text": "string (Hinglish situation)",
  "context_category": "Academic & Competitive Achievement",
  "coping_mechanism": "{final_coping}",
  "emotion seed": "dominant emotion (e.g. shame, panic, exhaustion, inadequacy)",
  "intensity_seed": "integer 1-5",
  "cultural markers": ["list", "of", "markers"]
}}
"""
            prompts.append(detailed_prompt.strip())
    return prompts

# 6. EXECUTION
all_c1_prompts = generate_c1_prompts(STRESSORS, PERSONAS, TRIGGER_EVENTS)

print(f"Total C1 prompts generated: {len(all_c1_prompts)} (6 stressors x 3 personas)")
print(f"Total Situations this will yield: {len(all_c1_prompts) * 5}\n")

print("--- PROMPT EXAMPLE: JEE Repeat Attempt x P2 (Female Student) ---")
# Index 4 is Stressor 1 (JEE), Persona 1 (P2). Base coping was Duty-Based, but script overrides to Sequential.
print(all_c1_prompts[17])