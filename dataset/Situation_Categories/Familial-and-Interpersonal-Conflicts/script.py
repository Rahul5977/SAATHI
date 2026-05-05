"""Family/interpersonal situation seeds for dataset generation."""

import json
PERSONAS = {
    "P1": "P1 — Male college student (18–22), Hostel, first-gen or aspirational family, JEE/NEET dropout or placement pressure, Hindi belt or metro.",
    "P2": "P2 — Female college student (18–22), Safety concerns, restricted mobility, marriage pressure alongside studies.",
    "P3": "P3 — UPSC/PSC aspirant (22–30, M/F), Multi-year prep, coaching institute life, family investment, dropper stigma.",
    "P4": "P4 — IT/software professional (24–35, M/F), Tier-1 or Tier-2 city, MNC or startup, EMI pressure, WFH isolation.",
    "P5": "P5 — Married woman (25–40), homemaker, Joint family, saas conflict, career gap, child pressure, lost identity.",
    "P6": "P6 — Married woman (25–40), working, Dual burden — career + ghar, guilt, suppressed ambition.",
    "P7": "P7 — Farmer / rural man (30–55), Kisan identity, debt, crop failure, moneylender, no help-seeking norm.",
    "P8": "P8 — First-gen urban migrant (20–35, M), Metro isolation, remittance burden, no community, language gap.",
    "P9": "P9 — Middle-aged man (40–55), breadwinner, family head, health neglect, 'kuch nahi hone dunga' mask.",
    "P10": "P10 — Elderly parent / in-law (60+), Dependent, ignored, generational gap, loneliness, health.",
    "P11": "P11 — LGBTQ young adult (20–30), Hidden identity, family fear, urban but isolated.",
    "P12": "P12 — Single mother / divorced woman (28–45), Stigma, economic precarity, children's wellbeing, community exclusion."
}

# 2. TRIGGER EVENTS
TRIGGER_EVENTS = [
    "Argument last night that ended without resolution",
    "Father announced a decision without asking me",
    "Saas compared me to her daughter again today",
    "Rishta party visited and rejected",
    "Brother got larger share of property",
    "In-laws announced they're moving in permanently",
    "Heard parents discussing my 'settling down' on phone"
]

# 3. CATEGORY 2 STRESSORS
STRESSORS = [
    {"SUB": "Father's authoritarian control", "DESC": "Career/life decisions overridden; no space to disagree", "COP": "Relational Preservation"},
    {"SUB": "Mother's emotional manipulation", "DESC": "Guilt-based control; 'tumhari wajah se hota hai'", "COP": "Sequential Coping"},
    {"SUB": "Joint family living conflict", "DESC": "Privacy loss; saas domination; no autonomy", "COP": "Somatization"},
    {"SUB": "Parental comparison", "DESC": "'Sharma ji ka beta' dynamic; constant inadequacy", "COP": "Sequential Coping"},
    {"SUB": "Elder sibling obligation", "DESC": "Sacrificing own goals for younger siblings' future", "COP": "Duty-Based Coping"},
    {"SUB": "Property/inheritance dispute", "DESC": "Post-death family fracture; sibling estrangement", "COP": "Relational Preservation"}
]

# 4. ELABORATE PROMPT GENERATOR
def generate_elaborate_prompts(stressor_list, persona_dict, trigger_list):
    prompts = []
    for s in stressor_list:
        for p_id, p_text in persona_dict.items():
            # Pairing Trigger to Stressor (cycling through) 
            trigger = trigger_list[stressor_list.index(s) % len(trigger_list)]
            
            detailed_prompt = f"""
You are an expert in Indian socio-cultural psychology and mental health dataset curation. [cite: 2]
Generate a series of realistic, emotionally rich situations faced by Indian help-seekers. [cite: 3]
Each situation must clearly embed the assigned cultural coping mechanism so that it can later be used for CoT reasoning. [cite: 4]

TARGET DIMENSIONS FOR THIS BATCH:
- Indian Context Category: Joint family tensions, izzat, parental authority
- Core Stressor: {s['SUB']} ({s['DESC']}) [cite: 34]
- Persona: {p_text} [cite: 35]
- Immediate Trigger Event: {trigger} 
- Target Coping Mechanism: {s['COP']} [cite: 37]

COPING MECHANISM DEFINITIONS (STRICT ADHERENCE): 
1. Sequential (Emotion-First): Seeker is flooded with emotion and cannot think about solutions; problem is secondary. [cite: 7-10]
2. Somatization (Idioms of Distress): Pain is expressed through physical symptoms (sir dard, thakan, body ache) instead of mental health terms. [cite: 12-16]
3. Relational Preservation: Conflict with authority is suppressed due to duty, guilt, or fear of damaging 'izzat'. [cite: 18-22]
4. Duty-Based (Structural Endurance): Seeker is exhausted by obligations and feels they 'cannot afford to stop'. [cite: 24-28]

RULES: [cite: 38-47]
- Generate EXACTLY 5 highly distinct situations. [cite: 39]
- Each situation must be 2-4 sentences long, in FIRST PERSON and HINGLISH format. [cite: 41-42]
- Embed authentic markers: joint family, saas-bahu dynamics, log kya kahenge, izzat, rishta, etc. [cite: 44]
- Do NOT write Western-centric situations (e.g., "seeing a therapist"). [cite: 45]
- Output MUST be a valid JSON array of 5 objects. No conversational text outside the array. [cite: 46-47]

JSON SCHEMA PER OBJECT: [cite: 54-67]
{{
  "situation id": "SXXX",
  "text": "string (Hinglish situation)",
  "context_category": "Joint family tensions & Parental Authority",
  "coping_mechanism": "{s['COP']}",
  "emotion seed": "dominant emotion (e.g. guilt, shame, resentment, despair)",
  "intensity_seed": "integer 1-5",
  "cultural markers": ["list", "of", "markers"]
}}
"""
            prompts.append(detailed_prompt.strip())
    return prompts

# 5. EXECUTION
all_prompts = generate_elaborate_prompts(STRESSORS, PERSONAS, TRIGGER_EVENTS)

# Example: Output the first combination (Authoritarian Father x P1)
print(all_prompts[71])