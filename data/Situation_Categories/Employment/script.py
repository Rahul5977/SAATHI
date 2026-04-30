import json

# 1. PERSONAS COMPATIBLE WITH C4
PERSONAS = {
    "P1": "P1 — Male college student (18–22), Hostel, first-gen or aspirational family, JEE/NEET dropout or placement pressure.",
    "P3": "P3 — UPSC/PSC aspirant (22–30, M/F), Multi-year prep, coaching institute life, family investment, dropper stigma.",
    "P4": "P4 — IT/software professional (24–35, M/F), Tier-1 or Tier-2 city, MNC or startup, EMI pressure, WFH isolation.",
    "P6": "P6 — Married woman (25–40), working, Dual burden — career + ghar, guilt, suppressed ambition.",
    "P7": "P7 — Farmer / rural man (30–55), Kisan identity, debt, crop failure, moneylender, no help-seeking norm.",
    "P8": "P8 — First-gen urban migrant (20–35, M), Metro isolation, remittance burden, no community, language gap.",
    "P9": "P9 — Middle-aged man (40–55), breadwinner, Business owner or salaried, family head, health neglect."
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

# 3. CATEGORY 4 STRESSORS & COPING MAPPING
STRESSORS = [
    {"SUB": "Sarkari naukri failure", "DESC": "Years of coaching; family honour tied to govt job", "COP": "Somatization"},
    {"SUB": "Tech layoff", "DESC": "IT sector; EMIs to pay; identity tied to MNC job", "COP": "Sequential Coping"},
    {"SUB": "First-generation earner burden", "DESC": "Sole income; family looks up; no margin for failure", "COP": "Duty-Based Coping"},
    {"SUB": "Toxic boss/workplace", "DESC": "Screaming boss; unpaid overtime; no exit option felt", "COP": "Relational Preservation"},
    {"SUB": "Work-life conflict (new parent)", "DESC": "Office hours vs child; guilt of absence", "COP": "Duty-Based Coping"},
    {"SUB": "Business failure (small trader)", "DESC": "Shop/kirana loss; neighborhood shame; debt spiral", "COP": "Sequential Coping"}
]

# 4. ELABORATE PROMPT GENERATOR
def generate_c4_prompts(stressor_list, persona_dict, trigger_list):
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
- Indian Context Category: Employment & Livelihood
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
- Embed authentic markers: sarkari naukri, EMI, kirana shop, kisan, layoff, MNC, startup, etc.
- Do NOT write Western-centric situations.
- Output MUST be a valid JSON array of 5 objects. No conversational text outside the array.

JSON SCHEMA PER OBJECT:
{{
  "situation id": "SXXX",
  "text": "string (Hinglish situation)",
  "context_category": "Employment & Livelihood",
  "coping_mechanism": "{s['COP']}",
  "emotion seed": "dominant emotion (e.g. panic, shame, exhaustion)",
  "intensity_seed": "integer 1-5",
  "cultural markers": ["list", "of", "markers"]
}}
"""
            prompts.append(detailed_prompt.strip())
    return prompts

# 5. EXECUTION
all_c4_prompts = generate_c4_prompts(STRESSORS, PERSONAS, TRIGGER_EVENTS)

# Example: Output the logic volume and the first combination 
print(f"Total C4 prompts generated: {len(all_c4_prompts)} (6 stressors x 7 personas)")
print(f"Total Situations this will yield: {len(all_c4_prompts) * 5}\n")

print("--- PROMPT EXAMPLE: Sarkari Naukri Failure x P1 (Male College Student) ---")
print(all_c4_prompts[41])