"""Step 2 dataset preparation for Gemma fine-tuning.

This module reads ``data_hindi.json``, filters to non-crisis medium/high quality
conversations, and exports supporter-turn prompt/response pairs using the fixed
5-block prompt structure required for Gemma fine-tuning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from fine_tuning.muril_dataset_prep import filter_non_crisis_conversations, load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



DATASET_PATH = PROJECT_ROOT / "data" / "data_hindi.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
GEMMA_OUTPUT_PATH = OUTPUT_DIR / "gemma_dataset.jsonl"


SYSTEM_INSTRUCTION = (
    "Aap ek madadgaar Hindi-bolne wale emotional support assistant hain. "
    "Aapko sirf Hindi Devanagari script mein jawab dena hai. "
    "Aapko diye gaye strategy aur response style ko follow karna hai. "
    "Aapko Bharatiya parivarik aur samajik sandarbhon ko samajhte hue sahaanubhooti ke saath jawab dena hai."
)


STRATEGY_DESCRIPTIONS = {
    "Question": "Is turn mein khule aur surakshit tareeke se aur jaankari lene ke liye prashn poochna hai.",
    "Restatement or Paraphrasing": "Is turn mein user ki baat ko apne shabdon mein saaf tareeke se dobara rakhna hai.",
    "Reflection of Feelings": "Is turn mein user ki bhavnaon ko seedhe tarah se pehchaan kar unka pratibimban karna hai.",
    "Self-disclosure": "Is turn mein bahut saavdhani se ek halka, sambandhit aur supportive self-disclosure istemal karna hai.",
    "Affirmation and Reassurance": "Is turn mein user ke anubhav ko validate karke unhe ashwasan dena hai.",
    "Providing Suggestions": "Is turn mein chhote, vyavaharik aur dayalu actionable suggestions dene hain.",
    "Information": "Is turn mein spasht, upyogi aur non-judgmental jaankari deni hai.",
}


def filter_gemma_training_conversations(conversations: list[dict]) -> list[dict]:
    """Keep only non-crisis conversations with medium or high quality tiers"""
    non_crisis_conversations = filter_non_crisis_conversations(conversations)
    filtered: list[dict] = []

    for conversation in non_crisis_conversations:
        survey_score = conversation.get("survey_score") or {}
        quality_tier = survey_score.get("quality_tier")
        if quality_tier in {"high", "medium"}:
            filtered.append(conversation)

    return filtered


def build_seeker_state_block(previous_seeker_turn: dict) -> str:
    """Format Block 2 using seeker_analysis from the immediately previous seeker turn"""
    seeker_analysis = previous_seeker_turn.get("seeker_analysis") or {}
    emotion = seeker_analysis.get("emotion") or "neutral"
    intensity = seeker_analysis.get("intensity")
    sentiment = seeker_analysis.get("sentiment") or "neutral"
    cultural_signal = seeker_analysis.get("cultural_signal") or "other"
    relationship_signal = seeker_analysis.get("relationship_signal") or "self"
    coping_behavior = seeker_analysis.get("coping_behavior") or "contact"

    return "\n".join(
        [
            "[Block 2: Seeker State]",
            f"emotion: {emotion}",
            f"intensity: {intensity}",
            f"sentiment: {sentiment}",
            f"cultural_signal: {cultural_signal}",
            f"relationship_signal: {relationship_signal}",
            f"coping_behavior: {coping_behavior}",
        ]
    )


def build_stage_strategy_block(supporter_turn: dict) -> str:
    """Format Block 3 using current supporter turn planning metadata"""
    supporter_plan = supporter_turn.get("supporter_plan") or {}
    strategy = supporter_plan.get("esconv_strategy") or "Affirmation and Reassurance"
    response_style = supporter_plan.get("response_style") or "warm_supportive"
    esc_stage = supporter_turn.get("esc_stage") or supporter_plan.get("stage_at_generation") or "exploration"
    strategy_description = STRATEGY_DESCRIPTIONS.get(
        strategy,
        "Is turn mein supportive aur culturally aware response dena hai.",
    )

    return "\n".join(
        [
            "[Block 3: Stage and Strategy]",
            f"esc_stage: {esc_stage}",
            f"esconv_strategy: {strategy}",
            f"strategy_instruction: {strategy_description}",
            f"response_style: {response_style}",
        ]
    )


def format_history_turn(turn: dict) -> str:
    """Format a single historical turn for Block 4"""
    speaker = turn.get("speaker")
    if speaker == "seeker":
        content = (turn.get("content_hinglish") or "").strip()
        return f"Seeker: {content}"

    supporter_plan = turn.get("supporter_plan") or {}
    strategy = supporter_plan.get("esconv_strategy") or "Unknown"
    content = (turn.get("content_devanagari") or "").strip()
    return f"Supporter [{strategy}]: {content}"


def build_history_block(dialog: list[dict], supporter_index: int) -> str:
    """Format Block 4 from the last 6 turns before the current supporter turn"""
    history_turns = dialog[max(0, supporter_index - 6):supporter_index]
    formatted_history = [format_history_turn(turn) for turn in history_turns if turn.get("speaker") in {"seeker", "supporter"}]

    if not formatted_history:
        formatted_history.append("No previous conversation history.")

    return "\n".join(["[Block 4: Conversation History]"] + formatted_history)


def build_current_message_block(previous_seeker_turn: dict) -> str:
    """Format Block 5 using the Hinglish text from the immediately previous seeker turn"""
    current_message = (previous_seeker_turn.get("content_hinglish") or "").strip()
    return "\n".join([
        "[Block 5: Current Seeker Message]",
        current_message,
    ])


def build_training_prompt(dialog: list[dict], supporter_index: int) -> str:
    """Build the exact 5-block user prompt wrapped in Gemma chat-template markers"""
    supporter_turn = dialog[supporter_index]
    previous_seeker_turn = dialog[supporter_index - 1]

    user_content = "\n\n".join(
        [
            "[Block 1: System Instruction]\n" + SYSTEM_INSTRUCTION,
            build_seeker_state_block(previous_seeker_turn),
            build_stage_strategy_block(supporter_turn),
            build_history_block(dialog, supporter_index),
            build_current_message_block(previous_seeker_turn),
        ]
    )

    return "\n".join(
        [
            "<start_of_turn>user",
            user_content,
            "<end_of_turn>",
            "<start_of_turn>model",
            "",
        ]
    )


def compute_sample_weight(conversation: dict) -> float:
    """Upweight samples from conversations with stronger negative emotion deltas.
    More negative values indicate stronger improvement
    """
    conversation_context = conversation.get("conversation_context") or {}
    emotion_delta = conversation_context.get("emotion_delta")

    if emotion_delta is None:
        seeker_scores = ((conversation.get("survey_score") or {}).get("seeker") or {})
        emotion_delta = seeker_scores.get("emotion_delta", 0)

    try:
        delta_value = int(emotion_delta)
    except (TypeError, ValueError):
        delta_value = 0

    return float(1 + max(0, -delta_value))


def extract_gemma_samples(conversations: list[dict]) -> tuple[list[dict], dict]:
    """Extract supporter-turn prompt/response records for Gemma fine-tuning"""
    records: list[dict] = []
    stats = {
        "supporter_turns_seen": 0,
        "exported_samples": 0,
        "skipped_turns": 0,
    }

    for conversation in conversations:
        dialog = conversation.get("dialog") or []
        survey_score = conversation.get("survey_score") or {}
        quality_tier = survey_score.get("quality_tier")
        sample_weight = compute_sample_weight(conversation)

        for index, turn in enumerate(dialog):
            if turn.get("speaker") != "supporter":
                continue

            stats["supporter_turns_seen"] += 1

            if index == 0:
                stats["skipped_turns"] += 1
                continue

            previous_turn = dialog[index - 1]
            if previous_turn.get("speaker") != "seeker":
                stats["skipped_turns"] += 1
                continue

            target = turn.get("content_devanagari")
            previous_message = previous_turn.get("content_hinglish")
            supporter_plan = turn.get("supporter_plan") or {}
            seeker_analysis = previous_turn.get("seeker_analysis") or {}

            if not isinstance(target, str) or not target.strip():
                stats["skipped_turns"] += 1
                continue

            if not isinstance(previous_message, str) or not previous_message.strip():
                stats["skipped_turns"] += 1
                continue

            if not isinstance(seeker_analysis, dict) or not seeker_analysis:
                stats["skipped_turns"] += 1
                continue

            prompt = build_training_prompt(dialog, index)
            record = {
                "prompt": prompt,
                "target": target.strip(),
                "training_text": prompt + target.strip() + "\n<end_of_turn>",
                "sample_weight": sample_weight,
                "metadata": {
                    "conversation_id": conversation.get("conversation_id"),
                    "supporter_turn_id": turn.get("turn_id"),
                    "seeker_turn_id": previous_turn.get("turn_id"),
                    "quality_tier": quality_tier,
                    "esc_stage": turn.get("esc_stage"),
                    "esconv_strategy": supporter_plan.get("esconv_strategy"),
                    "response_style": supporter_plan.get("response_style"),
                },
            }
            records.append(record)
            stats["exported_samples"] += 1

    return records, stats


def save_jsonl(records: list[dict], output_path: Path) -> None:
    """Write records in JSONL format"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_handle:
        for record in records:
            file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    """Run dataset preparation for Gemma"""
    conversations = load_dataset(DATASET_PATH)
    filtered_conversations = filter_gemma_training_conversations(conversations)
    records, stats = extract_gemma_samples(filtered_conversations)
    save_jsonl(records, GEMMA_OUTPUT_PATH)

    print("=" * 72)
    print(" SAATHI STEP 2: GEMMA DATASET PREPARATION ".center(72))
    print("=" * 72)
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Total conversations loaded: {len(conversations)}")
    print(f"Gemma-eligible conversations kept: {len(filtered_conversations)}")
    print(f"Supporter turns seen: {stats['supporter_turns_seen']}")
    print(f"Supporter turns exported: {stats['exported_samples']}")
    print(f"Malformed/skipped turns: {stats['skipped_turns']}")
    print(f"Saved Gemma dataset: {GEMMA_OUTPUT_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()