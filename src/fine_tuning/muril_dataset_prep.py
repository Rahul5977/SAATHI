"""Step 1 dataset preparation for MuRIL fine-tuning.

This module reads the raw ``data_hindi.json`` dataset, removes crisis
conversations, extracts seeker turns, encodes the six MuRIL labels using fixed
vocabularies, and writes ``muril_dataset.jsonl`` plus ``label_mappings.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "data" / "data_hindi.json"
OUTPUT_DIR = ROOT / "data" / "processed"
MURIL_OUTPUT_PATH = OUTPUT_DIR / "muril_dataset.jsonl"
LABEL_MAPPING_PATH = OUTPUT_DIR / "label_mappings.json"

# Fixed vocabularies for each MuRIL head, used for stable label encoding and decoding
LABEL_VOCAB: dict[str, list[Any]] = {
    "emotion": [
        "anxiety",
        "sadness",
        "anger",
        "fear",
        "grief",
        "disgust",
        "frustration",
        "joy",
        "neutral",
    ],
    "intensity": [0, 1, 2],
    "sentiment": ["negative", "neutral", "positive"],
    "intent": ["venting", "advice-seeking", "validation-seeking", "crisis"],
    "cultural_signal": [
        "academic_stress",
        "family_pressure",
        "career_stress",
        "relationship_issue",
        "financial_stress",
        "social_pressure",
        "other",
    ],
    "relationship_signal": [
        "parent",
        "sibling",
        "romantic_partner",
        "friend",
        "colleague",
        "self",
    ],
}


def load_dataset(dataset_path: Path) -> list[dict]:
    """Load the raw JSON dataset from disk"""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)

    if not isinstance(data, list):
        raise ValueError("Expected top-level dataset to be a list of conversations")

    return data


def is_crisis_conversation(conversation: dict) -> bool:
    """Return True when the conversation is marked as crisis"""
    session_metadata = conversation.get("session_metadata") or {}
    crisis_flag = session_metadata.get("crisis_flag") or {}

    if isinstance(crisis_flag, dict) and "is_crisis" in crisis_flag:
        return bool(crisis_flag.get("is_crisis"))

    return False


def filter_non_crisis_conversations(conversations: list[dict]) -> list[dict]:
    """Keep only conversations that are not marked as crisis"""
    return [conversation for conversation in conversations if not is_crisis_conversation(conversation)]


def build_label_mappings() -> dict:
    """Build fixed label-to-id and id-to-label mappings for all MuRIL heads"""
    mappings: dict[str, dict[str, dict[str, Any]]] = {}

    for head_name, vocabulary in LABEL_VOCAB.items():
        label_to_id = {str(label): index for index, label in enumerate(vocabulary)}
        id_to_label = {str(index): label for index, label in enumerate(vocabulary)}
        mappings[head_name] = {
            "label_to_id": label_to_id,
            "id_to_label": id_to_label,
        }

    return mappings


def normalize_seeker_labels(seeker_analysis: dict) -> dict:
    """Normalize seeker labels and apply required null handling"""
    normalized = dict(seeker_analysis)
    normalized["cultural_signal"] = normalized.get("cultural_signal") or "other"
    normalized["relationship_signal"] = normalized.get("relationship_signal") or "self"

    if normalized.get("intensity") is None:
        raise ValueError("Missing required intensity label")

    normalized["intensity"] = int(normalized["intensity"])
    return normalized


def _encode_label(label_name: str, raw_value: Any, label_mappings: dict) -> int:
    """Convert a raw label into a stable integer id"""
    label_lookup = label_mappings[label_name]["label_to_id"]
    key = str(raw_value)

    if key not in label_lookup:
        raise ValueError(f"Unknown label for {label_name}: {raw_value}")

    return int(label_lookup[key])


def extract_muril_samples(conversations: list[dict], label_mappings: dict) -> tuple[list[dict], dict]:
    """Extract seeker-turn records for MuRIL training"""
    samples: list[dict] = []
    stats = {
        "total_turns_seen": 0,
        "seeker_turns_seen": 0,
        "exported_samples": 0,
        "skipped_turns": 0,
    }

    for conversation in conversations:
        dialog = conversation.get("dialog") or []
        session_metadata = conversation.get("session_metadata") or {}
        conversation_context = conversation.get("conversation_context") or {}
        conversation_id = conversation.get("conversation_id")

        for turn in dialog:
            stats["total_turns_seen"] += 1

            if turn.get("speaker") != "seeker":
                continue

            stats["seeker_turns_seen"] += 1
            text = turn.get("content_hinglish")
            seeker_analysis = turn.get("seeker_analysis")

            if not isinstance(text, str) or not text.strip():
                stats["skipped_turns"] += 1
                continue

            if not isinstance(seeker_analysis, dict):
                stats["skipped_turns"] += 1
                continue

            try:
                normalized_labels = normalize_seeker_labels(seeker_analysis)
                encoded_labels = {
                    "emotion": _encode_label("emotion", normalized_labels["emotion"], label_mappings),
                    "intensity": _encode_label("intensity", normalized_labels["intensity"], label_mappings),
                    "sentiment": _encode_label("sentiment", normalized_labels["sentiment"], label_mappings),
                    "intent": _encode_label("intent", normalized_labels["intent"], label_mappings),
                    "cultural_signal": _encode_label(
                        "cultural_signal",
                        normalized_labels["cultural_signal"],
                        label_mappings,
                    ),
                    "relationship_signal": _encode_label(
                        "relationship_signal",
                        normalized_labels["relationship_signal"],
                        label_mappings,
                    ),
                }
            except (KeyError, TypeError, ValueError):
                stats["skipped_turns"] += 1
                continue

            sample = {
                "text": text.strip(),
                "labels": encoded_labels,
                "metadata": {
                    "conversation_id": conversation_id,
                    "turn_id": turn.get("turn_id"),
                    "problem_type": session_metadata.get("problem_type"),
                    "emotion_type": conversation_context.get("emotion_type"),
                },
            }
            samples.append(sample)
            stats["exported_samples"] += 1

    return samples, stats


def save_jsonl(records: list[dict], output_path: Path) -> None:
    """Write records in JSONL format"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_handle:
        for record in records:
            file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_json(data: dict, output_path: Path) -> None:
    """Write JSON data with UTF-8 encoding"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(data, file_handle, ensure_ascii=False, indent=2)


def main() -> None:
    """Run dataset preparation for MuRIL"""
    conversations = load_dataset(DATASET_PATH)
    non_crisis_conversations = filter_non_crisis_conversations(conversations)
    label_mappings = build_label_mappings()
    muril_samples, stats = extract_muril_samples(non_crisis_conversations, label_mappings)

    save_jsonl(muril_samples, MURIL_OUTPUT_PATH)
    save_json(label_mappings, LABEL_MAPPING_PATH)

    print("=" * 72)
    print(" SAATHI STEP 1: MURIL DATASET PREPARATION ".center(72))
    print("=" * 72)
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Total conversations loaded: {len(conversations)}")
    print(f"Non-crisis conversations kept: {len(non_crisis_conversations)}")
    print(f"Total dialog turns seen: {stats['total_turns_seen']}")
    print(f"Seeker turns seen: {stats['seeker_turns_seen']}")
    print(f"Seeker turns exported: {stats['exported_samples']}")
    print(f"Malformed/skipped turns: {stats['skipped_turns']}")
    print(f"Saved MuRIL dataset: {MURIL_OUTPUT_PATH}")
    print(f"Saved label mappings: {LABEL_MAPPING_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()