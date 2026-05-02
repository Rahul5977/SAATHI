"""
SAATHI dataset parser.

Two-stage workflow:
  1. Auto-detect the JSON structure of the dataset (run with --detect-only first).
  2. Parse every Supporter turn into a flat record list saved to data/parsed_records.json.

Usage:
    python -m indexing.parse_conversations --detect-only
    python -m indexing.parse_conversations
    python -m indexing.parse_conversations --sample 3
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

from config import DATASET_DIR, DATA_DIR, PARSED_RECORDS_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("parse_conversations")



# Candidate field names — used by both the detector and the parser.
# Order matters: the first match wins.

CANDIDATES = {
    "conversations_key": ["conversations", "dialogues", "data", "examples", "samples"],
    "turns_key": ["turns", "dialogue", "messages", "utterances", "conversation"],
    "speaker_key": ["role", "speaker", "from", "type", "actor"],
    "text_key": ["content", "text", "utterance", "message", "value"],
    "strategy_key": ["strategy", "support_strategy", "response_strategy"],
    "phase_key": ["phase", "stage", "conversation_phase", "support_phase"],
    "coping_key": ["coping_mechanism_active", "coping_mechanism", "coping", "coping_style"],
    "emotion_key": ["emotion_type", "emotion", "seeker_emotion"],
    "intensity_key": ["emotion_intensity", "intensity", "emotion_level"],
    "lens_key": ["lens", "cultural_lens", "frame"],
    "rationale_key": ["rationale", "reason", "explanation", "justification"],
    "coping_shade_key": ["coping_shade_signal", "coping_shade", "shade_signal"],
    "stigma_key": ["stigma_cue", "stigma", "stigma_signal"],
    "persona_key": ["persona_code", "persona", "persona_id"],
}

SEEKER_LABELS = {"seeker", "user", "client", "human", "patient", "student"}
SUPPORTER_LABELS = {"supporter", "assistant", "counselor", "therapist", "helper", "saathi", "bot"}



# Helpers
def _find_first_key(d: dict, candidates: list[str]) -> Optional[str]:
    """Return the first candidate name present in dict d (case-insensitive)."""
    if not isinstance(d, dict):
        return None
    lower_map = {k.lower(): k for k in d.keys()}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _candidate_json_files_for_detection(root: Path) -> list[Path]:
    """
    Return JSON files ordered by likelihood of being conversation data.

    Priority:
      1) files under a `conversations` path segment
      2) all remaining JSON files
    """
    all_files = sorted(root.rglob("*.json"))
    convo_files: list[Path] = []
    other_files: list[Path] = []
    for p in all_files:
        if "conversations" in {part.lower() for part in p.parts}:
            convo_files.append(p)
        else:
            other_files.append(p)
    return convo_files + other_files


def _extract_conversations_and_turns(data: Any) -> tuple[list, Optional[str], list, Optional[str]]:
    """
    Best-effort extractor used by structure detection.

    Returns:
      (conversations, conversations_key, turns, turns_key)
    """
    conversations: list = []
    conversations_key: Optional[str] = None
    turns: list = []
    turns_key: Optional[str] = None

    if isinstance(data, list):
        conversations = data
        conversations_key = "<root list>"
    elif isinstance(data, dict):
        conversations_key = _find_first_key(data, CANDIDATES["conversations_key"])
        if conversations_key and isinstance(data[conversations_key], list):
            conversations = data[conversations_key]
        else:
            turns_key_top = _find_first_key(data, CANDIDATES["turns_key"])
            if turns_key_top and isinstance(data[turns_key_top], list):
                conversations = [data]
                conversations_key = "<single-conversation file>"
            else:
                for k, v in data.items():
                    if isinstance(v, list) and v and isinstance(v[0], dict):
                        conversations = v
                        conversations_key = k
                        break

    if not conversations:
        return [], conversations_key, [], turns_key

    first_conv = conversations[0]
    if isinstance(first_conv, list):
        turns = first_conv
        turns_key = "<root list>"
    elif isinstance(first_conv, dict):
        turns_key = _find_first_key(first_conv, CANDIDATES["turns_key"])
        if turns_key and isinstance(first_conv[turns_key], list):
            turns = first_conv[turns_key]
        else:
            for k, v in first_conv.items():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    turns = v
                    turns_key = k
                    break

    return conversations, conversations_key, turns, turns_key


def _looks_like_conversation_turn(turn: dict) -> bool:
    """Heuristic: a turn should have at least a speaker-ish field and text-ish field."""
    if not isinstance(turn, dict):
        return False
    speaker_key = _find_first_key(turn, CANDIDATES["speaker_key"])
    text_key = _find_first_key(turn, CANDIDATES["text_key"])
    return speaker_key is not None and text_key is not None


def _all_json_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.json"))


def _category_for(path: Path, dataset_root: Path) -> str:
    """Infer the category folder name from a JSON file's path relative to the dataset."""
    try:
        rel = path.relative_to(dataset_root)
    except ValueError:
        return "unknown"
    parts = rel.parts
    if len(parts) >= 2:
        # e.g. dataset/conversations/Academic-Presuure/Prompt-10.json
        # take the last directory in the path (closest to the file) as category
        return parts[-2]
    return "unknown"


def _normalize_speaker(value: Any) -> str:
    """Normalize a speaker label to 'seeker' / 'supporter' / 'other'."""
    if value is None:
        return "other"
    s = str(value).strip().lower()
    if s in SEEKER_LABELS:
        return "seeker"
    if s in SUPPORTER_LABELS:
        return "supporter"
    # heuristic substring match
    if any(lbl in s for lbl in SEEKER_LABELS):
        return "seeker"
    if any(lbl in s for lbl in SUPPORTER_LABELS):
        return "supporter"
    return "other"


def _safe_get(turn: dict, key: Optional[str], default: Any = None) -> Any:
    if key is None:
        return default
    return turn.get(key, default)

# Stage 1: detect_structure
def detect_structure(dataset_dir: Path) -> dict:
    """
    Inspect the first JSON file and propose a field-name mapping.
    Returns a dict that can be passed to parse_all().
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    candidate_files = _candidate_json_files_for_detection(dataset_dir)
    if not candidate_files:
        raise FileNotFoundError(f"No .json files found anywhere under: {dataset_dir}")
    sample_file: Optional[Path] = None
    conversations: list = []
    turns: list = []
    conversations_key: Optional[str] = None
    turns_key: Optional[str] = None

    # Scan candidate files and pick the first one that genuinely looks like conversation data.
    for candidate in candidate_files:
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        convs, convs_key, candidate_turns, candidate_turns_key = _extract_conversations_and_turns(data)
        if convs and candidate_turns and isinstance(candidate_turns[0], dict) and _looks_like_conversation_turn(candidate_turns[0]):
            sample_file = candidate
            conversations = convs
            turns = candidate_turns
            conversations_key = convs_key
            turns_key = candidate_turns_key
            break

    if sample_file is None:
        raise ValueError(
            "Could not auto-detect a conversation-format JSON file. "
            "Please verify files under dataset/conversations/ contain turn-level speaker/text fields."
        )

    logger.info(f"Detecting structure from sample file: {sample_file}")

    # ----- detect per-turn field names -----
    sample_turn = turns[0] if isinstance(turns[0], dict) else {}
    mapping = {
        "conversations_key": conversations_key,
        "turns_key": turns_key,
        "speaker_key": _find_first_key(sample_turn, CANDIDATES["speaker_key"]),
        "text_key": _find_first_key(sample_turn, CANDIDATES["text_key"]),
        "strategy_key": _find_first_key(sample_turn, CANDIDATES["strategy_key"]),
        "phase_key": _find_first_key(sample_turn, CANDIDATES["phase_key"]),
        "coping_key": _find_first_key(sample_turn, CANDIDATES["coping_key"]),
        "emotion_key": _find_first_key(sample_turn, CANDIDATES["emotion_key"]),
        "intensity_key": _find_first_key(sample_turn, CANDIDATES["intensity_key"]),
        "lens_key": _find_first_key(sample_turn, CANDIDATES["lens_key"]),
        "rationale_key": _find_first_key(sample_turn, CANDIDATES["rationale_key"]),
        "coping_shade_key": _find_first_key(sample_turn, CANDIDATES["coping_shade_key"]),
        "stigma_key": _find_first_key(sample_turn, CANDIDATES["stigma_key"]),
        "persona_key": _find_first_key(sample_turn, CANDIDATES["persona_key"]),
    }

    # ----- pretty-print the report -----
    print("\n" + "=" * 78)
    print("STRUCTURE DETECTION REPORT")
    print("=" * 78)
    print(f"Sample file        : {sample_file}")
    print(f"File has {len(conversations)} conversations. "
          f"First conversation has {len(turns)} turns.")
    print(f"Conversations key  : {mapping['conversations_key']!r}")
    print(f"Turns key          : {mapping['turns_key']!r}")
    print()
    print("First 2 turns of first conversation:")
    print("-" * 78)
    for i, t in enumerate(turns[:2]):
        print(f"--- Turn {i} ---")
        print(json.dumps(t, indent=2, ensure_ascii=False)[:1500])
        print()

    print("Detected field mapping (None = field not present in sample):")
    print("-" * 78)
    for k, v in mapping.items():
        print(f"  {k:20s} -> {v!r}")
    print("=" * 78)
    print(
        "If any of these look wrong, re-run with --detect-only after editing CANDIDATES "
        "in indexing/parse_conversations.py, or pass overrides via the parse() API."
    )
    print()

    return mapping



# Stage 2: parse_all

def _extract_conversations(data: Any, mapping: dict) -> list:
    """Pull the list of conversations out of one loaded JSON file."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        ck = mapping.get("conversations_key")
        if ck and ck in data and isinstance(data[ck], list):
            return data[ck]
        # Single-conversation file?
        tk = mapping.get("turns_key")
        if tk and tk in data and isinstance(data[tk], list):
            return [data]
        # Fallback: first list-of-dicts value.
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    return []


def _extract_turns(conv: Any, mapping: dict) -> list:
    if isinstance(conv, list):
        return conv
    if isinstance(conv, dict):
        tk = mapping.get("turns_key")
        if tk and tk in conv and isinstance(conv[tk], list):
            return conv[tk]
        for v in conv.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    return []


def _build_record(
    *,
    category: str,
    filename: str,
    conversation_id: str,
    turn_number: int,
    seeker_turn: dict,
    supporter_turn: dict,
    mapping: dict,
) -> dict:
    text_key = mapping.get("text_key")

    seeker_text = _safe_get(seeker_turn, text_key, "") if seeker_turn else ""
    supporter_text = _safe_get(supporter_turn, text_key, "")

    # Most metadata lives on the Supporter turn (it's the annotated response).
    # Fall back to the Seeker turn if a field is missing on the Supporter side.
    def from_supporter_or_seeker(key_name: str, default: Any) -> Any:
        key = mapping.get(key_name)
        if key is None:
            return default
        if supporter_turn and key in supporter_turn and supporter_turn[key] not in (None, ""):
            return supporter_turn[key]
        if seeker_turn and key in seeker_turn and seeker_turn[key] not in (None, ""):
            return seeker_turn[key]
        return default

    try:
        intensity_raw = from_supporter_or_seeker("intensity_key", 3)
        seeker_intensity = int(intensity_raw) if intensity_raw is not None else 3
    except (TypeError, ValueError):
        seeker_intensity = 3

    stigma_raw = from_supporter_or_seeker("stigma_key", False)
    if isinstance(stigma_raw, str):
        seeker_stigma_cue = stigma_raw.strip().lower() in {"true", "yes", "1"}
    else:
        seeker_stigma_cue = bool(stigma_raw)

    return {
        "record_id": str(uuid.uuid4()),
        "category": category,
        "filename": filename,
        "conversation_id": conversation_id,
        "turn_number": turn_number,
        "seeker_text": str(seeker_text or ""),
        "seeker_emotion": str(from_supporter_or_seeker("emotion_key", "unknown")),
        "seeker_intensity": seeker_intensity,
        "seeker_coping": str(from_supporter_or_seeker("coping_key", "unknown")),
        "seeker_coping_shade": str(from_supporter_or_seeker("coping_shade_key", "")),
        "seeker_stigma_cue": seeker_stigma_cue,
        "supporter_text": str(supporter_text or ""),
        "strategy": str(from_supporter_or_seeker("strategy_key", "")),
        "phase": str(from_supporter_or_seeker("phase_key", "")),
        "lens": from_supporter_or_seeker("lens_key", None),
        "rationale": str(from_supporter_or_seeker("rationale_key", "")),
        "persona_code": str(from_supporter_or_seeker("persona_key", "P0")),
    }


def parse_all(dataset_dir: Path, mapping: dict) -> list[dict]:
    """Walk every JSON file under dataset_dir and produce a flat list of records."""
    files = _all_json_files(dataset_dir)
    if not files:
        raise FileNotFoundError(f"No .json files found under {dataset_dir}")

    logger.info(f"Found {len(files)} JSON files under {dataset_dir}")

    records: list[dict] = []
    speaker_key = mapping.get("speaker_key")

    files_skipped = 0
    convs_skipped = 0
    invalid_conversation_files: list[str] = []
    files_with_no_supporter_turns: list[str] = []

    for fp in files:
        category = _category_for(fp, dataset_dir)
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Skipping unreadable file {fp}: {e}")
            files_skipped += 1
            continue

        conversations = _extract_conversations(data, mapping)
        if not conversations:
            convs_skipped += 1
            continue

        for conv_idx, conv in enumerate(conversations):
            turns = _extract_turns(conv, mapping)
            if not turns:
                continue

            conv_id = f"{category}_{fp.stem}_{conv_idx}"
            supporter_turn_count = 0
            seen_turn_like_object = False

            for t_idx, turn in enumerate(turns):
                if not isinstance(turn, dict):
                    continue
                seen_turn_like_object = True

                # Some source files may contain generator error payloads instead of chat turns.
                # Example: {"error": "COMPAT_FAIL", "reason": "..."}.
                if "error" in turn and mapping.get("speaker_key") not in turn:
                    continue
                speaker_val = turn.get(speaker_key) if speaker_key else turn.get("role")
                speaker = _normalize_speaker(speaker_val)
                if speaker != "supporter":
                    continue

                # Find the Seeker turn immediately before this Supporter turn.
                seeker_turn: Optional[dict] = None
                for j in range(t_idx - 1, -1, -1):
                    prev = turns[j]
                    if not isinstance(prev, dict):
                        continue
                    prev_speaker = _normalize_speaker(
                        prev.get(speaker_key) if speaker_key else prev.get("role")
                    )
                    if prev_speaker == "seeker":
                        seeker_turn = prev
                        break

                rec = _build_record(
                    category=category,
                    filename=fp.name,
                    conversation_id=conv_id,
                    turn_number=supporter_turn_count,
                    seeker_turn=seeker_turn or {},
                    supporter_turn=turn,
                    mapping=mapping,
                )
                records.append(rec)
                supporter_turn_count += 1

            if supporter_turn_count == 0 and seen_turn_like_object:
                files_with_no_supporter_turns.append(str(fp))
                # Mark as invalid when it looks like a non-conversation payload file.
                # (all turn-like dicts missing speaker key)
                speaker_key_name = mapping.get("speaker_key")
                all_missing_speaker = all(
                    isinstance(t, dict) and (not speaker_key_name or speaker_key_name not in t)
                    for t in turns
                    if isinstance(t, dict)
                )
                if all_missing_speaker:
                    invalid_conversation_files.append(str(fp))

    if files_skipped:
        logger.warning(f"{files_skipped} file(s) were skipped due to read errors.")
    if convs_skipped:
        logger.warning(f"{convs_skipped} file(s) had no recognizable conversations list.")
    if files_with_no_supporter_turns:
        logger.warning(
            f"{len(files_with_no_supporter_turns)} file(s) had zero Supporter turns "
            "(likely incomplete/invalid conversation outputs)."
        )
    if invalid_conversation_files:
        logger.warning("Invalid conversation payload files detected (showing up to 10):")
        for p in invalid_conversation_files[:10]:
            logger.warning(f"  - {p}")

    return records



# Reporting

def print_summary(records: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("PARSE SUMMARY")
    print("=" * 78)
    print(f"Total records: {len(records)}")

    by_category = Counter(r["category"] for r in records)
    print("\nRecords per category:")
    print("-" * 50)
    print(f"{'Category':<35s}{'Count':>10s}")
    print("-" * 50)
    for cat, n in sorted(by_category.items(), key=lambda x: -x[1]):
        print(f"{cat:<35s}{n:>10d}")

    def _print_counts(title: str, key: str, top: int = 25) -> None:
        c = Counter((r.get(key) or "").strip() for r in records)
        c.pop("", None)
        print(f"\nUnique {title} ({len(c)} distinct values):")
        print("-" * 50)
        for val, n in c.most_common(top):
            print(f"  {n:>5d}  {val}")
        if len(c) > top:
            print(f"  ... ({len(c) - top} more)")

    _print_counts("strategies", "strategy")
    _print_counts("coping mechanisms", "seeker_coping")
    _print_counts("phases", "phase")

    missing_strategy = sum(1 for r in records if not (r.get("strategy") or "").strip())
    missing_seeker = sum(1 for r in records if not (r.get("seeker_text") or "").strip())
    if missing_strategy or missing_seeker:
        print("\nRecords with missing critical fields:")
        print(f"  missing strategy   : {missing_strategy}")
        print(f"  missing seeker_text: {missing_seeker}")
    else:
        print("\nAll records have non-empty strategy and seeker_text.")
    print("=" * 78)


def print_samples(records: list[dict], n: int) -> None:
    if not records:
        return
    print("\n" + "=" * 78)
    print(f"SAMPLE RECORDS (n={n})")
    print("=" * 78)
    chosen = random.sample(records, min(n, len(records)))
    for i, r in enumerate(chosen, 1):
        print(f"\n--- Sample {i}/{len(chosen)} ---")
        print(json.dumps(r, indent=2, ensure_ascii=False))
    print("=" * 78)



# CLI

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="parse_conversations",
        description="Detect dataset structure and/or parse SAATHI conversations into flat records.",
    )
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="Only run structure detection on the first JSON file and stop.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="After parsing, print N random records in full.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Override dataset directory (defaults to config.DATASET_DIR).",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve() if args.dataset_dir else DATASET_DIR
    logger.info(f"Using dataset directory: {dataset_dir}")

    mapping = detect_structure(dataset_dir)

    if args.detect_only:
        logger.info("--detect-only specified; stopping after structure detection.")
        return 0

    logger.info("Starting full parse...")
    records = parse_all(dataset_dir, mapping)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PARSED_RECORDS_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    logger.info(f"Wrote {len(records)} records to {PARSED_RECORDS_PATH}")

    print_summary(records)
    if args.sample > 0:
        print_samples(records, args.sample)

    return 0


if __name__ == "__main__":
    sys.exit(main())
