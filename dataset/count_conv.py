"""
count_conversations.py  (v2)
============================
Counts conversations generated vs situations, with:
  - Fuzzy folder name matching (handles typos + case differences)
  - New file structure: list[ list[turn_dict] | error_dict ]
  - Skips error records like {"error": "COMPAT_FAIL", ...}
  - Handles files where multiple conversations are merged into one array

Run
---
    python count_conversations.py --root /home/suraj/Desktop/dataset/dataset
"""

import json
import re
import argparse
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Folder-name normalisation + fuzzy matching
# ---------------------------------------------------------------------------

def normalise_folder_name(name: str) -> str:
    """
    Aggressively normalise a folder name for comparison:
      - lowercase
      - remove hyphens, underscores, spaces
      - fix known typos phonetically

    Examples:
      'Academic-Presuure'  → 'academicpressure'
      'Academic-Pressure'  → 'academicpressure'
      'Employement'        → 'employment'
      'Employment'         → 'employment'
      'Familial-and-Interpersonal-Conflicts' → 'familialandinterpersonalconflicts'
      'financial'          → 'financial'
      'Financial'          → 'financial'
      'Marriage-and-Pressure' → 'marriageandpressure'
      'marriage'           → 'marriage'   ← won't match above!
    """
    s = name.lower().strip()
    # remove separators
    s = re.sub(r"[-_\s]+", "", s)
    # fix known typos
    typo_fixes = {
        "presuure": "pressure",   # Academic-Presuure
        "employement": "employment",
    }
    for wrong, right in typo_fixes.items():
        s = s.replace(wrong, right)
    return s


# Manually define the canonical mapping from situation-folder name
# to all possible conversation-folder name variants.
# Key   = normalised situation folder name
# Value = set of normalised conversation folder names that match it
MANUAL_MATCH = {
    # situation folder norm      : conv folder norm(s)
    "academicpressure":          {"academicpressure", "academicpresuure"},
    "employment":                {"employment", "employement"},
    "familialandinterpersonalconflicts": {"familialandinterpersonalconflicts",
                                          "familial", "familialandinterpersonal"},
    "financial":                 {"financial"},
    "gender":                    {"gender"},
    "health":                    {"health"},
    "marriageandpressure":       {"marriageandpressure", "marriage",
                                  "marriagepressure"},
    "migration":                 {"migration"},
}


def build_sit_to_conv_map(all_sit_cats: set, all_conv_cats: set) -> dict:
    """
    Returns {sit_cat_original: conv_cat_original | None}
    Uses MANUAL_MATCH first, then falls back to normalised string equality.
    """
    mapping = {}
    for sit_cat in all_sit_cats:
        sit_norm = normalise_folder_name(sit_cat)

        matched = None
        # try manual map first
        allowed_conv_norms = MANUAL_MATCH.get(sit_norm, {sit_norm})

        for conv_cat in all_conv_cats:
            conv_norm = normalise_folder_name(conv_cat)
            if conv_norm in allowed_conv_norms:
                matched = conv_cat
                break

        mapping[sit_cat] = matched  # None if no match
    return mapping


# ---------------------------------------------------------------------------
# Prompt-stem normalisation
# ---------------------------------------------------------------------------

def normalise_prompt_stem(stem: str) -> str:
    """
    Prompt-1, Prompt1, prompt1, PROMPT-1 → 'prompt1'
    Strips non-alphanumeric, lowercases.
    """
    return re.sub(r"[^a-z0-9]", "", stem.lower())


# ---------------------------------------------------------------------------
# Situation counter
# ---------------------------------------------------------------------------

def count_situations(sit_root: Path) -> dict:
    """
    Returns {category_folder_name: {prompt_stem: n_situations}}
    """
    result = defaultdict(dict)
    for p in sorted(sit_root.rglob("*")):
        if not (p.is_file() and p.suffix.lower() == ".json"):
            continue
        category = p.parent.name
        stem     = normalise_prompt_stem(p.stem)
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            count = len(data) if isinstance(data, list) else 1
        except Exception as e:
            print(f"  ⚠ Skipping situation file {p}: {e}")
            count = 0
        result[category][stem] = count
    return dict(result)


# ---------------------------------------------------------------------------
# Conversation counter  (handles new mixed structure)
# ---------------------------------------------------------------------------

def is_error_record(item) -> bool:
    """Return True if item is an error dict like {"error": "COMPAT_FAIL", ...}"""
    return isinstance(item, dict) and "error" in item


def is_valid_conversation(item) -> bool:
    """
    A valid conversation is a list of turn dicts (at least one non-error dict).
    Rejects:
      - error dicts  {"error": ...}
      - empty lists
      - lists whose first element is an error dict
    """
    if not isinstance(item, list):
        return False
    if len(item) == 0:
        return False
    # check first element
    first = item[0]
    if is_error_record(first):
        return False
    if not isinstance(first, dict):
        return False
    return True


def count_conversations(conv_root: Path) -> dict:
    """
    Returns {category_folder_name: {prompt_stem: n_valid_conversations}}

    File structure (confirmed):
        list[
            list[turn_dict, ...]   ← valid conversation
          | {"error": ..., ...}    ← skip
        ]

    'n_valid_conversations' = number of inner lists that pass is_valid_conversation().
    """
    result    = defaultdict(dict)
    err_log   = defaultdict(lambda: defaultdict(int))   # category/stem → error count

    for p in sorted(conv_root.rglob("*")):
        if not (p.is_file() and p.suffix.lower() == ".json"):
            continue
        category = p.parent.name
        stem     = normalise_prompt_stem(p.stem)
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ⚠ Skipping conv file {p}: {e}")
            result[category][stem] = 0
            continue

        if not isinstance(data, list):
            print(f"  ⚠ Unexpected top-level type {type(data)} in {p}")
            result[category][stem] = 0
            continue

        valid_count = 0
        error_count = 0

        for item in data:
            if is_error_record(item):
                error_count += 1
            elif is_valid_conversation(item):
                valid_count += 1
            else:
                # unknown structure — log but don't count
                error_count += 1

        result[category][stem]        = valid_count
        err_log[category][stem]       = error_count

    # print error summary
    total_errors = sum(
        v for cat in err_log.values() for v in cat.values()
    )
    if total_errors:
        print(f"\n  ── Error/skipped conversation records ──────────────────")
        for cat in sorted(err_log):
            for stem in sorted(err_log[cat]):
                n = err_log[cat][stem]
                if n:
                    print(f"    {cat}/{stem}: {n} error record(s) skipped")
        print(f"  Total error records skipped: {total_errors}\n")

    return dict(result)


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def compare(sit_root: Path, conv_root: Path):

    sit_data  = count_situations(sit_root)
    conv_data = count_conversations(conv_root)

    all_sit_cats  = set(sit_data.keys())
    all_conv_cats = set(conv_data.keys())

    # build fuzzy sit→conv category map
    cat_map = build_sit_to_conv_map(all_sit_cats, all_conv_cats)

    print(f"\n  Category mapping resolved:")
    for sit_cat, conv_cat in sorted(cat_map.items()):
        status = conv_cat if conv_cat else "✗ NO MATCH"
        print(f"    {sit_cat:<45} → {status}")

    total_situations    = 0
    total_conversations = 0
    total_missing       = 0
    total_partial       = 0
    total_error_skipped = 0

    print("\n" + "=" * 80)
    print("  SITUATION vs CONVERSATION COUNT")
    print("=" * 80)

    for sit_cat in sorted(all_sit_cats):
        matched_conv_cat = cat_map.get(sit_cat)

        print(f"\n{'─' * 80}")
        print(f"  SITUATION CATEGORY : {sit_cat}")

        if matched_conv_cat is None:
            print(f"  ✗ NO matching conversation folder found!")
            cat_sit_total = sum(sit_data[sit_cat].values())
            total_situations += cat_sit_total
            total_missing    += cat_sit_total
            print(f"    Situations  : {cat_sit_total}")
            print(f"    Convs found : 0")
            continue

        print(f"  CONV CATEGORY      : {matched_conv_cat}")
        print(f"{'─' * 80}")
        print(f"  {'Prompt':<15} {'Situations':>12} {'Convs Generated':>16} "
              f"{'Missing':>9} {'Status':>10}")
        print(f"  {'─'*15} {'─'*12} {'─'*16} {'─'*9} {'─'*10}")

        all_stems = sorted(
            set(sit_data[sit_cat].keys()) |
            set(conv_data[matched_conv_cat].keys())
        )

        cat_sit_total  = 0
        cat_conv_total = 0

        for stem in all_stems:
            n_sit  = sit_data[sit_cat].get(stem, 0)
            n_conv = conv_data[matched_conv_cat].get(stem, 0)
            n_miss = max(n_sit - n_conv, 0)

            cat_sit_total  += n_sit
            cat_conv_total += n_conv

            if n_conv == 0 and n_sit > 0:
                status = "MISSING"
                total_missing += n_sit
            elif n_conv < n_sit:
                status = "PARTIAL"
                total_partial += n_miss
            elif n_conv == n_sit:
                status = "✓ OK"
            else:
                status = "EXTRA"   # more convs than situations (shouldn't happen)

            print(f"  {stem:<15} {n_sit:>12} {n_conv:>16} {n_miss:>9} {status:>10}")

        total_situations    += cat_sit_total
        total_conversations += cat_conv_total

        print(f"  {'─'*15} {'─'*12} {'─'*16} {'─'*9} {'─'*10}")
        print(f"  {'CATEGORY TOTAL':<15} {cat_sit_total:>12} {cat_conv_total:>16} "
              f"{max(cat_sit_total - cat_conv_total, 0):>9}")

    # ── conversation folders with no situation match ──────────────────────────
    matched_conv_cats = set(v for v in cat_map.values() if v is not None)
    unmatched_conv    = sorted(all_conv_cats - matched_conv_cats)
    if unmatched_conv:
        print(f"\n{'─' * 80}")
        print("  CONVERSATION FOLDERS WITH NO MATCHING SITUATION CATEGORY:")
        for c in unmatched_conv:
            cat_total = sum(conv_data[c].values())
            total_conversations += cat_total
            print(f"    {c}  →  {cat_total} conversations (unmatched)")

    # ── grand summary ─────────────────────────────────────────────────────────
    missing_total = max(total_situations - total_conversations, 0)
    coverage_pct  = (
        round(100 * total_conversations / total_situations, 2)
        if total_situations else 0.0
    )

    print(f"\n{'=' * 80}")
    print("  GRAND SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Total situations                    : {total_situations}")
    print(f"  Total conversations generated        : {total_conversations}")
    print(f"  Total missing (not yet generated)    : {missing_total}")
    print(f"  Coverage                             : {coverage_pct}%")
    print(f"{'=' * 80}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str, required=True,
        help="Path to inner dataset folder "
             "(contains conversations/ and Situation_Categories/)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    conv_root = sit_root = None
    for child in root.iterdir():
        lname = child.name.lower()
        if lname == "conversations":
            conv_root = child
        elif lname in ("situation_categories", "situations"):
            sit_root = child

    if conv_root is None:
        raise FileNotFoundError(f"No 'conversations' folder under {root}")
    if sit_root is None:
        raise FileNotFoundError(f"No 'Situation_Categories' folder under {root}")

    print(f"  Situations   folder : {sit_root}")
    print(f"  Conversations folder: {conv_root}")

    compare(sit_root, conv_root)