"""
    python count_conversations.py --root /home/suraj/Desktop/dataset/dataset
"""

import json
import re
import argparse
from collections import defaultdict
from pathlib import Path


# Folder-name normalisation + fuzzy matching

def normalise_folder_name(name: str) -> str:
    """
    Aggressively normalise a folder name for comparison
    """
    s = name.lower().strip()
    # remove separators
    s = re.sub(r"[-_\s]+", "", s)
    # fix known typos
    typo_fixes = {
        "presuure":    "pressure",    # Academic-Presuure
        "employement": "employment",  # Employement
    }
    for wrong, right in typo_fixes.items():
        s = s.replace(wrong, right)
    return s


# Manually define the canonical mapping from situation-folder name
# to all possible conversation-folder name variants.
# Key   = normalised situation folder name
# Value = set of normalised conversation folder names that match it

MANUAL_MATCH = {
    # situation folder norm               : conv folder norm(s)
    "academicpressure":                   {"academicpressure", "academicpresuure"},
    "employment":                         {"employment", "employement"},
    "familialandinterpersonalconflicts":  {
                                            "familialandinterpersonalconflicts",
                                            "familial",
                                            "familialandinterpersonal",
                                            # Family-Pressure folder variant
                                            "familypressure",
                                            "family",
                                            "familyconflict",
                                            "familyandinterpersonal",
                                          },
    # In case the situation folder itself is named Family-Pressure
    "familypressure":                     {
                                            "familypressure",
                                            "family",
                                            "familyconflict",
                                            "familialandinterpersonalconflicts",
                                            "familialandinterpersonal",
                                            "familial",
                                          },
    "financial":                          {"financial"},
    "gender":                             {"gender"},
    "health":                             {"health"},
    "marriageandpressure":                {
                                            "marriageandpressure",
                                            "marriage",
                                            "marriagepressure",
                                          },
    "migration":                          {"migration"},
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


# Prompt-stem normalisation

# Prefixes to strip from filename stems before normalising.
_STEM_STRIP_PREFIXES = (
    "conversation_",
    "conversation-",
    "conversation",
    "conv_",
    "conv-",
    "conv",
)


def normalise_prompt_stem(stem: str) -> str:
    """
    Normalise a filename stem for comparison across situation and conversation
    files.
    """
    s = stem.lower().strip()

    # strip known prefixes (order matters — longest first wins)
    for prefix in sorted(_STEM_STRIP_PREFIXES, key=len, reverse=True):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break  # only strip one prefix

    # remove all non-alphanumeric
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


# Build a "display stem" that is human-readable in the table

def display_stem(raw_stem: str) -> str:
    """Return a cleaned but still human-readable version of the stem."""
    return raw_stem.lower().strip()


# Situation counter

def count_situations(sit_root: Path) -> dict:
    """
    Returns {category_folder_name: {norm_stem: (n_situations, raw_stem)}}
    """
    result = defaultdict(dict)
    for p in sorted(sit_root.rglob("*")):
        if not (p.is_file() and p.suffix.lower() == ".json"):
            continue
        category = p.parent.name
        raw_stem  = p.stem
        norm      = normalise_prompt_stem(raw_stem)
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            count = len(data) if isinstance(data, list) else 1
        except Exception as e:
            print(f"  ⚠ Skipping situation file {p}: {e}")
            count = 0
        result[category][norm] = (count, raw_stem)
    return dict(result)


# Conversation counter  (handles new mixed structure)

def is_error_record(item) -> bool:
    """Return True if item is an error dict like {"error": "COMPAT_FAIL", ...}"""
    return isinstance(item, dict) and "error" in item


def is_valid_conversation(item) -> bool:
    """
    A valid conversation is a list of turn dicts (at least one non-error dict).
    """
    if not isinstance(item, list):
        return False
    if len(item) == 0:
        return False
    first = item[0]
    if is_error_record(first):
        return False
    if not isinstance(first, dict):
        return False
    return True


def count_conversations(conv_root: Path) -> dict:
    """
    Returns {category_folder_name: {norm_stem: (n_valid_conversations, raw_stem)}}
    """
    result  = defaultdict(dict)
    err_log = defaultdict(lambda: defaultdict(int))

    for p in sorted(conv_root.rglob("*")):
        if not (p.is_file() and p.suffix.lower() == ".json"):
            continue
        category = p.parent.name
        raw_stem  = p.stem
        norm      = normalise_prompt_stem(raw_stem)

        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ⚠ Skipping conv file {p}: {e}")
            result[category][norm] = (0, raw_stem)
            continue

        if not isinstance(data, list):
            print(f"  ⚠ Unexpected top-level type {type(data)} in {p}")
            result[category][norm] = (0, raw_stem)
            continue

        valid_count = 0
        error_count = 0

        for item in data:
            if is_error_record(item):
                error_count += 1
            elif is_valid_conversation(item):
                valid_count += 1
            else:
                error_count += 1

        result[category][norm]   = (valid_count, raw_stem)
        err_log[category][norm]  = error_count

    # print error summary
    total_errors = sum(v for cat in err_log.values() for v in cat.values())
    if total_errors:
        print(f"\n  ── Error/skipped conversation records ──────────────────")
        for cat in sorted(err_log):
            for norm in sorted(err_log[cat]):
                n = err_log[cat][norm]
                if n:
                    raw = result[cat][norm][1]
                    print(f"    {cat}/{raw}: {n} error record(s) skipped")
        print(f"  Total error records skipped: {total_errors}\n")

    return dict(result)


# Main comparison

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
        print(f"    {sit_cat:<50} → {status}")

    total_situations    = 0
    total_conversations = 0
    total_missing       = 0
    total_partial       = 0

    print("\n" + "=" * 90)
    print("  SITUATION vs CONVERSATION COUNT")
    print("=" * 90)

    for sit_cat in sorted(all_sit_cats):
        matched_conv_cat = cat_map.get(sit_cat)

        print(f"\n{'─' * 90}")
        print(f"  SITUATION CATEGORY : {sit_cat}")

        if matched_conv_cat is None:
            print(f"  ✗ NO matching conversation folder found!")
            cat_sit_total = sum(v for v, _ in sit_data[sit_cat].values())
            total_situations += cat_sit_total
            total_missing    += cat_sit_total
            print(f"    Situations  : {cat_sit_total}")
            print(f"    Convs found : 0")
            continue

        print(f"  CONV CATEGORY      : {matched_conv_cat}")

        # resolve stems 
        # Build a unified view:
        #   norm_stem → (n_sit, sit_raw, n_conv, conv_raw)
        sit_stems  = sit_data[sit_cat]           # {norm: (count, raw)}
        conv_stems = conv_data[matched_conv_cat]  # {norm: (count, raw)}

        # Try to align stems.  If the sets of norm-stems differ entirely
        # (e.g. sit has 'prompt1' and conv has 'family'), we do a positional
        # merge so the numbers still appear side-by-side.
        sit_norms  = list(sit_stems.keys())
        conv_norms = list(conv_stems.keys())

        # all unique stems
        all_norms  = sorted(set(sit_norms) | set(conv_norms))

        # If there is ZERO overlap between sit norms and conv norms,
        # it's likely the file naming is completely different (e.g. Prompt-1
        # vs Conversation_Family).  In that case do a positional merge.
        overlap = set(sit_norms) & set(conv_norms)
        if not overlap and sit_norms and conv_norms:
            print(f"\n  ⚠  Stem mismatch detected — using positional merge.")
            print(f"     Situation stems : {sit_norms}")
            print(f"     Conv stems      : {conv_norms}")

            # zip longest
            from itertools import zip_longest
            rows = []
            for s_norm, c_norm in zip_longest(sit_norms, conv_norms):
                s_count, s_raw = sit_stems[s_norm]  if s_norm else (0, "—")
                c_count, c_raw = conv_stems[c_norm] if c_norm else (0, "—")
                rows.append((s_raw, s_norm, s_count, c_raw, c_norm, c_count))
        else:
            rows = []
            for norm in all_norms:
                s_count, s_raw = sit_stems.get(norm,  (0, "—"))
                c_count, c_raw = conv_stems.get(norm, (0, "—"))
                rows.append((s_raw, norm, s_count, c_raw, norm, c_count))

        # print table 
        print(f"{'─' * 90}")
        print(f"  {'Sit File':<22} {'Sit N':>7}  {'Conv File':<28} "
              f"{'Conv N':>7}  {'Missing':>8}  {'Status':>10}")
        print(f"  {'─'*22} {'─'*7}  {'─'*28} {'─'*7}  {'─'*8}  {'─'*10}")

        cat_sit_total  = 0
        cat_conv_total = 0

        for (s_raw, s_norm, n_sit, c_raw, c_norm, n_conv) in rows:
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
                status = "EXTRA"

            print(f"  {s_raw:<22} {n_sit:>7}  {c_raw:<28} "
                  f"{n_conv:>7}  {n_miss:>8}  {status:>10}")

        total_situations    += cat_sit_total
        total_conversations += cat_conv_total

        print(f"  {'─'*22} {'─'*7}  {'─'*28} {'─'*7}  {'─'*8}  {'─'*10}")
        print(f"  {'CATEGORY TOTAL':<22} {cat_sit_total:>7}  {'':28} "
              f"{cat_conv_total:>7}  {max(cat_sit_total - cat_conv_total, 0):>8}")

    # conversation folders with no situation match 
    matched_conv_cats = {v for v in cat_map.values() if v is not None}
    unmatched_conv    = sorted(all_conv_cats - matched_conv_cats)
    if unmatched_conv:
        print(f"\n{'─' * 90}")
        print("  CONVERSATION FOLDERS WITH NO MATCHING SITUATION CATEGORY:")
        for c in unmatched_conv:
            cat_total = sum(v for v, _ in conv_data[c].values())
            total_conversations += cat_total
            print(f"    {c}  →  {cat_total} conversations (unmatched)")

    # summary
    missing_total = max(total_situations - total_conversations, 0)
    coverage_pct  = (
        round(100 * total_conversations / total_situations, 2)
        if total_situations else 0.0
    )

    print(f"\n{'=' * 90}")
    print(" SUMMARY")
    print(f"{'=' * 90}")
    print(f"  Total situations                    : {total_situations}")
    print(f"  Total conversations generated        : {total_conversations}")
    print(f"  Total missing (not yet generated)    : {missing_total}")
    print(f"  Coverage                             : {coverage_pct}%")
    print(f"{'=' * 90}\n")


# CLI

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