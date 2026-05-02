"""
python calculate_report_metrics.py --root /path/to/dataset/dataset
python calculate_report_metrics.py --root /path/to/dataset/dataset --output metrics.json
python calculate_report_metrics.py --root /path/to/dataset/dataset --debug
"""

import json
import re
import argparse
from collections import defaultdict, Counter
from pathlib import Path


# Exact field names in the files 
# Conversation turn fields (all use underscores, standard naming)
T_TURN_ID          = "turn_id"
T_SPEAKER          = "speaker"
T_TEXT             = "text"
T_STAGE            = "stage"
T_STRATEGY         = "strategy"
T_RESTATEMENT_LENS = "restatement_lens"
T_STIGMA_RESPONSE  = "stigma_response"
T_STIGMA_CUE       = "stigma_cue"
T_EMOTION          = "emotion"
T_INTENSITY        = "intensity"
T_INTENSITY_SHIFT  = "intensity_shift"
T_COPING_SIGNAL    = "coping_shade_signal"
T_RISK_FLAG        = "risk_flag"

# Situation fields (note: spaces in key names!)
S_SIT_ID           = "situation id"        
S_TEXT             = "text"
S_CONTEXT_CAT      = "context_category"
S_COPING_MECH      = "coping_mechanism"
S_EMOTION_SEED     = "emotion seed"        
S_INTENSITY_SEED   = "intensity_seed"
S_CULTURAL_MARKERS = "cultural markers"    


# File-name normalisation

def normalise_prompt_stem(stem: str) -> str:
    """
    Normalise prompt file stems so all of these map to the same key:
      Prompt-1  Prompt1  prompt1  PROMPT-1  prompt-10  Prompt10
    Result is always like  'prompt1'  'prompt10'
    """
    s = stem.lower()
    # remove any non-alphanumeric characters (hyphens, underscores, spaces)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s          # e.g. "prompt1", "prompt10"


def iter_json_files(folder: Path):
    """
    Yield (Path, category_name) for every .json file found recursively.
    Case-insensitive suffix check handles .JSON, .Json, etc.
    """
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() == ".json":
            yield p, p.parent.name



# Situation loader

def load_situations(sit_root: Path, debug: bool = False) -> list:
    """
    Returns flat list of situation dicts.
    Each file: list[situation_dict]  (5 per file)
    Injects __category__ and __file_stem__ for traceability.
    """
    situations = []
    for p, category in iter_json_files(sit_root):
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"  Skipping {p}: {e}")
            continue

        if not isinstance(data, list):
            if debug:
                print(f"  [DEBUG] Situation file not a list: {p}")
            continue

        for obj in data:
            if isinstance(obj, dict):
                obj["__category__"]  = category
                obj["__file_stem__"] = normalise_prompt_stem(p.stem)
                situations.append(obj)
            else:
                if debug:
                    print(f"  [DEBUG] Non-dict item in situation file {p}: {type(obj)}")

    return situations



# Conversation loader


def load_conversations(conv_root: Path, debug: bool = False) -> dict:
    """
    Conversation file structure:
        list[list[turn_dict]]
        ^--- 5 conversations (one per situation variant S001-S005)
             ^--- ordered turns for that conversation

    Returns:
        {dialogue_id: [turn_dict, ...]}
        dialogue_id = "<category>/<prompt_stem>/<conv_index>"
        e.g.  "academic-presuure/prompt1/0"
    """
    dialogues: dict = {}

    for p, category in iter_json_files(conv_root):
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"  ⚠ Skipping {p}: {e}")
            continue

        if not isinstance(data, list):
            if debug:
                print(f"  [DEBUG] Conv file not a list: {p}")
            continue

        stem = normalise_prompt_stem(p.stem)
        cat  = category.lower()

        for conv_idx, conv in enumerate(data):
            # Each element should be a list of turn dicts
            if isinstance(conv, list):
                turns = [t for t in conv if isinstance(t, dict)]
            elif isinstance(conv, dict):
                # Fallback: single turn stored as dict (shouldn't happen per diagnosis)
                turns = [conv]
            else:
                if debug:
                    print(f"  [DEBUG] Unexpected conv item type {type(conv)} "
                          f"in {p}[{conv_idx}]")
                continue

            if not turns:
                continue

            # Sort turns by turn_id to ensure correct order
            turns.sort(key=lambda t: t.get(T_TURN_ID, 0))

            dialogue_id = f"{cat}/{stem}/{conv_idx}"
            dialogues[dialogue_id] = turns

    return dialogues



# Dataset entry point

def load_dataset(root: Path, debug: bool = False):
    """
    Locate conversations/ and Situation_Categories/ under root
    (case-insensitive) and load them.

    Returns (situations: list, dialogues: dict)
    """
    conv_root = sit_root = None
    for child in root.iterdir():
        lname = child.name.lower()
        if lname == "conversations":
            conv_root = child
        elif lname in ("situation_categories", "situations"):
            sit_root = child

    if conv_root is None:
        raise FileNotFoundError(f"No 'conversations' folder found under {root}")
    if sit_root is None:
        raise FileNotFoundError(f"No 'Situation_Categories'/'situations' folder under {root}")

    print(f"  Conversations folder : {conv_root}")
    print(f"  Situations   folder  : {sit_root}")

    situations = load_situations(sit_root,  debug=debug)
    dialogues  = load_conversations(conv_root, debug=debug)

    return situations, dialogues



# Simple helpers

def word_count(text) -> int:
    if not text or not isinstance(text, str):
        return 0
    return len(text.split())


def safe_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def mean(lst: list) -> float:
    return sum(lst) / len(lst) if lst else 0.0


def pct_dist(counter: Counter, total: int) -> dict:
    """Return {label: percentage} sorted by frequency descending."""
    if not total:
        return {}
    return {
        k: round(100 * v / total, 2)
        for k, v in counter.most_common()
    }



# Field-coverage report (printed before metrics)

def print_field_coverage(situations: list, dialogues: dict):
    all_turns = [t for turns in dialogues.values() for t in turns]

    print("\n Turn-level field coverage")
    turn_fields = [
        T_SPEAKER, T_TEXT, T_TURN_ID, T_STAGE,
        T_STRATEGY, T_RESTATEMENT_LENS, T_COPING_SIGNAL,
        T_STIGMA_CUE, T_STIGMA_RESPONSE, T_RISK_FLAG,
        T_EMOTION, T_INTENSITY, T_INTENSITY_SHIFT,
    ]
    n = len(all_turns)
    for f in turn_fields:
        non_null = sum(
            1 for t in all_turns
            if t.get(f) is not None and t.get(f) != ""
        )
        pct = 100 * non_null / n if n else 0
        bar = "█" * int(pct / 5)
        print(f"    {f:25s} {non_null:>5d}/{n:<5d} ({pct:5.1f}%)  {bar}")

    print("\n Situation-level field coverage")
    sit_fields = [
        S_EMOTION_SEED, S_INTENSITY_SEED, S_COPING_MECH,
        S_CONTEXT_CAT, S_CULTURAL_MARKERS, S_TEXT,
    ]
    m = len(situations)
    for f in sit_fields:
        non_null = sum(
            1 for s in situations
            if s.get(f) is not None and s.get(f) != ""
        )
        pct = 100 * non_null / m if m else 0
        bar = "█" * int(pct / 5)
        print(f"    {f:25s} {non_null:>5d}/{m:<5d} ({pct:5.1f}%)  {bar}")
    print()



# SECTION 1 : Corpus overview

def compute_corpus_overview(situations: list, dialogues: dict) -> dict:
    """
    num_situations               : total situation objects loaded
    num_dialogues                : unique (category/prompt/variant) conversations
    num_utterances_total         : all individual turns across all dialogues
    avg_turns_per_dialogue       : mean number of turns per dialogue
    avg_length_per_dialogue_wds  : mean total word count per dialogue
    avg_length_per_utterance_wds : mean word count per individual turn
    """
    num_situations       = len(situations)
    num_dialogues        = len(dialogues)

    all_turns            = [t for turns in dialogues.values() for t in turns]
    num_utterances_total = len(all_turns)

    turn_counts  = []
    dial_lengths = []
    utt_lengths  = []

    for turns in dialogues.values():
        turn_counts.append(len(turns))

        wc = sum(word_count(t.get(T_TEXT, "")) for t in turns)
        dial_lengths.append(wc)

        for t in turns:
            utt_lengths.append(word_count(t.get(T_TEXT, "")))

    return {
        "num_situations":               num_situations,
        "num_dialogues":                num_dialogues,
        "num_utterances_total":         num_utterances_total,
        "avg_turns_per_dialogue":       round(mean(turn_counts),  2),
        "avg_length_per_dialogue_wds":  round(mean(dial_lengths), 2),
        "avg_length_per_utterance_wds": round(mean(utt_lengths),  2),
    }



# SECTION 2 : Seeker vs Supporter split 

def compute_seeker_supporter_split(dialogues: dict) -> dict:
    seek_dial_lens   = []
    seek_utt_lens    = []
    supp_dial_lens   = []
    supp_utt_lens    = []
    seek_turn_counts = []
    supp_turn_counts = []

    for turns in dialogues.values():
        s_turns   = [t for t in turns if t.get(T_SPEAKER) == "Seeker"]
        sup_turns = [t for t in turns if t.get(T_SPEAKER) == "Supporter"]

        seek_dial_lens.append(sum(word_count(t.get(T_TEXT, "")) for t in s_turns))
        supp_dial_lens.append(sum(word_count(t.get(T_TEXT, "")) for t in sup_turns))
        seek_turn_counts.append(len(s_turns))
        supp_turn_counts.append(len(sup_turns))

        for t in s_turns:
            seek_utt_lens.append(word_count(t.get(T_TEXT, "")))
        for t in sup_turns:
            supp_utt_lens.append(word_count(t.get(T_TEXT, "")))

    return {
        "avg_seeker_turns_per_dialogue":       round(mean(seek_turn_counts), 2),
        "avg_supporter_turns_per_dialogue":    round(mean(supp_turn_counts), 2),
        "avg_seeker_len_per_dialogue_wds":     round(mean(seek_dial_lens),   2),
        "avg_seeker_len_per_utterance_wds":    round(mean(seek_utt_lens),    2),
        "avg_supporter_len_per_dialogue_wds":  round(mean(supp_dial_lens),   2),
        "avg_supporter_len_per_utterance_wds": round(mean(supp_utt_lens),    2),
    }



# SECTION 3 : Linguistic — Language & script

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def devanagari_ratio(text: str) -> float:
    if not text:
        return 0.0
    deva = len(DEVANAGARI_RE.findall(text))
    return deva / len(text)


def compute_linguistic(dialogues: dict) -> dict:
    """
    pct_hinglish_utterances           : % turns where coping_shade_signal is not None/empty
    avg_devanagari_char_ratio         : mean Devanagari fraction across all turn texts
    avg_hinglish_phrases_per_dialogue : mean count of non-null coping_shade_signal per dialogue
    intensity_shift_distribution_pct  : % of turns per intensity_shift value
    """
    total_turns       = 0
    hinglish_turns    = 0
    deva_ratios       = []
    hinglish_per_dial = []
    intensity_shifts  = []

    for turns in dialogues.values():
        dial_hinglish = 0
        for t in turns:
            total_turns += 1

            shade = t.get(T_COPING_SIGNAL)
            if shade is not None and shade != "":
                hinglish_turns += 1
                dial_hinglish  += 1

            text = t.get(T_TEXT, "")
            deva_ratios.append(devanagari_ratio(text))

            shift = t.get(T_INTENSITY_SHIFT)
            if shift is not None and shift != "":
                intensity_shifts.append(str(shift))

        hinglish_per_dial.append(dial_hinglish)

    shift_dist = pct_dist(Counter(intensity_shifts), len(intensity_shifts))

    return {
        "pct_hinglish_utterances":
            round(100 * hinglish_turns / total_turns, 2) if total_turns else 0.0,
        "avg_devanagari_char_ratio":
            round(mean(deva_ratios), 4),
        "avg_hinglish_phrases_per_dialogue":
            round(mean(hinglish_per_dial), 2),
        "intensity_shift_distribution_pct":
            shift_dist,
    }



# SECTION 4 : Emotion labels

def compute_emotion_labels(situations: list, dialogues: dict) -> dict:
    """
    From situations:
      num_unique_emotion_seeds          : distinct emotion_seed values
      emotion_seed_distribution_pct     : % per seed
      avg_initial_distress_intensity    : mean intensity_seed

    From dialogue turns (seeker only):
      avg_emotion_transitions_per_dial  : mean count of emotion-field changes
      avg_final_distress_intensity      : mean intensity on last seeker turn
      avg_intensity_delta               : mean (last − first seeker intensity)
      emotion_distribution_pct          : % of seeker turns per emotion label
    """
    # situation level 
    emotion_seeds = [
        s[S_EMOTION_SEED]
        for s in situations
        if s.get(S_EMOTION_SEED)
    ]
    intensity_seeds = [
        float(s[S_INTENSITY_SEED])
        for s in situations
        if s.get(S_INTENSITY_SEED) is not None
    ]

    seed_dist = pct_dist(Counter(emotion_seeds), len(emotion_seeds))

    # dialogue level 
    emotion_transitions = []
    final_intensities   = []
    intensity_deltas    = []
    all_seeker_emotions = []

    for turns in dialogues.values():
        s_turns = [t for t in turns if t.get(T_SPEAKER) == "Seeker"]
        if not s_turns:
            continue

        emotions = [t.get(T_EMOTION) for t in s_turns]
        all_seeker_emotions.extend([e for e in emotions if e])

        transitions = sum(
            1 for i in range(1, len(emotions))
            if emotions[i] != emotions[i - 1]
        )
        emotion_transitions.append(transitions)

        intensities = [
            safe_float(t.get(T_INTENSITY))
            for t in s_turns
            if safe_float(t.get(T_INTENSITY)) is not None
        ]
        if intensities:
            final_intensities.append(intensities[-1])
            intensity_deltas.append(intensities[-1] - intensities[0])

    emotion_dist = pct_dist(Counter(all_seeker_emotions), len(all_seeker_emotions))

    return {
        "num_unique_emotion_seeds":            len(set(emotion_seeds)),
        "emotion_seed_distribution_pct":       seed_dist,
        "avg_initial_distress_intensity":      round(mean(intensity_seeds),     2),
        "avg_emotion_transitions_per_dial":    round(mean(emotion_transitions), 2),
        "avg_final_distress_intensity":        round(mean(final_intensities),   2),
        "avg_intensity_delta_end_minus_start": round(mean(intensity_deltas),    2),
        "emotion_distribution_in_dialogues_pct": emotion_dist,
    }



# SECTION 5 : Support strategies 

def compute_strategies(dialogues: dict) -> dict:
    """
    num_unique_strategies_used          : distinct non-null strategy values
    strategy_distribution_pct           : % of supporter turns per strategy
    restatement_lens_distribution_pct   : % of supporter turns per lens value
    avg_strategies_per_dialogue         : mean distinct strategy count per dialogue
    stage_distribution_pct              : % of ALL turns per stage
    pct_supporter_turns_with_strategy   : % of supporter turns that have a strategy
    """
    all_strategies    = []
    all_lenses        = []
    strat_per_dial    = []
    all_stages        = []
    supp_total        = 0
    supp_with_strat   = 0

    for turns in dialogues.values():
        sup_turns = [t for t in turns if t.get(T_SPEAKER) == "Supporter"]
        dial_strategies = set()

        for t in turns:
            stage = t.get(T_STAGE)
            if stage:
                all_stages.append(stage)

        for t in sup_turns:
            supp_total += 1
            s = t.get(T_STRATEGY)
            if s is not None and s != "":
                all_strategies.append(s)
                dial_strategies.add(s)
                supp_with_strat += 1

            lens = t.get(T_RESTATEMENT_LENS)
            if lens is not None and lens != "":
                all_lenses.append(lens)

        strat_per_dial.append(len(dial_strategies))

    strat_dist  = pct_dist(Counter(all_strategies), len(all_strategies))
    lens_dist   = pct_dist(Counter(all_lenses),     len(all_lenses))
    stage_dist  = pct_dist(Counter(all_stages),     len(all_stages))

    return {
        "num_unique_strategies_used":
            len(set(all_strategies)),
        "pct_supporter_turns_with_strategy":
            round(100 * supp_with_strat / supp_total, 2) if supp_total else 0.0,
        "strategy_distribution_pct":
            strat_dist,
        "restatement_lens_distribution_pct":
            lens_dist,
        "avg_strategies_per_dialogue":
            round(mean(strat_per_dial), 2),
        "stage_distribution_pct":
            stage_dist,
    }



#  SECTION 6 : Cultural signals 

def compute_cultural(situations: list, dialogues: dict) -> dict:
    """
    From situations:
      coping_mechanism_distribution_pct  : % per coping type
      context_category_distribution_pct  : % per C1–C8 category
      avg_cultural_markers_per_situation : mean len(cultural markers list)

    From turns:
      pct_seeker_turns_with_stigma_cue       : stigma_cue is True
      pct_supporter_turns_with_stigma_resp   : stigma_response is True
      crisis_flag_distribution_pct           : % of seeker turns per risk_flag value
    """
    #  situation level 
    coping_mechs = [
        s[S_COPING_MECH]
        for s in situations
        if s.get(S_COPING_MECH)
    ]
    cat_vals = [
        s[S_CONTEXT_CAT]
        for s in situations
        if s.get(S_CONTEXT_CAT)
    ]
    marker_lens = [
        len(s[S_CULTURAL_MARKERS])
        for s in situations
        if isinstance(s.get(S_CULTURAL_MARKERS), list)
    ]

    cop_dist = pct_dist(Counter(coping_mechs), len(coping_mechs))
    cat_dist = pct_dist(Counter(cat_vals),     len(cat_vals))

    # turn level 
    seeker_total      = 0
    stigma_cue_total  = 0
    supp_total        = 0
    stigma_resp_total = 0
    risk_flags        = []

    for turns in dialogues.values():
        for t in turns:
            spk = t.get(T_SPEAKER)

            if spk == "Seeker":
                seeker_total += 1

                cue = t.get(T_STIGMA_CUE)
                if cue is True or cue == "true" or cue == 1:
                    stigma_cue_total += 1

                rf = t.get(T_RISK_FLAG)
                risk_flags.append(str(rf) if rf is not None else "none")

            elif spk == "Supporter":
                supp_total += 1

                resp = t.get(T_STIGMA_RESPONSE)
                if resp is True or resp == "true" or resp == 1:
                    stigma_resp_total += 1

    risk_dist = pct_dist(Counter(risk_flags), len(risk_flags))

    return {
        "coping_mechanism_distribution_pct":
            cop_dist,
        "context_category_distribution_pct":
            cat_dist,
        "avg_cultural_markers_per_situation":
            round(mean(marker_lens), 2),
        "pct_seeker_turns_with_stigma_cue":
            round(100 * stigma_cue_total / seeker_total, 2) if seeker_total else 0.0,
        "pct_supporter_turns_with_stigma_response":
            round(100 * stigma_resp_total / supp_total, 2)  if supp_total  else 0.0,
        "crisis_flag_distribution_pct":
            risk_dist,
    }



# Master runner

def compute_all_metrics(root: Path, debug: bool = False) -> dict:
    print(f"\nLoading dataset from: {root}")
    situations, dialogues = load_dataset(root, debug=debug)

    n_turns = sum(len(v) for v in dialogues.values())
    print(f"  Situations loaded  : {len(situations)}")
    print(f"  Dialogues grouped  : {len(dialogues)}")
    print(f"  Total turns        : {n_turns}")

    print_field_coverage(situations, dialogues)

    report: dict = {}

    print("Computing: Corpus overview …")
    report["corpus_overview"] = compute_corpus_overview(situations, dialogues)

    print("Computing: Seeker vs Supporter split …")
    report["seeker_supporter_split"] = compute_seeker_supporter_split(dialogues)

    print("Computing: Linguistic — Language & script …")
    report["linguistic_language_script"] = compute_linguistic(dialogues)

    print("Computing: Emotion labels …")
    report["emotion_labels"] = compute_emotion_labels(situations, dialogues)

    print("Computing: Support strategies …")
    report["support_strategies"] = compute_strategies(dialogues)

    print("Computing: Cultural — Indian-specific signals …")
    report["cultural_signals"] = compute_cultural(situations, dialogues)

    return report



# Pretty printer

def pretty_print(report: dict):
    sections = {
        "corpus_overview":            "Standard — Corpus Overview",
        "seeker_supporter_split":     "Standard — Seeker vs Supporter Split",
        "linguistic_language_script": "Linguistic — Language & Script",
        "emotion_labels":             "Emotion — Emotion Labels",
        "support_strategies":         "Strategy — Support Strategies",
        "cultural_signals":           "Cultural — Indian-specific Signals",
    }
    print("\n" + "=" * 70)
    print("  REPORT METRICS")
    print("=" * 70)
    for key, title in sections.items():
        print(f"\n{'─' * 70}")
        print(f"  {title}")
        print(f"{'─' * 70}")
        data = report.get(key, {})
        for field, value in data.items():
            if isinstance(value, dict):
                if not value:
                    print(f"  {field}: (empty — field not present in data)")
                else:
                    print(f"  {field}:")
                    for k, v in value.items():
                        print(f"      {k}: {v}")
            else:
                print(f"  {field}: {value}")



# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate report metrics for India Emotional Support Conversation dataset."
    )
    parser.add_argument(
        "--root", type=str, required=True,
        help="Path to inner dataset folder (contains conversations/ and Situation_Categories/)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional path to save metrics as JSON (e.g. metrics.json)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print detailed structure info during loading",
    )
    args = parser.parse_args()

    root_path = Path(args.root)
    if not root_path.exists():
        raise FileNotFoundError(f"Root path not found: {root_path}")

    metrics = compute_all_metrics(root_path, debug=args.debug)
    pretty_print(metrics)

    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Metrics saved to: {out_path}")