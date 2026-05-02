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

S_SIT_ID           = "situation id"        
S_TEXT             = "text"
S_CONTEXT_CAT      = "context_category"
S_COPING_MECH      = "coping_mechanism"
S_EMOTION_SEED     = "emotion seed"        
S_INTENSITY_SEED   = "intensity_seed"
S_CULTURAL_MARKERS = "cultural markers"    


# Normalisation helpers (NEW)

def normalise_folder_name(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"[-_\s]+", "", s)
    typo_fixes = {
        "presuure": "pressure",
        "employement": "employment",
    }
    for wrong, right in typo_fixes.items():
        s = s.replace(wrong, right)
    return s

_STEM_STRIP_PREFIXES = (
    "conversation_", "conversation-", "conversation",
    "conv_", "conv-", "conv",
)

def normalise_prompt_stem(stem: str) -> str:
    """
    Prompt-1 → prompt1
    Conversation_Family → family
    Conv-Health → health
    """
    s = stem.lower().strip()
    for prefix in sorted(_STEM_STRIP_PREFIXES, key=len, reverse=True):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def is_error_record(item) -> bool:
    return isinstance(item, dict) and "error" in item

def is_valid_conversation(item) -> bool:
    if not isinstance(item, list) or len(item) == 0:
        return False
    first = item[0]
    if is_error_record(first):
        return False
    return isinstance(first, dict)


def iter_json_files(folder: Path):
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() == ".json":
            yield p, p.parent.name


# Loaders

def load_situations(sit_root: Path, debug: bool = False) -> list:
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
                obj["__category__"]  = normalise_folder_name(category)
                obj["__file_stem__"] = normalise_prompt_stem(p.stem)
                obj["__category_raw__"] = category
                situations.append(obj)
    return situations


def load_conversations(conv_root: Path, debug: bool = False) -> dict:
    """
    Returns {dialogue_id: [turn_dict, ...]}
    Skips {"error":...} records entirely.
    """
    dialogues: dict = {}
    err_by_file = defaultdict(int)

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
        cat  = normalise_folder_name(category)

        for conv_idx, conv in enumerate(data):
            # --- NEW: skip error records ---
            if is_error_record(conv):
                err_by_file[p] += 1
                continue

            if is_valid_conversation(conv):
                turns = [t for t in conv if isinstance(t, dict) and not is_error_record(t)]
            elif isinstance(conv, dict):  # fallback single turn
                turns = [conv]
            else:
                err_by_file[p] += 1
                if debug:
                    print(f"  [DEBUG] Unexpected conv type {type(conv)} in {p}[{conv_idx}]")
                continue

            if not turns:
                continue

            turns.sort(key=lambda t: t.get(T_TURN_ID, 0))
            dialogue_id = f"{cat}/{stem}/{conv_idx}"
            dialogues[dialogue_id] = turns

    total_err = sum(err_by_file.values())
    if total_err:
        print(f"\n  ── Skipped {total_err} error/invalid conversation records ──")
        if debug:
            for fp, cnt in sorted(err_by_file.items()):
                print(f"    {fp.parent.name}/{fp.name}: {cnt}")

    return dialogues


def load_dataset(root: Path, debug: bool = False):
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
        raise FileNotFoundError(f"No 'Situation_Categories' folder under {root}")

    print(f"  Conversations folder : {conv_root}")
    print(f"  Situations   folder  : {sit_root}")

    situations = load_situations(sit_root,  debug=debug)
    dialogues  = load_conversations(conv_root, debug=debug)
    return situations, dialogues


# Helpers 

def word_count(text) -> int:
    if not text or not isinstance(text, str):
        return 0
    return len(text.split())

def safe_float(val):
    try: return float(val)
    except: return None

def mean(lst: list) -> float:
    return sum(lst) / len(lst) if lst else 0.0

def pct_dist(counter: Counter, total: int) -> dict:
    if not total: return {}
    return {k: round(100 * v / total, 2) for k, v in counter.most_common()}


def print_field_coverage(situations: list, dialogues: dict):
    all_turns = [t for turns in dialogues.values() for t in turns]
    print("\n Turn-level field coverage")
    turn_fields = [T_SPEAKER, T_TEXT, T_TURN_ID, T_STAGE, T_STRATEGY,
                   T_RESTATEMENT_LENS, T_COPING_SIGNAL, T_STIGMA_CUE,
                   T_STIGMA_RESPONSE, T_RISK_FLAG, T_EMOTION, T_INTENSITY,
                   T_INTENSITY_SHIFT]
    n = len(all_turns)
    for f in turn_fields:
        non_null = sum(1 for t in all_turns if t.get(f) not in (None, ""))
        pct = 100 * non_null / n if n else 0
        print(f"    {f:25s} {non_null:>5d}/{n:<5d} ({pct:5.1f}%)")
    print("\n Situation-level field coverage")
    sit_fields = [S_EMOTION_SEED, S_INTENSITY_SEED, S_COPING_MECH,
                  S_CONTEXT_CAT, S_CULTURAL_MARKERS, S_TEXT]
    m = len(situations)
    for f in sit_fields:
        non_null = sum(1 for s in situations if s.get(f) not in (None, ""))
        pct = 100 * non_null / m if m else 0
        print(f"    {f:25s} {non_null:>5d}/{m:<5d} ({pct:5.1f}%)")
    print()

def compute_corpus_overview(situations, dialogues): 
    all_turns = [t for turns in dialogues.values() for t in turns]
    turn_counts = [len(t) for t in dialogues.values()]
    dial_lengths = [sum(word_count(t.get(T_TEXT,"")) for t in turns) for turns in dialogues.values()]
    utt_lengths = [word_count(t.get(T_TEXT,"")) for t in all_turns]
    return {
        "num_situations": len(situations),
        "num_dialogues": len(dialogues),
        "num_utterances_total": len(all_turns),
        "avg_turns_per_dialogue": round(mean(turn_counts),2),
        "avg_length_per_dialogue_wds": round(mean(dial_lengths),2),
        "avg_length_per_utterance_wds": round(mean(utt_lengths),2),
    }

def compute_seeker_supporter_split(dialogues):
    seek_dial_lens=[]; supp_dial_lens=[]; seek_utt_lens=[]; supp_utt_lens=[]
    seek_turn_counts=[]; supp_turn_counts=[]
    for turns in dialogues.values():
        s=[t for t in turns if t.get(T_SPEAKER)=="Seeker"]
        sup=[t for t in turns if t.get(T_SPEAKER)=="Supporter"]
        seek_dial_lens.append(sum(word_count(t.get(T_TEXT,"")) for t in s))
        supp_dial_lens.append(sum(word_count(t.get(T_TEXT,"")) for t in sup))
        seek_turn_counts.append(len(s)); supp_turn_counts.append(len(sup))
        seek_utt_lens.extend(word_count(t.get(T_TEXT,"")) for t in s)
        supp_utt_lens.extend(word_count(t.get(T_TEXT,"")) for t in sup)
    return {
        "avg_seeker_turns_per_dialogue": round(mean(seek_turn_counts),2),
        "avg_supporter_turns_per_dialogue": round(mean(supp_turn_counts),2),
        "avg_seeker_len_per_dialogue_wds": round(mean(seek_dial_lens),2),
        "avg_seeker_len_per_utterance_wds": round(mean(seek_utt_lens),2),
        "avg_supporter_len_per_dialogue_wds": round(mean(supp_dial_lens),2),
        "avg_supporter_len_per_utterance_wds": round(mean(supp_utt_lens),2),
    }

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
def devanagari_ratio(text): return len(DEVANAGARI_RE.findall(text or ""))/len(text) if text else 0.0

def compute_linguistic(dialogues):
    total=hing=0; deva=[]; per_dial=[]; shifts=[]
    for turns in dialogues.values():
        d=0
        for t in turns:
            total+=1
            if t.get(T_COPING_SIGNAL): hing+=1; d+=1
            deva.append(devanagari_ratio(t.get(T_TEXT,"")))
            if t.get(T_INTENSITY_SHIFT) not in (None,""): shifts.append(str(t.get(T_INTENSITY_SHIFT)))
        per_dial.append(d)
    return {
        "pct_hinglish_utterances": round(100*hing/total,2) if total else 0,
        "avg_devanagari_char_ratio": round(mean(deva),4),
        "avg_hinglish_phrases_per_dialogue": round(mean(per_dial),2),
        "intensity_shift_distribution_pct": pct_dist(Counter(shifts), len(shifts)),
    }

def compute_emotion_labels(situations, dialogues):
    seeds=[s[S_EMOTION_SEED] for s in situations if s.get(S_EMOTION_SEED)]
    int_seeds=[float(s[S_INTENSITY_SEED]) for s in situations if s.get(S_INTENSITY_SEED) is not None]
    trans=[]; finals=[]; deltas=[]; emos=[]
    for turns in dialogues.values():
        s=[t for t in turns if t.get(T_SPEAKER)=="Seeker"]
        if not s: continue
        e=[t.get(T_EMOTION) for t in s]; emos.extend(x for x in e if x)
        trans.append(sum(1 for i in range(1,len(e)) if e[i]!=e[i-1]))
        ints=[safe_float(t.get(T_INTENSITY)) for t in s]; ints=[i for i in ints if i is not None]
        if ints: finals.append(ints[-1]); deltas.append(ints[-1]-ints[0])
    return {
        "num_unique_emotion_seeds": len(set(seeds)),
        "emotion_seed_distribution_pct": pct_dist(Counter(seeds), len(seeds)),
        "avg_initial_distress_intensity": round(mean(int_seeds),2),
        "avg_emotion_transitions_per_dial": round(mean(trans),2),
        "avg_final_distress_intensity": round(mean(finals),2),
        "avg_intensity_delta_end_minus_start": round(mean(deltas),2),
        "emotion_distribution_in_dialogues_pct": pct_dist(Counter(emos), len(emos)),
    }

def compute_strategies(dialogues):
    strats=[]; lenses=[]; per=[]; stages=[]; tot=with_s=0
    for turns in dialogues.values():
        sup=[t for t in turns if t.get(T_SPEAKER)=="Supporter"]; s=set()
        for t in turns:
            if t.get(T_STAGE): stages.append(t.get(T_STAGE))
        for t in sup:
            tot+=1
            if t.get(T_STRATEGY): strats.append(t.get(T_STRATEGY)); s.add(t.get(T_STRATEGY)); with_s+=1
            if t.get(T_RESTATEMENT_LENS): lenses.append(t.get(T_RESTATEMENT_LENS))
        per.append(len(s))
    return {
        "num_unique_strategies_used": len(set(strats)),
        "pct_supporter_turns_with_strategy": round(100*with_s/tot,2) if tot else 0,
        "strategy_distribution_pct": pct_dist(Counter(strats), len(strats)),
        "restatement_lens_distribution_pct": pct_dist(Counter(lenses), len(lenses)),
        "avg_strategies_per_dialogue": round(mean(per),2),
        "stage_distribution_pct": pct_dist(Counter(stages), len(stages)),
    }

def compute_cultural(situations, dialogues):
    cop=[s[S_COPING_MECH] for s in situations if s.get(S_COPING_MECH)]
    cats=[s[S_CONTEXT_CAT] for s in situations if s.get(S_CONTEXT_CAT)]
    marks=[len(s[S_CULTURAL_MARKERS]) for s in situations if isinstance(s.get(S_CULTURAL_MARKERS), list)]
    seek=stig_c=sup=stig_r=0; risks=[]
    for turns in dialogues.values():
        for t in turns:
            if t.get(T_SPEAKER)=="Seeker":
                seek+=1
                if t.get(T_STIGMA_CUE) in (True,"true",1): stig_c+=1
                risks.append(str(t.get(T_RISK_FLAG) or "none"))
            elif t.get(T_SPEAKER)=="Supporter":
                sup+=1
                if t.get(T_STIGMA_RESPONSE) in (True,"true",1): stig_r+=1
    return {
        "coping_mechanism_distribution_pct": pct_dist(Counter(cop), len(cop)),
        "context_category_distribution_pct": pct_dist(Counter(cats), len(cats)),
        "avg_cultural_markers_per_situation": round(mean(marks),2),
        "pct_seeker_turns_with_stigma_cue": round(100*stig_c/seek,2) if seek else 0,
        "pct_supporter_turns_with_stigma_response": round(100*stig_r/sup,2) if sup else 0,
        "crisis_flag_distribution_pct": pct_dist(Counter(risks), len(risks)),
    }



def compute_all_metrics(root: Path, debug: bool = False) -> dict:
    print(f"\nLoading dataset from: {root}")
    situations, dialogues = load_dataset(root, debug=debug)
    n_turns = sum(len(v) for v in dialogues.values())
    print(f"  Situations loaded  : {len(situations)}")
    print(f"  Dialogues grouped  : {len(dialogues)}")
    print(f"  Total turns        : {n_turns}")
    print_field_coverage(situations, dialogues)
    report = {}
    report["corpus_overview"] = compute_corpus_overview(situations, dialogues)
    report["seeker_supporter_split"] = compute_seeker_supporter_split(dialogues)
    report["linguistic_language_script"] = compute_linguistic(dialogues)
    report["emotion_labels"] = compute_emotion_labels(situations, dialogues)
    report["support_strategies"] = compute_strategies(dialogues)
    report["cultural_signals"] = compute_cultural(situations, dialogues)
    return report

def pretty_print(report: dict):
    sections = {
        "corpus_overview": "Standard — Corpus Overview",
        "seeker_supporter_split": "Standard — Seeker vs Supporter Split",
        "linguistic_language_script": "Linguistic — Language & Script",
        "emotion_labels": "Emotion — Emotion Labels",
        "support_strategies": "Strategy — Support Strategies",
        "cultural_signals": "Cultural — Indian-specific Signals",
    }
    print("\n" + "=" * 70)
    for key, title in sections.items():
        print(f"\n{title}\n{'-'*70}")
        for field, value in report.get(key, {}).items():
            if isinstance(value, dict):
                print(f"  {field}:")
                for k,v in (value.items() if value else []): print(f"      {k}: {v}")
            else: print(f"  {field}: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    metrics = compute_all_metrics(Path(args.root), debug=args.debug)
    pretty_print(metrics)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Metrics saved to: {args.output}")