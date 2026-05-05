"""FAISS few-shot retrieval: composite query embed, domain + behavioral filters, MMR."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from config import (
    EMBEDDING_BACKEND,
    INDEX_PATH,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
)
from core.schemas import normalize_category, normalize_coping


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generator_retriever")


# Tunables
FAISS_CANDIDATE_K = 80     # cast wide; we filter aggressively after
# When problem_type is set, pull more neighbors before domain filter — many
# top-by-text hits can be the wrong life-domain under short seeker lines.
FAISS_CANDIDATE_K_WITH_PROBLEM = 200
SOFT_TOP_N        = 20     # how many enter MMR after soft scoring
MMR_LAMBDA        = 0.7    # 1.0 = pure relevance, 0.0 = pure diversity

PERSONA_BOOST     = 0.15
INTENSITY_BOOST   = 0.10
EMOTION_BOOST     = 0.05
PROBLEM_CATEGORY_BOOST = 0.10

# Max chars for the SEEKER: segment in the *query* composite string. Slightly
# above the 200-char index cap so short turns still carry a topic prefix from
# session context without truncating the current utterance first.
SEEKER_QUERY_MAX_CHARS = 220

# Lowercase substrings of `record["category"]` (dataset folder name) used to
# nudge retrieval toward the analyzer's canonical problem when emotion-only
# similarity would drift (e.g. JEE thread + reply "Haan sach me").
PROBLEM_CATEGORY_ANCHORS: dict[str, tuple[str, ...]] = {
    "Academic_Pressure": (
        "academic",
        "presuure",
        "jee",
        "placement",
        "gate",
        "neet",
        "board",
        "study",
        "exam",
    ),
    "Family_Dynamics": ("family", "familial"),
    "Marriage_Rishta": ("marriage", "rishta", "shaadi", "wedding"),
    "Employment_Livelihood": (
        "employ",
        "office",
        "layoff",
        "livelihood",
        "promotion",
        "job",
    ),
    "Financial_Debt": ("financial", "debt", "emi", "loan"),
    "Gender_Identity": ("gender", "identity", "lgbt", "trans", "queer"),
    "Health_Chronic_Illness": (
        "health",
        "chronic",
        "illness",
        "hospital",
        "therapy",
        "mental",
    ),
    "Migration_Displacement": ("migration", "migrant", "foreign", "visa", "displacement"),
}


def _problem_category_match_bonus(
    target_problem: Optional[str],
    record_category: Optional[str],
) -> float:
    if not target_problem or not record_category:
        return 0.0
    cat = record_category.lower()
    anchors = PROBLEM_CATEGORY_ANCHORS.get(target_problem)
    if not anchors:
        return 0.0
    for a in anchors:
        if a in cat:
            return PROBLEM_CATEGORY_BOOST
    return 0.0


def _record_matches_problem_type(
    target_problem: str,
    record_category: Optional[str],
) -> bool:
    """True if indexed `category` (folder id) belongs to the same canonical
    problem domain as ``target_problem`` (analyzer PROBLEM_TYPE)."""
    pt = (target_problem or "").strip()
    if not pt or not record_category:
        return False
    canon = normalize_category(str(record_category))
    if canon == pt:
        return True
    return _problem_category_match_bonus(pt, record_category) > 0.0


@dataclass
class RetrievalOutcome:
    """Return value of `GeneratorRetriever.retrieve` — records plus debug
    fields for the dev UI / WebSocket meta."""

    records: list[dict]
    filter_level: str
    query_text: str


# Helpers
def _norm_strategy(s: Optional[str]) -> str:
    return (s or "").strip().upper()


def _norm_phase(p: Optional[str]) -> str:
    return (p or "").strip().capitalize()


def _norm_coping(c: Optional[str]) -> str:
    """Use core.schemas.normalize_coping to collapse dataset variants
    ("Sequential Coping" / "Sequential_Coping" -> "Sequential",
     "Duty-Based Coping" -> "Duty_Based", etc.)."""
    canon = normalize_coping(c or "")
    return (canon or (c or "")).strip()


def _safe_int(v, default: int = 3) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


# GeneratorRetriever
class GeneratorRetriever:
    """Loads the FAISS index + records once and serves retrieval queries."""

    # ---------- construction ----------
    def __init__(self, index_path: str | Path = INDEX_PATH):
        self.index_path = Path(index_path)
        index_file       = self.index_path / "index.faiss"
        records_file     = self.index_path / "records.json"
        embeddings_file  = self.index_path / "embeddings.npy"

        for p in (index_file, records_file, embeddings_file):
            if not p.exists():
                raise FileNotFoundError(
                    f"Required index artifact missing: {p}. "
                    "Run `python -m indexing.build_index` first."
                )

        logger.info(f"Loading FAISS index from {index_file}")
        self.index = faiss.read_index(str(index_file))

        logger.info(f"Loading records from {records_file}")
        self.records: list[dict] = json.loads(records_file.read_text())

        logger.info(f"Loading embeddings matrix from {embeddings_file}")
        self.embeddings: np.ndarray = np.load(str(embeddings_file))
        if self.embeddings.dtype != np.float32:
            self.embeddings = self.embeddings.astype(np.float32, copy=False)

        # Sanity invariants — these MUST line up or retrieval is meaningless.
        if self.index.ntotal != len(self.records):
            raise RuntimeError(
                f"FAISS ntotal ({self.index.ntotal}) != records "
                f"({len(self.records)}). Rebuild the index."
            )
        if self.embeddings.shape[0] != len(self.records):
            raise RuntimeError(
                f"embeddings rows ({self.embeddings.shape[0]}) != records "
                f"({len(self.records)}). Rebuild the index."
            )

        self.dim = int(self.index.d)
        self.backend = EMBEDDING_BACKEND
        self._openai_client = None  # lazy
        self._st_model      = None  # lazy

        logger.info(
            f"Retriever ready  |  ntotal={self.index.ntotal:,}  "
            f"dim={self.dim}  backend={self.backend}"
        )

    # ---------- query embedding ----------
    async def _embed_query(self, text: str) -> np.ndarray:
        """Embed a single query and return a (1, dim) float32 unit vector."""
        if self.backend == "openai":
            vec = await self._embed_query_openai(text)
        elif self.backend == "local":
            vec = await asyncio.to_thread(self._embed_query_local, text)
        else:
            raise ValueError(f"Unknown EMBEDDING_BACKEND: {self.backend}")

        vec = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return vec / norm

    async def _embed_query_openai(self, text: str) -> np.ndarray:
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "EMBEDDING_BACKEND=openai but OPENAI_API_KEY is empty."
            )
        if self._openai_client is None:
            from openai import AsyncOpenAI
            self._openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

        from openai import APIConnectionError, APITimeoutError, RateLimitError
        retryable = (APIConnectionError, APITimeoutError, RateLimitError)

        last: Exception | None = None
        delay = 1.0
        for attempt in range(1, 6):
            try:
                resp = await self._openai_client.embeddings.create(
                    model=OPENAI_EMBEDDING_MODEL,
                    input=[text],
                )
                return np.asarray(resp.data[0].embedding, dtype=np.float32)
            except retryable as e:
                last = e
                if attempt == 5:
                    break
                logger.warning(
                    f"Query embed transient error (attempt {attempt}/5): "
                    f"{type(e).__name__}: {e}. Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2.0, 30.0)
        assert last is not None
        raise last

    def _embed_query_local(self, text: str) -> np.ndarray:
        if self._st_model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except ImportError as e:
                raise RuntimeError(
                    "EMBEDDING_BACKEND=local requires sentence-transformers."
                ) from e
            logger.info(f"Loading local model: {LOCAL_EMBEDDING_MODEL}")
            self._st_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
        vec = self._st_model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=False,  # we normalize centrally
        )
        return np.asarray(vec[0], dtype=np.float32)

    # ---------- composite string ----------
    @staticmethod
    def _build_query_string(
        seeker_text: str,
        strategy: str,
        coping_mech: str,
        phase: str,
        intensity: int,
        emotion: Optional[str],
        session_context: Optional[str] = None,
    ) -> str:
        """Mirror `indexing.build_index.build_composite_string` so the query
        lives in the same embedding distribution as the indexed records.
        Emotion is included when known to align fully with the build template;
        otherwise we use the literal token 'unknown' (also the build default).

        `session_context` is prepended to the seeker line (topic pins from
        analyzer + facts + summary) so short replies still embed with the
        same thread as earlier turns — without reindexing.
        """
        raw = (seeker_text or "").strip()
        ctx = (session_context or "").strip()
        if ctx:
            seeker = f"{ctx} :: {raw}".strip()
        else:
            seeker = raw
        seeker = seeker[:SEEKER_QUERY_MAX_CHARS]
        emo = (emotion or "unknown").strip()
        return (
            f"[STRATEGY:{strategy}] "
            f"[COPING:{coping_mech}] "
            f"[PHASE:{phase}] "
            f"[INTENSITY:{intensity}] "
            f"[EMOTION:{emo}] "
            f"SEEKER: {seeker}"
        )

    # ---------- main retrieve ----------
    async def retrieve(
        self,
        seeker_text: str,
        strategy: str,
        coping_mech: str,
        phase: str,
        intensity: int,
        persona_code: str = "P0",
        emotion: Optional[str] = None,
        top_k: int = 6,
        session_context: Optional[str] = None,
        problem_type: Optional[str] = None,
    ) -> RetrievalOutcome:
        """End-to-end retrieval. Returns ``RetrievalOutcome`` with up to
        ``top_k`` record dicts plus the composite query string and
        filter-cascade label for observability."""
        if top_k <= 0:
            return RetrievalOutcome(records=[], filter_level="skipped", query_text="")

        s_strategy = _norm_strategy(strategy)
        s_coping   = _norm_coping(coping_mech)
        s_phase    = _norm_phase(phase)
        s_intensity = _safe_int(intensity)

        # ---- STEP 1: embed query and FAISS search top-K ----
        q_text = self._build_query_string(
            seeker_text,
            s_strategy,
            s_coping,
            s_phase,
            s_intensity,
            emotion,
            session_context=session_context,
        )
        q_vec = await self._embed_query(q_text)

        k_cap = (
            FAISS_CANDIDATE_K_WITH_PROBLEM
            if (problem_type or "").strip()
            else FAISS_CANDIDATE_K
        )
        k = min(k_cap, self.index.ntotal)
        scores, idxs = self.index.search(q_vec, k)
        scores = scores[0]
        idxs = idxs[0]

        candidates: list[dict] = []
        for s, i in zip(scores.tolist(), idxs.tolist()):
            if i < 0:  # FAISS pads with -1 when k > ntotal
                continue
            candidates.append({
                "idx": int(i),
                "faiss_score": float(s),
                "score": float(s),       # mutable; soft scoring updates this
                "record": self.records[i],
            })

        if not candidates:
            logger.warning("FAISS returned 0 candidates — empty index?")
            return RetrievalOutcome(
                records=[], filter_level="no_faiss_hits", query_text=q_text,
            )

        # ---- STEP 2: problem-domain-first cascade, then strategy/coping/phase ----
        filtered, level = self._filter_cascade(
            candidates,
            s_strategy,
            s_coping,
            s_phase,
            top_k,
            problem_type=(problem_type or "").strip() or None,
        )
        logger.info(
            f"Filter level={level}  candidates={len(filtered)}  "
            f"(problem_type={problem_type!r}, strategy={s_strategy}, "
            f"coping={s_coping}, phase={s_phase})"
        )

        # ---- STEP 3: soft scoring ----
        self._apply_soft_scoring(
            filtered,
            target_intensity=s_intensity,
            target_persona=persona_code,
            target_emotion=(emotion or "").strip().lower() or None,
            target_problem_type=(problem_type or "").strip() or None,
        )
        filtered.sort(key=lambda c: c["score"], reverse=True)

        # ---- STEP 4: MMR re-ranking on soft-sorted top-N ----
        pool = filtered[:SOFT_TOP_N]
        selected = self._mmr_rerank(pool, top_k=top_k, lam=MMR_LAMBDA)

        # ---- STEP 5: enforce conversation_id diversity ----
        selected = self._dedupe_by_conversation(selected, pool, top_k=top_k)

        # ---- STEP 5b: cap single-emotion dominance for behavioral variety ----
        # Prevents "5 fear examples in a row" tunnel vision where the
        # generator keys on one emotion vocabulary and ignores the rest.
        selected = self._cap_emotion_dominance(selected, pool, top_k=top_k)

        # ---- STEP 6: project to plain record dicts (plus debug score) ----
        out: list[dict] = []
        for c in selected[:top_k]:
            rec = dict(c["record"])  # shallow copy
            rec["_retrieval_score"] = round(c["score"], 4)
            rec["_faiss_score"]     = round(c["faiss_score"], 4)
            out.append(rec)
        return RetrievalOutcome(records=out, filter_level=level, query_text=q_text)

    def _strategy_coping_phase_cascade(
        self,
        base: list[dict],
        strategy: str,
        coping: str,
        phase: str,
        top_k: int,
    ) -> tuple[list[dict], str]:
        """Progressive relaxation of strategy / coping / phase filters on
        ``base``. Last resort: all of ``base`` (no behavioral hard filters)."""
        def by(strat: bool, cop: bool, ph: bool) -> list[dict]:
            out = []
            for c in base:
                r = c["record"]
                if strat and _norm_strategy(r.get("strategy")) != strategy:
                    continue
                if cop and _norm_coping(r.get("seeker_coping")) != coping:
                    continue
                if ph and _norm_phase(r.get("phase")) != phase:
                    continue
                out.append(c)
            return out

        strict = by(True, True, True)
        if len(strict) >= top_k:
            return strict, "strategy+coping+phase"

        no_phase = by(True, True, False)
        if len(no_phase) >= top_k:
            return no_phase, "strategy+coping"

        only_strat = by(True, False, False)
        if len(only_strat) >= top_k:
            return only_strat, "strategy_only"

        return base, "fallback_unfiltered"

    def _filter_cascade(
        self,
        candidates: list[dict],
        strategy: str,
        coping: str,
        phase: str,
        top_k: int,
        problem_type: Optional[str] = None,
    ) -> tuple[list[dict], str]:
        """Strategy/coping/phase cascade. When ``problem_type`` is set, the
        cascade runs on domain-matched records first; cross-domain pool is
        used only if fewer than ``top_k`` rows survive."""
        if not (problem_type or "").strip():
            return self._strategy_coping_phase_cascade(
                candidates, strategy, coping, phase, top_k,
            )

        domain = [
            c for c in candidates
            if _record_matches_problem_type(
                problem_type, c["record"].get("category"),
            )
        ]
        filtered, level = self._strategy_coping_phase_cascade(
            domain, strategy, coping, phase, top_k,
        )
        if len(filtered) >= top_k:
            return filtered, f"problem_domain+{level}"

        logger.info(
            "problem_domain pool size=%d (need %d) — cross_domain_fallback",
            len(filtered), top_k,
        )
        fb, fb_level = self._strategy_coping_phase_cascade(
            candidates, strategy, coping, phase, top_k,
        )
        return fb, f"cross_domain_fallback+{fb_level}"

    # ---------- step 3: soft scoring ----------
    @staticmethod
    def _apply_soft_scoring(
        candidates: list[dict],
        target_intensity: int,
        target_persona: str,
        target_emotion: Optional[str],
        target_problem_type: Optional[str] = None,
    ) -> None:
        """Mutates `candidates` in place: bumps `score` per spec."""
        for c in candidates:
            r = c["record"]
            score = c["faiss_score"]

            if target_persona and r.get("persona_code") == target_persona:
                score += PERSONA_BOOST

            rec_int = _safe_int(r.get("seeker_intensity"), default=3)
            if abs(rec_int - target_intensity) <= 1:
                score += INTENSITY_BOOST

            if target_emotion:
                rec_emo = (r.get("seeker_emotion") or "").strip().lower()
                if rec_emo and rec_emo == target_emotion:
                    score += EMOTION_BOOST

            score += _problem_category_match_bonus(
                target_problem_type,
                r.get("category"),
            )

            c["score"] = score

    # ---------- step 4: MMR ----------
    def _mmr_rerank(
        self,
        pool: list[dict],
        top_k: int,
        lam: float = MMR_LAMBDA,
    ) -> list[dict]:
        """Maximal Marginal Relevance over `pool`. Relevance = raw FAISS
        cosine sim (per spec). Redundancy = max cosine sim of the candidate
        against any already-selected vector, looked up in self.embeddings.
        """
        if not pool:
            return []
        n_target = min(top_k, len(pool))

        # Pre-fetch embedding rows for the pool once (O(N*dim) memory hit
        # bounded by SOFT_TOP_N).
        pool_idxs = [c["idx"] for c in pool]
        pool_vecs = self.embeddings[pool_idxs]  # (N, dim), already L2-norm

        selected: list[dict] = []
        selected_local_pos: list[int] = []  # positions inside `pool`
        remaining = list(range(len(pool)))

        while remaining and len(selected) < n_target:
            best_pos = -1
            best_score = -float("inf")

            if not selected_local_pos:
                # First pick: redundancy = 0, so pick max relevance.
                for pos in remaining:
                    rel = pool[pos]["faiss_score"]
                    mmr = lam * rel
                    if mmr > best_score:
                        best_score = mmr
                        best_pos = pos
            else:
                sel_vecs = pool_vecs[selected_local_pos]  # (k, dim)
                for pos in remaining:
                    rel = pool[pos]["faiss_score"]
                    sims = sel_vecs @ pool_vecs[pos]      # (k,)
                    redundancy = float(sims.max())
                    mmr = lam * rel - (1.0 - lam) * redundancy
                    if mmr > best_score:
                        best_score = mmr
                        best_pos = pos

            selected.append(pool[best_pos])
            selected_local_pos.append(best_pos)
            remaining.remove(best_pos)

        return selected

    # ---------- step 5: conversation_id dedup ----------
    @staticmethod
    def _dedupe_by_conversation(
        selected: list[dict],
        pool: list[dict],
        top_k: int,
    ) -> list[dict]:
        """If two selected records share `conversation_id`, drop the
        lower-scored one and replace it with the next non-duplicate from the
        pool that isn't already in `selected`."""
        seen_conv: set[str] = set()
        seen_idx: set[int] = set()
        kept: list[dict] = []

        for c in selected:
            conv = c["record"].get("conversation_id") or ""
            if conv and conv in seen_conv:
                continue
            kept.append(c)
            if conv:
                seen_conv.add(conv)
            seen_idx.add(c["idx"])
            if len(kept) >= top_k:
                return kept

        # Fill from the broader pool (sorted by current score) if we lost any.
        if len(kept) < top_k:
            backups = sorted(pool, key=lambda c: c["score"], reverse=True)
            for c in backups:
                if c["idx"] in seen_idx:
                    continue
                conv = c["record"].get("conversation_id") or ""
                if conv and conv in seen_conv:
                    continue
                kept.append(c)
                if conv:
                    seen_conv.add(conv)
                seen_idx.add(c["idx"])
                if len(kept) >= top_k:
                    break

        return kept

    # ---------- step 5b: emotion-diversity cap ----------
    @staticmethod
    def _cap_emotion_dominance(
        selected: list[dict],
        pool: list[dict],
        top_k: int,
        max_share_pct: int = 50,
    ) -> list[dict]:
        """If more than `max_share_pct`% of selected records share the same
        `seeker_emotion`, swap excess entries for the next non-dominant
        candidates from `pool`. Preserves overall ordering otherwise."""
        if top_k <= 2 or not selected:
            return selected

        # Compute the cap (e.g. top_k=6, 50% → max 3 of any emotion).
        cap = max(1, (top_k * max_share_pct) // 100)

        counts: dict[str, int] = {}
        for c in selected:
            emo = (c["record"].get("seeker_emotion") or "").lower().strip()
            if not emo:
                continue
            counts[emo] = counts.get(emo, 0) + 1

        # Find which emotion(s) exceed the cap.
        over = {emo for emo, n in counts.items() if n > cap}
        if not over:
            return selected

        # Walk the selected list keeping at most `cap` of each over-represented
        # emotion; mark the rest for replacement.
        kept: list[dict] = []
        kept_idx: set[int] = set()
        kept_counts: dict[str, int] = {}
        excess_slots = 0
        for c in selected:
            emo = (c["record"].get("seeker_emotion") or "").lower().strip()
            if emo in over and kept_counts.get(emo, 0) >= cap:
                excess_slots += 1
                continue
            kept.append(c)
            kept_idx.add(c["idx"])
            if emo:
                kept_counts[emo] = kept_counts.get(emo, 0) + 1

        if excess_slots == 0:
            return selected

        # Pull replacements from the pool that don't worsen emotion balance.
        for c in sorted(pool, key=lambda x: x["score"], reverse=True):
            if excess_slots <= 0:
                break
            if c["idx"] in kept_idx:
                continue
            emo = (c["record"].get("seeker_emotion") or "").lower().strip()
            if emo in over and kept_counts.get(emo, 0) >= cap:
                continue
            kept.append(c)
            kept_idx.add(c["idx"])
            if emo:
                kept_counts[emo] = kept_counts.get(emo, 0) + 1
            excess_slots -= 1

        return kept

    # ---------- prompt formatting ----------
    @staticmethod
    def format_for_prompt(records: list[dict]) -> str:
        """Render retrieved records as the EXAMPLES block of the Generator
        system prompt. Stable, human-readable, line-bounded."""
        if not records:
            return "(no examples available)"

        chunks: list[str] = []
        for i, r in enumerate(records):
            lens = r.get("lens") or "N/A"
            rationale = (r.get("rationale") or "").strip()
            why_line = (
                f"WHY THIS WORKS: {rationale[:150]}\n\n" if rationale else ""
            )
            chunks.append(
                f"--- EXAMPLE {i + 1} ---\n"
                f"STRATEGY: {r.get('strategy','')}  |  LENS: {lens}  |  "
                f"COPING: {r.get('seeker_coping','')}\n"
                f"PHASE: {r.get('phase','')}  |  "
                f"INTENSITY: {r.get('seeker_intensity','')}  |  "
                f"EMOTION: {r.get('seeker_emotion','')}\n\n"
                f"{why_line}"
                f"SEEKER SAID:\n"
                f"\"{(r.get('seeker_text') or '').strip()}\"\n\n"
                f"SAATHI SAID:\n"
                f"\"{(r.get('supporter_text') or '').strip()}\"\n"
            )
        return "\n".join(chunks)

    @staticmethod
    def format_negative_example() -> str:
        """Hardcoded one-shot anti-example. Always prepended/appended to the
        Generator's few-shot block to anchor the model away from clinical
        translation. Emojis are intentional and required by spec."""
        return (
            "--- WHAT NOT TO DO ---\n"
            "❌ WRONG (clinical vocabulary, loses trust instantly):\n"
            "\"It sounds like you might be experiencing burnout and anxiety. "
            "I'd suggest setting healthy boundaries and maybe exploring some "
            "mindfulness techniques.\"\n\n"
            "✅ RIGHT (mirrors the seeker's frame, stays in their world):\n"
            "\"Machine ki tarah chalte chalte thak gaye ho — bas ruk-na nahi, "
            "chal-te rehna, yahi toh hua hai na? Itni bhaari zimmedari akele "
            "utha rahe ho, waqai bahut zyada hai.\"\n\n"
            "WHY: The wrong version translates the seeker's lived experience "
            "into foreign clinical language. The right version uses their OWN "
            "words (\"machine ki tarah\") and stays inside their frame of "
            "reference. SAATHI never translates — it mirrors.\n"
        )


# __main__ smoke test
async def _smoke_test() -> None:
    retriever = GeneratorRetriever()

    test_queries = [
        {
            "label": "Duty-bound exhaustion (Exploration)",
            "seeker_text": (
                "Machine ki tarah chal raha hoon roz, ruk nahi sakta warna "
                "sab ruk jayega."
            ),
            "strategy": "RESTATEMENT_OR_PARAPHRASING",
            "coping_mech": "Duty_Based",
            "phase": "Exploration",
            "intensity": 5,
            "emotion": "exhaustion",
            "persona_code": "P0",
        },
        {
            "label": "Family-conflict avoidance via question (Insight)",
            "seeker_text": (
                "Bolne se ghar ka mahaul kharab hoga, isliye chup hi reh "
                "leta hoon."
            ),
            "strategy": "QUESTION",
            "coping_mech": "Relational_Preservation",
            "phase": "Insight",
            "intensity": 4,
            "emotion": "frustration",
            "persona_code": "P0",
        },
        {
            "label": "Body-pain expression of unspoken pain (Insight)",
            "seeker_text": (
                "Sar bhaari rehta hai, neend nahi aati. Kisi ko bata bhi nahi "
                "sakti ki andar kya chal raha hai."
            ),
            "strategy": "REFLECTION_OF_FEELINGS",
            "coping_mech": "Somatization",
            "phase": "Insight",
            "intensity": 3,
            "emotion": "sadness",
            "persona_code": "P0",
        },
    ]

    for tq in test_queries:
        label = tq.pop("label")
        print("\n" + "=" * 78)
        print(f"TEST: {label}")
        print(f"  strategy={tq['strategy']}  coping={tq['coping_mech']}  "
              f"phase={tq['phase']}  intensity={tq['intensity']}  "
              f"emotion={tq['emotion']}")
        print("=" * 78)

        t0 = time.perf_counter()
        outcome = await retriever.retrieve(top_k=4, **tq)
        results = outcome.records
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        print(f"Returned {len(results)} examples in {elapsed_ms:.1f} ms\n")
        print(retriever.format_for_prompt(results))

    print("\n" + "=" * 78)
    print("NEGATIVE EXAMPLE (always injected into Generator prompt)")
    print("=" * 78)
    print(retriever.format_negative_example())


if __name__ == "__main__":
    asyncio.run(_smoke_test())
