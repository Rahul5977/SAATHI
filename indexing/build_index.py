"""Builds the FAISS index and aligned `records.json` / `embeddings.npy` from parsed data."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from tqdm import tqdm

from config import (
    DATA_DIR,
    EMBEDDING_BACKEND,
    INDEX_PATH,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    PARSED_RECORDS_PATH,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_index")


# Output paths
INDEX_FILE      = INDEX_PATH / "index.faiss"
RECORDS_FILE    = INDEX_PATH / "records.json"
EMBEDDINGS_FILE = INDEX_PATH / "embeddings.npy"
CHECKPOINT_FILE = INDEX_PATH / "embeddings_checkpoint.npy"
CHECKPOINT_META = INDEX_PATH / "embeddings_checkpoint.meta.json"

# Backend batch sizes
OPENAI_BATCH = 500
LOCAL_BATCH  = 256


# Composite string builder
def build_composite_string(rec: dict) -> str:
    """Compose the embedding input for a single record.

    Format must stay stable: the same template is used by retrieval-time
    queries so the embedding distributions align.
    """
    seeker_text = (rec.get("seeker_text") or "").strip()
    return (
        f"[STRATEGY:{rec.get('strategy','')}] "
        f"[COPING:{rec.get('seeker_coping','unknown')}] "
        f"[PHASE:{rec.get('phase','')}] "
        f"[INTENSITY:{rec.get('seeker_intensity', 3)}] "
        f"[EMOTION:{rec.get('seeker_emotion','unknown')}] "
        f"SEEKER: {seeker_text[:200]}"
    )


# Checkpoint helpers (used by the OpenAI backend for resume)
def _save_checkpoint(arr: np.ndarray, total: int) -> None:
    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    # numpy auto-appends ".npy" only if missing — so name the temp file with
    # ".tmp.npy" to keep the suffix stable for the atomic rename.
    tmp = CHECKPOINT_FILE.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    tmp.replace(CHECKPOINT_FILE)
    CHECKPOINT_META.write_text(
        json.dumps({"completed": int(arr.shape[0]), "total": int(total)})
    )


def _load_checkpoint(total: int) -> Optional[np.ndarray]:
    """Return the partial embeddings array iff a consistent checkpoint exists."""
    if not (CHECKPOINT_FILE.exists() and CHECKPOINT_META.exists()):
        return None
    try:
        meta = json.loads(CHECKPOINT_META.read_text())
        if int(meta.get("total", -1)) != int(total):
            logger.warning(
                "Checkpoint total (%s) doesn't match expected (%s); ignoring.",
                meta.get("total"), total,
            )
            return None
        arr = np.load(CHECKPOINT_FILE)
        if arr.shape[0] > total:
            logger.warning("Checkpoint has more rows than expected; ignoring.")
            return None
        return arr.astype(np.float32, copy=False)
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        return None


def _clear_checkpoint() -> None:
    CHECKPOINT_FILE.unlink(missing_ok=True)
    CHECKPOINT_META.unlink(missing_ok=True)


# Backend: OpenAI (async, batched, resumable)
async def _embed_openai(texts: list[str]) -> np.ndarray:
    """Embed via OpenAI text-embedding-3-small with batch checkpointing.

    Returns a (N, dim) RAW (un-normalized) float32 array.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "EMBEDDING_BACKEND=openai but OPENAI_API_KEY is empty. Set it in .env."
        )

    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    n = len(texts)

    # ---- Resume from checkpoint if present ----
    completed_arr = _load_checkpoint(total=n)
    if completed_arr is not None and completed_arr.shape[0] > 0:
        logger.info(
            f"Resuming from checkpoint: {completed_arr.shape[0]}/{n} embeddings already done."
        )
        if completed_arr.shape[0] == n:
            return completed_arr
        new_chunks: list[np.ndarray] = [completed_arr]
        completed = int(completed_arr.shape[0])
    else:
        new_chunks = []
        completed = 0

    pbar = tqdm(
        total=n,
        initial=completed,
        desc=f"OpenAI embeddings ({OPENAI_EMBEDDING_MODEL})",
        unit="rec",
    )

    # Errors we will retry with exponential backoff. Pulled lazily so we
    # don't hard-import openai at module load time.
    from openai import APIConnectionError, APITimeoutError, RateLimitError
    RETRYABLE = (APIConnectionError, APITimeoutError, RateLimitError)

    async def _create_with_retry(batch: list[str], offset: int) -> list:
        """Call client.embeddings.create with exponential-backoff retry on
        transient errors (connection drops, TLS handshake hiccups, 429s).
        Returns resp.data on success; raises on persistent failure."""
        max_attempts = 5
        delay = 1.0
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                resp = await client.embeddings.create(
                    model=OPENAI_EMBEDDING_MODEL,
                    input=batch,
                )
                return resp.data
            except RETRYABLE as e:
                last_exc = e
                if attempt == max_attempts:
                    break
                logger.warning(
                    f"OpenAI transient error at offset {offset} "
                    f"(attempt {attempt}/{max_attempts}): {type(e).__name__}: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2.0, 30.0)
        assert last_exc is not None
        raise last_exc

    save_every_batches = 5
    batch_count = 0
    try:
        for start in range(completed, n, OPENAI_BATCH):
            end = min(start + OPENAI_BATCH, n)
            batch = texts[start:end]
            try:
                data = await _create_with_retry(batch, offset=start)
            except Exception as e:
                # Save what we have so the next run can resume.
                if new_chunks:
                    partial = np.vstack(new_chunks).astype(np.float32, copy=False)
                    _save_checkpoint(partial, total=n)
                    logger.error(
                        f"OpenAI batch failed at offset {start}: {e}. "
                        f"Saved checkpoint with {partial.shape[0]} vectors. Re-run to resume."
                    )
                raise

            vecs = np.asarray(
                [d.embedding for d in data], dtype=np.float32
            )
            new_chunks.append(vecs)
            pbar.update(end - start)

            batch_count += 1
            if batch_count % save_every_batches == 0:
                partial = np.vstack(new_chunks).astype(np.float32, copy=False)
                _save_checkpoint(partial, total=n)
    finally:
        pbar.close()

    full = np.vstack(new_chunks).astype(np.float32, copy=False)
    if full.shape[0] != n:
        raise RuntimeError(
            f"Embedding count mismatch: got {full.shape[0]}, expected {n}."
        )
    # Embed completed successfully for ALL `n` texts. Clear any stale
    # checkpoint so subsequent embed_texts() calls (e.g., per-query embeddings
    # during verify_retrieval) do NOT see a "1/1 already done" cache hit and
    # incorrectly reuse a previous query's vector.
    _clear_checkpoint()
    return full


# Backend: Local (sentence-transformers, lazy import)
def _embed_local_sync(texts: list[str]) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "EMBEDDING_BACKEND=local requires sentence-transformers.\n"
            "Install with: pip install sentence-transformers"
        ) from e

    model_attr = "_st_model"
    if not hasattr(_embed_local_sync, model_attr):
        logger.info(f"Loading local embedding model: {LOCAL_EMBEDDING_MODEL}")
        setattr(
            _embed_local_sync, model_attr, SentenceTransformer(LOCAL_EMBEDDING_MODEL)
        )
    model = getattr(_embed_local_sync, model_attr)

    vecs = model.encode(
        texts,
        batch_size=LOCAL_BATCH,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=False,  # we normalize centrally below
    )
    return np.asarray(vecs, dtype=np.float32)


# Public API: embed_texts
async def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed `texts` via the configured backend and return an (N, dim) float32
    numpy array L2-normalized to unit length.

    Mirrors the spec contract in Prompt 3 §3 — used by both build and verify.
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    if EMBEDDING_BACKEND == "openai":
        vecs = await _embed_openai(texts)
    elif EMBEDDING_BACKEND == "local":
        vecs = await asyncio.to_thread(_embed_local_sync, texts)
    else:
        raise ValueError(
            f"Unknown EMBEDDING_BACKEND: {EMBEDDING_BACKEND!r}. Use 'openai' or 'local'."
        )

    vecs = np.ascontiguousarray(vecs, dtype=np.float32)
    faiss.normalize_L2(vecs)
    return vecs


# Build
def _human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:,.2f} {unit}"
        size /= 1024.0
    return f"{size:,.2f} GB"


def _print_build_summary(
    n_vectors: int, dim: int, elapsed_s: float
) -> None:
    print("\n" + "=" * 78)
    print("INDEX BUILD SUMMARY")
    print("=" * 78)
    print(f"Total vectors indexed : {n_vectors:,}")
    print(f"Embedding dimensions  : {dim}")
    print(f"Embedding backend     : {EMBEDDING_BACKEND}")
    if INDEX_FILE.exists():
        print(f"Index file size       : {_human_size(INDEX_FILE.stat().st_size)}")
    if RECORDS_FILE.exists():
        print(f"Records file size     : {_human_size(RECORDS_FILE.stat().st_size)}")
    if EMBEDDINGS_FILE.exists():
        print(f"Embeddings file size  : {_human_size(EMBEDDINGS_FILE.stat().st_size)}")
    print(f"Time taken            : {elapsed_s:,.1f} s")
    print("=" * 78)


async def build_index() -> None:
    t0 = time.perf_counter()

    if not PARSED_RECORDS_PATH.exists():
        raise FileNotFoundError(
            f"{PARSED_RECORDS_PATH} not found. "
            "Run `python -m indexing.parse_conversations` first."
        )

    with PARSED_RECORDS_PATH.open("r", encoding="utf-8") as f:
        records = json.load(f)
    if not records:
        raise ValueError(f"{PARSED_RECORDS_PATH} is empty — nothing to index.")

    logger.info(f"Loaded {len(records):,} records from {PARSED_RECORDS_PATH}")

    INDEX_PATH.mkdir(parents=True, exist_ok=True)

    composites = [build_composite_string(r) for r in records]
    logger.info(
        f"Built {len(composites):,} composite strings. "
        f"Sample: {composites[0][:120]}..."
    )

    logger.info(f"Embedding via backend={EMBEDDING_BACKEND}...")
    embeddings = await embed_texts(composites)
    n, dim = embeddings.shape
    logger.info(f"Got embeddings shape={embeddings.shape}, dtype={embeddings.dtype}")

    if n != len(records):
        raise RuntimeError(
            f"Embedding row count ({n}) does not match records ({len(records)})."
        )

    logger.info(f"Building FAISS IndexFlatIP (dim={dim})...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info(f"Index ntotal={index.ntotal}")

    # ---- Save outputs ----
    logger.info(f"Writing index   -> {INDEX_FILE}")
    faiss.write_index(index, str(INDEX_FILE))

    logger.info(f"Writing records -> {RECORDS_FILE}")
    with RECORDS_FILE.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    logger.info(f"Writing vectors -> {EMBEDDINGS_FILE}")
    np.save(EMBEDDINGS_FILE, embeddings)

    # All persisted; safe to clear the checkpoint.
    _clear_checkpoint()

    elapsed = time.perf_counter() - t0
    _print_build_summary(n, dim, elapsed)


# Verify
VERIFY_QUERIES: list[tuple[str, dict]] = [
    (
        "[STRATEGY:RESTATEMENT_OR_PARAPHRASING] [COPING:Duty_Based] "
        "[PHASE:Exploration] [INTENSITY:5] [EMOTION:exhaustion] "
        "SEEKER: machine ki tarah chal raha hoon roz subah se",
        {
            "strategy": "RESTATEMENT_OR_PARAPHRASING",
            "seeker_coping": "Duty_Based",
            "phase": "Exploration",
        },
    ),
    (
        "[STRATEGY:QUESTION] [COPING:Relational_Preservation] "
        "[PHASE:Exploration] [INTENSITY:4] [EMOTION:shame] "
        "SEEKER: ghar mein koi nahi samajhta meri baat",
        {
            "strategy": "QUESTION",
            "seeker_coping": "Relational_Preservation",
            "phase": "Exploration",
        },
    ),
    (
        "[STRATEGY:REFLECTION_OF_FEELINGS] [COPING:Somatization] "
        "[PHASE:Insight] [INTENSITY:3] [EMOTION:overwhelm] "
        "SEEKER: sir mein bahut dard ho raha hai subah se",
        {
            "strategy": "REFLECTION_OF_FEELINGS",
            "seeker_coping": "Somatization",
            "phase": "Insight",
        },
    ),
]


async def verify_retrieval(top_k: int = 5) -> float:
    """Run the 3 spec test queries against the existing index and print results.

    Returns the mean retrieval quality across queries (in [0, 1]).
    """
    if not INDEX_FILE.exists() or not RECORDS_FILE.exists():
        raise FileNotFoundError(
            "Index files not found. Run `python -m indexing.build_index` first."
        )

    logger.info(f"Loading index from {INDEX_FILE}")
    index = faiss.read_index(str(INDEX_FILE))
    with RECORDS_FILE.open("r", encoding="utf-8") as f:
        records = json.load(f)
    logger.info(f"Loaded index with ntotal={index.ntotal} and {len(records):,} records.")

    qualities: list[float] = []

    for q_text, expected in VERIFY_QUERIES:
        q_vec = await embed_texts([q_text])  # already normalized
        D, I = index.search(q_vec, top_k)
        scores = D[0].tolist()
        idxs = I[0].tolist()
        results = [records[i] for i in idxs]

        matches = sum(
            1
            for r in results
            if all(r.get(k) == v for k, v in expected.items())
        )
        quality = matches / top_k if top_k else 0.0
        qualities.append(quality)

        print("\n" + "-" * 78)
        print(f"QUERY: {q_text[:80]}{'...' if len(q_text) > 80 else ''}")
        print(
            "EXPECT: "
            f"strategy={expected['strategy']}, "
            f"coping={expected['seeker_coping']}, "
            f"phase={expected['phase']}"
        )
        print("-" * 78)
        for rank, (r, s) in enumerate(zip(results, scores), 1):
            print(
                f"  #{rank}  score={s:+.3f}  "
                f"{r.get('strategy','?'):<30} | "
                f"{r.get('seeker_coping','?'):<24} | "
                f"{r.get('phase','?'):<12} | "
                f"int={r.get('seeker_intensity','?')}"
            )
            seeker = (r.get("seeker_text") or "").strip().replace("\n", " ")
            supp = (r.get("supporter_text") or "").strip().replace("\n", " ")
            print(f"      seeker:    {seeker[:80]}{'...' if len(seeker) > 80 else ''}")
            print(f"      supporter: {supp[:80]}{'...' if len(supp) > 80 else ''}")
        print(
            f"  retrieval_quality = {quality:.2f}  "
            f"({matches}/{top_k} match strategy+coping+phase)"
        )

    mean_q = sum(qualities) / len(qualities) if qualities else 0.0
    print("\n" + "=" * 78)
    print(f"OVERALL retrieval_quality (mean over {len(qualities)} queries): {mean_q:.2f}")
    print("=" * 78)
    return mean_q


# CLI
async def amain() -> int:
    parser = argparse.ArgumentParser(
        prog="build_index",
        description="Build / verify / rebuild the SAATHI FAISS retrieval index.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify an existing index by running 3 sample queries (no rebuild).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete existing index files and build from scratch.",
    )
    args = parser.parse_args()

    if args.verify and args.rebuild:
        parser.error("--verify and --rebuild are mutually exclusive.")

    if args.verify:
        await verify_retrieval()
        return 0

    if args.rebuild:
        if INDEX_PATH.exists():
            logger.info(f"Removing existing index directory: {INDEX_PATH}")
            # Preserve the directory itself; just remove its contents.
            for p in INDEX_PATH.iterdir():
                if p.is_file() or p.is_symlink():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p)
        INDEX_PATH.mkdir(parents=True, exist_ok=True)

    await build_index()
    print()
    await verify_retrieval()
    return 0


def main() -> int:
    try:
        return asyncio.run(amain())
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Partial checkpoint (if any) is preserved.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
