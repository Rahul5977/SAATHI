# SAATHI

SAATHI is a **peer-support style conversational assistant** aimed at Indian college students. The project spans **curated multi-turn dialogue data** (Hinglish / culturally grounded situations) through to a **working chat API** that combines safety checks, situation analysis, phased support strategies, generation, summarization, and memory.

---

## What we built (at a glance)

1. **Dataset** — Situation-grounded conversations across themes such as academic pressure, employment, finances, family conflict, marriage-related pressure, health, gender identity, and migration. Dialogues are stored as structured JSON (turn-level metadata: emotion, intensity, coping signals, strategies, phases, etc.).
2. **Preparation & analytics** — Scripts normalize folder names, validate and aggregate conversation trees, and produce **flattened records** (for example `data/parsed_records.json`) suitable for indexing and retrieval.
3. **Retrieval** — Records are embedded and indexed (FAISS under `data/faiss_index/`) so the live system can surface **similar past turns** as context for responses.
4. **Runtime architecture** — A **FastAPI** service exposes REST and WebSocket chat. Each turn flows through a **pipeline**: safety screening → analyzer (signals and facts) → **phase / strategy** selection → response generation, with **periodic summarization** and **Redis-backed session + user memory** for continuity across chats.
5. **Configuration** — Models, embeddings, paths, and behaviour tunables are centralized in `config.py` and environment variables (including optional local LLM / embedding backends).

---

## Repository layout (high level)

| Area | Role |
|------|------|
| `dataset/` | Source conversation JSONs by situation category; helper scripts per category |
| `dataset/calculate_fields.py` | Metrics and reporting over conversation trees |
| `data/` | Parsed records and FAISS index assets |
| `api/` | FastAPI app, chat routes |
| `pipeline/` | Orchestrator, sessions, memory |
| `agents/` | Analyzer, generator, safety, summarizer |
| `core/` | Schemas, phase gate, safety content |
| `prompts/` | LLM prompt templates |
| `eval/` | Evaluation / golden fixtures (e.g. memory continuity) |

---

## Contributors

| Name | ID |
|------|-----|
| Teesha Ramchandani | 12342260 |
| Rahul Raj | 12341680 |
| Suraj Kumar | 12342080 |

---

*B.Tech project — dataset preparation through production-oriented architecture for empathetic, safe peer-support dialogue.*
