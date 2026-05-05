"""Paths, LLM/embedding settings, and SAATHI feature tunables from the environment."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
DATASET_DIR = PROJECT_ROOT / os.getenv("DATASET_DIR", "dataset")
DATA_DIR = PROJECT_ROOT / "data"
INDEX_PATH = DATA_DIR / "faiss_index"
PARSED_RECORDS_PATH = DATA_DIR / "parsed_records.json"

LLM_BACKEND = os.getenv("LLM_BACKEND", "openai")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANALYZER_MODEL = "gpt-4o"
GENERATOR_MODEL = "gpt-4o"
SAFETY_MODEL = "gpt-4o-mini"
SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL", "gpt-4o-mini")

LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:8001/v1")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "saathi-7b")

EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "openai")
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
LOCAL_EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL",
    "paraphrase-multilingual-MiniLM-L12-v2",
)

ANALYZER_TEMPERATURE = 0.1
GENERATOR_TEMPERATURE = 0.75
SAFETY_TEMPERATURE = 0.0
SUMMARIZER_TEMPERATURE = 0.1

SAATHI_SUMMARY_EVERY_N_TURNS = int(os.getenv("SAATHI_SUMMARY_EVERY_N_TURNS", "4"))
SAATHI_SUMMARY_HISTORY_TRIGGER = int(
    os.getenv("SAATHI_SUMMARY_HISTORY_TRIGGER", "12")
)
SAATHI_SUMMARY_INCREMENTAL_WINDOW = int(
    os.getenv("SAATHI_SUMMARY_INCREMENTAL_WINDOW", "8")
)

SAATHI_CARE_TAG_FREQ = int(os.getenv("SAATHI_CARE_TAG_FREQ", "4"))
SAATHI_EMOJI_ENABLED = os.getenv("SAATHI_EMOJI_ENABLED", "false").lower() == "true"
SAATHI_FACTS_WINDOW = int(os.getenv("SAATHI_FACTS_WINDOW", "8"))

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

CATEGORIES = [
    "Academic",
    "Employment",
    "Financial",
    "Family_Conflict",
    "Marriage_Pressure",
    "Health",
    "Gender_Identity",
    "Migration",
]
