import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATASET_DIR = PROJECT_ROOT / os.getenv("DATASET_DIR", "dataset")
DATA_DIR = PROJECT_ROOT / "data"
INDEX_PATH = DATA_DIR / "faiss_index"
PARSED_RECORDS_PATH = DATA_DIR / "parsed_records.json"

# LLM configuration
LLM_BACKEND = os.getenv("LLM_BACKEND", "openai")  # "openai" or "local"

# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANALYZER_MODEL = "gpt-4o"
GENERATOR_MODEL = "gpt-4o"
SAFETY_MODEL = "gpt-4o-mini"

# Local LLM settings (future — vLLM or Ollama serving OpenAI-compatible endpoint)
LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:8001/v1")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "saathi-7b")

# Embedding configuration
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "openai")  # "openai" or "local"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

# Temperature settings
ANALYZER_TEMPERATURE = 0.1
GENERATOR_TEMPERATURE = 0.75
SAFETY_TEMPERATURE = 0.0

# Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Categories in the dataset
CATEGORIES = [
    "Academic", "Employment", "Financial", "Family_Conflict",
    "Marriage_Pressure", "Health", "Gender_Identity", "Migration"
]
