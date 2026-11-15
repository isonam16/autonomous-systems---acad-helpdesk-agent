import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Vercel's filesystem is read-only except /tmp
_DATA_OUT = Path("/tmp/helpdesk") if os.getenv("VERCEL") else ROOT / "data"
DOCS_DIR = ROOT / "data" / "docs"
TIMETABLE_CSV = DOCS_DIR / "csv" / "courses.csv"
FACULTY_CSV = DOCS_DIR / "csv" / "faculty.csv"
CHROMA_DIR = _DATA_OUT / "chroma"
CHUNKS_JSON = _DATA_OUT / "chunks.json"
UNANSWERED_LOG = _DATA_OUT / "unanswered_questions.csv"

# Ollama
OLLAMA_HOST = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
os.environ.setdefault("OLLAMA_API_BASE", OLLAMA_HOST)  # used by LiteLLM
# Swap the LLM here for the "replace the LLM" experiment, e.g. ollama_chat/qwen3:8b
# Provider choice: LLM_PROVIDER=gemini (default, needs GEMINI_API_KEY, see .env.example) or ollama
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini/gemini-3.5-flash")
if LLM_PROVIDER == "gemini":
    _key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not _key:
        raise RuntimeError("LLM_PROVIDER=gemini but GEMINI_API_KEY is not set (see .env.example)")
    os.environ["GEMINI_API_KEY"] = _key
    LLM_MODEL = os.getenv("LLM_MODEL", GEMINI_MODEL)
else:
    LLM_MODEL = os.getenv("LLM_MODEL", "ollama_chat/llama3.1:8b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Max texts embedded per minute while indexing (Gemini free tier = 100; set higher if billed)
EMBED_RPM = int(os.getenv("EMBED_RPM", "80"))

# Retrieval
CHUNK_CHARS = 900
CHUNK_OVERLAP = 150
TOP_K = 5
# Cosine distance of the best dense hit above which we say "nothing relevant".
# TUNE this on your own data using the in-scope vs out-of-scope questions.
MAX_BEST_DISTANCE = float(os.getenv("MAX_BEST_DISTANCE", "0.55"))
USE_CONTEXTUAL_RETRIEVAL = os.getenv("USE_CONTEXTUAL_RETRIEVAL", "0") == "1"
