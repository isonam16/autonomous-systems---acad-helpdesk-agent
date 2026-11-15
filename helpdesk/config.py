import os
from pathlib import Path
try:
 from dotenv import load_dotenv
 load_dotenv(Path(__file__).resolve().parent.parent/".env")
except ImportError: pass
ROOT=Path(__file__).resolve().parent.parent
DOCS_DIR=Path(os.getenv("DOCS_DIR",str(ROOT/"data"/"docs")))
TIMETABLE_CSV=ROOT/"data"/"timetable.csv"
FACULTY_CSV=ROOT/"data"/"faculty.csv"
CHROMA_DIR=Path(os.getenv("CHROMA_DIR",str(ROOT/"data"/"chroma")))
CHUNKS_JSON=Path(os.getenv("CHUNKS_JSON",str(ROOT/"data"/"chunks.json")))
UNANSWERED_LOG=Path(os.getenv("UNANSWERED_LOG",str(ROOT/"data"/"unanswered_questions.csv")))
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY: raise RuntimeError("GEMINI_API_KEY is required.")
os.environ["GEMINI_API_KEY"]=GEMINI_API_KEY
GEMINI_MODEL=os.getenv("GEMINI_MODEL","gemini-2.5-flash")
LLM_MODEL=f"gemini/{GEMINI_MODEL}"
EMBED_MODEL=os.getenv("EMBED_MODEL","gemini-embedding-001")
CHUNK_CHARS=900
CHUNK_OVERLAP=150
TOP_K=5
MAX_BEST_DISTANCE=float(os.getenv("MAX_BEST_DISTANCE","0.55"))
USE_CONTEXTUAL_RETRIEVAL=os.getenv("USE_CONTEXTUAL_RETRIEVAL","0")=="1"
