"""Settings shared by all scripts in this folder."""

from pathlib import Path

ROOT = Path(__file__).parent
PDF_PATH = ROOT / "data" / "NIST.AI.100-1.pdf"
INDEX_DIR = ROOT / "index"      # chunk files + FAISS indexes (re-creatable, not committed)
RESULTS_DIR = ROOT / "results"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# chunk size in words -> overlap in words (about 15 % of the chunk size)
CHUNK_SIZES = {150: 22, 500: 75}

OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "qwen2.5:1.5b"
