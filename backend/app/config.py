"""Runtime configuration, read from environment variables with local-dev defaults."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Loads KEY=VALUE lines from backend/.env without overriding real environment variables."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(BASE_DIR / ".env")
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Each dataset under backend/datasets/ is imported into its own SQLite file in DATA_DIR
# (read-only at query time). PostgreSQL is the Phase-2 target.
# Platform metadata: schema annotations, access policies, vector index, audit log.
APP_DB_PATH = Path(os.getenv("APP_DB_PATH", DATA_DIR / "platform.db"))

JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_TTL_SECONDS = int(os.getenv("JWT_TTL_SECONDS", str(12 * 3600)))

# This build is local-only. No hosted LLM is used by the pipeline.
OPENAI_API_KEY = ""
OPENAI_MODEL = "disabled"

SCHEMA_TOP_K = int(os.getenv("SCHEMA_TOP_K", "3"))
AMBIGUITY_THRESHOLD = float(os.getenv("AMBIGUITY_THRESHOLD", "0.6"))
MAX_REPAIR_ATTEMPTS = int(os.getenv("MAX_REPAIR_ATTEMPTS", "3"))
ML_HALLUCINATION_ENABLED = os.getenv("ML_HALLUCINATION_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
ML_HALLUCINATION_THRESHOLD = float(os.getenv("ML_HALLUCINATION_THRESHOLD", "0.80"))
ML_HALLUCINATION_MODEL_PATH = Path(os.getenv("ML_HALLUCINATION_MODEL_PATH", BASE_DIR / "models" / "sql_hallucination_classifier.pkl"))
FUZZY_MATCH_THRESHOLD = float(os.getenv("FUZZY_MATCH_THRESHOLD", "0.8"))
MAX_RESULT_ROWS = int(os.getenv("MAX_RESULT_ROWS", "500"))
HISTORY_WINDOW = 3

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
