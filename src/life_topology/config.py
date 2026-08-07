from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PACKAGE_ROOT / "output"
PARQUET_ROOT = OUTPUT_ROOT / "parquet"
DB_PATH = OUTPUT_ROOT / "daily_state.db"
VECTOR_PATH = OUTPUT_ROOT / "vectors.lance"
MANIFEST_PATH = OUTPUT_ROOT / "manifest.json"

# Model repositories are remote executable inputs, not ordinary data files.
# Resolve only this allowlist and pin every artifact to an audited commit.
MODEL_REVISIONS = {
    "sentence-transformers/all-MiniLM-L6-v2": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "BAAI/bge-m3": "5617a9f61b028005a4858fdac845db406aefb181",
}
SAFETENSORS_MODELS = frozenset({"sentence-transformers/all-MiniLM-L6-v2"})


@dataclass(frozen=True)
class Settings:
    timezone: str = "America/Chicago"
    behavioral_day_hour: int = 4
    ai_episode_gap_minutes: int = 75
    social_episode_gap_minutes: int = 180
    media_episode_gap_minutes: int = 45
    min_atomic_chars: int = 12
    embedding_batch_size: int = 64
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    windows: tuple[int, ...] = (7, 30, 90, 180)
    random_seed: int = 17032002


SETTINGS = Settings()


def ensure_output_dirs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PARQUET_ROOT.mkdir(parents=True, exist_ok=True)
