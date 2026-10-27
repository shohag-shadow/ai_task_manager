import os
from functools import lru_cache
from pathlib import Path

from sentence_transformers import SentenceTransformer

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    """Load the model once and reuse it for the life of the process."""
    return SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)


def _embed(text: str) -> list[float]:
    """Return a normalized 384-d embedding."""
    vec = _model().encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vec.tolist()
