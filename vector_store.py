import os
from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "tasks_minilm")


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


_client = chromadb.PersistentClient(
    path=str(DATA_DIR / "chroma"),
    settings=Settings(anonymized_telemetry=False),
)
_collection = _client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


def _task_text(task: dict) -> str:
    return (
        f"{task['title']}.{task.get('description') or ''}.Status: {task['status']}. Priority: {task['priority']}.Due:{task['due_date']}."
    )


def upsert(task: dict) -> None:
    text = _task_text(task)
    _collection.upsert(
        ids=[task["id"]],
        embeddings=[_embed(text)],
        documents=[text],
        metadatas=[{
            "title": task["title"],
            "status": task["status"],
            "priority": task["priority"],
        }],
    )


def remove(task_id: str) -> None:
    try:
        _collection.delete(ids=[task_id])
    except Exception:
        pass


def search(query: str, n: int = 10,
        max_distance: float = 0.65) -> list[tuple[str, float]]:
    """Return [(task_id, distance)] sorted by distance, closest first."""
    try:
        res = _collection.query(
            query_embeddings=[_embed(query)],
            n_results=n,
            include=["distances"],
        )
    except Exception:
        return []
    if not res["ids"] or not res["ids"][0]:
        return []
    pairs = list(zip(res["ids"][0], res["distances"][0]))
    return [(tid, d) for tid, d in pairs if d <= max_distance]
