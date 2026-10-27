"""SQLite storage for tasks."""
import sqlite3
import uuid
from datetime import date
from pathlib import Path

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "tasks.db"

NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id          TEXT PRIMARY KEY,
            title       TEXT NOT NULL,
            description TEXT,
            status      TEXT NOT NULL DEFAULT 'todo'
                        CHECK (status IN ('todo','in_progress','done')),
            priority    TEXT NOT NULL DEFAULT 'medium'
                        CHECK (priority IN ('low','medium','high','urgent')),
            due_date    TEXT,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        )
        """)


def make_deterministic_id(title: str) -> str:
    """Build a stable id from the title and the current date."""
    name = f"task:{title.strip().lower()}:{date.today().isoformat()}"
    return str(uuid.uuid5(NAMESPACE, name))


def _row_to_dict(row: sqlite3.Row) -> dict:
    return dict(row) if row else None
