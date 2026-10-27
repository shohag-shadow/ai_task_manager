"""SQLite storage for tasks."""
import sqlite3
import uuid
from datetime import datetime, date
from pathlib import Path
from typing import Optional

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


def create_task(
    title: str,
    description: Optional[str] = None,
    status: str = "todo",
    priority: str = "medium",
    due_date: Optional[str] = None,
) -> dict:
    task_id = make_deterministic_id(title)
    now = datetime.utcnow().isoformat()
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if existing:
            return {"ok": False, "error": "DUPLICATE",
                    "message": f"A task titled '{title}' already exists today.",
                    "task": _row_to_dict(existing)}
        conn.execute("""
            INSERT INTO tasks (id, title, description, status, priority,
                            due_date, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (task_id, title, description, status, priority, due_date, now, now))
        task = _row_to_dict(conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        ).fetchone())
    return {"ok": True, "task": task}


def get_task(task_id: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
    return _row_to_dict(row)


def list_tasks(status: Optional[str] = None,
            priority: Optional[str] = None) -> list[dict]:
    q = "SELECT * FROM tasks WHERE 1=1"
    params: list = []
    if status:
        q += " AND status = ?"
        params.append(status)
    if priority:
        q += " AND priority = ?"
        params.append(priority)
    q += " ORDER BY CASE priority WHEN 'urgent' THEN 0 WHEN 'high' THEN 1 " \
        "WHEN 'medium' THEN 2 ELSE 3 END, due_date IS NULL, due_date"
    with get_conn() as conn:
        rows = conn.execute(q, params).fetchall()
    return [_row_to_dict(r) for r in rows]


def update_task(task_id: str, **fields) -> dict:
    allowed = {"title", "description", "status", "priority", "due_date"}
    fields = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not fields:
        return {"ok": False, "error": "NO_FIELDS",
                "message": "No valid fields to update."}
    fields["updated_at"] = datetime.utcnow().isoformat()
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [task_id]
    with get_conn() as conn:
        cur = conn.execute(
            f"UPDATE tasks SET {set_clause} WHERE id = ?", values
        )
        if cur.rowcount == 0:
            return {"ok": False, "error": "NOT_FOUND",
                    "message": f"Task {task_id} not found."}
        task = _row_to_dict(conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        ).fetchone())
    return {"ok": True, "task": task}


def delete_task(task_id: str) -> dict:
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        if cur.rowcount == 0:
            return {"ok": False, "error": "NOT_FOUND",
                    "message": f"Task {task_id} not found."}
    return {"ok": True, "task_id": task_id}
