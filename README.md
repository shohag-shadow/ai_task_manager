# AI Task Manager

A task manager you control through chat. A LangGraph agent reads your message,
picks the right tool (create / list / search / update / delete), and writes to
SQLite. Tasks are also indexed in Chroma so "find the thing about the demo"
works without exact wording.

## How it works

| File | Role |
| --- | --- |
| `main.py` | FastAPI app, routes, static file mount |
| `agent.py` | Model client, system prompt, tool schemas, LangGraph loop |
| `db.py` | SQLite schema and CRUD (source of truth) |
| `vector_store.py` | MiniLM embeddings in a persistent Chroma collection |
| `static/` | Single page chat + task list, plain JS with Tailwind CDN |

The agent runs a two node graph: `agent` calls the model, `tools` executes any
requested tool, and control returns to `agent` until it answers without a tool
call. Every tool result is logged and returned to the browser with the reply.

SQLite owns task state. The vector store is a lookup index only: search hits are
re-resolved against the database, so anything deleted there is never returned.

Task ids are derived from the title and today's date, so creating the same task
twice in one day returns a duplicate error instead of a second row.

## Requirements

- Python 3.10+
- An OpenAI compatible chat completion endpoint and API key

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` in the project root:

```
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=                 # optional, for a compatible gateway
MODEL_NAME=gpt-4o-mini           # optional
```

Optional tuning: `EMBED_DEVICE` (default `cpu`), `CHROMA_COLLECTION`
(default `tasks_minilm`).

## Run

```bash
uvicorn main:app --reload
```

Open http://127.0.0.1:8000.

The first start downloads the embedding model (`all-MiniLM-L6-v2`) and builds
the Chroma collection. Run from the project root: `data/` is created relative to
the working directory and holds `tasks.db` plus `data/chroma/`. Both are
git-ignored, so deleting `data/` resets the app.

## Usage

Type in the chat panel. The task list on the left refreshes after every turn.

- `Create a high priority task to finish the demo tomorrow`
- `Show my tasks`
- `Move the demo task to in progress`
- `Delete the demo task` (asks for a yes before deleting)
- `Delete all finished tasks`

The agent only handles task work. Greetings are fine, anything unrelated is
declined. It always searches before updating or deleting, asks for a
confirmation, and reports back the exact titles it changed.

## API

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| `GET` | `/api/health` | | `{"ok": true, "model": "..."}` |
| `GET` | `/api/tasks` | | `{"tasks": [...]}` |
| `POST` | `/api/chat` | `{"message": str, "history": [{"role": "user"\|"assistant", "content": str}]}` | `{"reply": str, "tool_log": [...]}` |

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Create a task to finish the demo tomorrow", "history": []}'
```

## Task fields

- `status`: `todo`, `in_progress`, `done`
- `priority`: `low`, `medium`, `high`, `urgent`
- `due_date`: `YYYY-MM-DD`, optional

There is no time field, so times belong in the title (`"Meeting at 9 PM"`).
