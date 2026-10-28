# AI Task Manager

A task manager you control with plain chat. You type something like
"remind me to finish the demo tomorrow, high priority" and an LLM agent picks
the right tool (create, list, search, update, delete) and saves it to SQLite.
You can also search in your own words ("the thing about the client demo") and it
still finds the task.

- Repository: https://github.com/shohag-shadow/ai_task_manager

## Setup

Needs Python 3.10+ and an OpenAI compatible API key.

```bash
git clone https://github.com/shohag-shadow/ai_task_manager.git
cd ai_task_manager
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # Windows: copy .env.example .env
```

Now edit `.env` and fill in your key. 

| Variable | Required | What it is |
| --- | --- | --- |
| `OPENAI_API_KEY` | yes | your API key, e.g. `sk-...` |
| `OPENAI_BASE_URL` | no | only if you use a compatible gateway. Leave it empty for OpenAI directly |
| `MODEL_NAME` | no | chat model to use, default `gpt-4o-mini` |

There are two optional tuning variables if you ever need them: `EMBED_DEVICE`
(default `cpu`) and `CHROMA_COLLECTION` (default `tasks_minilm`).

Run it:

```bash
uvicorn main:app --reload
```

Open http://127.0.0.1:8000.

The first start downloads the embedding model (`all-MiniLM-L6-v2`, about 90 MB)
and creates the local database. Run it from the project root: everything is
stored in `data/` (`tasks.db` plus `data/chroma/`). Delete that folder to reset
the app. Secret files and data are git-ignored.

## What it can do

Type in the chat box on the right. The task list on the left refreshes after
every message.

| You type | What happens |
| --- | --- |
| `Create a high priority task to finish the demo tomorrow` | Creates the task, turns "tomorrow" into a real date |
| `Show my tasks` | Lists all tasks, grouped by status |
| `Show only urgent tasks` | Lists with a priority filter |
| `Move the demo task to in progress` | Updates one task |
| `Delete the demo task` | Asks for a `yes` before deleting |
| `Delete all finished tasks` | Asks to confirm a bulk delete |
| `Find the task about the client call` | Searches by meaning, not exact words |

The agent only does task work. Greetings are fine; anything else gets a short
polite refusal. It searches before updating or deleting, asks for confirmation
before deleting, and repeats the exact task titles it changed.

## How it works

```
browser  ->  FastAPI /api/chat  ->  LangGraph agent  ->  tools
                                          |               |
                                    OpenAI model     SQLite (real data)
                                                     Chroma (search index)
```

- **SQLite is the source of truth.** The vector store is only an index, so a
  task deleted from the database is never returned by a search.
- **The graph has two nodes.** `agent` asks the model which tool to call,
  `tools` runs it and sends the result back, and the loop ends when the model
  replies without a tool call.
- **Most wording is written in Python, not by the model.** Lists, delete
  prompts and errors come from code, so numbers and titles are always exact.
  When a tool already wrote the final reply, the second model call is skipped.

## Project structure

```
ai_task_manager/
├── main.py              FastAPI app, routes, static file mount
├── agent.py             model client, system prompt, tool schemas, graph loop
├── db.py                SQLite table and every query (source of truth)
├── vector_store.py      MiniLM embeddings in a persistent Chroma collection
├── requirements.txt     Python dependencies
├── .env.example         template for the three settings (safe to commit)
├── promt.txt            the prompts used to build this project
├── README.md
├── static/
│   ├── index.html       one page: task list on the left, chat on the right
│   └── app.js           fetch calls to /api and rendering, plain JS
└── data/                created on first run, git-ignored
    ├── tasks.db         SQLite database
    └── chroma/          vector index on disk
```

## API

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| `GET` | `/api/health` | | `{"ok": true, "model": "..."}` |
| `GET` | `/api/tasks` | | `{"tasks": [...]}` |
| `POST` | `/api/chat` | `{"message": str, "history": [{"role": "user"\|"assistant", "content": str}]}` | `{"reply": str, "tool_log": [...]}` |


Task fields: `status` is `todo`, `in_progress` or `done`; `priority` is `low`,
`medium`, `high` or `urgent`; `due_date` is `YYYY-MM-DD` and optional. 

## Problems I hit and how I fixed them

**1. "Show my tasks" was hiding the finished ones.**
The list tool kept coming back with only todo tasks. The model was adding
`status="todo"` to the call by itself, and the tool trusted it. Now the code
checks each filter against the user's actual message and drops any status or
priority they never said out loud. "All tasks" always means no filter, whatever
the model decides to send.

**2. The same task could be created twice.**
Two identical rows appeared with different ids, and then updating got
confusing because two tasks matched the same words. I gave tasks ids built from
the title plus the date, and added a unique index on active titles. Creating a
duplicate now returns a clear message instead of a second row.

**3. Delete was the messiest part.**
The model wrote a different confirmation every time, sometimes said a task was
deleted before it happened, and once forgot it had already asked. I moved the
whole flow into code: the delete tool never deletes, it just finds the matches
and stores them in memory. `yes` and `no` are checked in Python without calling
the model at all, and any other reply silently cancels. The final message is
built from what the database actually did.

**4. Prompt caching never kicked in.**
I had today's date at the top of the system prompt, so the long prompt counted
as new text every single day. Moving the date to the end keeps the whole static
part reusable, and the app still knows what "tomorrow" means.

**5. Search returned tasks I never had, or kept returning deleted ones.**
Vector search always returns its nearest neighbours, even when they are nowhere
near the question, and the index was outliving rows removed from the database.
I added a distance cutoff, and every hit is now looked up in SQLite before it is
shown. Ids that no longer exist are dropped, so the index can never show a task
the database does not have.  

**6. My system promt was too large and was consuming most of the token.**
The system promt that i created first was 1.5-2 k tokens and it was consuming most of the tokens used for the app.
I added most of the rules that were in the promt in code in db and other places.
This improved the consistancy and reduced the token cost.Model now only handle picking the tool that is needed and make texts like title, decide date etc.

## Notes and limits

- Single user, single process. Pending deletes are kept in memory, so restarting
  the app cancels them.
- Only one active task per title. Re-using a title after the first one is
  finished is allowed.
- Lists show up to 30 tasks; older ones stay in the database.
- Without `OPENAI_API_KEY` the app starts and shows the page, but chat returns a
  clear error instead of failing silently.
- There is no time field, so times go in the title ("Meeting at 9 PM").
