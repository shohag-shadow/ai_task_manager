import json
import os
import re
import sqlite3
from datetime import date
from typing import TypedDict

from langgraph.graph import StateGraph, END
from openai import OpenAI

import db
import vector_store

MODEL = os.getenv("MODEL_NAME", "gpt-4o-mini")
_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL") or None,
)

STATUSES = ("todo", "in_progress", "done")
ACTIVE = ("todo", "in_progress")
PRIORITIES = ("low", "medium", "high", "urgent")
UPDATABLE = ("title", "description", "status", "priority", "due_date")

# Semantic cutoff. Assumes vector_store.search returns (id, distance) with lower = closer.
# If it returns a similarity score instead, flip the comparison in _search().
MAX_DISTANCE = 1.0
MAX_TURNS = 6  # past user/assistant pairs sent to the model

SYSTEM_PROMPT = """You are a task manager assistant. You manage tasks ONLY through the provided tools.

1. SCOPE (most important): You only greet the user and manage tasks. Do not chat, give opinions, answer general questions, do math, or write text or code, even if asked nicely.
   - Greeting: greet back briefly and ask how you can help with their tasks.
   - Anything else: decline politely and humbly in one or two short sentences, then offer task help, e.g.
     "Sorry, I can't help with that. I'm a task manager assistant and can only help with your tasks, like adding, listing, updating, or deleting them. Is there a task I can help you with?"
   - Mixed request: do the task part, never answer the rest, and say so briefly, e.g.
     "Ignored the 'x' part of your request as it is out of my scope"

2. CREATE: call create_task with a clear title. Convert relative dates ("tomorrow", "next Monday", "in 3 days") to YYYY-MM-DD. Leave out due date and priority if the user did not mention them. There is no time field, so put any time in the title ("Meeting at 9 PM").

3. UPDATE / MOVE / COMPLETE: call update_task with `query` (keywords from the task's current title) and only the fields that change. The tool finds the task and asks the user if several match.

4. DELETE: call delete_tasks. Use `query` for one specific task. For bulk requests use filters, e.g. "delete all finished tasks" -> status "done" and all_matching true. The tool asks the user to confirm, so never say anything was deleted yourself.

5. LIST / SEARCH: use list_tasks for "show/list" requests (with status/priority filters if given) and find_tasks for descriptive queries like "tasks related to X".

6. VALUES: statuses are todo, in_progress, done ("in progress" -> in_progress, "complete" -> done). Priorities are low, medium, high, urgent.

7. REPLIES: keep them short and friendly. When confirming a change, quote the exact title, status, priority and due_date from the tool result. If a tool returns ok=false, say what went wrong. If you did not call a tool this turn, do not claim any change happened. Never invent task details."""


# --------------------------------------------------------------------------- tools

def _tool(name: str, description: str, properties: dict, required: tuple = ()) -> dict:
    params: dict = {"type": "object", "properties": properties}
    if required:
        params["required"] = list(required)
    return {"type": "function",
            "function": {"name": name, "description": description, "parameters": params}}


_text = {"type": "string"}
_status = {"type": "string", "enum": list(STATUSES)}
_priority = {"type": "string", "enum": list(PRIORITIES)}
_due = {"type": "string", "description": "YYYY-MM-DD"}

TOOLS = [
    _tool("create_task", "Create a new task.",
          {"title": _text, "description": _text, "status": _status,
           "priority": _priority, "due_date": _due},
          required=("title",)),
    _tool("list_tasks", "List tasks, optionally filtered by status or priority.",
          {"status": _status, "priority": _priority}),
    _tool("find_tasks", "Search tasks by keyword or natural language.",
          {"query": _text, "status": _status},
          required=("query",)),
    _tool("update_task",
          "Update ONE task. `query` is keywords from its current title. The tool finds "
          "the task itself and asks the user if several match.",
          {"query": _text, "title": _text, "description": _text, "status": _status,
           "priority": _priority, "due_date": _due},
          required=("query",)),
    _tool("delete_tasks",
          "Delete tasks. The tool finds the matches itself and asks the user to confirm; "
          "it never deletes directly. Use `query` for one specific task, or status/priority "
          "filters for bulk requests. Set all_matching=true only when the user wants ALL "
          "matching tasks deleted.",
          {"query": _text, "status": _status, "priority": _priority,
           "all_matching": {"type": "boolean"}}),
]

_REQUIRED = {"create_task": "title", "find_tasks": "query", "update_task": "query"}


def _validate(name: str, args: dict) -> str | None:
    key = _REQUIRED.get(name)
    if key and not str(args.get(key) or "").strip():
        return f"{key} is required"
    if "title" in args and not str(args["title"] or "").strip():
        return "title cannot be empty"
    if args.get("status") not in (None, *STATUSES):
        return f"status must be one of {', '.join(STATUSES)}"
    if args.get("priority") not in (None, *PRIORITIES):
        return f"priority must be one of {', '.join(PRIORITIES)}"
    if args.get("due_date"):
        try:
            date.fromisoformat(args["due_date"])
        except ValueError:
            return "due_date must be a valid date in YYYY-MM-DD format"
    return None


# --------------------------------------------------------------------------- helpers

def _compact(tasks: list[dict], with_description: bool = False) -> list[dict]:
    rows = []
    for t in tasks:
        row = {k: t.get(k) for k in ("id", "title", "status", "priority", "due_date")}
        if with_description:
            row["description"] = (t.get("description") or "")[:120] or None
        rows.append(row)
    return rows


def _numbered(tasks: list[dict]) -> str:
    return "\n".join(f"{i}. '{t['title']}' ({t['status']})" for i, t in enumerate(tasks, 1))


def _is_duplicate_title_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("unique" in msg and "title" in msg) or "uq_tasks_title_active" in msg


def _search(query: str, statuses: tuple | None = None) -> list[dict]:
    """Keyword hits first, then semantic hits that are actually close enough."""
    kw_ids = [t["id"] for t in db.keyword_search(query, limit=10)]
    vec_ids = [tid for tid, dist in vector_store.search(query, n=10) if dist <= MAX_DISTANCE]
    tasks = db.hydrate(list(dict.fromkeys(kw_ids + vec_ids)))
    if statuses:
        tasks = [t for t in tasks if t["status"] in statuses]
    return tasks


def _resolve(query: str, statuses: tuple | None = None) -> list[dict]:
    """The tasks the user most likely means. An exact title match wins."""
    tasks = _search(query, statuses)
    exact = [t for t in tasks if t["title"].strip().lower() == query.strip().lower()]
    return exact or tasks


# --------------------------------------------------------------------------- tool implementations
# A result with a "reply" key is sent to the user as is, which skips the second LLM call.

def _exec_create(args: dict) -> dict:
    title = args["title"].strip()
    try:
        result = db.create_task(
            title=title,
            description=args.get("description"),
            status=args.get("status", "todo"),
            priority=args.get("priority", "medium"),
            due_date=args.get("due_date"),
        )
    except sqlite3.IntegrityError as e:
        if _is_duplicate_title_error(e):
            return {"ok": False, "error": "DUPLICATE_TITLE",
                    "reply": f"There's already an active task called '{title}'. "
                             f"What should the new one be called?"}
        raise
    if result["ok"]:
        vector_store.upsert(result["task"])
    return result


def _exec_list(args: dict) -> dict:
    tasks = db.list_tasks(status=args.get("status"), priority=args.get("priority"))
    return {"ok": True, "count": len(tasks), "truncated": len(tasks) > 20,
            "tasks": _compact(tasks[:20], with_description=True)}


def _exec_find(args: dict) -> dict:
    status = args.get("status")
    tasks = _search(args["query"], (status,) if status else None)
    return {"ok": True, "count": len(tasks), "tasks": _compact(tasks[:5])}


def _exec_update(args: dict) -> dict:
    query = args["query"].strip()
    fields = {k: args[k] for k in UPDATABLE if args.get(k) is not None}
    if not fields:
        return {"ok": False, "error": "NOTHING_TO_UPDATE",
                "reply": "What would you like to change on that task?"}

    # Active tasks first. Completed ones only if nothing active matches (e.g. "reopen X").
    tasks = _resolve(query, ACTIVE) or _resolve(query)
    if not tasks:
        return {"ok": False, "error": "NOT_FOUND",
                "reply": f"I couldn't find a task matching '{query}'."}
    if len(tasks) > 1:
        return {"ok": False, "error": "AMBIGUOUS",
                "reply": "Which one do you mean?\n" + _numbered(tasks[:5])}

    try:
        result = db.update_task(tasks[0]["id"], **fields)
    except sqlite3.IntegrityError as e:
        if _is_duplicate_title_error(e):
            return {"ok": False, "error": "DUPLICATE_TITLE",
                    "reply": f"There's already an active task called '{fields.get('title')}'. "
                             f"What should it be called instead?"}
        raise
    if result["ok"]:
        vector_store.upsert(result["task"])
    return result


# Tasks waiting for the user's "yes". Single user, single process: held in memory.
_pending_delete: list[dict] = []


def _exec_delete(args: dict) -> dict:
    """Never deletes. Finds the targets, remembers them, and asks for confirmation."""
    global _pending_delete
    query = (args.get("query") or "").strip()
    status, priority = args.get("status"), args.get("priority")
    bulk = bool(args.get("all_matching"))

    if not (query or status or priority or bulk):
        return {"ok": False, "error": "MISSING_TARGET", "reply": "Which task should I delete?"}

    if query:
        tasks = [t for t in _resolve(query)
                 if (not status or t["status"] == status)
                 and (not priority or t["priority"] == priority)]
    else:
        tasks = db.list_tasks(status=status, priority=priority)

    if not tasks:
        return {"ok": False, "error": "NOT_FOUND", "reply": "I couldn't find any matching tasks."}
    if len(tasks) > 1 and not bulk:
        return {"ok": False, "error": "AMBIGUOUS",
                "reply": "Which one do you mean?\n" + _numbered(tasks[:5])}

    _pending_delete = [{"id": t["id"], "title": t["title"]} for t in tasks]
    if len(tasks) == 1:
        title = tasks[0]["title"]
        text = f"Delete '{title}'? Reply 'yes' to confirm."
    else:
        text = f"Delete these {len(tasks)} tasks?\n{_numbered(tasks)}\nReply 'yes' to confirm."
    return {"ok": True, "needs_confirmation": True, "reply": text}


_TOOL_MAP = {
    "create_task": _exec_create,
    "list_tasks": _exec_list,
    "find_tasks": _exec_find,
    "update_task": _exec_update,
    "delete_tasks": _exec_delete,
}


# --------------------------------------------------------------------------- graph

class State(TypedDict):
    messages: list
    tool_log: list[dict]


def call_model(state: State) -> State:
    resp = _client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system",
             "content": f"{SYSTEM_PROMPT}\n\nToday is {date.today().isoformat()}."},
            *state["messages"],
        ],
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.1,
        max_tokens=500,
    )
    msg = resp.choices[0].message
    out: dict = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        out["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]
    return {"messages": state["messages"] + [out], "tool_log": state["tool_log"]}


def call_tools(state: State) -> State:
    log = list(state["tool_log"])
    new_msgs: list[dict] = []
    results: list[dict] = []

    for tc in state["messages"][-1].get("tool_calls") or []:
        name = tc["function"]["name"]
        try:
            args = json.loads(tc["function"]["arguments"] or "{}")
        except json.JSONDecodeError:
            args = {}

        fn = _TOOL_MAP.get(name)
        problem = _validate(name, args)
        if fn is None:
            result = {"ok": False, "error": "UNKNOWN_TOOL", "message": f"Unknown tool: {name}"}
        elif problem:
            result = {"ok": False, "error": "INVALID_INPUT", "message": problem}
        else:
            try:
                result = fn(args)
            except Exception as e:  # noqa: BLE001
                result = {"ok": False, "error": "TOOL_ERROR", "message": str(e)}

        results.append(result)
        log.append({"name": name, "args": args, "result": result, "tool_call_id": tc["id"]})
        new_msgs.append({"role": "tool", "tool_call_id": tc["id"],
                         "content": json.dumps(result, default=str)})

    # If every tool already produced the final wording, skip the extra LLM call.
    if results and all("reply" in r for r in results):
        new_msgs.append({"role": "assistant", "content": "\n".join(r["reply"] for r in results)})

    return {"messages": state["messages"] + new_msgs, "tool_log": log}


def _after_agent(state: State) -> str:
    return "tools" if state["messages"][-1].get("tool_calls") else END


def _after_tools(state: State) -> str:
    return END if state["messages"][-1]["role"] == "assistant" else "agent"


_graph = StateGraph(State)
_graph.add_node("agent", call_model)
_graph.add_node("tools", call_tools)
_graph.set_entry_point("agent")
_graph.add_conditional_edges("agent", _after_agent, {"tools": "tools", END: END})
_graph.add_conditional_edges("tools", _after_tools, {"agent": "agent", END: END})
app_graph = _graph.compile()


# --------------------------------------------------------------------------- chat entry point

_YES = {"yes", "y", "yeah", "yep", "sure", "ok", "okay", "confirm", "do it", "go ahead"}
_NO = {"no", "n", "nope", "cancel", "stop", "dont", "never mind", "nevermind"}


def _answer(message: str) -> str:
    text = re.sub(r"[^a-z ]", "", message.lower()).replace("please", "")
    return " ".join(text.split())


def _handle_pending_delete(message: str) -> dict | None:
    """Handle yes/no to a pending delete without the LLM. Any other message cancels it."""
    global _pending_delete
    pending, _pending_delete = _pending_delete, []
    if not pending:
        return None

    answer = _answer(message)
    if answer in _NO:
        return {"reply": "Okay, nothing was deleted.", "tool_log": [], "last_turn": None}
    if answer not in _YES:
        return None

    deleted, failed, log = [], [], []
    for t in pending:
        try:
            r = db.delete_task(t["id"])
        except Exception as e:  # noqa: BLE001
            r = {"ok": False, "error": "TOOL_ERROR", "message": str(e)}
        log.append({"name": "delete_task", "args": {"task_id": t["id"]}, "result": r})
        if r.get("ok"):
            vector_store.remove(t["id"])
            deleted.append(f"'{t['title']}'")
        else:
            failed.append(f"'{t['title']}' ({r.get('message') or r.get('error')})")

    parts = []
    if deleted:
        parts.append("Deleted " + ", ".join(deleted) + ".")
    if failed:
        parts.append("Could not delete " + ", ".join(failed) + ".")
    return {"reply": " ".join(parts), "tool_log": log, "last_turn": None}


def _trim(history: list[dict]) -> list[dict]:
    keep = [m for m in history
            if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)]
    keep = keep[-MAX_TURNS * 2:]
    while keep and keep[0]["role"] != "user":
        keep = keep[1:]
    return keep


def run_chat(message: str, history: list[dict], last_turn: list[dict] | None = None) -> dict:
    """`last_turn` is kept so existing callers still work, but it is no longer needed."""
    handled = _handle_pending_delete(message)
    if handled:
        return handled

    messages = [{"role": m["role"], "content": m["content"]} for m in _trim(history)]
    messages.append({"role": "user", "content": message})

    final = app_graph.invoke({"messages": messages, "tool_log": []})
    return {
        "reply": final["messages"][-1].get("content") or "",
        "tool_log": final["tool_log"],
        "last_turn": None,
    }