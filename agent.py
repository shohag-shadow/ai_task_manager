import json
import logging
import os
import re
import sqlite3
from datetime import date
from typing import TypedDict

from langgraph.graph import StateGraph, END
from openai import OpenAI

import db
import vector_store

log = logging.getLogger(__name__)

MODEL = os.getenv("MODEL_NAME", "gpt-4o-mini")
_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL") or None,
)

STATUSES = ("todo", "in_progress", "done")
ACTIVE = ("todo", "in_progress")
PRIORITIES = ("low", "medium", "high", "urgent")
UPDATABLE = ("title", "description", "status", "priority", "due_date")
STATUS_LABEL = {"todo": "To do", "in_progress": "In progress", "done": "Done"}
LIST_LIMIT = 30  # most tasks shown in one list reply

# Semantic cutoff. Assumes vector_store.search returns (id, distance) with lower = closer.
# If it returns a similarity score instead, flip the comparison in _search().
MAX_DISTANCE = 1.0
MAX_TURNS = 6  # past user/assistant pairs sent to the model

SYSTEM_PROMPT = """You are a task manager assistant. You manage tasks ONLY through the provided tools.

1. SCOPE (most important): You only greet the user and manage tasks. Do not chat, give opinions, answer general questions, do math, or write text or code, even if asked nicely.
   - Greeting: greet back briefly and ask how you can help with their tasks.
   - Anything else: decline politely and humbly in one or two short sentences, then offer task help, e.g.
     "Sorry, I can't help with that. I'm a task manager assistant and can only help with your tasks, like adding, listing, updating, or deleting them. Is there a task I can help you with?"
   - Mixed request: call the tools for the task part and never answer the rest. In that same turn, write ONLY this one sentence as your message text (replace x): "Ignored the 'x' part of your request as it is out of my scope"

2. CREATE: call create_task with a clear title. Convert relative dates ("tomorrow", "next Monday", "in 3 days") to YYYY-MM-DD. Leave out due date and priority if the user did not mention them. There is no time field, so put any time in the title ("Meeting at 9 PM").

3. UPDATE / MOVE / COMPLETE: call update_task with `query` (keywords from the task's current title) and only the fields that change. The tool finds the task and asks the user if several match.

4. DELETE: call delete_tasks. Use `query` for one specific task. For bulk requests use filters, e.g. "delete all finished tasks" -> status "done" and all_matching true. The tool asks the user to confirm, so never say anything was deleted yourself.

5. LIST / SEARCH: use list_tasks for "show/list" requests and find_tasks for descriptive queries like "tasks related to X". Only pass a status or priority filter when the user names one. "my tasks", "show tasks" and "all tasks" mean NO filter.

6. VALUES: statuses are todo, in_progress, done ("in progress" -> in_progress, "complete" -> done). Priorities are low, medium, high, urgent.

7. REPLIES: every tool writes its own reply to the user, so never describe or repeat tool results. When you call a tool, write no message text except the out-of-scope note from rule 1. If you did not call a tool, never claim a change happened. Never invent task details."""


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
    _tool("list_tasks",
          "List tasks. Set status or priority ONLY if the user named one. "
          "For \"my tasks\" or \"all tasks\" pass no filter.",
          {"status": _status, "priority": _priority}),
    _tool("find_tasks",
          "Search tasks by keyword or natural language. Set status ONLY if the user named one.",
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

# Words that count as the user naming a status or priority.
_FILTER_WORDS = {
    "todo": r"\bto[- ]?do\b",
    "in_progress": r"\bin[- ]?progress\b|\bongoing\b",
    "done": r"\bdone\b|\bcomplete[d]?\b|\bfinished\b",
    "low": r"\blow\b",
    "medium": r"\bmedium\b",
    "high": r"\bhigh\b",
    "urgent": r"\burgent\b",
}


def _last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content") or ""
    return ""


def _drop_unrequested_filters(args: dict, user_text: str) -> dict:
    """Models like to add status='todo' on their own. Keep a list/search filter only
    if the user's message actually names it, so "list all my tasks" is never narrowed."""
    text = user_text.lower()
    return {k: v for k, v in args.items()
            if k not in ("status", "priority") or re.search(_FILTER_WORDS.get(v, "$^"), text)}


def _compact(tasks: list[dict]) -> list[dict]:
    keys = ("id", "title", "status", "priority", "due_date")
    return [{k: t.get(k) for k in keys} for t in tasks]


def _tasks_word(n: int) -> str:
    return f"{n} task" if n == 1 else f"{n} tasks"


def _numbered(tasks: list[dict], limit: int = 5) -> str:
    lines = [f"{i}. '{t['title']}' ({t['status']})" for i, t in enumerate(tasks[:limit], 1)]
    if len(tasks) > limit:
        lines.append(f"...and {len(tasks) - limit} more. Please be more specific.")
    return "\n".join(lines)


def _render_tasks(tasks: list[dict], header: str) -> str:
    """Plain-text list grouped by status. Written in code so nothing gets left out."""
    shown = tasks[:LIST_LIMIT]
    lines = [header]
    for status in STATUSES:
        group = [t for t in shown if t["status"] == status]
        if not group:
            continue
        lines.append(f"\n{STATUS_LABEL[status]}:")
        for t in group:
            due = f", due {t['due_date']}" if t.get("due_date") else ""
            lines.append(f"- {t['title']} ({t['priority']}{due})")
    if len(tasks) > len(shown):
        lines.append(f"\n...and {len(tasks) - len(shown)} more.")
    return "\n".join(lines)


def _is_duplicate_title_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("unique" in msg and "title" in msg) or "uq_tasks_title_active" in msg


def _task_summary(t: dict) -> str:
    due = f", due {t['due_date']}" if t.get("due_date") else ""
    return f"'{t['title']}' ({t['status']}, {t['priority']}{due})"


def _finish(result: dict, done: str, failed: str) -> dict:
    """Index the saved task and write the reply from the values the db actually returned."""
    if result.get("ok") and result.get("task"):
        vector_store.upsert(result["task"])
        result["reply"] = f"{done} {_task_summary(result['task'])}."
    else:
        result["reply"] = f"{failed}: {result.get('message') or result.get('error') or 'unknown error'}."
    return result


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
# Every result carries a "reply" that is sent to the user as is. The LLM only picks the tool.

def _exec_create(args: dict) -> dict:
    title = args["title"].strip()
    try:
        result = db.create_task(
            title=title,
            description=args.get("description") or None,
            status=args.get("status") or "todo",
            priority=args.get("priority") or "medium",
            due_date=args.get("due_date") or None,
        )
    except sqlite3.IntegrityError as e:
        if _is_duplicate_title_error(e):
            return {"ok": False, "error": "DUPLICATE_TITLE",
                    "reply": f"There's already an active task called '{title}'. "
                             f"What should the new one be called?"}
        raise
    return _finish(result, "Created", "Could not create the task")


def _exec_list(args: dict) -> dict:
    status, priority = args.get("status"), args.get("priority")
    tasks = db.list_tasks(status=status, priority=priority)

    if status or priority:
        total = len(db.list_tasks())
        scope = " and ".join(x for x in (f"status {status}" if status else "",
                                         f"priority {priority}" if priority else "") if x)
        if tasks:
            text = _render_tasks(
                tasks, f"Showing {len(tasks)} of your {_tasks_word(total)}, only those with {scope}:")
        else:
            text = f"No tasks match {scope}. You have {_tasks_word(total)} in total."
    elif tasks:
        text = _render_tasks(tasks, f"You have {_tasks_word(len(tasks))}:")
    else:
        text = "You have no tasks yet."

    return {"ok": True, "count": len(tasks), "filters": {"status": status, "priority": priority},
            "tasks": _compact(tasks), "reply": text}


def _exec_find(args: dict) -> dict:
    query, status = args["query"].strip(), args.get("status")
    tasks = _search(query, (status,) if status else None)
    scope = f" with status {status}" if status else ""
    if tasks:
        text = _render_tasks(tasks, f"Found {_tasks_word(len(tasks))} matching '{query}'{scope}:")
    else:
        text = f"No tasks match '{query}'{scope}."
    return {"ok": True, "count": len(tasks), "tasks": _compact(tasks), "reply": text}


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
                "reply": "Which one do you mean?\n" + _numbered(tasks)}

    try:
        result = db.update_task(tasks[0]["id"], **fields)
    except sqlite3.IntegrityError as e:
        if _is_duplicate_title_error(e):
            name = fields.get("title") or tasks[0]["title"]
            return {"ok": False, "error": "DUPLICATE_TITLE",
                    "reply": f"There's already an active task called '{name}'. "
                             f"Pick a different title, or finish that one first."}
        raise
    return _finish(result, "Updated", "Could not update the task")


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
        found = _search(query) if bulk else _resolve(query)
        tasks = [t for t in found
                 if (not status or t["status"] == status)
                 and (not priority or t["priority"] == priority)]
    else:
        tasks = db.list_tasks(status=status, priority=priority)

    if not tasks:
        return {"ok": False, "error": "NOT_FOUND", "reply": "I couldn't find any matching tasks."}
    if len(tasks) > 1 and not bulk:
        return {"ok": False, "error": "AMBIGUOUS",
                "reply": "Which one do you mean?\n" + _numbered(tasks)}

    _pending_delete = [{"id": t["id"], "title": t["title"]} for t in tasks]
    if len(tasks) == 1:
        title = tasks[0]["title"]
        text = f"Delete '{title}'? Reply 'yes' to confirm."
    else:
        text = f"Delete these {len(tasks)} tasks?\n{_numbered(tasks, limit=len(tasks))}\nReply 'yes' to confirm."
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
    last = state["messages"][-1]
    tool_log = list(state["tool_log"])
    replies: list[str] = []

    for tc in last.get("tool_calls") or []:
        name = tc["function"]["name"]
        try:
            args = json.loads(tc["function"]["arguments"] or "{}")
        except json.JSONDecodeError:
            args = {}

        if name in ("list_tasks", "find_tasks"):
            args = _drop_unrequested_filters(args, _last_user_text(state["messages"]))

        fn = _TOOL_MAP.get(name)
        problem = _validate(name, args)
        if fn is None:
            result = {"ok": False, "error": "UNKNOWN_TOOL", "reply": "Sorry, I can't do that."}
        elif problem:
            result = {"ok": False, "error": "INVALID_INPUT", "message": problem,
                      "reply": f"I couldn't do that: {problem}."}
        else:
            try:
                result = fn(args)
            except Exception as e:  # noqa: BLE001
                log.exception("tool %s failed", name)
                result = {"ok": False, "error": "TOOL_ERROR", "message": str(e),
                          "reply": "Sorry, that action failed. Please try again."}

        replies.append(result["reply"])
        tool_log.append({"name": name, "args": args, "result": result, "tool_call_id": tc["id"]})

    reply = "\n".join(replies)
    # The model may add only its out-of-scope note for mixed requests. Any other text is dropped.
    note = (last.get("content") or "").strip()
    if note.lower().startswith("ignored"):
        reply += "\n" + note

    return {"messages": state["messages"] + [{"role": "assistant", "content": reply}],
            "tool_log": tool_log}


def _after_agent(state: State) -> str:
    return "tools" if state["messages"][-1].get("tool_calls") else END


_graph = StateGraph(State)
_graph.add_node("agent", call_model)
_graph.add_node("tools", call_tools)
_graph.set_entry_point("agent")
_graph.add_conditional_edges("agent", _after_agent, {"tools": "tools", END: END})
_graph.add_edge("tools", END)
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

    try:
        final = app_graph.invoke({"messages": messages, "tool_log": []})
    except Exception:  # noqa: BLE001
        log.exception("chat request failed")
        return {"reply": "Sorry, something went wrong on my side. Please try again.",
                "tool_log": [], "last_turn": None}
    return {
        "reply": final["messages"][-1].get("content") or "",
        "tool_log": final["tool_log"],
        "last_turn": None,
    }