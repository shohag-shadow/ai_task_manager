import json
import os
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

SYSTEM_PROMPT = """You are a task manager assistant. You control tasks ONLY through the provided tools.

OPERATING RULES: follow strictly
0. **MOST IMPORTANT** SCOPE: You only do two things: say hello, and manage tasks using the PROVIDED TOOLS.
   - You are a tool, not a chatbot. Do not chat, give opinions, respond to feelings, or answer general questions.
   - Do not do math, write text or code, give advice, or help with anything unrelated to tasks, even if the request is small or the user insists.
   - If the user greets you, greet them back briefly and ask how you can help with their tasks.
   - For anything else, decline politely and humbly in one or two short sentences, then offer task help. Use wording like:
     "Sorry, I can't help with that. I'm a task manager assistant and can only help with your tasks, like adding, listing, updating, or deleting them. Is there a task I can help you with?"
   - If a message mixes task work with something else, do the task part and politely decline the rest. Never answer the out-of-scope part. When you reject a part of the work, tell the user it was out of your scope. Use wording like:
     "Ignored the 'x' part of your request as it is out of my scope"

1. CREATE: For "create/add/remind me" requests call create_task with a clear title.
   Parse relative dates ("tomorrow", "next Monday", "in 3 days") into YYYY-MM-DD.
   If due date or priority is not mentioned, omit it (defaults apply).
   - Before creating a task, call find_tasks to check whether one already exists for the same thing.
     Search ACTIVE tasks only (status 'todo' or 'in_progress'). Completed tasks ('done') never count as a duplicate.
   - If one or more active tasks match, show them to the user and ask for confirmation to create the new task with a different title.
   - Always add time in the title as the db dont have a time field. For example "Meeting at 9 PM" make sure TIME is in the title not DATE.

2. FIND BEFORE MUTATE: Before update_task or delete_task, you MUST first call find_tasks(query)
   with keywords from the user's request.
   - For update_task, search ACTIVE tasks only (status 'todo' or 'in_progress'). Exception: if the user clearly
     refers to a completed task (for example "reopen X"), include 'done' tasks in the search.
   - For delete_task, search tasks of every status.
   - If find_tasks cannot exclude completed tasks by itself, discard any result with status 'done' before counting matches.
   - 0 results -> tell the user nothing matched and stop.
   - 1 result -> proceed using that task_id.
   - 2+ results -> DO NOT mutate. List the candidates (title + status only) and ask
     "Which one do you mean?".

3. DELETE CONFIRMATION:
   - Single task: after rule 2, you must ask "Delete '<title>'? Reply 'yes' to confirm." first.
     Only call delete_task after the user's most recent message is an explicit confirmation
     (yes / confirm / sure / do it). Never call delete_task on the first mention.
   - Multiple tasks ("delete all finished tasks", "delete all low priority tasks", "delete everything in progress"):
     a. Find them with ONE list_tasks call using the matching filter. "finished/completed/done" -> status 'done'.
        If the request is by keyword instead, use find_tasks. Bulk requests skip the "Which one do you mean?" step.
     b. 0 results -> tell the user nothing matched and stop.
     c. Otherwise show a numbered list of EVERY task that will be deleted (title + status) and ask once:
        "Delete these N tasks? Reply 'yes' to confirm." Do NOT call delete_task yet.
     d. Only after the user's most recent message is an explicit confirmation, call delete_task once for each
        task in that list, using the task_ids already returned. If those ids are no longer available, repeat the
        same list_tasks call and delete only the tasks that match the confirmed list.
     e. Delete only the tasks that were shown, nothing more. If the user says no or changes the request, delete nothing.

4. LISTS & SEARCH: For "show/list" requests use list_tasks (with status/priority filters if
   stated). For descriptive queries like "tasks related to X" use find_tasks or list_tasks
   with a filter.

5. STATUS VALUES: 'todo', 'in_progress', 'done'. PRIORITY: 'low','medium','high','urgent'.
   "In progress" -> in_progress. "Done"/"complete" -> done.

6. CONFIRMATION GROUNDING: non-negotiable:
   - When confirming a tool result, quote the EXACT values returned by the tool
     (title, status, priority, due_date). Do NOT paraphrase the user's words as if they
     were the result.
   - If create_task returns title "Today's Assignment", reply: Created "Today's Assignment".
     Never rewrite it as "complete today's assignment" or any other phrasing.
   - After a bulk delete, list the exact titles that were deleted. If any delete returned ok=false
     or an error, name those tasks and the error.
   - If the tool returned ok=false or an error, say what the error was. Never pretend the
     action succeeded.
   - If you did not call a tool this turn, do not claim any change occurred.

7. Keep replies short and friendly. Never invent task IDs or task details.

Be deterministic. Follow the rules exactly."""

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "create_task",
            "description": "Create a new task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "status": {"type": "string", "enum": ["todo", "in_progress", "done"]},
                    "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
                    "due_date": {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_tasks",
            "description": "List all tasks, optionally filtered by status or priority.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["todo", "in_progress", "done"]},
                    "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_tasks",
            "description": "Search tasks by name/keyword (semantic + keyword). Use before update/delete.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_task",
            "description": "Update fields on ONE task by its id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "status": {"type": "string", "enum": ["todo", "in_progress", "done"]},
                    "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
                    "due_date": {"type": "string"},
                },
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_task",
            "description": "Delete ONE task by id. Only call after user explicitly confirms.",
            "parameters": {
                "type": "object",
                "properties": {"task_id": {"type": "string"}},
                "required": ["task_id"],
            },
        },
    },
]


class State(TypedDict):
    messages: list
    tool_log: list[dict]


def call_model(state: State) -> State:
    resp = _client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {"role": "system", "content": f"Today is {date.today().isoformat()}"},
            *state["messages"],
        ],
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.1,
        max_tokens=1024,
    )
    msg = resp.choices[0].message

    out: dict = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]

    return {"messages": state["messages"] + [out]}


def _compact(tasks: list[dict], *, include_description: bool = True) -> list[dict]:
    out = []
    for t in tasks:
        row = {"id": t["id"], "title": t["title"], "status": t["status"],
            "priority": t["priority"], "due_date": t.get("due_date")}
        if include_description:
            row["description"] = (t.get("description") or "")[:120] or None
        out.append(row)
    return out


def _exec_create(args: dict) -> dict:
    result = db.create_task(
        title=args["title"],
        description=args.get("description"),
        status=args.get("status", "todo"),
        priority=args.get("priority", "medium"),
        due_date=args.get("due_date"),
    )
    if result["ok"]:
        vector_store.upsert(result["task"])
    return result


def _exec_list(args: dict) -> dict:
    tasks = db.list_tasks(status=args.get("status"), priority=args.get("priority"))
    return {"ok": True, "count": len(tasks), "tasks": _compact(tasks)}


def _exec_find(args: dict) -> dict:
    query = args["query"]
    vec = vector_store.search(query, n=10)
    vec_ids = [tid for tid, _ in vec]
    kw = db.keyword_search(query, limit=10)
    kw_ids = [t["id"] for t in kw]

    seen = set()
    ids: list[str] = []
    for tid in vec_ids + kw_ids:
        if tid not in seen:
            seen.add(tid)
            ids.append(tid)

    tasks = db.hydrate(ids)[:5]
    return {"ok": True, "count": len(tasks), "tasks": _compact(tasks, include_description=False)}


def _exec_update(args: dict) -> dict:
    args = dict(args)
    task_id = args.pop("task_id")
    result = db.update_task(task_id, **args)
    if result["ok"]:
        vector_store.upsert(result["task"])
    return result


def _exec_delete(args: dict) -> dict:
    result = db.delete_task(args["task_id"])
    if result["ok"]:
        vector_store.remove(args["task_id"])
    return result


_TOOL_MAP = {
    "create_task": _exec_create,
    "list_tasks": _exec_list,
    "find_tasks": _exec_find,
    "update_task": _exec_update,
    "delete_task": _exec_delete,
}


def call_tools(state: State) -> State:
    last = state["messages"][-1]
    log = list(state.get("tool_log", []))
    tool_msgs: list[dict] = []

    for tc in (last.get("tool_calls") or []):
        fn_info = tc.get("function", {})
        name = fn_info.get("name")
        try:
            args = json.loads(fn_info.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}

        fn = _TOOL_MAP.get(name)
        if fn is None:
            result = {"ok": False, "error": "UNKNOWN_TOOL",
                    "message": f"Unknown tool: {name}"}
        else:
            try:
                result = fn(args)
            except Exception as e:  # noqa: BLE001
                result = {"ok": False, "error": "TOOL_ERROR", "message": str(e)}

        log.append({"name": name, "args": args, "result": result})
        tool_msgs.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": json.dumps(result, default=str),
        })

    return {"messages": state["messages"] + tool_msgs, "tool_log": log}


def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if last.get("tool_calls") else END


_graph = StateGraph(State)
_graph.add_node("agent", call_model)
_graph.add_node("tools", call_tools)
_graph.set_entry_point("agent")
_graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
_graph.add_edge("tools", "agent")
app_graph = _graph.compile()


def _ground_reply(reply: str, tool_log: list[dict]) -> str:
    """Append the saved task details when the reply leaves the title out."""
    for entry in tool_log:
        if entry.get("name") not in ("create_task", "update_task"):
            continue
        r = entry.get("result") or {}
        task = r.get("task")
        if not (r.get("ok") and task):
            continue
        title = task["title"]
        if title.lower() not in reply.lower():
            due = f", due {task['due_date']}" if task.get("due_date") else ""
            return (reply.rstrip() +
                    f"\n\n_(Saved as “{title}” · {task['status']} · "
                    f"{task['priority']}{due}.)_")
    return reply


def _extract_reply(last) -> str:
    """Return the text of the final assistant message."""
    if isinstance(last, dict):
        return last.get("content") or ""
    return getattr(last, "content", "") or ""

MAX_TURNS = 5

def _trim(history: list[dict]) -> list[dict]:
    return history[-MAX_TURNS * 2:]

def run_chat(message: str, history: list[dict]) -> dict:
    init_messages = [{"role": m["role"], "content": m["content"]} for m in _trim(history)]
    init_messages.append({"role": "user", "content": message})

    final = app_graph.invoke({"messages": init_messages, "tool_log": []})
    last = final["messages"][-1]

    reply = _extract_reply(last)
    tool_log = final.get("tool_log", [])
    reply = _ground_reply(reply, tool_log)

    return {"reply": reply or "", "tool_log": tool_log}
