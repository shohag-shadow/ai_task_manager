const messagesEl = document.getElementById("messages");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const tasksEl = document.getElementById("tasks");
const refreshBtn = document.getElementById("refresh-btn");

let history = [];

function renderMarkdown(text) {
    if (typeof text !== "string") return "";

    // escape HTML first so nothing the model emits can inject tags
    const escaped = text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");

    // then apply a small subset of markdown
    return escaped
        .replace(/`([^`]+)`/g, '<code class="bg-slate-200 px-1 rounded">$1</code>')
        .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
        .replace(/\*([^*]+)\*/g, "<i>$1</i>")
        .replace(/^### (.+)$/gm, '<div class="font-semibold">$1</div>')
        .replace(/^## (.+)$/gm, '<div class="font-semibold text-base">$1</div>')
        .replace(/^# (.+)$/gm, '<div class="font-bold text-lg">$1</div>')
        .replace(/^\s*[-*] (.+)$/gm, "• $1")
        .replace(/\n/g, "<br>");
}

function addBubble(role, text) {
    const div = document.createElement("div");
    div.className = role === "user" ? "flex justify-end" : "flex justify-start";

    const inner = document.createElement("div");
    inner.className = role === "user"
        ? "bg-indigo-600 text-white rounded-2xl rounded-br-sm px-3 py-2 max-w-[85%] whitespace-pre-wrap"
        : "bg-slate-100 text-slate-800 rounded-2xl rounded-bl-sm px-3 py-2 max-w-[85%] break-words";

    if (role === "user") {
        inner.textContent = text;
    } else {
        inner.innerHTML = renderMarkdown(text);
    }

    div.appendChild(inner);
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return inner;
}

function addThinking() {
    return addBubble("assistant", "Thinking…");
}

function extractReply(data) {
    const r = data?.reply;
    if (typeof r === "string") return r;
    if (r && typeof r === "object" && typeof r.content === "string") return r.content;
    return "";
}

formEl.addEventListener("submit", async (e) => {
    e.preventDefault();
    const message = inputEl.value.trim();
    if (!message) return;

    addBubble("user", message);
    history.push({ role: "user", content: message });
    inputEl.value = "";
    sendBtn.disabled = true;

    const thinking = addThinking();

    try {
        const res = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message, history: history.slice(0, -1) }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }
        const data = await res.json();
        const reply = extractReply(data) || "(no reply)";

        thinking.innerHTML = renderMarkdown(reply);

        history.push({ role: "assistant", content: reply });
        await loadTasks();

    } catch (err) {
        thinking.textContent = `⚠️ ${err.message}`;
    } finally {
        sendBtn.disabled = false;
        inputEl.focus();
    }
});

const STATUS_LABELS = {
    todo: "To do",
    in_progress: "In progress",
    done: "Done",
};
const PRIORITY_STYLES = {
    low: "bg-slate-100 text-slate-600",
    medium: "bg-blue-100 text-blue-700",
    high: "bg-orange-100 text-orange-700",
    urgent: "bg-red-100 text-red-700",
};

async function loadTasks() {
    try {
        const res = await fetch("/api/tasks");
        const data = await res.json();
        renderTasks(data.tasks || []);
    } catch (e) {
        tasksEl.innerHTML = `<p class="text-sm text-red-500">Failed to load tasks.</p>`;
    }
}

function renderTasks(tasks) {
    if (!tasks.length) {
        tasksEl.innerHTML = `<p class="text-sm text-slate-400 text-center mt-6">No tasks yet. Try: “Create a task to finish the demo tomorrow.”</p>`;
        return;
    }
    const groups = { todo: [], in_progress: [], done: [] };
    tasks.forEach(t => { (groups[t.status] || groups.todo).push(t); });

    tasksEl.innerHTML = "";
    for (const status of ["todo", "in_progress", "done"]) {
        const items = groups[status];
        if (!items.length) continue;

        const header = document.createElement("h3");
        header.className = "text-xs font-semibold text-slate-500 uppercase mt-2 mb-1";
        header.textContent = `${STATUS_LABELS[status]} (${items.length})`;
        tasksEl.appendChild(header);

        items.forEach(t => tasksEl.appendChild(taskCard(t)));
    }
}

function taskCard(t) {
    const card = document.createElement("div");
    card.className = "border rounded-lg p-3 hover:shadow-sm transition bg-white";

    const titleRow = document.createElement("div");
    titleRow.className = "flex items-start justify-between gap-2";

    const title = document.createElement("div");
    title.className = "font-medium text-sm " + (t.status === "done" ? "line-through text-slate-400" : "");
    title.textContent = t.title;

    const prio = document.createElement("span");
    prio.className = "text-[10px] font-semibold px-2 py-0.5 rounded-full " + (PRIORITY_STYLES[t.priority] || "");
    prio.textContent = t.priority;

    titleRow.appendChild(title);
    titleRow.appendChild(prio);
    card.appendChild(titleRow);

    if (t.description) {
        const desc = document.createElement("p");
        desc.className = "text-xs text-slate-500 mt-1";
        desc.textContent = t.description;
        card.appendChild(desc);
    }

    if (t.due_date) {
        const due = document.createElement("p");
        due.className = "text-xs text-slate-500 mt-1";
        due.textContent = `📅 Due ${t.due_date}`;
        card.appendChild(due);
    }

    return card;
}

loadTasks();
inputEl.focus();
refreshBtn.addEventListener("click", loadTasks);
