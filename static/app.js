const messagesEl = document.getElementById("messages");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");

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

    } catch (err) {
        thinking.textContent = `⚠️ ${err.message}`;
    } finally {
        sendBtn.disabled = false;
        inputEl.focus();
    }
});

inputEl.focus();
