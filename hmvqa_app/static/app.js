const state = { sessionId: null, ready: false, polling: null };

const videoInput = document.getElementById("videoInput");
const chooseVideoBtn = document.getElementById("chooseVideoBtn");
const attachBtn = document.getElementById("attachBtn");
const dropZone = document.getElementById("dropZone");
const progressPanel = document.getElementById("progressPanel");
const progressBar = document.getElementById("progressBar");
const progressValue = document.getElementById("progressValue");
const progressMessage = document.getElementById("progressMessage");
const progressLabel = document.getElementById("progressLabel");
const sessionStatus = document.getElementById("sessionStatus");
const messages = document.getElementById("messages");
const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");
const sendBtn = document.getElementById("sendBtn");
const newSessionBtn = document.getElementById("newSessionBtn");
const clearSessionBtn = document.getElementById("clearSessionBtn");
const clearCacheBtn = document.getElementById("clearCacheBtn");
const refreshSessionsBtn = document.getElementById("refreshSessionsBtn");
const sessionList = document.getElementById("sessionList");
const imageViewer = document.getElementById("imageViewer");
const viewerImage = document.getElementById("viewerImage");
const viewerCaption = document.getElementById("viewerCaption");
const viewerCloseBtn = document.getElementById("viewerCloseBtn");
const viewerPrevBtn = document.getElementById("viewerPrevBtn");
const viewerNextBtn = document.getElementById("viewerNextBtn");

const modeSelect = document.getElementById("modeSelect");
const modelIdInput = document.getElementById("modelIdInput");
const thinkingInput = document.getElementById("thinkingInput");
const evidenceFramesInput = document.getElementById("evidenceFramesInput");
const sampleFpsInput = document.getElementById("sampleFpsInput");
let viewerItems = [];
let viewerIndex = 0;

function setReady(ready) {
  state.ready = ready;
  questionInput.disabled = !ready;
  sendBtn.disabled = !ready;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderInlineMarkdown(value) {
  return escapeHtml(value)
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>')
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>");
}

function renderMarkdown(value) {
  const lines = String(value ?? "").trim().split(/\r?\n/);
  const html = [];
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const heading = trimmed.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      const level = heading[1].length + 2;
      html.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
    } else if (/^[-*]\s+/.test(trimmed)) {
      html.push(`<p class="bullet">${renderInlineMarkdown(trimmed.replace(/^[-*]\s+/, ""))}</p>`);
    } else {
      html.push(`<p>${renderInlineMarkdown(trimmed)}</p>`);
    }
  }
  return html.join("");
}

function appendMessage(role, text, options = {}) {
  const row = document.createElement("div");
  row.className = `message ${role}`;
  const stack = document.createElement("div");
  stack.className = "message-stack";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  const meta = document.createElement("div");
  meta.className = "message-meta";
  meta.hidden = true;
  stack.appendChild(bubble);
  stack.appendChild(meta);
  row.appendChild(stack);
  messages.appendChild(row);
  setMessageContent(row, text, options);
  messages.scrollTop = messages.scrollHeight;
  return row;
}

function appendStoredMessage(message) {
  const role = message.role === "assistant" ? "assistant" : "user";
  const row = appendMessage(role, message.text || "", {
    markdown: role === "assistant",
    meta: role === "assistant" ? messageMeta(message) : "",
  });
  if (role === "assistant") {
    renderEvidenceInMessage(row, message.evidence);
  }
}

function messageMeta(message) {
  const timing = message.timing || {};
  const evidenceCount = Array.isArray(message.evidence) ? message.evidence.length : 0;
  const retrieval = Number(timing.retrieval_sec || 0).toFixed(2);
  return `${message.mode || "answer"} · ${evidenceCount} frames · retrieval ${retrieval}s`;
}

function setMessageContent(row, text, { markdown = false, meta = "" } = {}) {
  const bubble = row.querySelector(".bubble");
  bubble.innerHTML = markdown ? renderMarkdown(text) : escapeHtml(text);
  const metaNode = row.querySelector(".message-meta");
  if (metaNode) {
    metaNode.textContent = meta;
    metaNode.hidden = !meta;
  }
}

function renderEvidenceInMessage(row, evidence) {
  const stack = row.querySelector(".message-stack");
  const old = stack.querySelector(".evidence-strip");
  if (old) old.remove();
  const items = Array.isArray(evidence) ? evidence : [];
  if (!items.length) return;
  const strip = document.createElement("div");
  strip.className = "evidence-strip";
  items.forEach((item, index) => {
    const card = document.createElement("article");
    card.className = "evidence-card";
    card.tabIndex = 0;
    card.setAttribute("role", "button");
    card.setAttribute("aria-label", `Open evidence frame ${index + 1}`);
    const image = document.createElement("img");
    image.src = item.url;
    image.alt = `Evidence frame at ${Number(item.timestamp).toFixed(2)} seconds`;
    const meta = document.createElement("div");
    meta.className = "evidence-card-meta";
    const score = item.score === null || item.score === undefined ? "" : ` · ${Number(item.score).toFixed(4)}`;
    meta.innerHTML = `<strong>${Number(item.timestamp).toFixed(2)}s</strong><span>${escapeHtml(item.source)} · #${item.rank}${score}</span>`;
    card.appendChild(image);
    card.appendChild(meta);
    card.addEventListener("click", () => openViewer(items, index));
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openViewer(items, index);
      }
    });
    strip.appendChild(card);
  });
  stack.appendChild(strip);
}

function openViewer(items, index) {
  viewerItems = Array.isArray(items) ? items : [];
  if (!viewerItems.length || !imageViewer) return;
  viewerIndex = Math.max(0, Math.min(index, viewerItems.length - 1));
  imageViewer.classList.remove("hidden");
  document.body.classList.add("viewer-open");
  renderViewer();
}

function closeViewer() {
  imageViewer.classList.add("hidden");
  document.body.classList.remove("viewer-open");
}

function moveViewer(delta) {
  if (!viewerItems.length) return;
  viewerIndex = (viewerIndex + delta + viewerItems.length) % viewerItems.length;
  renderViewer();
}

function renderViewer() {
  const item = viewerItems[viewerIndex];
  if (!item) return;
  viewerImage.src = item.url;
  viewerImage.alt = `Evidence frame ${viewerIndex + 1} at ${Number(item.timestamp).toFixed(2)} seconds`;
  const score = item.score === null || item.score === undefined ? "" : ` · score ${Number(item.score).toFixed(4)}`;
  viewerCaption.textContent = `${viewerIndex + 1}/${viewerItems.length} · ${Number(item.timestamp).toFixed(2)}s · ${item.source}${score}`;
}

function setProgress(payload) {
  const value = Number(payload.progress || 0);
  progressPanel.classList.remove("hidden");
  progressBar.style.width = `${Math.max(0, Math.min(100, value))}%`;
  progressValue.textContent = `${value}%`;
  progressMessage.textContent = payload.error || payload.message || "";
  progressLabel.textContent = payload.video_name || "Processing video";
  sessionStatus.textContent = payload.message || "Processing video.";
}

async function uploadVideo(file) {
  if (!file) return;
  const sampleFps = Math.max(0.25, Math.min(4, Number.parseFloat(sampleFpsInput.value || "1")));
  sampleFpsInput.value = String(sampleFps);
  resetForNewVideo(false);
  dropZone.classList.add("hidden");
  setProgress({
    progress: 1,
    message: `Uploading ${file.name}. This can take a moment for large videos.`,
    video_name: file.name,
  });
  appendMessage("system", `Uploading ${file.name} at ${sampleFps} FPS`);
  try {
    const form = new FormData();
    form.append("video", file);
    form.append("sample_fps", String(sampleFps));
    const response = await fetch("/api/videos", { method: "POST", body: form });
    if (!response.ok) {
      appendMessage("system", `Upload failed: ${await response.text()}`);
      dropZone.classList.remove("hidden");
      return;
    }
    const payload = await response.json();
    state.sessionId = payload.session_id;
    refreshSessions();
    pollProgress();
  } catch (error) {
    appendMessage("system", `Upload failed: ${error.message}`);
    dropZone.classList.remove("hidden");
  }
}

async function pollProgress() {
  if (!state.sessionId) return;
  window.clearTimeout(state.polling);
  try {
    const response = await fetch(`/api/sessions/${state.sessionId}/progress`);
    if (!response.ok) throw new Error(await response.text());
    const payload = await response.json();
    setProgress(payload);
    if (payload.status === "ready") {
      setReady(true);
      appendMessage("system", payload.cache_hit ? "Cached memory loaded. Ask a question." : "Ingestion complete. Ask a question.");
      refreshSessions();
      return;
    }
    if (payload.status === "error") {
      appendMessage("system", `Ingestion failed: ${payload.error || payload.message}`);
      return;
    }
  } catch (error) {
    sessionStatus.textContent = `Progress check failed: ${error.message}`;
  }
  state.polling = window.setTimeout(pollProgress, 1000);
}

async function refreshSessions() {
  if (!sessionList) return;
  try {
    const response = await fetch("/api/sessions");
    if (!response.ok) throw new Error(await response.text());
    const payload = await response.json();
    renderSessionList(payload.sessions || []);
  } catch (error) {
    sessionList.innerHTML = `<p class="session-empty">Failed to load sessions.</p>`;
  }
}

function renderSessionList(items) {
  sessionList.innerHTML = "";
  if (!items.length) {
    sessionList.innerHTML = `<p class="session-empty">No video sessions yet.</p>`;
    return;
  }
  for (const item of items) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `session-card${item.session_id === state.sessionId ? " active" : ""}`;
    const duration = item.duration_sec === null || item.duration_sec === undefined ? "" : ` · ${Number(item.duration_sec).toFixed(1)}s`;
    button.innerHTML = `
      <strong>${escapeHtml(item.video_name || item.session_id)}</strong>
      <span>${escapeHtml(item.status || "unknown")}${duration}</span>
      <span>${Number(item.chat_count || 0)} chat messages</span>
    `;
    button.addEventListener("click", () => loadSession(item.session_id));
    sessionList.appendChild(button);
  }
}

async function loadSession(sessionId) {
  window.clearTimeout(state.polling);
  state.sessionId = sessionId;
  messages.innerHTML = "";
  dropZone.classList.add("hidden");
  setReady(false);
  try {
    const progressResponse = await fetch(`/api/sessions/${sessionId}/progress`);
    if (progressResponse.ok) {
      const progressPayload = await progressResponse.json();
      setProgress(progressPayload);
      setReady(progressPayload.status === "ready");
      if (progressPayload.status !== "ready" && progressPayload.status !== "error") {
        pollProgress();
      }
    }
    const historyResponse = await fetch(`/api/sessions/${sessionId}/history`);
    if (historyResponse.ok) {
      const historyPayload = await historyResponse.json();
      for (const message of historyPayload.messages || []) {
        appendStoredMessage(message);
      }
    }
  } catch (error) {
    appendMessage("system", `Failed to load session: ${error.message}`);
  }
  refreshSessions();
}

function modelPayload(question) {
  const evidenceFrames = Math.max(1, Math.min(64, Number.parseInt(evidenceFramesInput.value || "16", 10)));
  evidenceFramesInput.value = String(evidenceFrames);
  return {
    question,
    mode: modeSelect.value,
    evidence_frames: evidenceFrames,
    model_id: modelIdInput.value,
    enable_thinking: thinkingInput.checked,
    max_new_tokens: 384,
  };
}

async function askQuestion(question) {
  if (!state.sessionId || !state.ready) return;
  appendMessage("user", question);
  questionInput.value = "";
  questionInput.style.height = "auto";
  setReady(false);
  const pending = appendMessage("assistant", "Retrieving evidence and generating answer...");
  try {
    const response = await fetch(`/api/sessions/${state.sessionId}/answer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(modelPayload(question)),
    });
    if (!response.ok) throw new Error(await response.text());
    const payload = await response.json();
    setMessageContent(pending, payload.answer_text, { markdown: true, meta: messageMeta({ ...payload, text: payload.answer_text }) });
    renderEvidenceInMessage(pending, payload.evidence);
    refreshSessions();
  } catch (error) {
    setMessageContent(pending, `Request failed: ${error.message}`);
  } finally {
    setReady(true);
    messages.scrollTop = messages.scrollHeight;
  }
}

function resetForNewVideo(showDrop = true) {
  window.clearTimeout(state.polling);
  state.sessionId = null;
  state.ready = false;
  messages.innerHTML = "";
  setReady(false);
  progressPanel.classList.add("hidden");
  progressBar.style.width = "0%";
  progressValue.textContent = "0%";
  progressMessage.textContent = "";
  sessionStatus.textContent = "Upload a video to start a session.";
  if (showDrop) dropZone.classList.remove("hidden");
  refreshSessions();
}

chooseVideoBtn.addEventListener("click", () => videoInput.click());
attachBtn.addEventListener("click", () => videoInput.click());
newSessionBtn.addEventListener("click", () => resetForNewVideo(true));
if (clearSessionBtn) {
  clearSessionBtn.addEventListener("click", async () => {
    if (!state.sessionId) return;
    const sessionId = state.sessionId;
    await fetch(`/api/sessions/${sessionId}`, { method: "DELETE" });
    resetForNewVideo(true);
    appendMessage("system", `Cleared session ${sessionId}.`);
    refreshSessions();
  });
}
if (clearCacheBtn) {
  clearCacheBtn.addEventListener("click", async () => {
    await fetch("/api/cache", { method: "DELETE" });
    resetForNewVideo(true);
    appendMessage("system", "Cleared all HM-VQA app cache.");
    refreshSessions();
  });
}
if (refreshSessionsBtn) {
  refreshSessionsBtn.addEventListener("click", refreshSessions);
}
viewerCloseBtn.addEventListener("click", closeViewer);
viewerPrevBtn.addEventListener("click", () => moveViewer(-1));
viewerNextBtn.addEventListener("click", () => moveViewer(1));
imageViewer.addEventListener("click", (event) => {
  if (event.target === imageViewer) closeViewer();
});
videoInput.addEventListener("change", () => {
  const [file] = videoInput.files;
  uploadVideo(file);
  videoInput.value = "";
});
for (const eventName of ["dragenter", "dragover"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("drag-over");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("drag-over");
  });
}
dropZone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  uploadVideo(file);
});
chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (question) askQuestion(question);
});
questionInput.addEventListener("input", () => {
  questionInput.style.height = "auto";
  questionInput.style.height = `${Math.min(questionInput.scrollHeight, 140)}px`;
});
questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});
document.addEventListener("keydown", (event) => {
  if (imageViewer.classList.contains("hidden")) return;
  if (event.key === "Escape") closeViewer();
  if (event.key === "ArrowLeft") moveViewer(-1);
  if (event.key === "ArrowRight") moveViewer(1);
});
refreshSessions();
