const state = {
  sessionId: null,
  ready: false,
  polling: null,
};

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
const evidenceList = document.getElementById("evidenceList");
const evidenceCount = document.getElementById("evidenceCount");
const newSessionBtn = document.getElementById("newSessionBtn");

const backendSelect = document.getElementById("backendSelect");
const modelIdInput = document.getElementById("modelIdInput");
const apiBaseInput = document.getElementById("apiBaseInput");
const apiEnvInput = document.getElementById("apiEnvInput");
const apiKeyInput = document.getElementById("apiKeyInput");
const thinkingInput = document.getElementById("thinkingInput");
const evidenceModeSelect = document.getElementById("evidenceModeSelect");
const evidenceFramesInput = document.getElementById("evidenceFramesInput");
const sampleFpsInput = document.getElementById("sampleFpsInput");

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
  const source = String(value ?? "").trim();
  if (!source) return "";

  const codeBlocks = [];
  const withoutCode = source.replace(/```(\w+)?\n?([\s\S]*?)```/g, (_, lang, code) => {
    const token = `\u0000CODE${codeBlocks.length}\u0000`;
    codeBlocks.push(
      `<pre><code${lang ? ` class="language-${escapeHtml(lang)}"` : ""}>${escapeHtml(code.trim())}</code></pre>`,
    );
    return token;
  });

  const lines = withoutCode.split(/\r?\n/);
  const html = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!trimmed) {
      i += 1;
      continue;
    }

    const codeMatch = trimmed.match(/^\u0000CODE(\d+)\u0000$/);
    if (codeMatch) {
      html.push(codeBlocks[Number(codeMatch[1])] || "");
      i += 1;
      continue;
    }

    const headingMatch = trimmed.match(/^(#{1,3})\s+(.+)$/);
    if (headingMatch) {
      const level = headingMatch[1].length + 2;
      html.push(`<h${level}>${renderInlineMarkdown(headingMatch[2])}</h${level}>`);
      i += 1;
      continue;
    }

    if (/^[-*]\s+/.test(trimmed)) {
      const items = [];
      while (i < lines.length && /^[-*]\s+/.test(lines[i].trim())) {
        items.push(`<li>${renderInlineMarkdown(lines[i].trim().replace(/^[-*]\s+/, ""))}</li>`);
        i += 1;
      }
      html.push(`<ul>${items.join("")}</ul>`);
      continue;
    }

    if (/^\d+\.\s+/.test(trimmed)) {
      const items = [];
      while (i < lines.length && /^\d+\.\s+/.test(lines[i].trim())) {
        items.push(`<li>${renderInlineMarkdown(lines[i].trim().replace(/^\d+\.\s+/, ""))}</li>`);
        i += 1;
      }
      html.push(`<ol>${items.join("")}</ol>`);
      continue;
    }

    const paragraph = [];
    while (
      i < lines.length &&
      lines[i].trim() &&
      !/^\u0000CODE\d+\u0000$/.test(lines[i].trim()) &&
      !/^(#{1,3})\s+/.test(lines[i].trim()) &&
      !/^[-*]\s+/.test(lines[i].trim()) &&
      !/^\d+\.\s+/.test(lines[i].trim())
    ) {
      paragraph.push(renderInlineMarkdown(lines[i].trim()));
      i += 1;
    }
    html.push(`<p>${paragraph.join("<br>")}</p>`);
  }

  return html.join("");
}

function setMessageContent(row, text, { markdown = false, meta = "" } = {}) {
  const bubble = row.querySelector(".bubble");
  if (markdown) {
    bubble.innerHTML = renderMarkdown(text);
  } else {
    bubble.textContent = text;
  }
  const metaNode = row.querySelector(".message-meta");
  if (metaNode) {
    metaNode.textContent = meta;
    metaNode.hidden = !meta;
  }
}

function appendMessage(role, text, options = {}) {
  const row = document.createElement("div");
  row.className = `message ${role}`;
  const stack = document.createElement("div");
  stack.className = "message-stack";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  stack.appendChild(bubble);
  const meta = document.createElement("div");
  meta.className = "message-meta";
  meta.hidden = true;
  stack.appendChild(meta);
  row.appendChild(stack);
  messages.appendChild(row);
  setMessageContent(row, text, options);
  messages.scrollTop = messages.scrollHeight;
  return row;
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

function clearEvidence() {
  evidenceCount.textContent = "0 frames";
  evidenceList.innerHTML = '<p class="empty-state">Evidence frames will appear here after each answer.</p>';
}

function renderEvidence(items) {
  const evidence = Array.isArray(items) ? items : [];
  evidenceCount.textContent = `${evidence.length} frame${evidence.length === 1 ? "" : "s"}`;
  evidenceList.innerHTML = "";
  if (!evidence.length) {
    clearEvidence();
    return;
  }
  for (const item of evidence) {
    const card = document.createElement("article");
    card.className = "evidence-card";

    const image = document.createElement("img");
    image.src = item.url;
    image.alt = `Evidence frame at ${Number(item.timestamp).toFixed(2)} seconds`;
    card.appendChild(image);

    const meta = document.createElement("div");
    meta.className = "evidence-meta";
    const source = item.source === "uniform" ? "uniform" : "HM-VQA";
    const score = item.score === null || item.score === undefined ? "" : ` · score ${item.score}`;
    meta.innerHTML = `<strong>${Number(item.timestamp).toFixed(2)}s</strong>Frame ${item.rank} · ${source}${score}`;
    card.appendChild(meta);

    evidenceList.appendChild(card);
  }
}

async function uploadVideo(file) {
  if (!file) return;
  const sampleFps = Math.max(0.25, Math.min(4, Number.parseFloat(sampleFpsInput.value || "1")));
  sampleFpsInput.value = String(sampleFps);
  resetForNewVideo(false);
  appendMessage("system", `Uploading ${file.name} at ${sampleFps} FPS`);
  sessionStatus.textContent = "Uploading video.";
  progressPanel.classList.remove("hidden");
  setReady(false);

  const form = new FormData();
  form.append("video", file);
  form.append("sample_fps", String(sampleFps));
  const response = await fetch("/api/sessions", {
    method: "POST",
    body: form,
  });
  if (!response.ok) {
    const text = await response.text();
    appendMessage("system", `Upload failed: ${text}`);
    return;
  }
  const payload = await response.json();
  state.sessionId = payload.session_id;
  dropZone.classList.add("hidden");
  pollProgress();
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
      appendMessage("system", "Ingestion complete. Ask a question about this video.");
      return;
    }
    if (payload.status === "error") {
      setReady(false);
      appendMessage("system", `Ingestion failed: ${payload.error || payload.message}`);
      return;
    }
  } catch (error) {
    sessionStatus.textContent = `Progress check failed: ${error.message}`;
  }
  state.polling = window.setTimeout(pollProgress, 1000);
}

function modelPayload(question) {
  const evidenceFrames = Math.max(1, Math.min(64, Number.parseInt(evidenceFramesInput.value || "16", 10)));
  evidenceFramesInput.value = String(evidenceFrames);
  return {
    question,
    backend: backendSelect.value,
    model_id: modelIdInput.value.trim(),
    api_base_url: apiBaseInput.value.trim(),
    api_key_env_var: apiEnvInput.value.trim() || "HMVQA_DEMO_API_KEY",
    api_key: apiKeyInput.value.trim() || null,
    enable_thinking: thinkingInput.checked,
    retrieval_mode: evidenceModeSelect.value,
    evidence_frames: evidenceFrames,
    max_new_tokens: 384,
  };
}

async function askQuestion(question) {
  if (!state.sessionId || !state.ready) return;
  appendMessage("user", question);
  questionInput.value = "";
  questionInput.style.height = "auto";
  setReady(false);
  const modeText = evidenceModeSelect.value === "uniform" ? "Sampling uniform frames" : "Retrieving HM-VQA evidence";
  const pending = appendMessage("assistant", `${modeText} and generating answer...`);
  try {
    const response = await fetch(`/api/sessions/${state.sessionId}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(modelPayload(question)),
    });
    if (!response.ok) throw new Error(await response.text());
    const payload = await response.json();
    const mode = payload.retrieval?.mode === "uniform" ? "uniform" : "HM-VQA";
    const meta = payload.generation_sec
      ? `${mode} · ${payload.evidence?.length ?? 0} frames · generated in ${Number(payload.generation_sec).toFixed(2)}s`
      : `${mode} · ${payload.evidence?.length ?? 0} frames`;
    setMessageContent(pending, payload.answer, { markdown: true, meta });
    renderEvidence(payload.evidence);
  } catch (error) {
    setMessageContent(pending, `Request failed: ${error.message}`);
  } finally {
    setReady(true);
  }
}

function resetForNewVideo(showDrop = true) {
  window.clearTimeout(state.polling);
  state.sessionId = null;
  state.ready = false;
  messages.innerHTML = "";
  clearEvidence();
  setReady(false);
  progressPanel.classList.add("hidden");
  progressBar.style.width = "0%";
  progressValue.textContent = "0%";
  progressMessage.textContent = "";
  sessionStatus.textContent = "Upload a video to start a session.";
  if (showDrop) {
    dropZone.classList.remove("hidden");
  }
}

chooseVideoBtn.addEventListener("click", () => videoInput.click());
attachBtn.addEventListener("click", () => videoInput.click());
newSessionBtn.addEventListener("click", () => resetForNewVideo(true));

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
