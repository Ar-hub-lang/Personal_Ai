const chatMode = document.querySelector("#chat-mode");
const voiceMode = document.querySelector("#voice-mode");
const chatPanel = document.querySelector("#chat");
const voicePanel = document.querySelector("#voice");
const diagnosticsMode = document.querySelector("#diagnostics-mode");
const diagnosticsPanel = document.querySelector("#diagnostics");
const diagnosticForm = document.querySelector("#diagnostic-form");
const diagnosticPicker = document.querySelector("#diagnostic-check");
const diagnosticPortField = document.querySelector("#diagnostic-port-field");
const diagnosticPort = document.querySelector("#diagnostic-port");
const runDiagnostic = document.querySelector("#run-diagnostic");
const diagnosticStatus = document.querySelector("#diagnostic-status");
const diagnosticResult = document.querySelector("#diagnostic-result");
const messages = document.querySelector("#messages");
const chatForm = document.querySelector("#chat-form");
const promptInput = document.querySelector("#prompt");
const sendButton = document.querySelector("#send");
const chatStatus = document.querySelector("#chat-status");
const pdfFile = document.querySelector("#pdf-file");
const uploadPdf = document.querySelector("#upload-pdf");
const askPdf = document.querySelector("#ask-pdf");
const pdfStatus = document.querySelector("#pdf-status");
const voiceStatus = document.querySelector("#voice-status");
const recordButton = document.querySelector("#record");
const voiceReview = document.querySelector("#voice-review");
const voiceTranscript = document.querySelector("#voice-transcript");
const sendTranscript = document.querySelector("#send-transcript");
const voiceResult = document.querySelector("#voice-result");
const playback = document.querySelector("#playback");
const voiceSampleRate = 16000;
const maxRecordingSeconds = 20;
let recording;
let playbackUrl;
let previousPdfQuestion = "";
let pdfRevision = 0;
let supportProduct = false;
let supportTopics = [];
let conversationRevision = 0;
const modelPicker = document.querySelector("#model");
const modelStatus = document.querySelector("#model-status");
const newConversation = document.querySelector("#new-conversation");
const summaryButton = document.querySelector("#summary-button");
const summaryPanel = document.querySelector("#chat-summary");
const stopAnswer = document.querySelector("#stop-answer");
const pdfScopeControl = document.querySelector("#pdf-scope-control");
const pdfPicker = document.querySelector("#pdf-scope");
const pdfScopeStatus = document.querySelector("#pdf-scope-status");
let activePdf = "";
let activeAnswer;
const knowledgeForm = document.querySelector("#knowledge-form");
const knowledgeStatus = document.querySelector("#knowledge-status");
const approvalHeaders = { "X-Knowledge-Approval": "yes" };

function setStatus(element, text, isError = false) {
  element.textContent = text;
  element.dataset.state = isError ? "error" : "";
}

function selectMode(mode) {
  const showVoice = mode === "voice";
  chatMode.setAttribute("aria-pressed", String(mode === "chat"));
  voiceMode.setAttribute("aria-pressed", String(showVoice));
  diagnosticsMode.setAttribute("aria-pressed", String(mode === "diagnostics"));
  chatPanel.hidden = mode !== "chat";
  voicePanel.hidden = !showVoice;
  diagnosticsPanel.hidden = mode !== "diagnostics";

  if (!showVoice) {
    playback.pause();
    if (recording) {
      recording.discard = true;
      stopRecording(recording);
      setStatus(voiceStatus, "");
    }
  }
}

chatMode.addEventListener("click", () => selectMode("chat"));
voiceMode.addEventListener("click", () => selectMode("voice"));
diagnosticsMode.addEventListener("click", () => {
  selectMode("diagnostics");
  void loadDiagnosticChecks();
});

let diagnosticChecksLoading = false;

async function loadDiagnosticChecks() {
  if (diagnosticChecksLoading || diagnosticPicker.options.length) return;
  diagnosticChecksLoading = true;
  setStatus(diagnosticStatus, "Loading checks…");
  try {
    const response = await fetch("/diagnostics", { cache: "no-store" });
    if (!response.ok) throw new Error("Diagnostic checks could not be loaded.");
    const { checks } = await response.json();
    for (const { id, label } of checks) {
      const option = document.createElement("option");
      option.value = id;
      option.textContent = label;
      diagnosticPicker.append(option);
    }
    diagnosticPicker.disabled = false;
    runDiagnostic.disabled = false;
    setStatus(diagnosticStatus, "");
  } catch (error) {
    setStatus(diagnosticStatus, error.message, true);
  } finally {
    diagnosticChecksLoading = false;
  }
}

diagnosticPicker.addEventListener("change", () => {
  diagnosticPortField.hidden = diagnosticPicker.value !== "ports";
  diagnosticResult.hidden = true;
  setStatus(diagnosticStatus, "");
});

diagnosticForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (runDiagnostic.disabled) return;
  const request = { check: diagnosticPicker.value };
  if (request.check === "ports" && diagnosticPort.value !== "") {
    request.port = diagnosticPort.valueAsNumber;
  }
  runDiagnostic.disabled = true;
  diagnosticPicker.disabled = true;
  diagnosticPort.disabled = true;
  diagnosticResult.hidden = true;
  setStatus(diagnosticStatus, "Running check…");
  try {
    const { label, finding } = await sendRequest("/diagnostics/run", "application/json", JSON.stringify(request));
    const state = document.querySelector("#diagnostic-finding-state");
    document.querySelector("#diagnostic-result-title").textContent = label;
    state.dataset.state = finding.status;
    state.textContent = { ok: "Check complete", attention: "Needs attention", unavailable: "Unavailable" }[finding.status];
    document.querySelector("#diagnostic-summary").textContent = finding.summary;
    const details = document.querySelector("#diagnostic-details");
    details.replaceChildren();
    for (const [name, value] of Object.entries(finding.details)) {
      const term = document.createElement("dt");
      term.textContent = name;
      const description = document.createElement("dd");
      description.textContent = typeof value === "boolean" ? (value ? "Yes" : "No") : String(value);
      details.append(term, description);
    }
    diagnosticResult.hidden = false;
    setStatus(diagnosticStatus, "");
  } catch (error) {
    setStatus(diagnosticStatus, error.message, true);
  } finally {
    runDiagnostic.disabled = false;
    diagnosticPicker.disabled = false;
    diagnosticPort.disabled = false;
  }
});

async function checkConnection() {
  const connection = document.querySelector("#connection");
  try {
    const response = await fetch("/health", { cache: "no-store" });
    if (!response.ok) throw new Error();
    connection.textContent = "Backend connected";
    connection.dataset.state = "ready";
  } catch {
    connection.textContent = "Backend unavailable";
    connection.dataset.state = "";
  }
}

async function sendRequest(url, contentType, body, headers = {}) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": contentType, ...headers },
    body,
  });
  const result = await response.json();
  if (!response.ok) {
    throw new Error(typeof result.detail === "string" ? result.detail : "The request could not be completed.");
  }
  return result;
}

async function streamAnswer(body, conversation, signal) {
  const response = await fetch("/documents/ask/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok || !response.body) throw new Error("The request could not be completed.");
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let pending = "";
  let result;
  let draft;
  const previews = [];
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      pending += value;
      const lines = pending.split("\n");
      pending = lines.pop();
      for (const line of lines) {
        if (!line) continue;
        const event = JSON.parse(line);
        if (event.error) throw new Error(event.error);
        if (conversation !== conversationRevision) continue;
        if (event.status) setStatus(chatStatus, event.status);
        if (event.part) previews.push(appendMessage("Assistant", event.part.reply, event.part.sources));
        if (event.delta) {
          if (!draft) {
            for (const preview of previews.splice(0)) preview.remove();
            draft = appendMessage("Assistant", "");
            previews.push(draft);
          }
          draft.querySelector("p").append(document.createTextNode(event.delta));
          messages.scrollTop = messages.scrollHeight;
        }
        if (event.result) result = event.result;
      }
    }
  } finally {
    for (const preview of previews) preview.remove();
    await reader.cancel();
  }
  if (conversation !== conversationRevision) return null;
  if (!result) throw new Error("IT Support did not return a verified response.");
  return result;
}

function renderSources(container, sources = []) {
  const list = document.createElement("ul");
  list.className = "citations";
  for (const { filename, page, section, url } of sources) {
    const item = document.createElement("li");
    const label = `${filename}, ${section || `page ${page}`}`;
    if (url?.startsWith("https://")) {
      const link = document.createElement("a");
      link.href = url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = label;
      item.append(link);
    } else {
      item.textContent = label;
    }
    list.append(item);
  }
  if (sources.length) container.append(list);
}

function appendMessage(speaker, text, sources) {
  document.querySelector("#welcome").hidden = true;
  const message = document.createElement("div");
  message.className = "message";
  message.dataset.speaker = speaker;
  const name = document.createElement("strong");
  name.textContent = speaker;
  const content = document.createElement("p");
  content.textContent = text;
  message.append(name, content);
  renderSources(message, sources);
  messages.append(message);
  messages.scrollTop = messages.scrollHeight;
  return message;
}

let collectionMode = false;

function showDocument(payload) {
  const document = payload.document;
  collectionMode = Boolean(payload.collection);
  const firstSupportLoad = !supportProduct && payload.support_product;
  supportProduct = Boolean(payload.support_product);
  if (firstSupportLoad) {
    window.document.title = "IT Support";
    window.document.querySelector("#brand-name").textContent = "IT Support";
    window.document.querySelector("#chat-title").textContent = "What IT issue can I help with?";
    window.document.querySelector("#welcome-text").textContent = "Ask about VPN, Wi-Fi, passwords, printers, or software.";
  }
  diagnosticsMode.hidden = !supportProduct;
  newConversation.hidden = !supportProduct;
  summaryButton.hidden = !supportProduct || Boolean(activePdf);
  pdfScopeControl.hidden = !supportProduct;
  window.document.querySelector("#knowledge-list").hidden = !supportProduct;
  window.document.querySelector("#knowledge").hidden = !supportProduct;
  window.document.querySelector("#ask-document-choice").hidden = supportProduct;
  window.document.querySelector("#ask-document-label").textContent = collectionMode ? "Ask IT Support guides" : "Ask this PDF";
  previousPdfQuestion = "";
  pdfRevision += 1;
  askPdf.disabled = !document;
  if (supportProduct || !document) askPdf.checked = Boolean(supportProduct && document);
  pdfFile.accept = collectionMode ? ".pdf,.docx,.txt,.md,.markdown" : ".pdf,application/pdf";
  uploadPdf.textContent = collectionMode ? "Add guide" : document ? "Replace PDF" : "Upload PDF";
  if (collectionMode) {
    setStatus(pdfStatus, `${payload.documents.length} indexed sources`);
    if (supportProduct) {
      showKnowledge(payload.documents);
      showPdfChoices(payload.documents);
    }
  } else {
    setStatus(pdfStatus, document ? `${document.filename} · ${document.pages} indexed pages` : "");
  }
}

async function loadModels() {
  try {
    const response = await fetch("/models", { cache: "no-store" });
    if (!response.ok) throw new Error();
    const { models, default: preferred, backup } = await response.json();
    const available = new Set(models);
    const automatic = available.has(preferred) ? preferred : backup;
    const [automaticOption, ...choices] = modelPicker.options;
    const activeChoice = choices.find((option) => option.value === automatic);
    if (activeChoice && available.has(automatic)) {
      automaticOption.text = `Automatic (${activeChoice.text})`;
    }
    for (const option of choices) {
      option.disabled = !available.has(option.value);
      if (option.disabled) option.text += " (unavailable)";
    }
    if (!choices.some((option) => !option.disabled)) {
      setStatus(modelStatus, "Local Qwen models unavailable.", true);
    }
  } catch {
    setStatus(modelStatus, "Model status unavailable.", true);
  }
}

function showPdfChoices(guides) {
  pdfPicker.replaceChildren();
  const all = document.createElement("option");
  all.value = "";
  all.textContent = "All guides";
  pdfPicker.append(all);

  const pdfs = guides.filter(({ filename }) => filename.toLowerCase().endsWith(".pdf"));
  for (const { filename } of pdfs) {
    const option = document.createElement("option");
    option.value = filename;
    option.textContent = filename;
    pdfPicker.append(option);
  }
  const missing = activePdf && !pdfs.some(({ filename }) => filename === activePdf);
  if (missing) {
    const option = document.createElement("option");
    option.value = activePdf;
    option.textContent = activePdf + " (unavailable)";
    option.disabled = true;
    pdfPicker.append(option);
  }
  pdfPicker.value = activePdf;
  setStatus(pdfScopeStatus, missing ? "Selected PDF unavailable." : "", Boolean(missing));
}

function showKnowledge(guides) {
  const list = document.querySelector("#knowledge-list");
  list.replaceChildren();
  for (const guide of guides) {
    const row = document.createElement("li");
    const name = document.createElement("span");
    name.textContent = guide.filename;
    const remove = document.createElement("button");
    remove.className = "secondary";
    remove.type = "button";
    remove.textContent = "Remove";
    remove.setAttribute("aria-label", `Remove ${guide.filename}`);
    remove.addEventListener("click", async () => {
      if (!window.confirm(`Remove ${guide.filename} from the searchable index?`)) return;
      remove.disabled = true;
      try {
        const query = new URLSearchParams({ filename: guide.filename });
        const response = await fetch(`/documents?${query}`, { method: "DELETE" });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail);
        showDocument(result);
        setStatus(knowledgeStatus, "Source removed from the index.");
      } catch (error) {
        setStatus(knowledgeStatus, error.message, true);
        remove.disabled = false;
      }
    });
    row.append(name, remove);
    list.append(row);
  }
}

knowledgeForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const title = document.querySelector("#knowledge-title").value.trim();
  const text = document.querySelector("#knowledge-text").value.trim();
  const url = document.querySelector("#knowledge-url").value.trim();
  const save = knowledgeForm.querySelector("button");
  save.disabled = true;
  try {
    setStatus(knowledgeStatus, url && !text ? "Importing approved page…" : "Saving approved knowledge…");
    const result = await sendRequest("/documents/notes", "application/json",
      JSON.stringify({ title, text: text || null, url: url || null }), approvalHeaders);
    showDocument(result);
    knowledgeForm.reset();
    setStatus(knowledgeStatus, "Approved knowledge saved.");
  } catch (error) {
    setStatus(knowledgeStatus, error.message, true);
  } finally {
    save.disabled = false;
  }
});

function supportQuestion(prompt) {
  return {
    prompt,
    model: modelPicker.value || null,
    previous_question: activePdf ? null : supportTopics.join("\n") || null,
    pdf_filename: activePdf || null,
  };
}

function rememberTopic(prompt, result, revision, conversation) {
  if (!result.sources?.length || revision !== pdfRevision || conversation !== conversationRevision) return;
  const topic = prompt.replace(/\s+/g, " ").slice(0, 160);
  supportTopics = [topic, ...supportTopics.filter((item) => item !== topic)].slice(0, 3);
}

function closeSummary() {
  summaryPanel.hidden = true;
  summaryButton.setAttribute("aria-expanded", "false");
}

summaryButton.addEventListener("click", async () => {
  if (!summaryPanel.hidden) {
    closeSummary();
    return;
  }
  const revision = conversationRevision;
  summaryButton.disabled = true;
  try {
    const response = await fetch("/conversations/current/summary", { cache: "no-store" });
    if (!response.ok) throw new Error("Conversation summary is unavailable.");
    const result = await response.json();
    if (revision !== conversationRevision) return;
    summaryPanel.textContent = result.summary;
    summaryPanel.hidden = false;
    summaryButton.setAttribute("aria-expanded", "true");
  } catch (error) {
    setStatus(chatStatus, error.message, true);
  } finally {
    summaryButton.disabled = false;
  }
});

stopAnswer.addEventListener("click", () => activeAnswer?.abort());

function clearConversationView() {
  conversationRevision += 1;
  supportTopics = [];
  previousPdfQuestion = "";
  setStatus(chatStatus, "");
  setStatus(voiceStatus, "");
  messages.querySelectorAll(".message").forEach((message) => message.remove());
  document.querySelector("#welcome").hidden = false;
  closeSummary();
  voiceResult.hidden = true;
  playback.pause();
}

pdfPicker.addEventListener("change", async () => {
  activePdf = pdfPicker.value;
  summaryButton.hidden = Boolean(activePdf);
  setStatus(pdfScopeStatus, "");
  clearConversationView();
  pdfPicker.disabled = true;
  sendButton.disabled = true;
  try {
    await restoreConversation();
  } catch (error) {
    setStatus(chatStatus, error.message, true);
  } finally {
    pdfPicker.disabled = false;
    sendButton.disabled = false;
  }
});

newConversation.addEventListener("click", async () => {
  newConversation.disabled = true;
  try {
    activeAnswer?.abort();
    await sendRequest("/conversations/new", "application/json", "{}");
    clearConversationView();
    promptInput.value = "";
    voiceTranscript.value = "";
    if (recording) {
      recording.discard = true;
      stopRecording(recording);
    }
    voiceReview.hidden = true;
  } catch (error) {
    setStatus(chatStatus, error.message, true);
  } finally {
    newConversation.disabled = false;
  }
});

async function restoreConversation() {
  const revision = conversationRevision;
  const query = activePdf ? `?${new URLSearchParams({ pdf_filename: activePdf })}` : "";
  const response = await fetch(`/conversations/current${query}`, { cache: "no-store" });
  if (!response.ok) throw new Error("Saved conversation could not be loaded.");
  const { turns } = await response.json();
  if (revision !== conversationRevision) return;
  for (const turn of turns) {
    appendMessage("You", turn.prompt);
    appendMessage("Assistant", turn.reply, turn.sources);
    rememberTopic(turn.prompt, turn, pdfRevision, conversationRevision);
  }
}

uploadPdf.addEventListener("click", () => pdfFile.click());
pdfFile.addEventListener("change", async () => {
  const file = pdfFile.files[0];
  if (!file) return;
  if (supportProduct && !window.confirm(`Approve ${file.name} for the searchable IT Support collection?`)) {
    pdfFile.value = "";
    return;
  }

  uploadPdf.disabled = true;
  setStatus(pdfStatus, collectionMode ? "Indexing guide…" : "Indexing PDF…");
  try {
    const query = new URLSearchParams({ filename: file.name });
    const result = await sendRequest(`/documents?${query}`, file.type || "application/octet-stream", file,
      supportProduct ? approvalHeaders : {});
    showDocument(result);
    askPdf.checked = true;
  } catch (error) {
    setStatus(pdfStatus, error.message, true);
  } finally {
    pdfFile.value = "";
    uploadPdf.disabled = false;
  }
});

async function loadDocument() {
  try {
    const response = await fetch("/documents", { cache: "no-store" });
    if (!response.ok) throw new Error();
    showDocument(await response.json());
    if (supportProduct) await restoreConversation();
  } catch (error) {
    setStatus(pdfStatus, error.message || "Guide status unavailable.", true);
  } finally {
    uploadPdf.disabled = false;
    sendButton.disabled = false;
    pdfPicker.disabled = false;
  }
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const prompt = promptInput.value.trim();
  if (!prompt || sendButton.disabled || (supportProduct && (pdfPicker.disabled || newConversation.disabled))) return;

  sendButton.disabled = true;
  pdfPicker.disabled = supportProduct;
  promptInput.value = "";
  appendMessage("You", prompt);
  setStatus(chatStatus, "Waiting for a reply…");
  const conversation = conversationRevision;
  try {
    const documentMode = askPdf.checked;
    const revision = pdfRevision;
    if (supportProduct) {
      activeAnswer = new AbortController();
      stopAnswer.hidden = false;
      closeSummary();
      const result = await streamAnswer(supportQuestion(prompt), conversation, activeAnswer.signal);
      if (conversation !== conversationRevision) return;
      rememberTopic(prompt, result, revision, conversation);
      appendMessage("Assistant", result.reply, result.sources);
      setStatus(chatStatus, "");
      return;
    }
    const endpoint = documentMode ? "/documents/ask" : "/chat";
    const body = documentMode
      ? { prompt, model: modelPicker.value || null, previous_question: previousPdfQuestion || null }
      : { prompt, model: modelPicker.value || null };
    const result = await sendRequest(endpoint, "application/json", JSON.stringify(body));
    if (documentMode && result.sources?.length && pdfRevision === revision) previousPdfQuestion = prompt;
    appendMessage("Assistant", result.reply, result.sources);
    setStatus(chatStatus, "");
  } catch (error) {
    if (conversation !== conversationRevision) return;
    if (!promptInput.value) promptInput.value = prompt;
    setStatus(chatStatus, error.name === "AbortError" ? "Response stopped." : error.message, error.name !== "AbortError");
  } finally {
    activeAnswer = undefined;
    stopAnswer.hidden = true;
    sendButton.disabled = false;
    pdfPicker.disabled = false;
  }
});

promptInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

function stopRecording(session) {
  clearTimeout(session.limit);
  if (session.recorder.state !== "inactive") session.recorder.stop();
  session.stream.getTracks().forEach((track) => track.stop());
  recordButton.disabled = true;
}

function resetRecordButton() {
  recordButton.disabled = false;
  recordButton.textContent = "Start recording";
  recordButton.dataset.state = "";
  sendTranscript.disabled = false;
}

function encodeWav(samples) {
  const data = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(data);
  const writeText = (offset, value) => {
    for (let index = 0; index < value.length; index += 1) {
      view.setUint8(offset + index, value.charCodeAt(index));
    }
  };

  writeText(0, "RIFF");
  view.setUint32(4, data.byteLength - 8, true);
  writeText(8, "WAVE");
  writeText(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, voiceSampleRate, true);
  view.setUint32(28, voiceSampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeText(36, "data");
  view.setUint32(40, samples.length * 2, true);

  for (let index = 0; index < samples.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, samples[index]));
    view.setInt16(44 + index * 2, sample * (sample < 0 ? 32768 : 32767), true);
  }
  return data;
}

async function convertRecording(blob) {
  const decoder = new AudioContext({ sampleRate: voiceSampleRate });
  try {
    const decoded = await decoder.decodeAudioData(await blob.arrayBuffer());
    const frames = Math.min(
      Math.ceil(decoded.duration * voiceSampleRate),
      maxRecordingSeconds * voiceSampleRate,
    );
    const resampler = new OfflineAudioContext(1, frames, voiceSampleRate);
    const source = resampler.createBufferSource();
    source.buffer = decoded;
    source.connect(resampler.destination);
    source.start();
    const mono = await resampler.startRendering();
    return encodeWav(mono.getChannelData(0));
  } finally {
    await decoder.close();
  }
}

async function finishRecording(session) {
  if (recording === session) recording = undefined;
  if (session.discard) {
    resetRecordButton();
    return;
  }

  recordButton.disabled = true;
  recordButton.textContent = "Processing…";
  recordButton.dataset.state = "";
  setStatus(voiceStatus, "Transcribing…");
  try {
    const clip = new Blob(session.chunks, { type: session.recorder.mimeType });
    if (!clip.size) throw new Error("No audio was recorded. Try again.");
    const wav = await convertRecording(clip);
    const result = await sendRequest("/voice/transcribe", "audio/wav", wav);
    voiceResult.hidden = true;
    voiceTranscript.value = result.transcript;
    voiceReview.hidden = false;
    if (!voicePanel.hidden) voiceTranscript.focus();
    setStatus(voiceStatus, "");
  } catch (error) {
    setStatus(voiceStatus, error.message, true);
  } finally {
    resetRecordButton();
  }
}

voiceReview.addEventListener("submit", async (event) => {
  event.preventDefault();
  const prompt = voiceTranscript.value.trim();
  if (!prompt || sendTranscript.disabled || (supportProduct && (pdfPicker.disabled || newConversation.disabled))) return;

  sendTranscript.disabled = true;
  pdfPicker.disabled = supportProduct;
  recordButton.disabled = true;
  setStatus(voiceStatus, "Preparing a spoken reply…");
  try {
    const revision = pdfRevision;
    const conversation = conversationRevision;
    const body = supportProduct ? supportQuestion(prompt) : { prompt, model: modelPicker.value || null };
    const result = await sendRequest("/voice/reply", "application/json", JSON.stringify(body));
    if (conversation !== conversationRevision) return;
    if (supportProduct) rememberTopic(prompt, result, revision, conversation);
    document.querySelector("#transcript").textContent = prompt;
    document.querySelector("#voice-reply").textContent = result.reply;
    const sources = document.querySelector("#voice-sources");
    sources.replaceChildren();
    renderSources(sources, result.sources);
    if (playbackUrl) URL.revokeObjectURL(playbackUrl);
    playback.hidden = !result.audio;
    if (result.audio) {
      const audioBytes = Uint8Array.from(atob(result.audio), (byte) => byte.charCodeAt(0));
      playbackUrl = URL.createObjectURL(new Blob([audioBytes], { type: "audio/wav" }));
      playback.src = playbackUrl;
    } else {
      playbackUrl = undefined;
      playback.removeAttribute("src");
    }
    voiceReview.hidden = true;
    voiceResult.hidden = false;
    setStatus(voiceStatus, result.audio_error || "", Boolean(result.audio_error));
    if (result.audio && !voicePanel.hidden) {
      playback.play().catch(() => setStatus(voiceStatus, "Press Play to hear the reply."));
    }
  } catch (error) {
    setStatus(voiceStatus, error.message, true);
  } finally {
    pdfPicker.disabled = false;
    resetRecordButton();
  }
});

recordButton.addEventListener("click", async () => {
  if (recording) {
    stopRecording(recording);
    return;
  }

  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    setStatus(voiceStatus, "Microphone recording is unavailable in this browser. Open the local page in a browser that supports recording.", true);
    return;
  }

  recordButton.disabled = true;
  sendTranscript.disabled = true;
  setStatus(voiceStatus, "Waiting for microphone access…");
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    if (voicePanel.hidden) {
      stream.getTracks().forEach((track) => track.stop());
      setStatus(voiceStatus, "");
      resetRecordButton();
      return;
    }

    playback.pause();
    const recorder = new MediaRecorder(stream);
    const session = { recorder, stream, chunks: [], discard: false };
    recorder.addEventListener("dataavailable", (event) => {
      if (event.data.size) session.chunks.push(event.data);
    });
    recorder.addEventListener("stop", () => {
      clearTimeout(session.limit);
      stream.getTracks().forEach((track) => track.stop());
      void finishRecording(session);
    }, { once: true });
    recorder.addEventListener("error", () => {
      session.discard = true;
      stopRecording(session);
      setStatus(voiceStatus, "The recording stopped unexpectedly. Try again.", true);
    }, { once: true });
    recorder.start();
    recording = session;
    session.limit = setTimeout(() => stopRecording(session), maxRecordingSeconds * 1000);
    recordButton.disabled = false;
    recordButton.textContent = "Stop recording";
    recordButton.dataset.state = "recording";
    setStatus(voiceStatus, "Recording. Stop when you’ve finished speaking.");
  } catch (error) {
    stream?.getTracks().forEach((track) => track.stop());
    setStatus(
      voiceStatus,
      error.name === "NotAllowedError" ? "Allow microphone access in your browser and try again." : error.message,
      true,
    );
    resetRecordButton();
  }
});

checkConnection();
loadModels();
loadDocument();
