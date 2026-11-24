const chatMode = document.querySelector("#chat-mode");
const voiceMode = document.querySelector("#voice-mode");
const chatPanel = document.querySelector("#chat");
const voicePanel = document.querySelector("#voice");
const messages = document.querySelector("#messages");
const chatForm = document.querySelector("#chat-form");
const promptInput = document.querySelector("#prompt");
const sendButton = document.querySelector("#send");
const chatStatus = document.querySelector("#chat-status");
const pdfFile = document.querySelector("#pdf-file");
const uploadPdf = document.querySelector("#upload-pdf");
const askPdf = document.querySelector("#ask-pdf");
const pdfReader = document.querySelector("#pdf-reader");
const pdfReaderChoice = document.querySelector("#pdf-reader-choice");
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

function setStatus(element, text, isError = false) {
  element.textContent = text;
  element.dataset.state = isError ? "error" : "";
}

function selectMode(mode) {
  const showVoice = mode === "voice";
  chatMode.setAttribute("aria-pressed", String(!showVoice));
  voiceMode.setAttribute("aria-pressed", String(showVoice));
  chatPanel.hidden = showVoice;
  voicePanel.hidden = !showVoice;

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

async function sendRequest(url, contentType, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": contentType },
    body,
  });
  const result = await response.json();
  if (!response.ok) {
    throw new Error(typeof result.detail === "string" ? result.detail : "The request could not be completed.");
  }
  return result;
}

function appendMessage(speaker, text) {
  document.querySelector("#welcome").hidden = true;
  const message = document.createElement("div");
  message.className = "message";
  message.dataset.speaker = speaker;
  const name = document.createElement("strong");
  name.textContent = speaker;
  const content = document.createElement("p");
  content.textContent = text;
  message.append(name, content);
  messages.append(message);
  messages.scrollTop = messages.scrollHeight;
}

let collectionMode = false;

function showDocument(payload) {
  const document = payload.document;
  collectionMode = Boolean(payload.collection);
  window.document.querySelector("#ask-document-label").textContent = collectionMode ? "Ask IT Support guides" : "Ask this PDF";
  previousPdfQuestion = "";
  pdfRevision += 1;
  askPdf.disabled = !document;
  if (!document) askPdf.checked = false;
  pdfReaderChoice.hidden = !askPdf.checked;
  pdfFile.accept = collectionMode ? ".pdf,.docx,.txt,.md,.markdown" : ".pdf,application/pdf";
  uploadPdf.textContent = collectionMode ? "Add guide" : document ? "Replace PDF" : "Upload PDF";
  if (collectionMode) {
    setStatus(pdfStatus, `IT Support · ${payload.documents.length} indexed guides`);
  } else {
    setStatus(pdfStatus, document ? `${document.filename} · ${document.pages} indexed pages` : "");
  }
}

uploadPdf.addEventListener("click", () => pdfFile.click());
askPdf.addEventListener("change", () => {
  pdfReaderChoice.hidden = !askPdf.checked;
});
pdfFile.addEventListener("change", async () => {
  const file = pdfFile.files[0];
  if (!file) return;

  uploadPdf.disabled = true;
  setStatus(pdfStatus, collectionMode ? "Indexing guide…" : "Indexing PDF…");
  try {
    const query = new URLSearchParams({ filename: file.name });
    const result = await sendRequest(`/documents?${query}`, file.type || "application/octet-stream", file);
    showDocument(result);
    askPdf.checked = true;
    pdfReaderChoice.hidden = false;
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
  } catch {
    setStatus(pdfStatus, "PDF status unavailable.", true);
  } finally {
    uploadPdf.disabled = false;
  }
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const prompt = promptInput.value.trim();
  if (!prompt || sendButton.disabled) return;

  sendButton.disabled = true;
  promptInput.value = "";
  appendMessage("You", prompt);
  setStatus(chatStatus, "Waiting for a reply…");
  try {
    const documentMode = askPdf.checked;
    const revision = pdfRevision;
    const endpoint = documentMode ? "/documents/ask" : "/chat";
    const body = documentMode
      ? { prompt, previous_question: previousPdfQuestion || null, reader: pdfReader.value }
      : { prompt };
    const result = await sendRequest(endpoint, "application/json", JSON.stringify(body));
    if (documentMode && result.sources?.length && pdfRevision === revision) previousPdfQuestion = prompt;
    const sources = result.sources?.map(({ filename, page, section }) => `${filename}, ${section || `page ${page}`}`) ?? [];
    appendMessage("Assistant", [result.reply, ...sources].join("\n"));
    setStatus(chatStatus, "");
  } catch (error) {
    if (!promptInput.value) promptInput.value = prompt;
    setStatus(chatStatus, error.message, true);
  } finally {
    sendButton.disabled = false;
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
  if (!prompt || sendTranscript.disabled) return;

  sendTranscript.disabled = true;
  recordButton.disabled = true;
  setStatus(voiceStatus, "Preparing a spoken reply…");
  try {
    const result = await sendRequest("/voice/reply", "application/json", JSON.stringify({ prompt }));
    document.querySelector("#transcript").textContent = prompt;
    document.querySelector("#voice-reply").textContent = result.reply;
    if (playbackUrl) URL.revokeObjectURL(playbackUrl);
    const audioBytes = Uint8Array.from(atob(result.audio), (byte) => byte.charCodeAt(0));
    playbackUrl = URL.createObjectURL(new Blob([audioBytes], { type: "audio/wav" }));
    playback.src = playbackUrl;
    voiceReview.hidden = true;
    voiceResult.hidden = false;
    setStatus(voiceStatus, "");
    if (!voicePanel.hidden) {
      playback.play().catch(() => setStatus(voiceStatus, "Press Play to hear the reply."));
    }
  } catch (error) {
    setStatus(voiceStatus, error.message, true);
  } finally {
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
loadDocument();
