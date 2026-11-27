# IT Support

A local IT helpdesk assistant. It searches approved guides and cites the evidence
for its answers. The included pilot collection covers VPN, Wi-Fi, password
resets, printers, and software installation.

## Start and stop on macOS

Run these commands from the project folder. The launcher starts both Ollama and
FastAPI. It uses the IT Support collection by default.

```bash
./.venv/bin/python run_local.py
```

Open [IT Support](http://127.0.0.1:8000/). Press **Ctrl+C** in the launcher
terminal to stop both servers. From another terminal, find the launcher PID and
replace `PID` with that number.

```bash
pgrep -fl '(^|[[:space:]/])run_local[.]py([[:space:]]|$)'
kill -TERM PID
```

Use `curl --fail http://127.0.0.1:8000/health` to check FastAPI. Use
`curl --fail http://127.0.0.1:8000/models` to see the configured and installed
models. Use the **Model** menu to select an installed model for typed and spoken
questions. When the configured default is 4B and it is missing, automatic model
selection uses 1.7B. Benchmarks and explicit model choices use the selected
model directly.

## Install on macOS

Use Python 3.12. If the existing virtual environment works, start with the
dependency command.

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

If `.ollama-runtime/ollama` is missing, download the [Ollama macOS CLI](https://github.com/ollama/ollama/releases/tag/v0.40.4) and extract it into the project.

```bash
mkdir -p .ollama-runtime
curl -fL https://github.com/ollama/ollama/releases/download/v0.40.4/ollama-darwin.tgz \
  -o .ollama-runtime/ollama-darwin.tgz &&
  printf '%s  %s\n' a4ceae4b2aaf18df37ebcbfb56ba3a37a87838631112d66c2be07268693c8d58 \
    .ollama-runtime/ollama-darwin.tgz | shasum -a 256 -c - &&
  tar -xzf .ollama-runtime/ollama-darwin.tgz -C .ollama-runtime
```

For the first model download, start Ollama in one terminal. This keeps the model files in the project's ignored `models/` directory and enables local-only mode.

```bash
mkdir -p models
OLLAMA_HOST=127.0.0.1:11434 OLLAMA_MODELS="$PWD/models" \
  OLLAMA_NO_CLOUD=1 OLLAMA_NUM_PARALLEL=1 ./.ollama-runtime/ollama serve
```

With Ollama listening on port 11434, download the comparison model, the backup,
and the embedding model from a second terminal.

```bash
./.ollama-runtime/ollama pull qwen3:4b
./.ollama-runtime/ollama pull qwen3:1.7b
./.ollama-runtime/ollama pull nomic-embed-text:v1.5
```

Stop the temporary Ollama server before starting the combined launcher. IT
Support starts with Qwen3 4B. This command uses Qwen3 1.7B for the entire session.

```bash
IT_SUPPORT_MODEL=qwen3:1.7b ./.venv/bin/python run_local.py
```

## Install and run on Windows

Install Python 3.12 and open [Ollama for Windows](https://docs.ollama.com/windows).
Ollama runs in the background and serves its local API. In PowerShell, run these
commands from the project folder.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull qwen3:4b
ollama pull qwen3:1.7b
ollama pull nomic-embed-text:v1.5
.\.venv\Scripts\python.exe run_local.py
```

Open [IT Support](http://127.0.0.1:8000/). **Ctrl+C** stops FastAPI. Manage
Ollama from the Windows system tray. If `ollama` is unavailable immediately
after installation, open a new PowerShell window.

For the first test, use typed chat and a PDF with selectable text. Scanned PDF
OCR and voice still use macOS tooling.

## Week 1 · Core infrastructure

Use the same index, prompt, and retrieval settings for both models. Keep the
holdout split for the final comparison once you finish tuning. The model
benchmark records correct answers with citations, retrieval hits, unsupported
answers on missing facts, and median response time.

```bash
RAG_EVIDENCE_IDS=1 RAG_PARTWISE_EVIDENCE=1 RAG_SECTION_CHUNKS=0 \
./.venv/bin/python -m benchmarks.it_collection_eval --model qwen3:1.7b \
  --split tuning --noisy --index collections/it-support/index/documents.sqlite3 \
  --report collections/it-support/evaluation/qwen-1.7b-tuning-report.json
```

Repeat with `--model qwen3:4b` and a separate report filename. For the held-out
run use `--split holdout` and omit `--noisy`. Follow `Personal_AI_Project_Tracker.pdf` for the
next week after reviewing the scores. Chat streams verified answer text and
displays citations with the final reply. Use **Stop** to cancel a reply.
The archived TinyGPT reader has a JSON configuration beside its checkpoint.
The reader validates that configuration when restoring the checkpoint.

### Model comparison on the M1 MacBook Air

Each model run answered the same 40 cases across the 15-guide collection. The tuning split included the existing spelling variations.

| Model | Tuning correct | Holdout correct | Retrieval hits | Unsupported missing-fact answers | Answerable cases with no cited evidence | Tuning median | Holdout median |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3 1.7B | 21/30 | 8/10 | 31/31 | 0/9 | 0 | 2.16 s | 3.33 s |
| Qwen3 4B with original format | 11/30 | 4/10 | 31/31 | 0/9 | 20 | 3.99 s | 2.86 s |
| Qwen3 4B with updated format | 26/30 | 9/10 | 31/31 | 0/9 | 0 | 4.59 s | 3.30 s |

The updated 4B response schema puts the evidence IDs before the nullable answer. With the original order, 4B returned an empty answer on 20 of 31 answerable cases. Qwen3 4B is the IT Support default. The **Model** menu also offers Qwen3 1.7B.

### Week 1 smoke checks

1. Open the local page and check that the guide count and model menu load.
2. Ask a question about a named guide, such as “How do I install Company Portal
   at St. Cloud State?” Check the source under the answer.
3. Ask for a fact the guides do not contain. Check that the answer says it
   couldn’t find evidence.
4. In Voice, record a guide question, review the transcript, and send it. Verify
   the text, its sources, and playback.

## Week 2 · Memory and knowledge

- **Add guide** accepts PDF, Word, TXT, and Markdown. Confirm approval when
  prompted. The same filename replaces the previous indexed version. Scanned
  PDF pages use macOS Vision OCR when the embedded text is missing or corrupt.
- Open **Add a note or website** in the knowledge library. Enter a title and
  an approved public HTTPS URL to import the page. You can also paste an
  excerpt or add a local note. URL imports need an internet connection.
  The app indexes the approved text locally and cites the source URL.
- Use **Remove** to delete a guide or note from the searchable index. Original
  files in the guide folder stay available for manual rebuilding.
- Chat and Voice save turns in the browser's current SQLite-backed conversation.
  Reload the page to see the transcript, and ask what you said earlier to recall
  a conversation fact. **Summary** extracts key user statements from the conversation.
  **New chat** starts a separate conversation. Approved guides and notes persist
  in the collection database.

To load the bundled guides into an empty index or refresh them from disk, run
the intake command below. It uses the aliases in the source manifest.

```bash
mkdir -p collections/it-support/index
PERSONAL_AI_COLLECTION=1 \
PERSONAL_AI_DOCUMENT_DB="$PWD/collections/it-support/index/documents.sqlite3" \
./.venv/bin/python -m backend.document_intake collections/it-support/documents \
  --sources collections/it-support/sources.json
```

### Week 2 smoke checks

1. Approve a test note with a distinctive troubleshooting fact. Ask for the
   fact and check that the answer cites the note.
2. Save the title again with an updated fact. Check after a restart that the
   new answer is available. Remove the note and check that its fact is absent.
3. Approve a public HTTPS guide, then ask a question about its content and
   verify the citation.
4. Share a fact in Chat, reload the page, then ask what you shared. Open
   **Summary** to check the stored statements and **New chat** to start a separate conversation.

## Week 3 · System diagnostics

Open **Diagnostics**, choose a check, and click **Run check**. Results come
directly from the macOS or Windows computer hosting the app.
The panel covers tracker items 20–33.

- Hardware and OS, CPU usage, RAM, and disk space.
- IPv4 and IPv6 default routes, a DNS lookup for example.com, TCP listener
  owners, top processes, and counts of recent system log events. Windows reads
  error and critical events from the System and Application logs.
- Git branch and conflict state, the active Python environment, Node.js and npm
  versions, read-only SQLite connectivity, and the local Docker engine.

The TCP check defaults to the app and Ollama ports. Enter a port to see whether
a process is already listening on it. The log check returns totals only.
Docker reports unavailable when the local engine is stopped or inaccessible.
Windows collects its system metrics through built-in Windows PowerShell. Git and
Node.js require their tools on PATH. Start Docker Desktop to check its engine.

```bash
curl --fail http://127.0.0.1:8000/diagnostics
curl --fail-with-body http://127.0.0.1:8000/diagnostics/run \
  -H 'Content-Type: application/json' -d '{"check":"ports","port":8000}'
```

### Windows smoke test

In IT Support, select **Diagnostics → Hardware and OS → Run check**. Then run
**TCP port usage** and confirm the live app appears on port 8000. The native
pytest cases verify the Windows measurements and use a temporary local listener
to check port ownership and release.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_windows_diagnostics.py -v
```

## Optional voice setup

[Whisper multilingual models](https://github.com/openai/whisper/blob/main/model-card.md) run locally through whisper.cpp. The runtime and weights go in Git-ignored directories. To set them up on another Mac, use CMake and the Xcode Command Line Tools to build the CLI and download the Base fallback.

```bash
mkdir -p .whisper-runtime/src models
curl -fL https://codeload.github.com/ggml-org/whisper.cpp/tar.gz/refs/tags/v1.9.5 \
  -o .whisper-runtime/source.tar.gz &&
  tar -xzf .whisper-runtime/source.tar.gz --strip-components 1 -C .whisper-runtime/src &&
  cmake -S .whisper-runtime/src -B .whisper-runtime/build -DCMAKE_BUILD_TYPE=Release \
    -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_SERVER=OFF &&
  cmake --build .whisper-runtime/build --config Release --target whisper-cli -j 4

curl -fL https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin \
  -o models/ggml-base.bin.partial &&
  printf '%s  %s\n' 60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe \
    models/ggml-base.bin.partial | shasum -a 256 -c - &&
  mv models/ggml-base.bin.partial models/ggml-base.bin
```

To use [Whisper Small multilingual](https://huggingface.co/ggerganov/whisper.cpp/blob/main/ggml-small.bin), download its 488 MB model file. Voice mode uses Small when the verified file is in `models/`.

```bash
mkdir -p models
curl -fL https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin \
  -o models/ggml-small.bin.partial &&
  printf '%s  %s\n' 1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b \
    models/ggml-small.bin.partial | shasum -a 256 -c - &&
  mv models/ggml-small.bin.partial models/ggml-small.bin
```

To test a recorded WAV file directly, use `-l en` for English or set it to the language of the recording. Substitute `models/ggml-base.bin` to test Base.

```bash
./.whisper-runtime/build/bin/whisper-cli -m models/ggml-small.bin \
  -f /path/to/recording.wav -l en -nt
```

Add `--prompt 'Custom command name: KodeKove.'` to give Whisper a vocabulary hint for that recording. Hints take effect immediately and leave the model weights unchanged.

## Automated tests

```bash
./.venv/bin/python -m pytest -q
```

The local IT Support mode isolates its knowledge index from the earlier
personal assistant's database. Start that older mode with
`PERSONAL_AI_COLLECTION=0 ./.venv/bin/python run_local.py` when you need it.
The experimental training source remains in the repository. Retired weights and
the old project reports have a recoverable archive next to the project.
