# Personal AI Assistant

The FastAPI backend serves a local chat model through Ollama and provides a health endpoint.

## Setup

From the project root, use Python 3.12 and create a virtual environment if you do not already have one.

```bash
python3.12 -m venv .venv
```
#$ For restart(
pgrep -fl 'run_local\.py'
kill -TERM <PID>
)
Install the recorded dependencies into the project environment.

```bash
./.venv/bin/python -m pip install -r requirements.txt
```

## Run the local model on macOS

The project uses [Qwen3 1.7B](https://ollama.com/library/qwen3:1.7b), a 1.4 GB download. Run all commands from the project root.

If `.ollama-runtime/ollama` is missing, download the [Ollama macOS CLI](https://github.com/ollama/ollama/releases/tag/v0.40.4) and extract it into the project.

```bash
mkdir -p .ollama-runtime
curl -fL https://github.com/ollama/ollama/releases/download/v0.40.4/ollama-darwin.tgz \
  -o .ollama-runtime/ollama-darwin.tgz &&
  printf '%s  %s\n' a4ceae4b2aaf18df37ebcbfb56ba3a37a87838631112d66c2be07268693c8d58 \
    .ollama-runtime/ollama-darwin.tgz | shasum -a 256 -c - &&
  tar -xzf .ollama-runtime/ollama-darwin.tgz -C .ollama-runtime
```

Start Ollama in the first terminal. This keeps the model files in the project's ignored `models/` directory and enables local-only mode.

```bash
mkdir -p models
OLLAMA_HOST=127.0.0.1:11434 OLLAMA_MODELS="$PWD/models" \
  OLLAMA_NO_CLOUD=1 OLLAMA_NUM_PARALLEL=1 ./.ollama-runtime/ollama serve
```

In a second terminal, download the model once.

```bash
OLLAMA_HOST=127.0.0.1:11434 ./.ollama-runtime/ollama pull qwen3:1.7b
```

After the initial download, press **Ctrl+C** in the terminal running Ollama before using the combined launcher below.

## Chat in the terminal

With Ollama running, open a second terminal in the project root and start an interactive chat.

```bash
./.ollama-runtime/ollama run qwen3:1.7b
```

When you see `>>>`, type a prompt as plain text and press Enter. You can keep typing prompts in the same session.

## Start both servers

From the project root, run Ollama and FastAPI in one terminal.

```bash
./.venv/bin/python run_local.py
```

Open [Personal AI](http://127.0.0.1:8000/) when the launcher reports that it is ready. Press **Ctrl+C** in the same terminal to stop both servers. If you already started either server separately, stop it before running the launcher.

- **Chat** lets you type a prompt and read the reply.
- **Voice** lets you record up to 20 seconds. Review and edit the transcript, then click **Send transcript** to get a spoken reply. If playback does not start automatically, press **Play**.

Voice mode transcribes locally with Whisper Small multilingual when its weights are installed. It uses Whisper Base multilingual otherwise. macOS text-to-speech reads the reply. Set up Whisper below if its CLI or weights are missing.

In another terminal, check the backend and send it a prompt.

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail-with-body http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"In one sentence, what is FastAPI used for?"}'
```

The health check returns `{"status":"ok"}`. The chat endpoint returns a JSON object with a `reply` field. Each request sends one prompt to the model.

### Stop both servers

From another terminal, find the process running the Personal AI launcher.

```bash
pgrep -fl '(^|[[:space:]/])run_local[.]py([[:space:]]|$)'
```

Replace `PID` with the process ID of the Python launcher. Sending `TERM`
lets it shut down the FastAPI and Ollama processes it started.

```bash
kill -TERM PID
```

## Ask a PDF

In **Chat**, click **Upload PDF** and choose a PDF with selectable text (up to
10 MB and 100 pages). Once indexing finishes, leave **Ask this PDF** checked
and type your question. Replies cite the filename and PDF page numbers. You
can ask for a section by heading or count a numbered feature list. Uncheck
**Ask this PDF** for regular chat.

One PDF can be active at a time. **Replace PDF** changes the active document
after a successful upload. The app stores extracted passages locally in
`memory/documents.sqlite3` and sends relevant passages to your local Ollama.
Scanned pages need OCR before they can be searched.

Short PDFs that fit in the model's context are sent to Ollama in full. Longer
PDFs use local keyword search and, when available, semantic search. To enable
semantic search on a new setup, run the following once while Ollama is running.

```bash
./.ollama-runtime/ollama pull nomic-embed-text:v1.5
```

The first question on a longer PDF can take longer while the app embeds its
passages. Follow-up questions can use the last answered PDF question in the
same browser session.

With local Ollama running, run the PDF answer and citation benchmark from the
project root.

```bash
./.venv/bin/python -m benchmarks.pdf_qa
```

The benchmark uploads three synthetic PDFs to a temporary database, asks each
question once, and reports evidence, citation and missing-answer failures. It
leaves your active PDF in place.

### Run the IT Support collection

The IT Support index is separate from the personal PDF index. From the
project folder, run the collection on unused local ports.

```bash
PERSONAL_AI_COLLECTION=1 \
PERSONAL_AI_DOCUMENT_DB="$PWD/collections/it-support/index/documents.sqlite3" \
PERSONAL_AI_OLLAMA_PORT=11439 PERSONAL_AI_API_PORT=8001 \
RAG_EVIDENCE_IDS=1 RAG_PARTWISE_EVIDENCE=1 RAG_SECTION_CHUNKS=0 \
./.venv/bin/python run_local.py
```

Open `http://127.0.0.1:8001` and select **Ask IT Support guides**. **Add guide**
accepts PDF, DOCX, TXT, and Markdown. Adding the same filename updates that
guide. PDFs cite their original page numbers. DOCX, TXT, and Markdown cite
sections because they have no fixed page numbers in the index. PDF pages with
empty or visibly corrupt text are read through macOS Vision using Swift.

To rebuild or update the index from the guide folder, use the existing
collection path and the source aliases in `sources.json`.

```bash
PERSONAL_AI_COLLECTION=1 \
PERSONAL_AI_DOCUMENT_DB="$PWD/collections/it-support/index/documents.sqlite3" \
./.venv/bin/python -m backend.document_intake collections/it-support/documents \
  --sources collections/it-support/sources.json
```

To measure collection-wide retrieval with common typos, keep the same local
Ollama server running and use the tuning set.

```bash
PERSONAL_AI_OLLAMA_PORT=11439 RAG_EVIDENCE_IDS=1 RAG_PARTWISE_EVIDENCE=1 \
./.venv/bin/python -m benchmarks.it_collection_eval --noisy \
  --index collections/it-support/index/documents.sqlite3 \
  --report /tmp/it-support-collection-report.json
```

The report checks the expected filename as well as the page. Edit
`backend/Correct_Spelling.json` to add known prompt typos. Restart the app
after changing the dictionary.

### Diagnose PDF accuracy

Launch the app with terminal traces for PDF questions.

```bash
RAG_DEBUG=1 ./.venv/bin/python run_local.py
```

Traces include PDF excerpts, FTS and embedding scores, the context that reached
Ollama, the messages sent to it, and quote verification decisions. Enable
traces during local debugging because they can show private document text.

Run the 30-question evaluation with Ollama running and the mini project report
as the active PDF, or include a separate `--pdf` path for each source file.
The evaluator uses temporary SQLite copies.

```bash
./.venv/bin/python -m benchmarks.rag_eval --reader qwen \
  --pdf "$HOME/Downloads/Arpit_Resume.pdf" --report /tmp/personal-ai-rag-report.json
```

Use `--reader tinygpt` for the same questions with the local evidence reader.
Use `--traces` with `--report` for per-question diagnostics. The report scores
evidence retrieval, answer and citation correctness, and unsupported answers
on the missing-fact questions. Structured lookups are scored for answer
correctness and excluded from retrieval rates when they bypass retrieval.

With Qwen, add `--retrieval-only --keyword-only` to test the SQLite retriever
without Ollama. Add `--section-chunks` to re-chunk the temporary index using
500–1,000-character targets and overlap at sentence or list boundaries.
Set `RAG_SECTION_CHUNKS=1` when starting the app to try that chunking on future
uploads. Set `RAG_QUERY_REWRITE=1` to test standalone Qwen queries for follow-ups
with an available previous question.

Use the **PDF model** menu to try **TinyGPT (preview)** after training its
shared evidence reader. The reader selects excerpts from indexed text and
shows their original PDF pages. Qwen is the default PDF model.
TinyGPT's question matching handles common typos, related terms such as
`fee` and `cost`, and flexible résumé questions such as “Where did I study?”
It matches table values to their row and column labels and checks whether
dates describe the event asked about. It cites the evidence for each
supported part of a multi-part question and identifies any missing parts.

## Use local memory

In the browser's Chat or Voice mode, type or say **“Remember that I prefer 25-minute study sessions.”** The assistant confirms the save. You can then ask **“What do you remember about me?”** to see your latest saved facts, even after restarting FastAPI.

SQLite creates `memory/assistant.sqlite3` automatically. Use **“Remember that…”** to save a fact. The app uses your eight most recent facts as context for relevant replies.

To replace a saved address in Chat or Voice, say or type **“Update my address to 14 Rose Lane.”** This replaces earlier address statements saved as “My address is…”, “My current address is…”, “My home address is…”, or “I live at…”. Ask **“What is my current address?”** to read the saved address directly from SQLite.

To replace another saved fact, ask **“What do you remember about me?”** and copy the exact old fact. Then type this in Chat.

```text
Update memory: I prefer 25-minute study sessions. -> I prefer 40-minute study sessions.
```

The app replaces that fact in SQLite and keeps your other memories.

To remove a fact, copy it exactly from the memory list and type `Forget memory: I prefer 25-minute study sessions.` in Chat.

Use **“Update my name to Arpit”** to replace earlier saved name statements. Ask **“What is my name?”** to verify the saved spelling.

For names that Whisper misspells, **type** “Remember that my name is Arpit” in Chat. Voice mode uses the latest saved name as a spelling hint. Check the transcript after speaking to see what Whisper heard.

For other names you use often, type one of these commands in Chat with the exact spelling. Replace the examples and placeholders with the names you want to save.

```text
Remember voice name: Anubhav Tomer
Remember voice name for friends: Anubhav Tomer
Remember voice name for family: <full name>
Remember voice name for contacts: <full name>
```

Voice uses recent names across groups alongside your own saved name. Ask **“List voice names”** to check your entries. Name hints can improve transcription. Check the spelling before sending, especially when you intend to save a fact.

## Test speech recognition on macOS

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

### Correct recurring voice spellings

Voice mode checks [speech/spelling_corrections.json](speech/spelling_corrections.json) before displaying a transcript or sending it to Qwen. Each key is the exact word or phrase Whisper gets wrong, and its value is the spelling you want. The file includes `"arpitu": "Arpit"` and common technical terms and typos.

If a word keeps coming out wrong, copy the misheard spelling from the transcript, add it to the JSON with the desired spelling, and record again. Saved edits take effect on the next voice request.

The expanded English typo entries are adapted from the [Codespell v2.4.1 dictionary](https://github.com/codespell-project/codespell/blob/v2.4.1/codespell_lib/data/dictionary.txt), which derives from English Wikipedia and is licensed under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). The entries were filtered to single-suggestion misspellings, prioritized using the [Google 10,000 English list](https://github.com/first20hours/google-10000-english), and converted to JSON.

The same JSON file includes Indian given names and place spellings from Uttar Pradesh, Uttarakhand, and Haryana. It corrects variants such as `Uttargadesh` to `Uttar Pradesh`, `Haripoor` to `Haripur`, and `Beheet` to `Behat`. Place spellings were adapted from the [May 2026 Local Government Directory export](https://github.com/vanga/india-local-government-directory/tree/main/data) and a [2024 India Post directory snapshot](https://github.com/dropdevrahul/pincodes-india/blob/master/pincode.csv) under the [Government Open Data License – India](https://www.data.gov.in/government-open-data-license-india).

The additional Indian name entries combine the project's existing given names with names from [Faker's Indian English provider](https://github.com/joke2k/faker/blob/master/faker/providers/person/en_IN/__init__.py). The full-name combinations were generated for spelling coverage. For a recurring misheard name, add the exact full phrase and preferred spelling to the dictionary.

<details>
<summary>Faker MIT license notice</summary>

```text
Copyright (c) 2012 Daniele Faraglia

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

</details>

## Week 4 model prototype

`training/tokenizer.py` encodes UTF-8 text as byte tokens. `training/model.py` defines a small GPT-style Transformer with causal attention and a 128-token context. The context counts bytes, so a character can take more than one token.

```bash
./.venv/bin/python -m training.check_model
./.venv/bin/python -m training.check_model --device mps
```

The smoke test checks text round trips, next-token output shape, causal masking, and backpropagation with randomly initialized weights. Dataset preparation, training, and evaluation are Week 5.

## Train the Week 5 prototype

Put one example per line in `training/train.txt` and different examples in
`training/validation.txt`. Run from the project root.

```bash
./.venv/bin/python -m training.train --device mps --epochs 20
```

The script prints the untrained validation loss and the loss after each epoch.
Lower validation loss means better next-token prediction on the held-out text.
The best trained epoch is saved to `models/tiny-gpt.pt`. Use `--output` to keep
checkpoints from separate runs. For a quick test, use `--epochs 2`.

Evaluate the saved checkpoint on CPU and optionally try a text completion.

```bash
./.venv/bin/python -m training.evaluate
./.venv/bin/python -m training.evaluate --prompt "A Python function" --max-tokens 30
```

### Kaggle Python experiment

[kaggle_python_sample.csv](training/kaggle_python_sample.csv) contains short prose extracts from [Python Questions from Stack Overflow on Kaggle](https://www.kaggle.com/datasets/stackoverflow/pythonquestions). The CSV credits the original answers and authors by URL. The excerpts were filtered by topic, score, and length, with HTML and code blocks removed. Stack Overflow contributions are licensed under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).

`training/kaggle_train.txt` combines the sample with the original training sentences. Use `training/validation.txt` to compare the original and Kaggle runs.

```bash
./.venv/bin/python -m training.train --device mps --epochs 20 \
  --train training/kaggle_train.txt --output models/tiny-gpt-kaggle.pt
./.venv/bin/python -m training.evaluate --checkpoint models/tiny-gpt-kaggle.pt \
  --prompt "A Python function"
```

For a separate Kaggle test, `training/kaggle_test.txt` contains 120 excerpts
from question threads excluded from training. [The test CSV](training/kaggle_test_sample.csv)
contains the same excerpts with links to the original answers and authors.

```bash
./.venv/bin/python -m training.evaluate --checkpoint models/tiny-gpt-kaggle.pt \
  --validation training/kaggle_test.txt --prompt "A Python function"
```

### Focused Python Q&A experiment

The Q&A files contain 18 beginner Python topics, with six training questions,
one validation question, and one test question per topic. The test questions
use unseen wording for topics covered in training.

```bash
./.venv/bin/python -m training.train --device mps --epochs 200 \
  --train training/qa_train.txt --validation training/qa_validation.txt \
  --output models/tiny-gpt-qa.pt
./.venv/bin/python -m training.evaluate --checkpoint models/tiny-gpt-qa.pt \
  --validation training/qa_test.txt --qa
```

The evaluator shows the generated and expected answers, followed by an exact
match count. Review different wording for factual correctness before judging it.

### TinyGPT document evidence dataset

The `training/document_qa/` dataset contains 80 training, 22 validation and 22
test examples from eight public PDFs. The splits use separate PDFs. The Energy
Saver and Yosemite benchmark questions stay held out for model comparison.

Each example includes a question and four candidate excerpts with evidence IDs
and PDF page numbers. The target is the matching choice `A` to `D` or
`NO_ANSWER` when those excerpts lack the answer. Some examples remove only the
supporting passage to simulate a retrieval miss. Other examples ask for a missing detail.

Run the local dataset check from the project root.

```bash
./.venv/bin/python -m training.document_qa_data
```

`sources.json` contains the source URLs, PDF checksums and annotated excerpts.
Use `--write` to regenerate the JSONL files from it. Pass `--pdf-dir <folder>`
to verify the evidence against local copies named by source ID, such as
`insight.pdf`. The longest formatted example is 525 byte tokens, including
the target and boundary tokens. The document model uses a 640-byte-token
context so the question, candidates and target fit together. The original text
model configuration uses a 128-byte-token context.

Check the document context with the existing dataset before training.

```bash
./.venv/bin/python -m training.check_model --document-qa --device cpu
```

Train the document evidence selector from a fresh TinyGPT initialization.

```bash
./.venv/bin/python -m training.train_document_qa --device cpu --epochs 24
```

The trainer scores choices `A` through `D` and `NO_ANSWER` after reading the
complete prompt. It saves the best validation checkpoint to
`models/tiny-gpt-document-qa.pt` and reports exact evidence and abstention
scores on the separate test PDFs. Evaluate the saved checkpoint again with
`./.venv/bin/python -m training.train_document_qa --evaluate-only`.

The shared evidence reader trains TinyGPT on each question–excerpt pair along
with word-overlap features. It selects a cited excerpt or abstains. The next
command uses the public [SQuAD 2.0 training file](https://rajpurkar.github.io/SQuAD-explorer/)
under CC BY-SA 4.0 to augment the labeled PDF examples. Keep a local copy of
`train-v2.0.json` in `models/training-data/squad-v2-train.json`, or give its path
with `--squad`.

```bash
./.venv/bin/python -m training.train_document_reader --device cpu
./.venv/bin/python -m training.train_document_reader --evaluate-only
```

The trained reader checkpoint is `models/tiny-gpt-document-reader.pt`.
The validation and test scores measure selection from supplied candidates.
Run the end-to-end synthetic PDF benchmark with
`./.venv/bin/python -m benchmarks.pdf_qa --reader tinygpt`.

To fine-tune on typos and alternate phrasing, pass `--question-variants` and
`--initialize-from models/tiny-gpt-document-reader.pt`. Use a separate
`--checkpoint` path to compare the candidate with the current reader before
replacing it. Add `--question-variants` to `--evaluate-only` to score the
misspelled and rephrased test questions alongside the original ones.

Typed PDF questions use [`backend/Correct_Spelling.json`](backend/Correct_Spelling.json).
Add a lowercase typo and its corrected word, such as `"databse": "database"`,
then restart the app to load the file. The correction applies to the question.
The cited answer keeps the PDF's original wording.

Add `--resume-skills` to train or evaluate on fictional résumés. These questions
cover misspelled skills, lookup by category, and cases where the requested
information is missing.
Add `--evidence-examples` for fictional equipment records with confusable dates,
roles, and table values.

## Run the tests

```bash
./.venv/bin/python -m pytest
```
