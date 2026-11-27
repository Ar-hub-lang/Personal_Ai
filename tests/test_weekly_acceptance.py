"""Live Week 1 and Week 2 acceptance checks. Run with pytest tests/ -v from the project venv."""

import asyncio
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import closing, contextmanager
from io import BytesIO
from pathlib import Path
from statistics import mean

import httpx
import pytest
from pypdf import PdfReader
from fastapi.testclient import TestClient

from backend import documents, main, rag_search
from backend.website_intake import ReadableHTML, public_addresses
from backend.rag_trace import capture_trace
from benchmarks.pdf_qa import make_pdf
from benchmarks.rag_eval import contains, normalized
from training.document_reader import load_reader

ROOT = Path(__file__).resolve().parents[1]
COLLECTION = ROOT / "collections/it-support"
MODELS = ("qwen3:1.7b", "qwen3:4b")
APPROVAL = {"X-Knowledge-Approval": "yes"}
GUIDE = "weekly-fabledesk-vpn.pdf"
HOSTNAME = "vpn.fabledesk.invalid"
VPN_QUESTION = "What is the FableDesk VPN gateway hostname?"
VPN_PAGE = f"FableDesk VPN\nThe FableDesk VPN gateway hostname is {HOSTNAME}.\nIt supports macOS and Windows."
ROWS = {
    "01": "FastAPI docs, health, and malformed body",
    "02a": "Ollama answers while running",
    "02b": "Clean error after the test Ollama stops",
    "03 / 06": "Same 10 IT questions with evidence and citations",
    "04": "Ordinary SQLite chat transcript survives restart",
    "05": "Answers use an uploaded document",
    "07": "Switch 1.7B → 4B → 1.7B and retain same-tab history",
    "07 UI": "Visual model picker and history in a browser",
    "08a": "Verified answer text streams incrementally",
    "08b": "Cancel a live stream and use the API again",
    "09": "Warm Ollama TTFT, tokens/sec, latency and loaded memory",
    "10a": "Archive is hidden and its checkpoint restores in isolation",
    "10b": "Archived model configuration is saved and restorable",
    "11": "Recall a conversation-only fact after a few turns",
    "12": "Approved fact persists through restart and a new session",
    "13": "20-turn context retains key facts and invents none",
    "14": "Semantic retrieval finds a reworded query's passage",
    "15a": "Normal and multi-page PDFs",
    "15b": "Empty and corrupt PDFs return clean errors",
    "15c": "Image-only PDF uses OCR",
    "16a": "Approved website excerpt and URL, unapproved URL rejected",
    "16b": "Approved URL-only ingestion through a controlled fetch fixture",
    "17": "PDF page and Markdown section agree with the evidence",
    "18": "Update changes answer, deletion removes the evidence and citation",
    "19": "Add guide asks and honors No and Yes",
    "Hallucination": "Five out-of-document questions should abstain",
}
QUESTION_ROWS = {
    q["id"]: (document["filename"], q)
    for document in json.loads((COLLECTION / "evaluation/tuning.json").read_text())["documents"]
    for q in document["questions"]
}
IT_IDS = ("IT-T01", "IT-T02", "IT-T03", "IT-T10", "IT-T13", "IT-T14", "IT-T19", "IT-T21", "IT-T22", "IT-T29")
MISSING_IDS = ("IT-T05", "IT-T06", "IT-T11", "IT-T18", "IT-T24")


def mean_speed(rows):
    return (
        f"TTFT {mean(row['ttft'] for row in rows):.2f}s, "
        f"{mean(row['tokens_per_second'] for row in rows):.1f} tok/s, "
        f"latency {mean(row['latency'] for row in rows):.2f}s"
    )


class Comparison:
    def __init__(self):
        self.records = defaultdict(list)
        self.records["07 UI"] = [
            {"model": model, "status": "MANUAL", "detail": "Inspect the rendered controls and history in a real browser."}
            for model in MODELS
        ]

    @contextmanager
    def check(self, feature, model, case=""):
        result = {"model": model, "status": "PASS", "case": case}
        started = time.perf_counter()
        try:
            yield result
        except pytest.skip.Exception as error:
            result.update(status="MANUAL", detail=str(error))
            raise
        except Exception as error:
            result.update(status="FAIL", detail=str(error).splitlines()[0] if str(error) else type(error).__name__)
            raise
        finally:
            result["seconds"] = time.perf_counter() - started
            self.records[feature].append(result)

    def by_model(self, rows, model):
        return [row for row in rows if row["model"] in (model, "both")]

    def write(self):
        def summary(rows):
            automatic = [row for row in rows if row["status"] in {"PASS", "FAIL"}]
            manual = sum(row["status"] == "MANUAL" for row in rows)
            if not automatic:
                return "MANUAL" if manual else "NOT RUN"
            passed = sum(row["status"] == "PASS" for row in automatic)
            return f"{passed}/{len(automatic)}" + (f", {manual} MANUAL" if manual else "")

        def status(rows):
            states = {row["status"] for row in rows}
            return "FAIL" if "FAIL" in states else "MANUAL" if "MANUAL" in states else "PASS" if states else "NOT RUN"

        def ram(rows):
            samples = [row["ram_mib"] for row in rows if "ram_mib" in row]
            return f"{mean(samples):.0f} MiB" if samples else "—"

        lines = [
            "# Week 1 and Week 2 acceptance comparison",
            "",
            "Run `./.venv/bin/python -m pytest tests/ -v` from this checkout. "
            "A working environment on PATH also supports `pytest tests/ -v`. "
            "This checkout's generated activation script and pytest launcher point to the project's old path. "
            "Tests use temporary databases and loopback servers. The existing 15-guide index is copied read-only.",
            "",
            "The same 10 clean prompts and five missing-fact prompts run on both models. "
            "Answers are checked for expected facts and source pages. Feature 09 measures the first nonblank "
            "content token, generation rate, and total wall time directly from warmed Ollama streams. "
            "RAM means Ollama's loaded model allocation, including context. Other times are whole test durations.",
            "",
            "| Feature ID | Test | 1.7B | 4B | PASS/FAIL | Speed (1.7B / 4B) | RAM (1.7B / 4B) |",
            "|---|---|---:|---:|---|---|---|",
        ]
        for feature, label in ROWS.items():
            rows = self.records[feature]
            pair = [self.by_model(rows, model) for model in MODELS]
            speed = []
            for group in pair:
                measurements = [row for row in group if "ttft" in row]
                seconds = [row["seconds"] for row in group if row["status"] != "MANUAL"]
                speed.append(mean_speed(measurements) if measurements else f"{mean(seconds):.2f}s" if seconds else "—")
            lines.append(f"| {feature} | {label} | {summary(pair[0])} | {summary(pair[1])} | {status(rows)} | "
                         f"{' / '.join(speed)} | {' / '.join(map(ram, pair))} |")
        lines.extend(["", "## Per-model summary", "", "Pass rates count individual automated acceptance cases, including shared checks, and exclude MANUAL.", ""])
        for model in MODELS:
            checks = [row for rows in self.records.values() for row in self.by_model(rows, model)
                      if row["status"] in {"PASS", "FAIL"}]
            passed = sum(row["status"] == "PASS" for row in checks)
            speed = self.by_model([row for row in self.records["09"] if "ttft" in row], model)
            rate = f"{passed}/{len(checks)} ({passed / len(checks):.1%})" if checks else "unmeasured"
            lines.append(f"- **{model}** · {rate} · {mean_speed(speed) if speed else 'speed unmeasured'} · average RAM {ram(speed)}")
        lines.extend(["", "## Failures and manual checks", ""])
        for feature in ROWS:
            for row in self.records[feature]:
                if row["status"] != "PASS":
                    detail = row.get("detail", "").replace("`", "'")
                    lines.append(f"- **{feature} · {row['model']} {row.get('case', '')} · {row['status']}** — {detail}")
        lines.extend(["", "## Recommendation", ""])
        questions = [self.by_model(self.records["03 / 06"], model) for model in MODELS]
        scores = [sum(row["status"] == "PASS" for row in group) for group in questions]
        if scores[0] == scores[1] and scores[0]:
            routine = f"Use **qwen3:1.7b** for routine guide lookups. Both models passed {scores[0]} of the same {len(questions[0])} cases."
        elif max(scores):
            preferred = MODELS[scores.index(max(scores))]
            routine = f"Use **{preferred}** for routine guide lookups. It had the higher cited-answer score on the shared questions."
        else:
            routine = "Run the live cited-answer checks before choosing a model for routine guide lookups."
        advanced = (
            "Use **qwen3:4b** for more involved document questions when answer quality is the priority. "
            "It led the broader 40-question comparison on the same collection. "
            "[Broader benchmark](../README.md#model-comparison-on-the-m1-macbook-air)"
        )
        resource = "Use **qwen3:1.7b** when the Mac has limited available memory. See the measured loaded RAM in feature 09."
        lines.extend([f"- {routine}", f"- {advanced}", f"- {resource}", ""])
        path = ROOT / "results/comparison.md"
        path.parent.mkdir(exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")


@pytest.fixture(scope="module")
def comparison():
    result = Comparison()
    yield result
    result.write()


@pytest.fixture(scope="module", params=MODELS)
def model(request):
    return request.param


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_ready(url, process, log):
    deadline = time.monotonic() + 30
    with httpx.Client(timeout=1, trust_env=False) as client:
        while time.monotonic() < deadline and process.poll() is None:
            try:
                if client.get(url).is_success:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(.1)
    raise RuntimeError(f"Test server did not become ready at {url}. {log.read_text(errors='replace')[-1200:]}")


def stop_owned(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=8)


@contextmanager
def serve_ollama(path, port, models):
    binary = ROOT / ".ollama-runtime/ollama"
    if not binary.is_file():
        pytest.skip("The project Ollama executable is required to start and stop an isolated daemon.")
    log = path / "ollama.log"
    addr = f"127.0.0.1:{port}"
    env = os.environ | {"OLLAMA_HOST": addr, "OLLAMA_MODELS": str(models), "OLLAMA_NO_CLOUD": "1",
                        "OLLAMA_NUM_PARALLEL": "1"}
    with log.open("w") as output:
        process = subprocess.Popen([str(binary), "serve"], env=env, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
        try:
            wait_ready(f"http://{addr}/api/version", process, log)
            yield
        finally:
            stop_owned(process)


@pytest.fixture(scope="module")
def ollama_port(tmp_path_factory):
    port = int(os.environ.get("PERSONAL_AI_OLLAMA_PORT", "11434"))
    try:
        with httpx.Client(timeout=3, trust_env=False) as client:
            tags = client.get(f"http://127.0.0.1:{port}/api/tags").raise_for_status().json()
        if set(MODELS).issubset({item["name"] for item in tags["models"]}):
            yield port
            return
    except httpx.HTTPError:
        pass
    if not (ROOT / ".ollama-runtime/ollama").is_file():
        yield None
        return
    port = free_port()
    with serve_ollama(tmp_path_factory.mktemp("weekly-ollama"), port, ROOT / "models"):
        yield port


class Lab:
    def __init__(self, path, ollama):
        self.path = path
        self.port = free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.document_db = path / "documents.sqlite3"
        self.memory_db = path / "memory.sqlite3"
        self.process = None
        self.env = os.environ | {
            "PYTHONDONTWRITEBYTECODE": "1", "PERSONAL_AI_COLLECTION": "1",
            "IT_SUPPORT_PRODUCT": "1", "PERSONAL_AI_DOCUMENT_DB": str(self.document_db),
            "PERSONAL_AI_MEMORY_DB": str(self.memory_db), "PERSONAL_AI_OLLAMA_PORT": str(ollama or free_port()),
            "RAG_EVIDENCE_IDS": "1", "RAG_PARTWISE_EVIDENCE": "1", "RAG_SECTION_CHUNKS": "0",
        }
        self.env["CLANG_MODULE_CACHE_PATH"] = str(path / "swift-cache")

    def start(self):
        log = self.path / "fastapi.log"
        with log.open("a") as output:
            self.process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", str(self.port)],
                env=self.env, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
        wait_ready(self.base_url + "/health", self.process, log)
        return self

    def stop(self):
        if self.process:
            stop_owned(self.process)
            self.process = None

    def restart(self):
        self.stop()
        return self.start()

    def client(self):
        return httpx.Client(base_url=self.base_url, timeout=120, trust_env=False)


@pytest.fixture(scope="module")
def product(tmp_path_factory, ollama_port):
    lab = Lab(tmp_path_factory.mktemp("weekly-collection"), ollama_port)
    index = COLLECTION / "index/documents.sqlite3"
    if index.exists():
        with closing(sqlite3.connect(index.as_uri() + "?mode=ro", uri=True)) as source:
            with closing(sqlite3.connect(lab.document_db)) as target:
                source.backup(target)
    try:
        yield lab.start()
    finally:
        lab.stop()


@pytest.fixture
def lab(tmp_path, ollama_port):
    service = Lab(tmp_path, ollama_port)
    try:
        yield service
    finally:
        service.stop()


def json_ok(response):
    assert response.is_success, f"HTTP {response.status_code}: {response.text}"
    return response.json()


def require_model(client, model):
    models = json_ok(client.get("/models"))["models"]
    if model not in models:
        pytest.skip(f"Install {model} in the configured local Ollama to run this live case.")


def ask(client, model, prompt, path="/documents/ask", **fields):
    return json_ok(client.post(path, json={"prompt": prompt, "model": model, **fields}))


def upload(client, filename, data):
    return client.post("/documents", params={"filename": filename}, headers=APPROVAL, content=data)


def note(client, title, text, url=None, headers=APPROVAL):
    return client.post("/documents/notes", headers=headers, json={"title": title, "text": text, "url": url})


def reply_has(answer, *values):
    assert contains(normalized(answer["reply"]), values), f"Expected {values!r}. Reply: {answer['reply']}"


def abstains(answer):
    assert answer["sources"] == [] and any(phrase in answer["reply"].casefold()
                                          for phrase in ("couldn’t find", "couldn't find", "i don't know", "i don’t know")), answer


def run_ui(lab, scenario, model=""):
    if not (node := shutil.which("node")):
        pytest.skip("Node.js with fetch and TextDecoderStream is required for the frontend event-handler check.")
    result = subprocess.run([node, str(ROOT / "tests/weekly_ui_checks.cjs"), lab.base_url, scenario, model],
                            cwd=ROOT, capture_output=True, text=True, timeout=150)
    assert result.returncode == 0, result.stderr.strip() or result.stdout.strip()


def test_01_fastapi_contract(comparison, product, model):
    with comparison.check("01", model), product.client() as client:
        docs = client.get("/docs")
        assert docs.status_code == 200 and "Swagger UI" in docs.text
        assert json_ok(client.get("/health")) == {"status": "ok"}
        invalid = client.post("/chat", json={"prompt": " ", "model": model})
        assert invalid.status_code == 422
        assert any(error["loc"][-1] == "prompt" for error in invalid.json()["detail"])


def test_02a_running_ollama(comparison, lab, model):
    with comparison.check("02a", model), lab.start().client() as client:
        require_model(client, model)
        json_ok(upload(client, GUIDE, make_pdf([VPN_PAGE])))
        reply_has(ask(client, model, VPN_QUESTION), HOSTNAME)


def test_02b_stopped_ollama(comparison, tmp_path, model):
    with comparison.check("02b", model):
        port = free_port()
        app = Lab(tmp_path, port)
        with serve_ollama(tmp_path, port, tmp_path / "unused-models"):
            with httpx.Client(trust_env=False) as transport:
                assert transport.get(f"http://127.0.0.1:{port}/api/version").is_success
        try:
            with app.start().client() as client:
                json_ok(upload(client, GUIDE, make_pdf([VPN_PAGE])))
                response = client.post("/chat", json={"prompt": VPN_QUESTION, "model": model})
                assert response.status_code == 503, response.text
                assert "Start Ollama" in response.json()["detail"]
                assert json_ok(client.get("/health")) == {"status": "ok"}
        finally:
            app.stop()


@pytest.mark.parametrize("case_id", IT_IDS)
def test_03_06_same_it_questions(comparison, product, model, case_id):
    with comparison.check("03 / 06", model, case_id), product.client() as client:
        require_model(client, model)
        filename, question = QUESTION_ROWS[case_id]
        if filename not in {row["filename"] for row in json_ok(client.get("/documents"))["documents"]}:
            pytest.skip(f"The project index needs {filename} for this published benchmark case.")
        answer = ask(client, model, question["question"])
        reply_has(answer, *question["contains"])
        actual = {source["page"] for source in answer["sources"]}
        expected = [set(source["pages"]) for source in question["sources"]]
        assert all(source["filename"] == filename for source in answer["sources"]), answer
        assert actual <= set.union(*expected) and all(actual & pages for pages in expected), (
            f"Expected page groups {[sorted(pages) for pages in expected]} in {filename}. "
            f"Cited pages {sorted(actual)}."
        )


def test_04_sqlite_chat_survives_restart(comparison, lab, model):
    with comparison.check("04", model):
        initial = "For this conversation, my workstation inventory tag is FROST-691."
        with lab.start().client() as client:
            require_model(client, model)
            json_ok(client.post("/chat", json={"prompt": initial, "model": model}))
            lab.restart()
            recall = ask(client, model, "What workstation inventory tag did I give you in my first chat message?", "/chat")
            assert "FROST-691" in recall["reply"], f"Ordinary chat was unavailable after restarting the same SQLite-backed app: {recall['reply']}"


def test_05_uses_uploaded_guide(comparison, lab, model):
    with comparison.check("05", model), lab.start().client() as client:
        require_model(client, model)
        json_ok(upload(client, GUIDE, make_pdf([VPN_PAGE])))
        answer = ask(client, model, VPN_QUESTION)
        reply_has(answer, HOSTNAME)
        assert {"filename": GUIDE, "page": 1} in answer["sources"]


def test_07_switch_models_with_tab_history(comparison, lab):
    with comparison.check("07", "both"), lab.start().client() as client:
        for selected in MODELS:
            require_model(client, selected)
        json_ok(upload(client, GUIDE, make_pdf([VPN_PAGE])))
        run_ui(lab, "switch")


def seed_stream(client):
    json_ok(upload(client, GUIDE, make_pdf([
        VPN_PAGE, "FableDesk VPN reconnect\nThe reconnect wait time is 35 seconds.",
    ])))


def test_08a_incremental_answer_tokens(comparison, lab, model):
    with comparison.check("08a", model), lab.start().client() as client:
        require_model(client, model)
        seed_stream(client)
        prompts = (
            (VPN_QUESTION, (HOSTNAME,), 1),
            ("Which operating systems does the FableDesk VPN support?", ("macOS", "Windows"), 2),
        )
        for prompt, facts, minimum_chunks in prompts:
            with client.stream("POST", "/documents/ask/stream", json={"prompt": prompt, "model": model}) as response:
                response.raise_for_status()
                events = [json.loads(line) for line in response.iter_lines() if line]
            assert events and "result" in events[-1], events
            result = events[-1]["result"]
            chunks = [event["delta"] for event in events[:-1] if event.get("delta")]
            assert len(chunks) >= minimum_chunks, events
            assert "".join(chunks) == result["reply"]
            reply_has(result, *facts)
            assert {"filename": GUIDE, "page": 1} in result["sources"]


def test_08b_cancel_stream(comparison, lab, model):
    with comparison.check("08b", model), lab.start().client() as client:
        require_model(client, model)
        seed_stream(client)
        with client.stream("POST", "/documents/ask/stream", json={"prompt": VPN_QUESTION, "model": model}) as response:
            response.raise_for_status()
            events = response.iter_lines()
            assert "Searching" in next(events)
            assert "Checking evidence" in next(events)
            start = time.perf_counter()
        assert time.perf_counter() - start < 5, "Closing the stream hung"
        assert json_ok(client.get("/health", timeout=5)) == {"status": "ok"}
        reply_has(json_ok(client.post("/documents/ask", json={"prompt": VPN_QUESTION, "model": model}, timeout=45)), HOSTNAME)


@pytest.mark.parametrize("case_id", IT_IDS)
def test_09_ollama_performance(comparison, ollama_port, model, case_id):
    with comparison.check("09", model, case_id) as report:
        if not ollama_port:
            pytest.skip("Local Ollama is required for performance measurements.")
        with httpx.Client(base_url=f"http://127.0.0.1:{ollama_port}", timeout=120, trust_env=False) as client:
            installed = {item["name"] for item in json_ok(client.get("/api/tags"))["models"]}
            if model not in installed:
                pytest.skip(f"Install {model} to measure its performance.")
            payload = {"model": model, "think": False, "keep_alive": "5m",
                       "options": {"num_ctx": 8192, "num_predict": 96, "temperature": 0},
                       "messages": [{"role": "user", "content": QUESTION_ROWS[case_id][1]["question"]}]}
            if case_id == IT_IDS[0]:
                warmup = payload | {"stream": False, "options": payload["options"] | {"num_predict": 3}}
                json_ok(client.post("/api/chat", json=warmup))
            start = time.perf_counter()
            ttft = None
            final = {}
            with client.stream("POST", "/api/chat", json=payload | {"stream": True}) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    event = json.loads(line)
                    assert "error" not in event, event
                    if ttft is None and event.get("message", {}).get("content", "").strip():
                        ttft = time.perf_counter() - start
                    if event.get("done"):
                        final = event
            latency = time.perf_counter() - start
            active = next((row for row in json_ok(client.get("/api/ps"))["models"] if row["name"] == model), None)
            assert active and active["size"] > 0, "Ollama did not report memory for the loaded model."
            assert ttft is not None and final.get("eval_count", 0) > 0 and final.get("eval_duration", 0) > 0
            report.update(ttft=ttft, latency=latency, ram_mib=active["size"] / 1048576,
                          tokens_per_second=final["eval_count"] * 1e9 / final["eval_duration"])


def archived_checkpoint():
    archive = ROOT.parent / "Personal_Ai_archive"
    checkpoints = sorted(archive.glob("*/models/tiny-gpt-document-reader.pt"))
    assert checkpoints, f"The archived reader checkpoint is absent from {archive}."
    return checkpoints[-1]


def test_10a_archive_hidden_and_restorable(comparison, product, tmp_path):
    with comparison.check("10a", "both"), product.client() as client:
        assert "tinygpt" not in client.get("/").raise_for_status().text.casefold()
        assert not any("tiny" in model.lower() for model in json_ok(client.get("/models"))["models"])
        checkpoint = archived_checkpoint()
        assert (checkpoint.parents[1] / "README-Personal-AI.md").is_file(), "Archived setup instructions are absent."
        restored = tmp_path / checkpoint.name
        shutil.copyfile(checkpoint, restored)
        reader, saved = load_reader(restored)
        assert next(reader.parameters()).numel() and isinstance(saved, dict)


def test_10b_saved_archive_configuration(comparison):
    with comparison.check("10b", "both"):
        checkpoint = archived_checkpoint()
        _, saved = load_reader(checkpoint)
        config = saved.get("model_config") or saved.get("config")
        sidecar = checkpoint.with_suffix(".json")
        if not config and sidecar.is_file():
            config = json.loads(sidecar.read_text())
        assert isinstance(config, dict) and config, "The archived checkpoint needs a saved model configuration."


def test_11_short_term_conversation_fact(comparison, lab, model):
    with comparison.check("11", model), lab.start().client() as client:
        require_model(client, model)
        json_ok(upload(client, GUIDE, make_pdf([VPN_PAGE])))
        turns = [
            "My workstation asset tag is FROST-691. What is the FableDesk VPN gateway hostname?",
            "Which operating systems does the FableDesk VPN support?",
            "What hostname does the FableDesk VPN use?",
        ]
        recent = []
        for prompt in turns:
            result = ask(client, model, prompt, "/chat", previous_question="\n".join(recent) or None)
            if result["sources"]:
                recent = [prompt[:160], *recent][:3]
        recall = ask(client, model, "What was that workstation asset tag I told you?", "/chat",
                     previous_question="\n".join(recent))
        assert "FROST-691" in recall["reply"], f"Conversation-only fact was not recalled after three turns: {recall['reply']}"


def test_12_approved_knowledge_survives_restart(comparison, lab, model):
    with comparison.check("12", model):
        with lab.start().client() as old:
            require_model(old, model)
            json_ok(note(old, "FableDesk credential recovery", "# Recovery\nThe approved recovery reference is AMBER-827."))
        with lab.restart().client() as new:
            answer = ask(new, model, "What is the approved recovery reference?")
            reply_has(answer, "AMBER-827")
            assert any(source["filename"] == "Note - FableDesk credential recovery.md" for source in answer["sources"])


def test_13_twenty_turn_summary(comparison, lab, model):
    with comparison.check("13", model), lab.start().client() as client:
        turns = [
            "FableDesk printer incident INC-4782",
            "Sam approved the shift handover on Tuesday",
            *(f"Routine follow-up {index}" for index in range(3, 21)),
        ]
        for prompt in turns:
            ask(client, model, prompt, "/chat")
        history = json_ok(client.get("/conversations/current"))
        assert [turn["prompt"] for turn in history["turns"]] == turns
        result = json_ok(client.get("/conversations/current/summary"))
        assert result["turn_count"] == 20
        assert "INC-4782" in result["summary"] and "Sam approved" in result["summary"]
        assert all(line[2:] in turns for line in result["summary"].splitlines()[1:])
        assert ask(client, model, "Summarize our conversation", "/chat")["reply"] == result["summary"]
        json_ok(client.post("/conversations/new"))
        assert json_ok(client.get("/conversations/current"))["turns"] == []


def test_14_semantic_rephrasing(comparison, lab, ollama_port, model, monkeypatch):
    with comparison.check("14", model), lab.start().client() as client:
        require_model(client, model)
        filename = "weekly-credential-recovery.txt"
        text = "Workstation sign-in credential recovery. The recovery password is VIOLET-731."
        json_ok(upload(client, filename, text.encode()))
        distractor = make_pdf([f"Office inventory {i}. " + "Budget receipt shelving cabinet storage audit. " * 45
                               for i in range(7)])
        json_ok(upload(client, "weekly-distractors.pdf", distractor))
        monkeypatch.setenv("PERSONAL_AI_COLLECTION", "1")
        monkeypatch.setattr(documents, "DATABASE_PATH", lab.document_db)
        monkeypatch.setattr(rag_search, "EMBED_URL", f"http://127.0.0.1:{ollama_port}/api/embed")
        monkeypatch.setattr(rag_search, "_cached_rows", ())
        monkeypatch.setattr(rag_search, "_cached_vectors", [])
        question = "I cannot remember what gets me into my laptop. Which passphrase unlocks it?"
        with capture_trace() as trace:
            rows = asyncio.run(rag_search.retrieve(question, max_full_text_bytes=7000))
        assert any(row[0] == filename for row in rows), rows
        evidence = [row for event in trace if event["stage"] == "retrieval" for row in event["rows"]
                    if row["passage"][0] == filename]
        assert evidence and any(row["semantic_rank"] and row["keyword_rank"] is None for row in evidence), evidence
        reply_has(ask(client, model, question), "VIOLET-731")


def test_15a_normal_and_multipage_pdf(comparison, lab, model):
    with comparison.check("15a", model), lab.start().client() as client:
        single = json_ok(upload(client, "single.pdf", make_pdf([VPN_PAGE])))
        assert single["documents"][0]["pages"] == 1
        multi = json_ok(upload(client, "two.pdf", make_pdf(["Page one. ALPHA-128.", "Page two. BRAVO-268."])))
        assert next(row for row in multi["documents"] if row["filename"] == "two.pdf")["pages"] == 2
        with closing(sqlite3.connect(lab.document_db)) as db:
            assert db.execute("select page from passages where filename='two.pdf' order by page").fetchall() == [(1,), (2,)]


def require_ocr():
    if sys.platform != "darwin" or not shutil.which("swift"):
        pytest.skip("macOS Swift/Vision is required to exercise empty and image-only PDF ingestion.")


def test_15b_empty_and_corrupt_pdf(comparison, lab, model):
    with comparison.check("15b", model), lab.start().client() as client:
        corrupt = upload(client, "corrupt.pdf", b"%PDF-1.7\ncorrupt data")
        assert corrupt.status_code == 422 and "PDF" in corrupt.json()["detail"]
        require_ocr()
        empty = upload(client, "empty.pdf", make_pdf([""]))
        assert empty.status_code == 422 and "No readable text" in empty.json()["detail"]
        assert json_ok(client.get("/documents"))["documents"] == []


def test_15c_scanned_pdf(comparison, lab, model):
    with comparison.check("15c", model), lab.start().client() as client:
        require_ocr()
        path = lab.path / "scanned.pdf"
        subprocess.run(["swift", str(ROOT / "tests/weekly_scan.swift"), str(path)], env=lab.env,
                       cwd=ROOT, timeout=120, check=True, capture_output=True)
        data = path.read_bytes()
        assert not PdfReader(BytesIO(data)).pages[0].extract_text().strip(), "The fixture must be image-only."
        json_ok(upload(client, path.name, data))
        with closing(sqlite3.connect(lab.document_db)) as db:
            page, text = db.execute("select page, text from passages where filename=?", (path.name,)).fetchone()
        assert page == 1 and "ZEBRA 741" in text


def test_16a_website_excerpt_requires_approval(comparison, lab, model):
    with comparison.check("16a", model), lab.start().client() as client:
        require_model(client, model)
        url = "https://help.example.org/fabledesk"
        title, excerpt = "FableDesk Wi-Fi", "# SSID\nThe FableDesk wireless SSID is CEDAR-NET."
        assert note(client, title, excerpt, url, headers={}).status_code == 403
        assert json_ok(client.get("/documents"))["documents"] == []
        assert note(client, title, excerpt, "http://help.example.org/fabledesk").status_code == 422
        json_ok(note(client, title, excerpt, url))
        answer = ask(client, model, "What is the FableDesk wireless SSID?")
        reply_has(answer, "CEDAR-NET")
        assert any(source.get("url") == url and source.get("section") == "SSID" for source in answer["sources"])


def test_16b_approved_url_without_excerpt(comparison, tmp_path, monkeypatch, model):
    with comparison.check("16b", model):
        monkeypatch.setenv("PERSONAL_AI_COLLECTION", "1")
        monkeypatch.setenv("IT_SUPPORT_PRODUCT", "1")
        monkeypatch.setattr(documents, "DATABASE_PATH", tmp_path / "website.sqlite3")
        url = "https://help.example.org/fabledesk"
        downloads = []
        def fetch_page(source):
            downloads.append(source)
            return source, "# Support code\nThe published desk code is FOREST-285."
        monkeypatch.setattr(main, "read_website", fetch_page)
        payload = {"title": "FableDesk reference", "url": url}
        with TestClient(main.app) as client:
            assert client.post("/documents/notes", json=payload).status_code == 403
            assert downloads == []
            reply = json_ok(client.post("/documents/notes", headers=APPROVAL, json=payload))
        assert downloads == [url]
        assert reply["documents"][0]["filename"] == "Website - FableDesk reference.md"
        metadata = documents.source_metadata("Website - FableDesk reference.md", 1)
        assert metadata["url"] == url and metadata["section"] == "Support code"
        assert "FOREST-285" in documents.indexed_passages()[0][2]


def test_website_import_rejects_private_addresses_and_ignores_scripts():
    with pytest.raises(ValueError, match="publicly routed"):
        public_addresses("127.0.0.1")
    parser = ReadableHTML()
    parser.feed("<main><h1>VPN</h1><script>untrusted script</script><p>Open GlobalProtect.</p></main>")
    parser.close()
    parser.flush()
    assert parser.lines == ["# VPN", "Open GlobalProtect."]

def test_17_exact_pdf_page_and_markdown_section(comparison, lab, model):
    with comparison.check("17", model), lab.start().client() as client:
        require_model(client, model)
        json_ok(upload(client, "weekly-printer.pdf", make_pdf([
            "FableDesk printer. Paper trays must be loaded before use.",
            "FableDesk printer setup. The printer pairing code is MAPLE-463.",
        ])))
        json_ok(note(client, "FableDesk driver", "# Installation\nThe signed printer driver version is FIR-804."))
        pdf = ask(client, model, "What is the FableDesk printer pairing code?")
        reply_has(pdf, "MAPLE-463")
        assert pdf["sources"] == [{"filename": "weekly-printer.pdf", "page": 2}], pdf
        markdown = ask(client, model, "What is the signed printer driver version?")
        reply_has(markdown, "FIR-804")
        assert markdown["sources"] == [{"filename": "Note - FableDesk driver.md", "page": 1, "section": "Installation"}], markdown


def test_18_update_and_delete_changes_retrieval(comparison, lab, model):
    with comparison.check("18", model), lab.start().client() as client:
        require_model(client, model)
        name = "weekly-maintenance.txt"
        prompt = "What is the FableDesk maintenance passcode?"
        json_ok(upload(client, name, b"The FableDesk maintenance passcode is WILLOW-464."))
        original = ask(client, model, prompt)
        reply_has(original, "WILLOW-464")
        json_ok(upload(client, name, b"The FableDesk maintenance passcode is WILLOW-512."))
        updated = ask(client, model, prompt)
        reply_has(updated, "WILLOW-512")
        assert "WILLOW-464" not in updated["reply"]
        json_ok(client.delete("/documents", params={"filename": name}))
        abstains(ask(client, model, prompt))


def test_19_permission_dialog_and_direct_denial(comparison, lab, model):
    with comparison.check("19", model), lab.start().client() as client:
        denied = client.post("/documents", params={"filename": "denied.txt"}, content=b"NIGHT-272",
                             headers={"X-Knowledge-Approval": "no"})
        assert denied.status_code == 403
        assert json_ok(client.get("/documents"))["documents"] == []
        run_ui(lab, "permission", model)


@pytest.mark.parametrize("case_id", MISSING_IDS)
def test_out_of_document_questions_abstain(comparison, product, model, case_id):
    with comparison.check("Hallucination", model, case_id), product.client() as client:
        require_model(client, model)
        filename, question = QUESTION_ROWS[case_id]
        if filename not in {row["filename"] for row in json_ok(client.get("/documents"))["documents"]}:
            pytest.skip(f"The project index needs {filename} for this missing-fact case.")
        abstains(ask(client, model, question["question"]))
