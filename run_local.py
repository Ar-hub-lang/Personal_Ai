import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from backend.local_settings import OLLAMA_PORT, local_port

PROJECT_ROOT = Path(__file__).resolve().parent
OLLAMA = PROJECT_ROOT / ".ollama-runtime" / "ollama"
HOST = "127.0.0.1"
API_PORT = local_port("PERSONAL_AI_API_PORT", 8000)


def port_in_use(port: int) -> bool:
    try:
        with socket.create_connection((HOST, port), timeout=0.3):
            return True
    except OSError:
        return False


def service_ready(port: int, path: str, *, require_version: bool = False) -> bool:
    connection = http.client.HTTPConnection(HOST, port, timeout=0.5)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        if response.status != 200:
            return False
        if require_version:
            info = json.loads(response.read(8192))
            return isinstance(info, dict) and isinstance(info.get("version"), str) and bool(info["version"])
        return True
    except (OSError, http.client.HTTPException, ValueError):
        return False
    finally:
        connection.close()


def wait_until_ready(
    name: str,
    port: int,
    path: str,
    services: dict[str, subprocess.Popen],
    should_stop: Callable[[], bool],
) -> None:
    deadline = time.monotonic() + 45
    while not should_stop():
        check_processes(services)
        if service_ready(port, path):
            return

        if time.monotonic() >= deadline:
            raise RuntimeError(f"{name} did not become ready on port {port}.")
        time.sleep(0.2)


def check_processes(services: dict[str, subprocess.Popen]) -> None:
    for name, process in services.items():
        status = process.poll()
        if status is not None:
            raise RuntimeError(f"{name} exited with status {status}.")


def signal_group(process: subprocess.Popen, signum: int) -> bool:
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        return False
    return True


def stop_services(services: dict[str, subprocess.Popen]) -> None:
    processes = list(reversed(services.values()))
    if sys.platform == "win32":
        for process in processes:
            # The console sends Ctrl+C to FastAPI as well as the launcher.
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait()
        return
    for process in processes:
        signal_group(process, signal.SIGTERM)

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        for process in processes:
            process.poll()
        if not any(signal_group(process, 0) for process in processes):
            break
        time.sleep(0.1)
    else:
        for process in processes:
            signal_group(process, signal.SIGKILL)

    for process in processes:
        process.wait()


def main() -> int:
    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True

    def should_stop() -> bool:
        return stopping

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    services = {}
    try:
        windows = sys.platform == "win32"
        if port_in_use(API_PORT):
            raise RuntimeError(f"Port {API_PORT} is in use. Stop the existing FastAPI server and try again.")
        if windows:
            if not service_ready(OLLAMA_PORT, "/api/version", require_version=True):
                raise RuntimeError(f"Start the Ollama Windows app and check that it is ready on port {OLLAMA_PORT}.")
        else:
            if not OLLAMA.is_file():
                raise RuntimeError("Ollama is missing. Follow the setup steps in README.md.")
            if port_in_use(OLLAMA_PORT):
                raise RuntimeError(f"Port {OLLAMA_PORT} is in use. Stop the existing Ollama server and try again.")

            models = PROJECT_ROOT / "models"
            models.mkdir(exist_ok=True)
            ollama_env = os.environ | {
                "OLLAMA_HOST": f"{HOST}:{OLLAMA_PORT}",
                "OLLAMA_MODELS": str(models),
                "OLLAMA_NO_CLOUD": "1",
                "OLLAMA_NUM_PARALLEL": "1",
            }
            services["Ollama"] = subprocess.Popen(
                [str(OLLAMA), "serve"], cwd=PROJECT_ROOT, env=ollama_env, start_new_session=True
            )
            wait_until_ready("Ollama", OLLAMA_PORT, "/api/version", services, should_stop)
        if stopping:
            return 0

        api_env = os.environ.copy()
        api_env.setdefault("PERSONAL_AI_COLLECTION", "1")
        if api_env["PERSONAL_AI_COLLECTION"] == "1":
            api_env.setdefault("IT_SUPPORT_PRODUCT", "1")
            index = PROJECT_ROOT / "collections" / "it-support" / "index" / "documents.sqlite3"
            api_env["PERSONAL_AI_DOCUMENT_DB"] = api_env.get("PERSONAL_AI_DOCUMENT_DB") or str(index)
            Path(api_env["PERSONAL_AI_DOCUMENT_DB"]).parent.mkdir(parents=True, exist_ok=True)
            api_env.setdefault("RAG_PARTWISE_EVIDENCE", "1")

        api_args = [
            sys.executable, "-m", "uvicorn", "backend.main:app",
            "--host", HOST, "--port", str(API_PORT),
        ]
        api_args.extend(("--workers", "1") if windows else ("--reload",))
        services["FastAPI"] = subprocess.Popen(
            api_args, cwd=PROJECT_ROOT, env=api_env, start_new_session=not windows,
        )
        wait_until_ready("FastAPI", API_PORT, "/health", services, should_stop)
        if not stopping:
            product = api_env["PERSONAL_AI_COLLECTION"] == "1" and api_env.get("IT_SUPPORT_PRODUCT") == "1"
            name = "IT Support" if product else "Personal AI"
            print(f"{name} is ready at http://{HOST}:{API_PORT}/", flush=True)
            if windows:
                print("Press Ctrl+C to stop FastAPI. Manage Ollama from the Windows tray.", flush=True)
            else:
                print("Press Ctrl+C to stop both servers.", flush=True)
        while not stopping:
            check_processes(services)
            time.sleep(0.25)
        return 0
    except (OSError, RuntimeError) as error:
        print(f"Local launcher: {error}", file=sys.stderr)
        return 1
    finally:
        if services:
            print("Stopping FastAPI…" if windows else "Stopping local servers…", flush=True)
            stop_services(services)


if __name__ == "__main__":
    raise SystemExit(main())
