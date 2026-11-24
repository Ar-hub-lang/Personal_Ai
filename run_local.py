import http.client
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
        connection = http.client.HTTPConnection(HOST, port, timeout=0.5)
        try:
            connection.request("GET", path)
            if connection.getresponse().status == 200:
                return
        except (OSError, http.client.HTTPException):
            pass
        finally:
            connection.close()

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
        if not OLLAMA.is_file():
            raise RuntimeError("Ollama is missing. Follow the setup steps in README.md.")

        for name, port in (("Ollama", OLLAMA_PORT), ("FastAPI", API_PORT)):
            if port_in_use(port):
                raise RuntimeError(f"Port {port} is in use. Stop the existing {name} server and try again.")

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

        services["FastAPI"] = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn", "backend.main:app",
                "--host", HOST, "--port", str(API_PORT), "--reload",
            ],
            cwd=PROJECT_ROOT,
            start_new_session=True,
        )
        wait_until_ready("FastAPI", API_PORT, "/health", services, should_stop)
        if not stopping:
            print(f"Personal AI is ready at http://{HOST}:{API_PORT}/", flush=True)
            print("Press Ctrl+C to stop both servers.", flush=True)
        while not stopping:
            check_processes(services)
            time.sleep(0.25)
        return 0
    except (OSError, RuntimeError) as error:
        print(f"Personal AI: {error}", file=sys.stderr)
        return 1
    finally:
        if services:
            print("Stopping Personal AI servers…", flush=True)
            stop_services(services)


if __name__ == "__main__":
    raise SystemExit(main())
