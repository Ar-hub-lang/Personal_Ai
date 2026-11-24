"""Loopback addresses shared by the runner, chat, and PDF retrieval."""

import os


def local_port(name: str, default: int) -> int:
    port = int(os.environ.get(name, str(default)))
    if not 1 <= port <= 65535:
        raise ValueError(f"{name} must be between 1 and 65535.")
    return port


OLLAMA_PORT = local_port("PERSONAL_AI_OLLAMA_PORT", 11434)
OLLAMA_URL = f"http://127.0.0.1:{OLLAMA_PORT}"
