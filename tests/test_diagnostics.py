import asyncio
import sqlite3
import subprocess
from contextlib import closing

import httpx
import pytest

from backend import diagnostics, documents, main


def make_request(method, path, *, peer="127.0.0.1", host="127.0.0.1", **kwargs):
    async def call():
        transport = httpx.ASGITransport(app=main.app, client=(peer, 50133))
        async with httpx.AsyncClient(transport=transport, base_url=f"http://{host}") as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(call())


def test_catalog_lists_the_fourteen_week_3_checks():
    data = make_request("GET", "/diagnostics").raise_for_status().json()
    assert [check["feature"] for check in data["checks"]] == list(range(20, 34))
    assert len({check["id"] for check in data["checks"]}) == 14


@pytest.mark.parametrize("options", [
    {"peer": "192.0.2.10"},
    {"host": "other.example"},
    {"headers": {"origin": "http://other.example"}},
])
def test_rejects_nonlocal_access(options):
    response = make_request("POST", "/diagnostics/run", json={"check": "cpu"}, **options)
    assert response.status_code == 403


@pytest.mark.parametrize("payload", [
    {"check": "sh"},
    {"check": "ports", "port": 65536},
    {"check": "ports", "port": "8000"},
    {"check": "ports", "target": "example.com"},
    {"check": "hardware", "port": 8000},
    {"check": "logs", "command": "cat"},
])
def test_rejects_non_catalog_checks_and_invalid_parameters(payload):
    response = make_request("POST", "/diagnostics/run", json=payload)
    assert response.status_code == 422


def test_ports_report_named_process_owners(monkeypatch):
    monkeypatch.setattr(diagnostics.platform, "system", lambda: "Darwin")

    def command(*args, **options):
        assert "-iTCP:9001" in args
        return "p305\ncOllama\np305\ncOllama\np407\ncpython"

    monkeypatch.setattr(diagnostics, "command", command)
    data = make_request("POST", "/diagnostics/run", json={"check": "ports", "port": 9001}).json()
    assert data["finding"]["details"] == {"TCP 9001": "Ollama (PID 305), python (PID 407)"}


def test_log_counts_exclude_messages(monkeypatch):
    monkeypatch.setattr(diagnostics.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(diagnostics, "command", lambda *a, **kw: (
        '{"messageType":"Error","eventMessage":"example private log"}\n'
        '{"messageType":"Fault","eventMessage":"example private log"}\n'
        '{"messageType":"Default","eventMessage":"example private log"}\n'
    ))
    response = make_request("POST", "/diagnostics/run", json={"check": "logs"})
    assert response.json()["finding"]["details"] == {"Errors": 1, "Faults": 1}
    assert "private log" not in response.text


def test_database_handles_missing_and_open_indexes_without_writing(monkeypatch, tmp_path):
    monkeypatch.setattr(diagnostics.platform, "system", lambda: "Darwin")
    path = tmp_path / "name with spaces.sqlite3"
    monkeypatch.setattr(documents, "DATABASE_PATH", path)
    body = {"check": "database"}
    assert make_request("POST", "/diagnostics/run", json=body).json()["finding"]["status"] == "unavailable"
    assert not path.exists()
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("CREATE TABLE fixture (value TEXT)")
        db.execute("INSERT INTO fixture VALUES ('private fixture')")
    before = path.read_bytes()
    response = make_request("POST", "/diagnostics/run", json=body)
    assert response.json()["finding"]["details"]["Schema objects"] == 1
    assert "private fixture" not in response.text
    assert path.read_bytes() == before


def test_timeout_omits_command_and_releases_check_slot(monkeypatch):
    monkeypatch.setattr(diagnostics.platform, "system", lambda: "Darwin")

    def timeout(*args, **options):
        raise subprocess.TimeoutExpired("sensitive details", 5)

    monkeypatch.setattr(diagnostics, "command", timeout)
    for _ in range(3):
        response = make_request("POST", "/diagnostics/run", json={"check": "cpu"})
        assert response.json()["finding"]["status"] == "unavailable"
        assert "sensitive details" not in response.text


def test_docker_accesses_only_the_local_socket(monkeypatch, tmp_path):
    monkeypatch.setattr(diagnostics.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(diagnostics.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(diagnostics.Path, "is_socket", lambda path: path == tmp_path / ".docker/run/docker.sock")
    monkeypatch.setattr(diagnostics.shutil, "which", lambda tool: "/local/bin/docker")

    def command(*args, **options):
        assert str(tmp_path / ".docker/run/docker.sock") in args
        assert args[-1] == "http://localhost/_ping"
        assert "--disable" in args
        return "OK"

    monkeypatch.setattr(diagnostics, "command", command)
    response = make_request("POST", "/diagnostics/run", json={"check": "docker"})
    assert response.json()["finding"]["status"] == "ok"
