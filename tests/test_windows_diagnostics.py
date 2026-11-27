import asyncio
import base64
import os
import platform
import socket
import subprocess

import httpx
import pytest

from backend import diagnostics, main, windows_diagnostics
import run_local


def run_check(check, **fields):
    async def request():
        transport = httpx.ASGITransport(app=main.app, client=("127.0.0.1", 51111))
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
            response = await client.post("/diagnostics/run", json={"check": check, **fields})
            assert response.headers["Cache-Control"] == "no-store"
            return response.raise_for_status().json()["finding"]

    return asyncio.run(request())


@pytest.mark.parametrize("check, sample, status, expected", [
    ("hardware", {"OS": "Windows 11", "Build": "26200", "Model": "Example PC", "Cores": 8},
     "ok", {"OS": "Windows 11", "Logical CPUs": 8}),
    ("cpu", {"Busy": 95}, "attention", {"Busy %": 95, "Idle %": 5}),
    ("memory", {"TotalKiB": 8 * 2**20, "FreeKiB": 2**20},
     "attention", {"Physical RAM (GiB)": 8, "Available %": 12.5}),
    ("network", {"0.0.0.0/0": "Wi-Fi", "::/0": None},
     "ok", {"IPv4": "Wi-Fi", "IPv6": "No default interface reported"}),
    ("network", {"0.0.0.0/0": None, "::/0": None},
     "attention", {"IPv4": "No default interface reported"}),
    ("dns", {"Configured": 1, "Answers": 0}, "attention", {"Addresses returned": 0}),
    ("logs", {"Errors": 3, "Critical": 2, "LimitReached": False}, "ok", {"Errors": 3, "Critical": 2}),
])
def test_windows_api_uses_native_collectors(monkeypatch, check, sample, status, expected):
    monkeypatch.setattr(diagnostics.platform, "system", lambda: "Windows")
    monkeypatch.setattr(windows_diagnostics, "powershell", lambda script: sample)
    result = run_check(check)
    assert result["status"] == status
    assert expected.items() <= result["details"].items()


def test_windows_ports_use_numeric_local_filter_and_handle_no_owner(monkeypatch):
    monkeypatch.setattr(diagnostics.platform, "system", lambda: "Windows")

    def powershell(script):
        assert "$ports = @(9001)" in script
        assert "Get-NetTCPConnection" in script
        return {"Owners": [{"Port": 9001, "Name": None, "Pid": 1090}]}

    monkeypatch.setattr(windows_diagnostics, "powershell", powershell)
    assert run_check("ports", port=9001)["details"] == {"TCP 9001": "Unknown process (PID 1090)"}
    monkeypatch.setattr(windows_diagnostics, "powershell", lambda script: {"Owners": []})
    assert run_check("ports", port=9001)["details"] == {"TCP 9001": "No visible listener"}


def test_windows_process_monitor_normalizes_cpu_and_reports_working_set(monkeypatch):
    monkeypatch.setattr(diagnostics.platform, "system", lambda: "Windows")
    monkeypatch.setattr(windows_diagnostics.os, "cpu_count", lambda: 4)
    monkeypatch.setattr(windows_diagnostics, "powershell", lambda script: {"Processes": [
        {"Name": "python", "Pid": 42, "Cpu": 200, "Bytes": 64 * 2**20},
    ]})
    assert run_check("processes")["details"] == {"python (PID 42)": "50.0% CPU, 64.0 MiB RAM"}


def test_windows_powershell_uses_profile_free_utf8_json(monkeypatch):
    def run(argv, **kwargs):
        assert argv[1:5] == ["-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand"]
        script = base64.b64decode(argv[-1]).decode("utf-16-le")
        assert "OutputEncoding" in script and "Get-CimInstance" in script
        assert kwargs["encoding"] == "utf-8-sig"
        return subprocess.CompletedProcess(argv, 0, '{"OS":"Windows"}', "")

    monkeypatch.setattr(windows_diagnostics.subprocess, "run", run)
    assert windows_diagnostics.powershell("Get-CimInstance") == {"OS": "Windows"}


def test_windows_docker_uses_only_local_named_pipes(monkeypatch):
    monkeypatch.setattr(diagnostics.platform, "system", lambda: "Windows")
    monkeypatch.setattr(windows_diagnostics.shutil, "which", lambda name: "docker.exe")
    monkeypatch.setenv("DOCKER_CONTEXT", "remote")
    monkeypatch.setenv("DOCKER_HOST", "ssh://other.example")
    tried = []

    def run(argv, *, env, **kwargs):
        assert "DOCKER_CONTEXT" not in env and "DOCKER_HOST" not in env
        assert argv[1] == "--host" and argv[2].startswith("npipe:////./pipe/")
        tried.append(argv[2])
        return subprocess.CompletedProcess(argv, 1, "", "") if len(tried) == 1 else subprocess.CompletedProcess(
            argv, 0, "29.8.0", "",
        )

    monkeypatch.setattr(windows_diagnostics.subprocess, "run", run)
    assert run_check("docker")["details"] == {"Docker CLI installed": True, "Engine version": "29.8.0"}
    assert tried == ["npipe:////./pipe/docker_engine", "npipe:////./pipe/dockerDesktopLinuxEngine"]


def test_windows_launch_uses_existing_ollama_and_owns_fastapi_only(monkeypatch, tmp_path):
    monkeypatch.setattr(run_local.sys, "platform", "win32")
    monkeypatch.setattr(run_local, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(run_local, "port_in_use", lambda port: False)
    monkeypatch.delenv("PERSONAL_AI_DOCUMENT_DB", raising=False)
    handlers = {}
    calls = []

    class APIProcess:
        def poll(self):
            return None

        def terminate(self):
            calls.append("terminate API")

        def wait(self, timeout=None):
            assert timeout == 3
            calls.append("wait for API")

    def service_ready(port, path, *, require_version):
        assert (port, path, require_version) == (run_local.OLLAMA_PORT, "/api/version", True)
        return True

    def start(argv, **kwargs):
        assert "backend.main:app" in argv and "--reload" not in argv
        assert argv[-2:] == ["--workers", "1"]
        assert kwargs["start_new_session"] is False
        calls.append("start API")
        return APIProcess()

    def wait(name, port, path, services, should_stop):
        assert name == "FastAPI" and list(services) == ["FastAPI"]
        handlers[run_local.signal.SIGINT](None, None)

    monkeypatch.setattr(run_local.signal, "signal", handlers.__setitem__)
    monkeypatch.setattr(run_local, "service_ready", service_ready)
    monkeypatch.setattr(run_local.subprocess, "Popen", start)
    monkeypatch.setattr(run_local, "wait_until_ready", wait)
    assert run_local.main() == 0
    assert calls == ["start API", "wait for API"]


@pytest.mark.skipif(platform.system() != "Windows", reason="Requires native Windows PowerShell and CIM")
@pytest.mark.parametrize("check", ["hardware", "cpu", "memory", "disk", "network", "dns", "processes", "logs"])
def test_native_windows_collectors(check):
    finding = run_check(check)
    assert finding["status"] in {"ok", "attention"}, finding
    assert finding["details"]


@pytest.mark.skipif(platform.system() != "Windows", reason="Requires the Windows TCP table")
def test_native_windows_listener_owner():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        finding = run_check("ports", port=port)
        assert f"PID {os.getpid()}" in finding["details"][f"TCP {port}"]
    assert run_check("ports", port=port)["details"] == {f"TCP {port}": "No visible listener"}
