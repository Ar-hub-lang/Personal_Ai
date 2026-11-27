"""On-demand, read-only checks for the computer running IT Support."""

import json
import os
import platform
import re
import shutil
import sqlite3
import subprocess
import sys
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from threading import BoundedSemaphore
from typing import Annotated, Callable

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from backend import documents, windows_diagnostics
from backend.diagnostic_result import CheckUnavailable, Finding
from backend.local_settings import OLLAMA_PORT, local_port

PROJECT = Path(__file__).resolve().parents[1]
LOOPBACK_NAMES = {"127.0.0.1", "::1", "localhost"}
DNS_TEST_NAME = "example.com"
CHECK_SLOTS = BoundedSemaphore(2)


@dataclass(frozen=True)
class Check:
    feature: int
    label: str
    run: Callable[..., Finding]


def command(*args: str, timeout: int = 5, accepted: tuple[int, ...] = (0,)) -> str:
    env = os.environ.copy() if platform.system() == "Windows" else {
        "HOME": str(Path.home()),
        "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
    }
    env.update(LC_ALL="C", GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    result = subprocess.run(
        args, cwd=PROJECT, capture_output=True, text=True, errors="replace", timeout=timeout, env=env,
    )
    if result.returncode not in accepted or (result.returncode != 0 and result.stderr.strip()):
        raise CheckUnavailable(f"{Path(args[0]).name} could not complete the check.")
    return result.stdout.strip() if result.returncode == 0 else ""


def sysctl(name: str) -> str:
    return command("/usr/sbin/sysctl", "-n", name)


def machine() -> Finding:
    version = platform.mac_ver()[0]
    return Finding("Detected the host and operating system.", {
        "OS": f"macOS {version}", "Architecture": platform.machine(),
        "Model": sysctl("hw.model"), "Logical CPUs": os.cpu_count() or 0,
    })


def cpu() -> Finding:
    samples = re.findall(
        r"CPU usage: (\d+(?:\.\d+)?)% user,\s*(\d+(?:\.\d+)?)% sys,\s*(\d+(?:\.\d+)?)% idle",
        command("/usr/bin/top", "-l", "2", "-n", "0", "-s", "1"),
    )
    if len(samples) < 2:
        raise CheckUnavailable("macOS did not return two CPU samples.")
    user, system, idle = map(float, samples[-1])
    busy = round(100 - idle, 2)
    return Finding("Sampled CPU usage over one second.", {
        "Busy %": busy, "User %": user, "System %": system, "Idle %": idle,
    }, "attention" if busy >= 90 else "ok")


def memory() -> Finding:
    pressure = command("/usr/bin/memory_pressure", "-Q")
    match = re.search(r"System-wide memory free percentage:\s*(\d+)%", pressure)
    if not match:
        raise CheckUnavailable("macOS did not return its memory pressure estimate.")
    free = int(match[1])
    return Finding("Read physical RAM and macOS’s memory pressure estimate.", {
        "Physical RAM (GiB)": round(int(sysctl("hw.memsize")) / 2**30, 2),
        "macOS memory free estimate %": free,
    }, "attention" if free < 15 else "ok")


def disk() -> Finding:
    total, used, free = shutil.disk_usage(PROJECT)
    used_pct = round(100 * used / total, 1)
    return Finding("Measured the volume that holds the IT Support project.", {
        "Capacity (GiB)": round(total / 2**30, 2), "Available (GiB)": round(free / 2**30, 2),
        "Used %": used_pct,
    }, "attention" if free < 5 * 2**30 or used_pct >= 90 else "ok")


def default_interface(family: str) -> str:
    output = command("/sbin/route", "-n", "get", family, "default", accepted=(0, 1))
    match = re.search(r"^\s*interface:\s*(\S+)", output, re.M)
    return match[1] if match else ""


def network() -> Finding:
    routes = {}
    for family in ("inet", "inet6"):
        try:
            routes[family] = default_interface(f"-{family}")
        except CheckUnavailable:
            routes[family] = ""
    details = {label: routes[family] or "No default interface reported"
               for family, label in (("inet", "IPv4"), ("inet6", "IPv6"))}
    if not any(routes.values()):
        return Finding("No default network route was found.", details, "attention")
    return Finding("Found a default network route.", details)


def dns() -> Finding:
    config = command("/usr/sbin/scutil", "--dns")
    resolvers = len(re.findall(r"^resolver #\d+", config, re.M))
    resolution = command("/usr/bin/dscacheutil", "-q", "host", "-a", "name", DNS_TEST_NAME)
    answers = len(re.findall(r"^(?:ip_address|ipv6_address):", resolution, re.M))
    return Finding(
        "The system resolver returned an address." if answers else "The system resolver returned no addresses.",
        {"Test hostname": DNS_TEST_NAME, "Configured resolver entries": resolvers, "Addresses returned": answers},
        "ok" if answers else "attention",
    )


def listeners(port: int) -> list[str]:
    # A listener can report multiple sockets. Deduplicate by process ID.
    lines = command("/usr/sbin/lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fpc", accepted=(0, 1))
    processes = {}
    pid = None
    for line in lines.splitlines():
        if line.startswith("p") and line[1:].isdigit():
            pid = line[1:]
        elif line.startswith("c") and pid is not None:
            processes[pid] = line[1:]
    return [f"{name} (PID {pid})" for pid, name in list(processes.items())[:8]]


def ports(port: int | None) -> Finding:
    targets = [port] if port is not None else sorted({local_port("PERSONAL_AI_API_PORT", 8000), OLLAMA_PORT})
    owners = windows_diagnostics.listeners(targets) if platform.system() == "Windows" else {
        target: listeners(target) for target in targets
    }
    details = {f"TCP {target}": ", ".join(owners[target]) or "No visible listener" for target in targets}
    return Finding("Checked TCP listeners and their process owners.", details)


def processes() -> Finding:
    output = command("/bin/ps", "-axo", "pid=,pcpu=,pmem=,comm=")
    rows = []
    for line in output.splitlines():
        columns = line.split(None, 3)
        if len(columns) == 4 and columns[0].isdigit():
            pid, cpu_value, memory_value, executable = columns
            rows.append((float(cpu_value), float(memory_value), int(pid), Path(executable).name))
    if not rows:
        raise CheckUnavailable("No process measurements were returned.")
    highest = sorted(rows, reverse=True)[:5]
    return Finding("Listed the five visible processes with the highest sampled CPU usage.", {
        f"{name} (PID {pid})": f"{cpu_value:g}% CPU, {memory_value:g}% RAM"
        for cpu_value, memory_value, pid, name in highest
    })


def system_logs() -> Finding:
    output = command(
        "/usr/bin/log", "show", "--last", "2m", "--no-pager", "--style", "ndjson",
        "--predicate", '(logType == "error" OR logType == "fault")', timeout=10,
    )
    counts = {"Error": 0, "Fault": 0}
    for row in output.splitlines():
        try:
            level = json.loads(row).get("messageType")
        except (ValueError, AttributeError):
            continue
        if level in counts:
            counts[level] += 1
    return Finding("Counted macOS log errors and faults from the last two minutes.", {
        "Errors": counts["Error"], "Faults": counts["Fault"],
    })


def git() -> Finding:
    binary = shutil.which("git.exe") if platform.system() == "Windows" else "/usr/bin/git"
    if not binary:
        raise CheckUnavailable("Git was not found on PATH.")
    args = (binary, "--no-optional-locks", "-C", str(PROJECT))
    try:
        head = command(*args, "rev-parse", "--short", "HEAD")
    except CheckUnavailable:
        head = "No readable commit"
    branch = command(*args, "branch", "--show-current")
    conflicts = {row.split("\t", 1)[-1] for row in command(*args, "ls-files", "-u").splitlines()}
    git_dirs = command(
        *args, "rev-parse", "--path-format=absolute",
        "--git-path", "MERGE_HEAD", "--git-path", "rebase-merge",
        "--git-path", "rebase-apply", "--git-path", "CHERRY_PICK_HEAD",
    ).splitlines()
    operations = [Path(path).name for path in git_dirs if Path(path).exists()]
    return Finding("Checked the project’s Git branch, index conflicts and in-progress operations.", {
        "Branch": branch or "Detached HEAD", "Commit": head,
        "Files with unresolved conflicts": len(conflicts), "In-progress markers": ", ".join(operations) or "None",
    }, "attention" if conflicts or operations or not branch else "ok")


def python_environment() -> Finding:
    in_venv = sys.prefix != sys.base_prefix
    return Finding("Inspected the Python runtime serving IT Support.", {
        "Version": platform.python_version(), "Virtual environment": in_venv,
        "Environment": Path(sys.prefix).name,
    }, "ok" if in_venv else "attention")


def node() -> Finding:
    tools = ("node", "npm")
    executables = ("node.exe", "npm.cmd") if platform.system() == "Windows" else tools
    paths = {name: shutil.which(executable) for name, executable in zip(tools, executables)}
    versions = {name: command(path, "--version") if path else "Not found"
                for name, path in paths.items()}
    return Finding("Checked the installed Node.js and npm versions.", versions,
                   "ok" if all(paths.values()) else "unavailable")


def database() -> Finding:
    index = documents.DATABASE_PATH.resolve()
    if not index.is_file():
        return Finding("The IT Support SQLite index has not been created.", status="unavailable")
    try:
        with closing(sqlite3.connect(f"{index.as_uri()}?mode=ro", uri=True, timeout=1)) as db:
            db.execute("PRAGMA query_only = ON")
            schema_objects = db.execute("SELECT count(*) FROM sqlite_schema").fetchone()[0]
    except sqlite3.Error as error:
        raise CheckUnavailable("The IT Support SQLite index could not be read.") from error
    return Finding("Opened the IT Support SQLite index in read-only mode and queried its schema.", {
        "SQLite version": sqlite3.sqlite_version, "Schema objects": schema_objects,
    })


def docker() -> Finding:
    cli = shutil.which("docker")
    sockets = (Path.home() / ".docker/run/docker.sock", Path("/var/run/docker.sock"))
    for location in sockets:
        if not location.is_socket():
            continue
        try:
            reply = command(
                "/usr/bin/curl", "--disable", "--silent", "--show-error", "--fail", "--noproxy", "*",
                "--max-time", "2", "--unix-socket", str(location), "http://localhost/_ping", timeout=3,
            )
        except CheckUnavailable:
            continue
        if reply == "OK":
            return Finding("The local Docker engine responded.", {"Docker CLI installed": bool(cli)})
    return Finding("No local Docker engine responded.", {"Docker CLI installed": bool(cli)}, "unavailable")


CHECKS = {
    "hardware": Check(20, "Hardware and OS", machine),
    "cpu": Check(21, "CPU", cpu),
    "memory": Check(22, "RAM", memory),
    "disk": Check(23, "Disk space", disk),
    "network": Check(24, "Network routes", network),
    "dns": Check(25, "DNS lookup", dns),
    "ports": Check(26, "TCP port usage", ports),
    "processes": Check(27, "Processes", processes),
    "logs": Check(28, "System log counts", system_logs),
    "git": Check(29, "Git", git),
    "python": Check(30, "Python environment", python_environment),
    "node": Check(31, "Node.js and npm", node),
    "database": Check(32, "SQLite connectivity", database),
    "docker": Check(33, "Docker engine", docker),
}


WINDOWS_CHECKS = {
    "hardware": windows_diagnostics.machine,
    "cpu": windows_diagnostics.cpu,
    "memory": windows_diagnostics.memory,
    "network": windows_diagnostics.network,
    "dns": windows_diagnostics.dns,
    "processes": windows_diagnostics.processes,
    "logs": windows_diagnostics.system_logs,
    "docker": windows_diagnostics.docker,
}


class DiagnosticRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check: Annotated[str, Field(min_length=1, max_length=32)]
    port: Annotated[int, Field(strict=True, ge=1, le=65535)] | None = None


router = APIRouter(prefix="/diagnostics", tags=["diagnostics"])


def require_local(request: Request) -> None:
    # Check both the peer and the Host header so a public hostname cannot use DNS rebinding.
    origin = request.headers.get("origin")
    if (request.client is None or request.client.host not in LOOPBACK_NAMES
            or request.url.hostname not in LOOPBACK_NAMES
            or (origin and origin != f"{request.url.scheme}://{request.url.netloc}")):
        raise HTTPException(403, "Diagnostics are available only from the local app.")


@router.get("")
def diagnostic_catalog(request: Request) -> dict:
    require_local(request)
    return {"checks": [
        {"id": key, "feature": check.feature, "label": check.label}
        for key, check in CHECKS.items()
    ]}


@router.post("/run")
def run_diagnostic(request: Request, response: Response, body: DiagnosticRequest) -> dict:
    require_local(request)
    check = CHECKS.get(body.check)
    if check is None:
        raise HTTPException(422, "Choose a diagnostic from the list.")
    if body.port is not None and body.check != "ports":
        raise HTTPException(422, "The port field applies only to TCP port usage.")
    if not CHECK_SLOTS.acquire(blocking=False):
        raise HTTPException(429, "Other diagnostics are running. Try again in a moment.")
    try:
        system = platform.system()
        if system not in {"Darwin", "Windows"}:
            finding = Finding("Diagnostics support macOS and Windows.", status="unavailable")
        else:
            collect = WINDOWS_CHECKS.get(body.check, check.run) if system == "Windows" else check.run
            finding = collect(body.port) if body.check == "ports" else collect()
    except subprocess.TimeoutExpired:
        finding = Finding("The check timed out.", status="unavailable")
    except CheckUnavailable as error:
        finding = Finding(str(error), status="unavailable")
    except (OSError, ValueError, TypeError, KeyError):
        finding = Finding("The local tool or its measurements are unavailable.", status="unavailable")
    finally:
        CHECK_SLOTS.release()
    response.headers["Cache-Control"] = "no-store"
    return {"check": body.check, "feature": check.feature, "label": check.label, "finding": finding}
