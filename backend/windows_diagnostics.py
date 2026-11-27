"""Read-only Windows collectors for the existing diagnostics API."""

import base64
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

from backend.diagnostic_result import CheckUnavailable, Finding

WINDOWS_ROOT = Path(os.environ.get("SystemRoot", r"C:\Windows"))
POWERSHELL = WINDOWS_ROOT / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
POWERSHELL_SETUP = """
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$WarningPreference = 'SilentlyContinue'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
"""
GIB = 2**30


def powershell(script: str) -> dict:
    encoded = base64.b64encode((POWERSHELL_SETUP + script).encode("utf-16-le")).decode("ascii")
    result = subprocess.run(
        [str(POWERSHELL), "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        env=os.environ | {"PSModulePath": str(POWERSHELL.parent / "Modules")},
        capture_output=True, encoding="utf-8-sig", errors="replace", timeout=20,
    )
    if result.returncode:
        raise CheckUnavailable("Windows could not read these measurements.")
    try:
        data = json.loads(result.stdout)
    except ValueError as error:
        raise CheckUnavailable("Windows returned an unreadable diagnostic response.") from error
    if not isinstance(data, dict):
        raise CheckUnavailable("Windows returned an incomplete diagnostic response.")
    return data


def machine() -> Finding:
    data = powershell("""
$os = Get-CimInstance -ClassName Win32_OperatingSystem -Property Caption, BuildNumber
$machine = Get-CimInstance -ClassName Win32_ComputerSystem -Property Manufacturer, Model, NumberOfLogicalProcessors
@{
    OS = $os.Caption
    Build = $os.BuildNumber
    Model = "$($machine.Manufacturer) $($machine.Model)".Trim()
    Cores = $machine.NumberOfLogicalProcessors
} | ConvertTo-Json -Compress
""")
    return Finding("Detected the host and operating system.", {
        "OS": data["OS"], "Build": data["Build"], "Architecture": platform.machine(),
        "Model": data["Model"], "Logical CPUs": int(data["Cores"]),
    })


def cpu() -> Finding:
    data = powershell("""
$cpu = Get-CimInstance -ClassName Win32_PerfFormattedData_PerfOS_Processor -Filter "Name = '_Total'"
if ($null -eq $cpu.PercentProcessorTime) { throw 'CPU counters are unavailable' }
@{ Busy = $cpu.PercentProcessorTime } | ConvertTo-Json -Compress
""")
    busy = int(data["Busy"])
    if not 0 <= busy <= 100:
        raise CheckUnavailable("Windows returned an invalid CPU sample.")
    return Finding("Sampled total Windows CPU usage.", {
        "Busy %": busy, "Idle %": 100 - busy,
    }, "attention" if busy >= 90 else "ok")


def memory() -> Finding:
    data = powershell("""
$os = Get-CimInstance -ClassName Win32_OperatingSystem -Property TotalVisibleMemorySize, FreePhysicalMemory
@{ TotalKiB = $os.TotalVisibleMemorySize; FreeKiB = $os.FreePhysicalMemory } | ConvertTo-Json -Compress
""")
    total, free = int(data["TotalKiB"]), int(data["FreeKiB"])
    if not 0 <= free <= total or not total:
        raise CheckUnavailable("Windows returned incomplete physical memory readings.")
    available = round(100 * free / total, 1)
    return Finding("Read physical memory visible to Windows and available RAM.", {
        "Physical RAM (GiB)": round(total * 1024 / GIB, 2),
        "Available RAM (GiB)": round(free * 1024 / GIB, 2), "Available %": available,
    }, "attention" if available < 15 else "ok")


def network() -> Finding:
    data = powershell("""
$routes = @(Get-NetRoute -PolicyStore ActiveStore)
$details = @{}
foreach ($prefix in @('0.0.0.0/0', '::/0')) {
    $selected = $routes | Where-Object { $_.DestinationPrefix -eq $prefix } |
        Sort-Object -Property @{Expression={ [long]$_.RouteMetric + [long]$_.InterfaceMetric }} |
        Select-Object -First 1
    $details[$prefix] = $selected.InterfaceAlias
}
$details | ConvertTo-Json -Compress
""")
    details = {"IPv4": data["0.0.0.0/0"], "IPv6": data["::/0"]}
    return Finding(
        "Found a default network route." if any(details.values()) else "No default network route was found.",
        {family: alias or "No default interface reported" for family, alias in details.items()},
        "ok" if any(details.values()) else "attention",
    )


def dns() -> Finding:
    data = powershell("""
$rows = @(Get-DnsClientServerAddress | Where-Object { $_.ServerAddresses.Count -gt 0 })
$addresses = @()
try {
    $addresses = @([System.Net.Dns]::GetHostAddresses('example.com'))
} catch {}
@{ Configured = $rows.Count; Answers = $addresses.Count } | ConvertTo-Json -Compress
""")
    answers = int(data["Answers"])
    return Finding(
        "The system resolver returned an address." if answers else "The system resolver returned no addresses.",
        {"Test hostname": "example.com", "Configured interface/family entries": int(data["Configured"]),
         "Addresses returned": answers},
        "ok" if answers else "attention",
    )


def listeners(ports: list[int]) -> dict[int, list[str]]:
    if any(type(port) is not int or not 1 <= port <= 65535 for port in ports):
        raise ValueError("TCP ports must be integers from 1 to 65535.")
    numbers = ",".join(map(str, ports))
    data = powershell(f"""
$ports = @({numbers})
$rows = @(Get-NetTCPConnection | Where-Object {{
    $_.State -eq 'Listen' -and $_.LocalPort -in $ports
}} | Sort-Object -Property LocalPort, OwningProcess -Unique)
$owners = @(
    foreach ($row in $rows) {{
        $process = Get-Process -Id $row.OwningProcess -ErrorAction SilentlyContinue
        @{{ Port = $row.LocalPort; Pid = $row.OwningProcess; Name = $process.ProcessName }}
    }}
)
@{{ Owners = $owners }} | ConvertTo-Json -Compress -Depth 3
""")
    owners: dict[int, list[str]] = {port: [] for port in ports}
    for row in data["Owners"]:
        port = int(row["Port"])
        if port in owners and len(owners[port]) < 8:
            owners[port].append(f'{row["Name"] or "Unknown process"} (PID {int(row["Pid"])})')
    return owners


def processes() -> Finding:
    data = powershell("""
$sample = Get-CimInstance -ClassName Win32_PerfFormattedData_PerfProc_Process -Property Name, IDProcess, PercentProcessorTime, WorkingSet
$rows = @($sample | Where-Object { $_.IDProcess -gt 0 -and $_.Name -ne '_Total' } |
    Sort-Object -Property PercentProcessorTime, WorkingSet -Descending | Select-Object -First 5)
@{ Processes = @($rows | ForEach-Object {
    @{ Name = $_.Name; Pid = $_.IDProcess; Cpu = $_.PercentProcessorTime; Bytes = $_.WorkingSet }
}) } | ConvertTo-Json -Compress -Depth 3
""")
    count = os.cpu_count() or 1
    rows = data["Processes"]
    if not rows:
        raise CheckUnavailable("No Windows process measurements were returned.")
    return Finding("Listed the five visible processes with the highest sampled CPU usage.", {
        f'{row["Name"]} (PID {int(row["Pid"])})':
        f'{min(100, int(row["Cpu"]) / count):.1f}% CPU, {int(row["Bytes"]) / 2**20:.1f} MiB RAM'
        for row in rows
    })


def system_logs() -> Finding:
    data = powershell("""
$events = @()
try {
    $events = @(Get-WinEvent -FilterHashtable @{
        LogName = @('System', 'Application')
        StartTime = (Get-Date).AddMinutes(-2)
        Level = 1,2
    } -MaxEvents 200)
} catch {
    if ($_.FullyQualifiedErrorId -notmatch '^NoMatchingEventsFound(?:,|$)') { throw }
}
@{
    Errors = @($events | Where-Object { $_.Level -eq 2 }).Count
    Critical = @($events | Where-Object { $_.Level -eq 1 }).Count
    LimitReached = $events.Count -eq 200
} | ConvertTo-Json -Compress
""")
    return Finding("Counted recent errors and critical events in the Windows System and Application logs.", {
        "Errors": int(data["Errors"]), "Critical": int(data["Critical"]),
        "Reached 200-event limit": data["LimitReached"],
    })


def docker() -> Finding:
    executable = shutil.which("docker.exe")
    if not executable:
        return Finding("The Windows Docker CLI was not found.", {"Docker CLI installed": False}, "unavailable")
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("DOCKER_")}
    for pipe in ("docker_engine", "dockerDesktopLinuxEngine", "dockerDesktopWindowsEngine"):
        try:
            result = subprocess.run(
                [executable, "--host", f"npipe:////./pipe/{pipe}", "version", "--format", "{{.Server.Version}}"],
                env=env, capture_output=True, encoding="utf-8", errors="replace", timeout=4,
            )
        except subprocess.TimeoutExpired:
            continue
        version = result.stdout.strip()
        if result.returncode == 0 and version and version != "<no value>":
            return Finding("The local Docker engine responded.", {
                "Docker CLI installed": True, "Engine version": version,
            })
    return Finding("No local Docker engine responded.", {"Docker CLI installed": True}, "unavailable")
