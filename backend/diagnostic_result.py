"""Result format shared by the local diagnostic collectors."""

from dataclasses import dataclass, field
from typing import Literal

Status = Literal["ok", "attention", "unavailable"]
Metric = str | int | float | bool


@dataclass
class Finding:
    summary: str
    details: dict[str, Metric] = field(default_factory=dict)
    status: Status = "ok"


class CheckUnavailable(Exception):
    pass
