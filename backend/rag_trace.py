"""Opt-in diagnostic events for one local PDF request."""

import json
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar


_events: ContextVar[list[dict] | None] = ContextVar("rag_events", default=None)
_logger = logging.getLogger("uvicorn.error")


@contextmanager
def capture_trace():
    events = []
    token = _events.set(events)
    try:
        yield events
    finally:
        _events.reset(token)


@contextmanager
def request_trace():
    if os.environ.get("RAG_DEBUG") != "1":
        yield
        return
    parent = _events.get()
    with capture_trace() as events:
        try:
            yield
        finally:
            _logger.info("RAG trace %s", json.dumps(events, ensure_ascii=False))
            if parent is not None:
                parent.extend(events)


def tracing() -> bool:
    return _events.get() is not None


def trace_event(stage: str, **details):
    events = _events.get()
    if events is not None:
        events.append({"stage": stage, **details})
