from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

Action = Literal["wait", "respond", "interject", "continue"]
Role = Literal["system", "user", "assistant"]

ACTIONS: tuple[str, ...] = ("wait", "respond", "interject", "continue")
ROLES: tuple[str, ...] = ("system", "user", "assistant")


@dataclass(frozen=True)
class Event:
    role: Role
    text: str
    dt_ms: int = 0
    event_id: str | None = None
    partial: bool = False


@dataclass(frozen=True)
class Target:
    action: Action
    messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class Example:
    events: tuple[Event, ...]
    target: Target
    source: str = "unknown"
    case_id: str | None = None


def normalize_role(value: Any, fallback: Role | None = None) -> Role:
    raw = str(value or "").strip().lower()
    if raw in {"assistant", "agent", "bot", "model", "ai"}:
        return "assistant"
    if raw in {"user", "human", "customer", "client"}:
        return "user"
    if raw == "system":
        return "system"
    if fallback is not None:
        return fallback
    raise ValueError(f"Unsupported role: {value!r}")


def event_from_dict(raw: dict[str, Any], fallback_role: Role | None = None) -> Event:
    role = normalize_role(
        raw.get("role")
        or raw.get("speaker")
        or raw.get("author")
        or raw.get("from")
        or raw.get("name"),
        fallback=fallback_role,
    )
    text = (
        raw.get("text") or raw.get("content") or raw.get("message") or raw.get("body")
    )
    if text is None:
        raise ValueError(f"Event is missing text/content/message/body: {raw!r}")

    dt_ms_raw = raw.get("dt_ms", raw.get("delay_ms", raw.get("delta_ms", 0)))
    if "delay_seconds" in raw:
        dt_ms_raw = float(raw["delay_seconds"]) * 1000
    if "delay_minutes" in raw:
        dt_ms_raw = float(raw["delay_minutes"]) * 60_000

    return Event(
        role=role,
        text=str(text).strip(),
        dt_ms=max(0, int(float(dt_ms_raw or 0))),
        event_id=str(raw["id"]) if raw.get("id") is not None else None,
        partial=bool(raw.get("partial", False)),
    )


def target_to_dict(target: Target) -> dict[str, Any]:
    return {"action": target.action, "messages": list(target.messages)}


def event_to_dict(event: Event) -> dict[str, Any]:
    data: dict[str, Any] = {
        "role": event.role,
        "text": event.text,
        "dt_ms": event.dt_ms,
    }
    if event.event_id is not None:
        data["event_id"] = event.event_id
    if event.partial:
        data["partial"] = True
    return data


def example_to_dict(example: Example, text: str | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {
        "events": [event_to_dict(event) for event in example.events],
        "target": target_to_dict(example.target),
        "source": example.source,
    }
    if example.case_id:
        data["case_id"] = example.case_id
    if text is not None:
        data["text"] = text
    return data


def ensure_events(events: Iterable[Event]) -> tuple[Event, ...]:
    normalized = tuple(events)
    if not normalized:
        raise ValueError("At least one event is required.")
    return normalized
