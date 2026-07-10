from __future__ import annotations

import hashlib
import html
import json
import re
from collections.abc import Iterable
from typing import Any

ACTIONS = ("wait", "respond", "interject", "continue")
ROLES = ("user", "assistant")
BACKCHANNELS = {"yeah", "yep", "ok", "okay", "right", "mhm", "uh huh", "uh-huh"}
MAX_TEXT_CHARS = 1200
MAX_EVENTS = 8
WORD_RE = re.compile(r"[A-Za-z0-9']+")
PLACEHOLDER_RE = re.compile(
    r"\b(?:lorem ipsum|placeholder|example text|sample text|foo|bar|baz|xxx|todo)\b",
    re.IGNORECASE,
)
META_MARKERS = (
    "as an ai",
    "json object",
    "target action",
    "training example",
    "<conversation>",
    "<act>",
)


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        if start < 0:
            raise
        parsed, _ = json.JSONDecoder().raw_decode(cleaned[start:])

    if isinstance(parsed, dict) and isinstance(parsed.get("example"), dict):
        parsed = parsed["example"]
    if isinstance(parsed, dict) and isinstance(parsed.get("examples"), list):
        parsed = parsed["examples"][0]
    if isinstance(parsed, list):
        parsed = parsed[0]
    if not isinstance(parsed, dict):
        raise ValueError("Model response did not contain a JSON object.")
    return parsed


def _text(raw: Any) -> str:
    return str(raw or "").strip()


def _role(raw: Any) -> str:
    value = _text(raw).lower()
    if value in {"human", "customer", "client"}:
        return "user"
    if value in {"ai", "agent", "bot", "model"}:
        return "assistant"
    return value


def normalize_example(
    raw: dict[str, Any], *, source: str, case_id: str
) -> dict[str, Any]:
    raw_events = raw.get("events") or raw.get("conversation") or raw.get("messages")
    if not isinstance(raw_events, list):
        raise ValueError("Missing events list.")

    events: list[dict[str, Any]] = []
    for item in raw_events:
        if not isinstance(item, dict):
            continue
        event: dict[str, Any] = {
            "role": _role(item.get("role") or item.get("speaker") or item.get("from")),
            "text": _text(
                item.get("text") or item.get("content") or item.get("message")
            ),
            "dt_ms": int(float(item.get("dt_ms", item.get("delay_ms", 0)) or 0)),
        }
        if bool(item.get("partial", False)):
            event["partial"] = True
        events.append(event)

    raw_target = raw.get("target") if isinstance(raw.get("target"), dict) else raw
    action = _text(raw_target.get("action")).lower()
    messages_raw = raw_target.get("messages", raw_target.get("message", []))
    if isinstance(messages_raw, str):
        messages = [messages_raw.strip()] if messages_raw.strip() else []
    elif isinstance(messages_raw, list):
        messages = [_text(message) for message in messages_raw if _text(message)]
    else:
        messages = []

    if action == "wait":
        messages = []

    example = {
        "events": events,
        "target": {"action": action, "messages": messages},
        "source": source,
        "case_id": case_id,
    }
    timing_bucket = raw.get("timing_bucket")
    if isinstance(timing_bucket, str) and timing_bucket:
        example["timing_bucket"] = timing_bucket
    return example


def validation_errors(
    example: dict[str, Any], expected_action: str | None = None
) -> list[str]:
    errors: list[str] = []
    events = example.get("events")
    target = example.get("target")
    if not isinstance(events, list) or not events:
        return ["events must be a non-empty list"]
    if len(events) > MAX_EVENTS:
        errors.append(f"events must have at most {MAX_EVENTS} items")
    if not isinstance(target, dict):
        return ["target must be an object"]

    action = target.get("action")
    messages = target.get("messages")
    if action not in ACTIONS:
        errors.append("target.action is invalid")
    if expected_action and action != expected_action:
        errors.append(f"target.action must be {expected_action}")
    if not isinstance(messages, list):
        errors.append("target.messages must be a list")
    elif action == "wait" and messages:
        errors.append("wait examples must not include target messages")
    elif action in {"respond", "interject", "continue"} and not messages:
        errors.append(f"{action} examples need at least one target message")

    seen_assistant = False
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            errors.append(f"event {index} must be an object")
            continue
        role = event.get("role")
        text = event.get("text")
        dt_ms = event.get("dt_ms")
        if role not in ROLES:
            errors.append(f"event {index} role is invalid")
        if role == "assistant":
            seen_assistant = True
        if not isinstance(text, str) or not text.strip():
            errors.append(f"event {index} text is empty")
        elif len(text) > MAX_TEXT_CHARS:
            errors.append(f"event {index} text is too long")
        elif any(marker in text.lower() for marker in META_MARKERS):
            errors.append(f"event {index} text contains meta text")
        if not isinstance(dt_ms, int) or dt_ms < 0 or dt_ms > 120_000:
            errors.append(f"event {index} dt_ms is invalid")

    for index, message in enumerate(messages if isinstance(messages, list) else []):
        if not isinstance(message, str) or not message.strip():
            errors.append(f"target message {index} is empty")
        elif len(message) > MAX_TEXT_CHARS:
            errors.append(f"target message {index} is too long")
        elif any(marker in message.lower() for marker in META_MARKERS):
            errors.append(f"target message {index} contains meta text")

    last = events[-1] if events and isinstance(events[-1], dict) else {}
    last_role = last.get("role")
    last_text = str(last.get("text") or "").strip().lower().strip(".!,")
    if action in {"wait", "respond"} and last_role != "user":
        errors.append(f"{action} examples must end with a user event")
    if action == "interject":
        if last_role != "user":
            errors.append("interject examples must end with a user correction")
        if not seen_assistant:
            errors.append("interject examples need a prior assistant event")
    if action == "continue":
        if last_role == "user" and last_text not in BACKCHANNELS:
            errors.append(
                "continue examples ending with user must end with a backchannel"
            )
        if not seen_assistant:
            errors.append("continue examples need a prior assistant event")

    return errors


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def quality_errors(example: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    events = example.get("events", [])
    target = example.get("target", {})
    action = target.get("action") if isinstance(target, dict) else None
    messages = target.get("messages", []) if isinstance(target, dict) else []

    for index, event in enumerate(events if isinstance(events, list) else []):
        text = str(event.get("text") or "")
        normalized = text.strip().lower().strip(".!,")
        allowed_continue_backchannel = (
            action == "continue"
            and index == len(events) - 1
            and event.get("role") == "user"
            and normalized in BACKCHANNELS
        )
        if word_count(text) < 3 and not allowed_continue_backchannel:
            errors.append(f"event {index} text is too short")
        if PLACEHOLDER_RE.search(text):
            errors.append(f"event {index} text contains placeholder text")

    for index, message in enumerate(messages if isinstance(messages, list) else []):
        if word_count(str(message)) < 4:
            errors.append(f"target message {index} is too short")
        if PLACEHOLDER_RE.search(str(message)):
            errors.append(f"target message {index} contains placeholder text")

    if action == "continue":
        assistant_texts = {
            re.sub(r"\s+", " ", str(event.get("text") or "").strip().lower())
            for event in events
            if isinstance(event, dict) and event.get("role") == "assistant"
        }
        for index, message in enumerate(messages if isinstance(messages, list) else []):
            normalized = re.sub(r"\s+", " ", str(message).strip().lower())
            if normalized in assistant_texts:
                errors.append(f"target message {index} repeats prior assistant text")

    return errors


def fingerprint(example: dict[str, Any]) -> str:
    parts: list[str] = [str(example["target"]["action"])]
    for event in example["events"]:
        text = re.sub(r"\s+", " ", str(event["text"]).strip().lower())
        parts.append(f"{event['role']}:{text}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def target_to_text(target: dict[str, Any]) -> str:
    action = target["action"]
    lines = [f"<act>{html.escape(action)}</act>"]
    for message in target.get("messages", []):
        lines.append(f"<msg>{html.escape(message)}</msg>")
    lines.append("<done/>")
    return "\n".join(lines)


def format_training_text(events: list[dict[str, Any]], target: dict[str, Any]) -> str:
    lines = [
        "You are a text interaction model for typed chat.",
        "Read the event stream, then emit exactly one floor-control action and any assistant messages.",
        "Use wait when the user appears mid-thought or another same-speaker message is likely.",
        "Use respond when the assistant should answer now.",
        "Use interject when the latest user message changes or corrects an in-progress assistant response.",
        "Use continue when the assistant should keep sending the next part of an existing response.",
        "Output only:",
        "<act>wait|respond|interject|continue</act>",
        "<msg>optional assistant message</msg>",
        "<done/>",
        "",
        "<conversation>",
    ]
    for event in events:
        attrs = [
            f'role="{html.escape(event["role"], quote=True)}"',
            f'dt_ms="{event["dt_ms"]}"',
        ]
        if event.get("partial"):
            attrs.append('partial="true"')
        lines.append(f"<event {' '.join(attrs)}>{html.escape(event['text'])}</event>")
    lines.extend(["</conversation>", "", target_to_text(target)])
    return "\n".join(lines)


def add_training_text(example: dict[str, Any]) -> dict[str, Any]:
    row = dict(example)
    row["text"] = format_training_text(example["events"], example["target"])
    return row


def iter_jsonl(path: str) -> Iterable[tuple[int, dict[str, Any]]]:
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parsed = json.loads(stripped)
            if not isinstance(parsed, dict):
                raise ValueError(f"Line {line_number} is not a JSON object.")
            yield line_number, parsed
