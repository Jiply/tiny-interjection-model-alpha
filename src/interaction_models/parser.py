from __future__ import annotations

import re

from .schema import ACTIONS, Action, Target

ACT_RE = re.compile(r"<act>\s*(.*?)\s*</act>", re.IGNORECASE | re.DOTALL)
MSG_RE = re.compile(r"<msg>\s*(.*?)\s*</msg>", re.IGNORECASE | re.DOTALL)
DONE_RE = re.compile(r"<done\s*/>", re.IGNORECASE)


class TargetParseError(ValueError):
    """Raised when model output cannot be parsed as an interaction target."""


def target_to_text(target: Target) -> str:
    parts = [f"<act>{target.action}</act>"]
    parts.extend(f"<msg>{message}</msg>" for message in target.messages)
    parts.append("<done/>")
    return "\n".join(parts)


def parse_target(raw: str, *, strict: bool = True) -> Target:
    act_matches = ACT_RE.findall(raw)
    if len(act_matches) != 1:
        raise TargetParseError("Expected exactly one <act>...</act> block.")

    action = act_matches[0].strip().lower()
    if action not in ACTIONS:
        raise TargetParseError(f"Unsupported action: {action!r}")

    messages = tuple(
        message.strip() for message in MSG_RE.findall(raw) if message.strip()
    )

    if strict and not DONE_RE.search(raw):
        raise TargetParseError("Missing <done/> marker.")
    if strict and action in {"respond", "interject", "continue"} and not messages:
        raise TargetParseError(f"Action {action!r} requires at least one <msg> block.")
    if strict and action == "wait" and messages:
        raise TargetParseError("Action 'wait' must not include assistant messages.")

    return Target(action=action, messages=messages)  # type: ignore[arg-type]


def parse_target_or_wait(raw: str) -> Target:
    try:
        return parse_target(raw, strict=False)
    except TargetParseError:
        return Target(action="wait", messages=())
