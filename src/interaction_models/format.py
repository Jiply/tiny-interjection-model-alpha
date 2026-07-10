from __future__ import annotations

from html import escape

from .parser import target_to_text
from .schema import Event, Target

SYSTEM_PROMPT = """You are a text interaction model for typed chat.
Read the event stream, then emit exactly one floor-control action and any assistant messages.
Use wait when the user appears mid-thought or another same-speaker message is likely.
Use respond when the assistant should answer now.
Use interject when the latest user message changes or corrects an in-progress assistant response.
Use continue when the assistant should keep sending the next part of an existing response.
Output only:
<act>wait|respond|interject|continue</act>
<msg>optional assistant message</msg>
<done/>"""


def serialize_events(events: tuple[Event, ...] | list[Event]) -> str:
    lines = ["<conversation>"]
    for event in events:
        attrs = [
            f'role="{escape(event.role, quote=True)}"',
            f'dt_ms="{event.dt_ms}"',
        ]
        if event.partial:
            attrs.append('partial="true"')
        if event.event_id:
            attrs.append(f'id="{escape(event.event_id, quote=True)}"')
        lines.append(f"<event {' '.join(attrs)}>{escape(event.text)}</event>")
    lines.append("</conversation>")
    return "\n".join(lines)


def build_prompt(events: tuple[Event, ...] | list[Event]) -> str:
    return f"{SYSTEM_PROMPT}\n\n{serialize_events(events)}"


def format_training_text(
    events: tuple[Event, ...] | list[Event], target: Target
) -> str:
    return f"{build_prompt(events)}\n{target_to_text(target)}"
