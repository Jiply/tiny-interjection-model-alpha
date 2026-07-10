from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .inference import InteractionEngine
from .schema import Event
from .train import DEFAULT_MODEL


class EventIn(BaseModel):
    role: Literal["system", "user", "assistant"]
    text: str = Field(min_length=1)
    dt_ms: int = Field(default=0, ge=0)
    event_id: str | None = None
    partial: bool = False


class DecideRequest(BaseModel):
    events: list[EventIn] = Field(min_length=1)


class DecideResponse(BaseModel):
    action: Literal["wait", "respond", "interject", "continue"]
    messages: list[str]
    raw: str
    mode: str


def create_app(engine: InteractionEngine) -> FastAPI:
    app = FastAPI(title="Tiny Interjection Model Alpha")

    @app.post("/decide", response_model=DecideResponse)
    def decide(request: DecideRequest) -> DecideResponse:
        events = tuple(
            Event(
                role=event.role,
                text=event.text,
                dt_ms=event.dt_ms,
                event_id=event.event_id,
                partial=event.partial,
            )
            for event in request.events
        )
        decision = engine.decide(events)
        return DecideResponse(
            action=decision.target.action,
            messages=list(decision.target.messages),
            raw=decision.raw,
            mode=decision.mode,
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the TIM decision API.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument(
        "--adapter-dir", type=Path, default=Path("adapters/qwen3-4b-instruct-2507")
    )
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--heuristic", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    engine = InteractionEngine(
        model_name=args.model_name,
        adapter_dir=args.adapter_dir,
        base_only=args.base_only,
        heuristic=args.heuristic,
        use_4bit=not args.no_4bit,
    )
    app = create_app(engine)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
