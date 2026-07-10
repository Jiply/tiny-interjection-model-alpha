from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile

from candidate_utils import (
    ACTIONS,
    add_training_text,
    extract_json_object,
    fingerprint,
    normalize_example,
    quality_errors,
    validation_errors,
)

DOMAINS = (
    "customer support",
    "product launch",
    "investor update",
    "recruiting",
    "scheduling",
    "technical migration",
    "sales follow-up",
    "internal status",
    "design feedback",
    "bug triage",
)

SITUATIONS = (
    "a nonprofit administrator debugging intermittent sign-in failures",
    "a product manager revising an accessibility launch announcement",
    "an operations lead planning a weekend database cutover",
    "a recruiter coordinating a final interview across time zones",
    "a support agent explaining a delayed replacement shipment",
    "a designer revising mobile navigation after usability feedback",
    "a sales lead following up after a security review",
    "an engineering manager communicating an incident recovery plan",
    "a finance lead clarifying a quarterly forecast assumption",
    "a customer success manager preparing an onboarding checklist",
    "a founder shortening an investor update before distribution",
    "a clinic coordinator rescheduling a non-urgent appointment",
    "a teacher organizing feedback for a student project",
    "a conference organizer changing a workshop room and time",
    "a retail manager handling an inventory reconciliation problem",
    "a developer narrowing a bug report to one browser version",
    "a legal operations analyst revising a contract review request",
    "a marketing lead changing the audience for a campaign brief",
    "a data analyst correcting the date range in a dashboard request",
    "a team lead turning a long memo into a short status message",
    "a travel coordinator changing one leg of a group itinerary",
    "a facilities manager sequencing an office maintenance rollout",
    "a researcher refining the scope of a literature summary",
    "a procurement lead comparing two vendor renewal options",
    "a community manager responding to an event cancellation",
    "a security engineer prioritizing remediation steps after an audit",
    "a logistics planner revising a warehouse migration timeline",
    "a publisher changing the tone and audience of a release note",
    "a hiring manager correcting the seniority level of an open role",
    "a project lead continuing a multi-part implementation plan",
    "an account manager clarifying a renewal meeting agenda",
    "a QA lead redirecting a test plan toward a regression risk",
)

TIMING_RANGES = (
    ("instant", 1, 250),
    ("quick", 251, 1_000),
    ("natural", 1_001, 3_000),
    ("delayed", 3_001, 10_000),
)


def build_prompt(
    action: str,
    domain: str,
    timing_range: tuple[str, int, int] | None = None,
    situation: str | None = None,
    failure_focused: bool = False,
) -> str:
    action_rules = {
        "wait": """
WAIT example requirements:
- The final event must be a user message that is clearly unfinished.
- Prefer one user event with partial=true.
- Do not include an assistant event.
- target.messages must be [].
- Good unfinished user text examples: "for the launch note, can you make the second paragraph", "wait, the part about pricing should", "I need the migration plan to include".
- Bad wait examples: "...", "ok", complete questions, or anything the assistant should answer now.
""",
        "respond": """
RESPOND example requirements:
- The final event must be a complete user request or question.
- Do not include a prior assistant unless it is necessary context.
- target.messages must contain one concrete assistant reply.
- The reply must not apologize unless the user asked about an error.
""",
        "interject": """
INTERJECT example requirements:
- Include a prior assistant event that has started answering.
- The final event must be a user correction, constraint, or redirect.
- target.messages must acknowledge the correction and switch direction.
- This is not a normal complete request; it interrupts an in-progress assistant response.
""",
        "continue": """
CONTINUE example requirements:
- Include a prior assistant event that is clearly incomplete or part one of a longer answer.
- The final event may be assistant text, or a short user backchannel exactly like "okay", "yeah", or "right".
- target.messages must contain the next new assistant sentence.
- Do not repeat any prior assistant event text as the target message.
""",
    }[action]
    timing_rule = ""
    if timing_range:
        timing_name, timing_min, timing_max = timing_range
        timing_rule = f"""
Timing requirement:
- Use at least two events.
- The first event must have dt_ms=0.
- The final event must have dt_ms between {timing_min} and {timing_max} inclusive.
- This timing range is named {timing_name}; do not mention that name in the conversation.
- The action must follow from both the language and event structure, not timing alone.
"""
    focus_rule = ""
    if failure_focused:
        focus_rules = {
            "wait": """
Failure-focus requirements:
- Make this a difficult wait/respond boundary case.
- The final user text must be unmistakably unfinished: set partial=true or end it with a dependency such as "about", "where", "with", "that", or a colon.
- Do not use a complete request followed by punctuation.
""",
            "respond": """
Failure-focus requirements:
- Use two consecutive user events: an initial complete request, then a complete concrete constraint or detail.
- Add "grounding_phrase" at the top level. It must be an exact 1-3 word substring from the final user event.
- The target message must repeat grounding_phrase exactly so it proves the latest detail was incorporated.
""",
            "interject": """
Failure-focus requirements:
- Add "grounding_phrase" at the top level. It must be an exact 1-3 word substring from the final user correction.
- The target message must repeat grounding_phrase exactly while acknowledging the redirect.
""",
            "continue": """
Failure-focus requirements:
- The prior assistant event must clearly present the first item, first stage, or first part.
- Set top-level "grounding_phrase" to exactly "Second".
- The target message must begin with "Second" and provide a specific next item without repeating the first.
""",
        }
        focus_rule = focus_rules[action]

    return f"""Generate one realistic training example for a typed-chat floor-control model.

Target action: {action}
Domain: {domain}
Scenario seed: {situation or domain}

Action meanings:
- wait: latest user message is incomplete or mid-thought; assistant should not answer yet.
- respond: latest user message is complete; assistant should answer now.
- interject: assistant has started responding, then user quickly corrects or redirects it.
- continue: assistant should keep sending the next part of an existing response.

Return one JSON object only. No markdown. No explanation.

Schema:
{{
  "grounding_phrase": "required only when requested below",
  "events": [
    {{"role": "user", "text": "...", "dt_ms": 0}},
    {{"role": "assistant", "text": "...", "dt_ms": 800}}
  ],
  "target": {{
    "action": "{action}",
    "messages": ["assistant text, empty only for wait"]
  }}
}}

Rules:
- Use only roles "user" and "assistant".
- dt_ms is milliseconds since the prior event.
- Use 1-5 events.
- The target.action must be exactly "{action}".
- Make the example natural and specific.
- Ground the conversation in the scenario seed and vary concrete details.
- Do not copy wording from examples in these instructions.
- Every event text must have at least 3 words, except an allowed final continue backchannel.
- Every target message must have at least 4 words unless target.messages is empty.
- Do not use placeholders like "...", "TODO", "foo", or "sample text".
- Do not mention JSON, labels, datasets, or these instructions inside event text.

{timing_rule}
{action_rules}
{focus_rule}
"""


def timing_error(example: dict[str, object], timing_range: tuple[str, int, int]) -> str:
    """When balanced timing is requested, validate the final event delay."""
    timing_name, timing_min, timing_max = timing_range
    events = example.get("events")
    if not isinstance(events, list) or len(events) < 2:
        return f"{timing_name} timing requires at least two events"
    final_event = events[-1]
    if not isinstance(final_event, dict):
        return f"{timing_name} timing has an invalid final event"
    dt_ms = final_event.get("dt_ms")
    if not isinstance(dt_ms, int) or not timing_min <= dt_ms <= timing_max:
        return f"final dt_ms must be between {timing_min} and {timing_max}"
    return ""


def failure_focus_error(
    raw: dict[str, object], example: dict[str, object], expected_action: str
) -> str:
    """When failure-focused generation is requested, validate its hard-case contract."""
    events = example.get("events")
    target = example.get("target")
    if not isinstance(events, list) or not events or not isinstance(target, dict):
        return "failure-focused example is malformed"
    if expected_action == "wait":
        final = events[-1]
        if not isinstance(final, dict):
            return "failure-focused wait has an invalid final event"
        text = str(final.get("text") or "").strip().lower()
        partial = bool(final.get("partial"))
        unfinished = partial or text.endswith(
            (" about", " where", " with", " that", ":")
        )
        return (
            "" if unfinished else "failure-focused wait must be unmistakably unfinished"
        )

    phrase = str(raw.get("grounding_phrase") or "").strip()
    if not 1 <= len(phrase.split()) <= 3:
        return "failure-focused grounding_phrase must contain 1-3 words"
    messages = target.get("messages")
    message_text = " ".join(str(message) for message in messages or [])
    if phrase.lower() not in message_text.lower():
        return "target message must repeat grounding_phrase"
    if expected_action == "continue":
        if phrase.lower() != "second" or not message_text.lower().startswith("second"):
            return 'failure-focused continue must begin with "Second"'
        return ""
    final = events[-1]
    final_text = str(final.get("text") or "") if isinstance(final, dict) else ""
    if phrase.lower() not in final_text.lower():
        return "grounding_phrase must come from the final user event"
    return ""


def extract_chat_content(payload: object) -> str:
    if isinstance(payload, dict):
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                message = choice.get("message")
                if isinstance(message, dict) and isinstance(
                    message.get("content"), str
                ):
                    return message["content"]
                if isinstance(choice.get("text"), str):
                    return choice["text"]
        if isinstance(payload.get("response"), str):
            return payload["response"]
        if isinstance(payload.get("output_text"), str):
            return payload["output_text"]
    return str(payload)


def call_do_serverless(
    *,
    doctl_bin: str,
    model: str,
    prompt: str,
    temperature: float,
    timeout: float,
    num_predict: int,
    seed: int | None = None,
) -> str:
    request_body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Return only the requested JSON object. No markdown.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": num_predict,
        "reasoning_effort": "none",
        "response_format": {"type": "json_object"},
    }
    if seed is not None:
        request_body["seed"] = seed
    with NamedTemporaryFile("w", encoding="utf-8", suffix=".json") as request_file:
        json.dump(request_body, request_file)
        request_file.flush()
        completed = subprocess.run(
            [
                doctl_bin,
                "serverless-inference",
                "chat-completions",
                "create",
                "--request",
                request_file.name,
                "--output",
                "json",
            ],
            check=True,
            capture_output=True,
            encoding="utf-8",
            timeout=timeout,
        )
    return extract_chat_content(json.loads(completed.stdout))


def call_do_serverless_with_retries(
    *,
    doctl_bin: str,
    model: str,
    prompt: str,
    temperature: float,
    timeout: float,
    num_predict: int,
    retries: int,
    retry_sleep: float,
    seed: int | None = None,
) -> str:
    last_error: Exception | None = None
    for retry_index in range(retries + 1):
        try:
            return call_do_serverless(
                doctl_bin=doctl_bin,
                model=model,
                prompt=prompt,
                temperature=temperature,
                timeout=timeout,
                num_predict=num_predict,
                seed=seed,
            )
        except (subprocess.SubprocessError, json.JSONDecodeError) as exc:
            last_error = exc
            if retry_index >= retries:
                break
            time.sleep(retry_sleep * (retry_index + 1))
    raise RuntimeError(
        f"DigitalOcean serverless request failed after {retries + 1} attempts: {last_error}"
    )


def load_existing(path: Path) -> tuple[int, set[str], Counter[str]]:
    if not path.exists():
        return 0, set(), Counter()

    count = 0
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = extract_json_object(stripped)
                example = normalize_example(
                    raw,
                    source=str(raw.get("source") or "existing"),
                    case_id=str(raw.get("case_id") or "existing"),
                )
                if validation_errors(example):
                    continue
                if quality_errors(example):
                    continue
                key = fingerprint(example)
                if key in seen:
                    continue
                seen.add(key)
                counts[example["target"]["action"]] += 1
                count += 1
            except Exception:
                continue
    return count, seen, counts


def next_case_index(*paths: Path) -> int:
    """When generation resumes, continue after every recorded attempt ID."""
    highest = -1
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(raw, dict):
                    continue
                match = re.search(r"-(\d+)$", str(raw.get("case_id") or ""))
                if match:
                    highest = max(highest, int(match.group(1)))
    return highest + 1


def next_action(counts: Counter[str]) -> str:
    return min(ACTIONS, key=lambda action: (counts[action], ACTIONS.index(action)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate TIM synthetic candidates.")
    parser.add_argument("--doctl-bin", default="doctl")
    parser.add_argument("--model", default="qwen3.5-397b-a17b")
    parser.add_argument("--output", default="data/candidates.raw.jsonl")
    parser.add_argument("--rejects", default="data/candidates.rejects.jsonl")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--max-attempts", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--num-predict", type=int, default=512)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5)
    parser.add_argument("--max-consecutive-request-failures", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--append", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-text", action="store_true")
    parser.add_argument("--balanced-timing", action="store_true")
    parser.add_argument("--failure-focused", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output)
    rejects_path = Path(args.rejects)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejects_path.parent.mkdir(parents=True, exist_ok=True)

    max_attempts = args.max_attempts or max(args.count * 6, 100)
    resume = args.resume or args.append
    existing_count, existing_seen, action_counts = (
        load_existing(output_path) if resume else (0, set(), Counter())
    )
    mode = "a" if resume else "w"
    accepted = existing_count
    first_case_index = next_case_index(output_path, rejects_path) if resume else 0
    seen = set(existing_seen)
    consecutive_request_failures = 0
    started = time.time()

    with (
        output_path.open(mode, encoding="utf-8") as out,
        rejects_path.open(mode, encoding="utf-8") as rej,
    ):
        for attempt in range(max_attempts):
            if accepted >= args.count:
                break
            attempt_index = first_case_index + attempt
            action = next_action(action_counts)
            attempt_rng = random.Random(args.seed + attempt_index)
            domain = DOMAINS[attempt_rng.randrange(len(DOMAINS))]
            situation = SITUATIONS[attempt_rng.randrange(len(SITUATIONS))]
            timing_range = (
                TIMING_RANGES[(accepted // len(ACTIONS)) % len(TIMING_RANGES)]
                if args.balanced_timing
                else None
            )
            prompt = build_prompt(
                action,
                domain,
                timing_range,
                situation,
                failure_focused=args.failure_focused,
            )
            case_id = f"do-llm-{action}-{args.seed}-{attempt_index:06d}"
            model_slug = args.model.replace("/", "-").replace(":", "-")
            source = f"do-serverless-{model_slug}"

            try:
                response_text = call_do_serverless_with_retries(
                    doctl_bin=args.doctl_bin,
                    model=args.model,
                    prompt=prompt,
                    temperature=args.temperature,
                    timeout=args.timeout,
                    num_predict=args.num_predict,
                    retries=args.retries,
                    retry_sleep=args.retry_sleep,
                    seed=args.seed + attempt_index,
                )
                consecutive_request_failures = 0
                raw = extract_json_object(response_text)
                example = normalize_example(raw, source=source, case_id=case_id)
                errors = validation_errors(example, expected_action=action)
                if not errors:
                    errors.extend(quality_errors(example))
                if timing_range:
                    timing_validation_error = timing_error(example, timing_range)
                    if timing_validation_error:
                        errors.append(timing_validation_error)
                    else:
                        example["timing_bucket"] = timing_range[0]
                if args.failure_focused:
                    focus_validation_error = failure_focus_error(raw, example, action)
                    if focus_validation_error:
                        errors.append(focus_validation_error)
                key = fingerprint(example) if not errors else ""
                if key in seen:
                    errors.append("duplicate")
                if errors:
                    rej.write(
                        json.dumps(
                            {"case_id": case_id, "errors": errors, "raw": raw},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue

                seen.add(key)
                action_counts[action] += 1
                row = example if args.no_text else add_training_text(example)
                out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                out.flush()
                accepted += 1
                if args.progress_every and accepted % args.progress_every == 0:
                    elapsed = max(time.time() - started, 1)
                    print(
                        f"accepted={accepted} attempts={attempt + 1} rate={accepted / elapsed:.2f}/s",
                        file=sys.stderr,
                    )
            except RuntimeError as exc:
                consecutive_request_failures += 1
                rej.write(
                    json.dumps(
                        {"case_id": case_id, "errors": [str(exc)]}, ensure_ascii=False
                    )
                    + "\n"
                )
                print(str(exc), file=sys.stderr)
                if (
                    consecutive_request_failures
                    >= args.max_consecutive_request_failures
                ):
                    return 2
            except Exception as exc:
                rej.write(
                    json.dumps(
                        {"case_id": case_id, "errors": [str(exc)]}, ensure_ascii=False
                    )
                    + "\n"
                )

    print(f"wrote {accepted} accepted candidates to {output_path}", file=sys.stderr)
    if accepted < args.count:
        print(f"stopped short after {max_attempts} attempts", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
