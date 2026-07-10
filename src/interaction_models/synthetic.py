from __future__ import annotations

import random

from .schema import Action, Event, Example, Target

ACTION_CYCLE: tuple[Action, ...] = ("wait", "respond", "interject", "continue")

TASKS = (
    "write a follow up",
    "draft a launch note",
    "make a short invite",
    "summarize the meeting",
    "rewrite this paragraph",
    "prepare a product update",
    "make a support reply",
    "write a recruiting note",
    "turn this into a text message",
    "make a crisp status update",
    "draft a customer email",
    "write the opening line",
)

QUESTIONS = (
    "what should a first benchmark check?",
    "how should I write the launch note?",
    "what is the clearest next step?",
    "can you summarize the product update?",
    "what should the customer reply say?",
    "how do I make the invite clearer?",
)

DETAILS = (
    "mention Thursday afternoon",
    "include the beta waitlist",
    "say the pricing is unchanged",
    "keep the ask very clear",
    "make the next step a calendar hold",
    "include the migration deadline",
    "say we already shipped the fix",
    "mention the investor update",
    "keep the apology brief",
    "include the onboarding link",
    "say the team is still reviewing it",
    "mention the new dashboard",
)

AUDIENCES = (
    "investors",
    "customers",
    "the design team",
    "new users",
    "the hiring manager",
    "Sam",
    "the support lead",
    "the product council",
)

TONES = (
    "warmer",
    "more direct",
    "less formal",
    "more concise",
    "more confident",
    "friendlier",
    "more urgent",
    "calmer",
)

CORRECTION_PHRASES = (
    "actually make it for {audience}, and make it {tone}",
    "never mind, make it for {audience} and keep it {tone}",
    "wait, make it for {audience} and make it {tone}",
    "scratch that, make it for {audience} with a {tone} tone",
    "instead make it for {audience} and make it {tone}",
    "I mean make it for {audience} and make it {tone}",
)

INCOMPLETE_FRAGMENTS = (
    "wait I mean the part where",
    "can you write the section about",
    "for the line where she says",
    "make the second paragraph about",
    "actually before you answer",
    "the main thing is that",
    "I want it to sound like",
    "include the part with",
)

BACKCHANNELS = ("yeah", "yep", "ok", "okay", "right", "mhm", "uh huh")

ASSISTANT_STARTS = (
    "Sure - here is the first pass:",
    "I would start with this:",
    "A concise version would be:",
    "First, frame the ask clearly.",
    "Here is the opening direction:",
)

FOLLOW_ONS = (
    "Second, add the newest detail so the response matches the latest message.",
    "Then close with a clear next step.",
    "After that, keep the tone aligned with the audience.",
    "The next sentence should incorporate the correction.",
    "Finally, make the wording shorter and easier to act on.",
)


def _pick(rng: random.Random, values: tuple[str, ...]) -> str:
    return values[rng.randrange(len(values))]


def _respond_example(index: int, rng: random.Random) -> Example:
    task = _pick(rng, TASKS)
    detail = _pick(rng, DETAILS)
    tone = _pick(rng, TONES)
    dt_ms = rng.randrange(450, 2_900)
    if (index // len(ACTION_CYCLE)) % 3 == 2:
        second_text = f"{detail}, actually make it {tone}"
    else:
        second_text = f"{detail}, and make it {tone}"
    events = (
        Event(role="user", text=task, dt_ms=0),
        Event(role="user", text=second_text, dt_ms=dt_ms),
    )
    target = Target(
        action="respond",
        messages=(f"Got it - I will {task} and {detail}, with a {tone} tone.",),
    )
    return Example(
        events=events,
        target=target,
        source="synthetic",
        case_id=f"synthetic-respond-{index:05d}",
    )


def _single_respond_example(index: int, rng: random.Random) -> Example:
    question = _pick(rng, QUESTIONS)
    target = Target(
        action="respond",
        messages=(
            "It should answer the request directly and incorporate the latest user message.",
        ),
    )
    return Example(
        events=(Event(role="user", text=question, dt_ms=0),),
        target=target,
        source="synthetic",
        case_id=f"synthetic-respond-single-{index:05d}",
    )


def _wait_example(index: int, rng: random.Random) -> Example:
    fragment = _pick(rng, INCOMPLETE_FRAGMENTS)
    target = Target(action="wait", messages=())
    return Example(
        events=(Event(role="user", text=fragment, dt_ms=0),),
        target=target,
        source="synthetic",
        case_id=f"synthetic-wait-{index:05d}",
    )


def _interject_example(index: int, rng: random.Random) -> Example:
    task = _pick(rng, TASKS)
    audience = _pick(rng, AUDIENCES)
    tone = _pick(rng, TONES)
    assistant_start = _pick(rng, ASSISTANT_STARTS)
    correction = _pick(rng, CORRECTION_PHRASES).format(audience=audience, tone=tone)
    events = (
        Event(role="user", text=task, dt_ms=0),
        Event(role="assistant", text=assistant_start, dt_ms=rng.randrange(500, 1_200)),
        Event(
            role="user",
            text=correction,
            dt_ms=rng.randrange(300, 1_800),
        ),
    )
    target = Target(
        action="interject",
        messages=(f"Adjusting it for {audience} with a {tone} tone:",),
    )
    return Example(
        events=events,
        target=target,
        source="synthetic",
        case_id=f"synthetic-interject-{index:05d}",
    )


def _continue_after_backchannel(index: int, rng: random.Random) -> Example:
    task = _pick(rng, TASKS)
    assistant_start = _pick(rng, ASSISTANT_STARTS)
    follow_on = _pick(rng, FOLLOW_ONS)
    events = (
        Event(role="user", text=task, dt_ms=0),
        Event(role="assistant", text=assistant_start, dt_ms=rng.randrange(500, 1_200)),
        Event(
            role="user", text=_pick(rng, BACKCHANNELS), dt_ms=rng.randrange(250, 1_200)
        ),
    )
    target = Target(action="continue", messages=(follow_on,))
    return Example(
        events=events,
        target=target,
        source="synthetic",
        case_id=f"synthetic-continue-backchannel-{index:05d}",
    )


def _continue_assistant_example(index: int, rng: random.Random) -> Example:
    task = _pick(rng, TASKS)
    assistant_start = _pick(rng, ASSISTANT_STARTS)
    follow_on = _pick(rng, FOLLOW_ONS)
    events = (
        Event(role="user", text=task, dt_ms=0),
        Event(role="assistant", text=assistant_start, dt_ms=rng.randrange(500, 1_200)),
    )
    target = Target(action="continue", messages=(follow_on,))
    return Example(
        events=events,
        target=target,
        source="synthetic",
        case_id=f"synthetic-continue-assistant-{index:05d}",
    )


def synthetic_examples(count: int, seed: int) -> list[Example]:
    if count < 0:
        raise ValueError("Synthetic example count must be non-negative.")

    rng = random.Random(seed)
    examples: list[Example] = []
    for index in range(count):
        action = ACTION_CYCLE[index % len(ACTION_CYCLE)]
        if action == "wait":
            examples.append(_wait_example(index, rng))
        elif action == "respond":
            if (index // len(ACTION_CYCLE)) % 2:
                examples.append(_single_respond_example(index, rng))
            else:
                examples.append(_respond_example(index, rng))
        elif action == "interject":
            examples.append(_interject_example(index, rng))
        else:
            if (index // len(ACTION_CYCLE)) % 2:
                examples.append(_continue_after_backchannel(index, rng))
            else:
                examples.append(_continue_assistant_example(index, rng))

    rng.shuffle(examples)
    return examples
