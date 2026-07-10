from __future__ import annotations

import random

from .schema import Action, Event, Example, Target

ACTION_CYCLE: tuple[Action, ...] = ("wait", "respond", "interject", "continue")

BENCHMARK_ANSWER = "Double-text handling and incorporation of the latest user message."

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

QUESTION_ANSWERS = (
    (
        "what should a first benchmark check?",
        BENCHMARK_ANSWER,
    ),
    (
        "which behaviors belong in the initial benchmark?",
        BENCHMARK_ANSWER,
    ),
    (
        "what capabilities should the earliest benchmark cover?",
        BENCHMARK_ANSWER,
    ),
    (
        "what should an initial benchmark measure?",
        BENCHMARK_ANSWER,
    ),
    (
        "what must the first evaluation benchmark capture?",
        BENCHMARK_ANSWER,
    ),
    (
        "which interaction skills should a first benchmark measure?",
        BENCHMARK_ANSWER,
    ),
    (
        "how should I write the launch note?",
        "Write the launch note with the main change first, followed by the clearest supporting detail.",
    ),
    (
        "what is the clearest next step?",
        "The clearest next step is to run the verification gate and review any failed cases.",
    ),
    (
        "can you summarize the product update?",
        "Yes - I can summarize the product update and preserve its key decisions.",
    ),
    (
        "what should the customer reply say?",
        "The customer reply should state the resolution clearly and give the customer a useful next step.",
    ),
    (
        "how do I make the invite clearer?",
        "Make the invite clearer by stating the audience, purpose, and next step directly.",
    ),
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

TONE_CORRECTION_PHRASES = (
    "wait, make it {tone}",
    "not formal, make it {tone}",
    "actually make it {tone}",
    "scratch that, make it {tone}",
    "I mean make it {tone}",
)

FORMATS = (
    "two-line message",
    "three-bullet note",
    "single-paragraph update",
    "brief text",
)

FORMAT_CORRECTION_PHRASES = (
    "never mind, turn it into a {format}",
    "scratch that, make it a {format}",
    "wait, use a {format} instead",
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

CONTINUATION_PAIRS = (
    (
        "First, frame the ask clearly.",
        "Second, add the newest detail so the response matches the latest message.",
    ),
    (
        "First, state the main decision.",
        "Second, close with a clear next step.",
    ),
    (
        "Option one is faster to test.",
        "Second, option two gives stronger evidence.",
    ),
    (
        "Stage one is local verification.",
        "Second, move to a small neural LoRA run.",
    ),
    (
        "Subject: Quick update",
        "Second, add body copy with the decision and next step.",
    ),
    (
        "The first check is schema validity.",
        "Second, check the predicted action label.",
    ),
)


def _pick(rng: random.Random, values: tuple[str, ...]) -> str:
    return values[rng.randrange(len(values))]


def _respond_example(index: int, rng: random.Random) -> Example:
    task = _pick(rng, TASKS)
    detail = _pick(rng, DETAILS)
    tone = _pick(rng, TONES)
    dt_ms = rng.randrange(450, 2_900)
    variant = (index // len(ACTION_CYCLE)) % 6
    if variant in {2, 4}:
        second_text = detail
    elif variant == 3:
        second_text = f"{detail}, actually make it {tone}"
    else:
        second_text = f"{detail}, and make it {tone}"
    events = (
        Event(role="user", text=task, dt_ms=0),
        Event(role="user", text=second_text, dt_ms=dt_ms),
    )
    response = f"Got it - I will {task} and {detail}."
    if variant not in {2, 4}:
        response = f"Got it - I will {task} and {detail}, with a {tone} tone."
    target = Target(action="respond", messages=(response,))
    return Example(
        events=events,
        target=target,
        source="synthetic",
        case_id=f"synthetic-respond-{index:05d}",
    )


def _single_respond_example(index: int, rng: random.Random) -> Example:
    question, answer = QUESTION_ANSWERS[rng.randrange(len(QUESTION_ANSWERS))]
    target = Target(
        action="respond",
        messages=(answer,),
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
    assistant_start, _ = CONTINUATION_PAIRS[rng.randrange(len(CONTINUATION_PAIRS))]
    correction_variant = (index // len(ACTION_CYCLE)) % 3
    tone_only = correction_variant == 1
    format_only = correction_variant == 2
    if tone_only:
        correction = _pick(rng, TONE_CORRECTION_PHRASES).format(tone=tone)
    elif format_only:
        output_format = _pick(rng, FORMATS)
        correction = _pick(rng, FORMAT_CORRECTION_PHRASES).format(format=output_format)
    else:
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
    response = f"I'll make it {tone}."
    if format_only:
        response = f"Switching to a {output_format} instead:"
    elif not tone_only:
        response = f"Adjusting it for {audience} with a {tone} tone:"
    target = Target(action="interject", messages=(response,))
    return Example(
        events=events,
        target=target,
        source="synthetic",
        case_id=f"synthetic-interject-{index:05d}",
    )


def _continue_after_backchannel(index: int, rng: random.Random) -> Example:
    task = _pick(rng, TASKS)
    assistant_start, follow_on = CONTINUATION_PAIRS[
        rng.randrange(len(CONTINUATION_PAIRS))
    ]
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
    assistant_start, follow_on = CONTINUATION_PAIRS[
        rng.randrange(len(CONTINUATION_PAIRS))
    ]
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


def synthetic_question_examples(count: int, seed: int) -> list[Example]:
    if count < 0:
        raise ValueError("Synthetic question example count must be non-negative.")

    rng = random.Random(seed)
    examples: list[Example] = []
    for index in range(count):
        example = _single_respond_example(index, rng)
        examples.append(
            Example(
                events=example.events,
                target=example.target,
                source="synthetic-question-boost",
                case_id=f"synthetic-question-boost-{index:05d}",
            )
        )
    rng.shuffle(examples)
    return examples
