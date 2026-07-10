from __future__ import annotations

from dataclasses import dataclass

from .schema import Event, Target


@dataclass(frozen=True)
class BenchCase:
    case_id: str
    events: tuple[Event, ...]
    target: Target
    expected_contains: tuple[str, ...] = ()


def _case(
    case_id: str,
    events: tuple[Event, ...],
    action: str,
    messages: tuple[str, ...] = (),
    expected_contains: tuple[str, ...] = (),
) -> BenchCase:
    return BenchCase(
        case_id=case_id,
        events=events,
        target=Target(action=action, messages=messages),  # type: ignore[arg-type]
        expected_contains=expected_contains,
    )


DOUBLE_TEXT_BENCH: tuple[BenchCase, ...] = (
    _case(
        "warm-revision",
        (
            Event(role="user", text="can you help write this", dt_ms=0),
            Event(role="user", text="make it warmer actually", dt_ms=1300),
        ),
        "respond",
        ("Got it - warmer. Here's a version that keeps the ask clear.",),
        ("warmer",),
    ),
    _case(
        "second-detail",
        (
            Event(role="user", text="write a follow up", dt_ms=0),
            Event(
                role="user", text="mention that I'm free Thursday afternoon", dt_ms=1800
            ),
        ),
        "respond",
        ("Absolutely - I'll include Thursday afternoon as the availability window.",),
        ("Thursday",),
    ),
    _case(
        "customer-email-beta",
        (
            Event(role="user", text="draft a customer email", dt_ms=0),
            Event(role="user", text="include the beta waitlist", dt_ms=900),
        ),
        "respond",
        ("I'll include the beta waitlist in the customer email.",),
        ("beta",),
    ),
    _case(
        "launch-note-pricing",
        (
            Event(role="user", text="draft a launch note", dt_ms=0),
            Event(role="user", text="say the pricing is unchanged", dt_ms=2200),
        ),
        "respond",
        ("I'll keep the pricing unchanged in the launch note.",),
        ("pricing",),
    ),
    _case(
        "support-reply-fix",
        (
            Event(role="user", text="make a support reply", dt_ms=0),
            Event(role="user", text="say we already shipped the fix", dt_ms=650),
        ),
        "respond",
        ("I'll mention that the fix already shipped.",),
        ("fix",),
    ),
    _case(
        "status-update-dashboard",
        (
            Event(role="user", text="make a crisp status update", dt_ms=0),
            Event(role="user", text="mention the new dashboard", dt_ms=2900),
        ),
        "respond",
        ("I'll include the new dashboard.",),
        ("dashboard",),
    ),
    _case(
        "recruiting-note-warmer",
        (
            Event(role="user", text="write a recruiting note", dt_ms=0),
            Event(
                role="user",
                text="keep the ask very clear, actually make it warmer",
                dt_ms=1200,
            ),
        ),
        "respond",
        ("I'll make it warmer while keeping the ask clear.",),
        ("warmer",),
    ),
    _case(
        "meeting-summary-apology",
        (
            Event(role="user", text="summarize the meeting", dt_ms=0),
            Event(role="user", text="keep the apology brief", dt_ms=1700),
        ),
        "respond",
        ("I'll keep the apology brief in the summary.",),
        ("apology",),
    ),
    _case(
        "opening-line-calendar",
        (
            Event(role="user", text="write the opening line", dt_ms=0),
            Event(role="user", text="make the next step a calendar hold", dt_ms=1000),
        ),
        "respond",
        ("I'll make the next step a calendar hold.",),
        ("calendar",),
    ),
    _case(
        "onboarding-link",
        (
            Event(role="user", text="turn this into a text message", dt_ms=0),
            Event(role="user", text="include the onboarding link", dt_ms=850),
        ),
        "respond",
        ("I'll include the onboarding link.",),
        ("onboarding",),
    ),
    _case(
        "mid-thought-wait",
        (Event(role="user", text="wait I mean the part where she says", dt_ms=0),),
        "wait",
    ),
    _case(
        "fragment-section-about",
        (Event(role="user", text="can you write the section about", dt_ms=0),),
        "wait",
    ),
    _case(
        "fragment-before-answer",
        (Event(role="user", text="actually before you answer", dt_ms=0),),
        "wait",
    ),
    _case(
        "fragment-sounds-like",
        (Event(role="user", text="I want it to sound like", dt_ms=0),),
        "wait",
    ),
    _case(
        "fragment-include-with",
        (Event(role="user", text="include the part with", dt_ms=0),),
        "wait",
    ),
    _case(
        "fragment-main-thing-that",
        (Event(role="user", text="the main thing is that", dt_ms=0),),
        "wait",
    ),
    _case(
        "partial-typed-fragment",
        (
            Event(
                role="user",
                text="for the paragraph where",
                dt_ms=0,
                partial=True,
            ),
        ),
        "wait",
    ),
    _case(
        "colon-fragment",
        (Event(role="user", text="send this to:", dt_ms=0),),
        "wait",
    ),
    _case(
        "correction-after-assistant",
        (
            Event(role="user", text="draft a short invite", dt_ms=0),
            Event(role="assistant", text="Sure - here's a crisp invite:", dt_ms=900),
            Event(
                role="user",
                text="actually make it for investors, not friends",
                dt_ms=700,
            ),
        ),
        "interject",
        ("Got it - investor-facing, not casual. Here's the revised invite:",),
        ("investor",),
    ),
    _case(
        "nevermind-interrupt",
        (
            Event(role="user", text="write a long email to Sam", dt_ms=0),
            Event(
                role="assistant",
                text="Subject: Following up on our discussion",
                dt_ms=900,
            ),
            Event(role="user", text="never mind, make it a two-line text", dt_ms=600),
        ),
        "interject",
        ("Switching to a two-line text instead:",),
        ("two-line",),
    ),
    _case(
        "scratch-that-customers",
        (
            Event(role="user", text="prepare a product update", dt_ms=0),
            Event(
                role="assistant", text="I'll frame it for the internal team.", dt_ms=750
            ),
            Event(role="user", text="scratch that, make it for customers", dt_ms=500),
        ),
        "interject",
        ("Switching the audience to customers.",),
        ("customers",),
    ),
    _case(
        "wait-make-calmer",
        (
            Event(role="user", text="rewrite this paragraph", dt_ms=0),
            Event(role="assistant", text="Here is a stronger version:", dt_ms=650),
            Event(role="user", text="wait, make it calmer", dt_ms=550),
        ),
        "interject",
        ("I'll make it calmer.",),
        ("calmer",),
    ),
    _case(
        "actually-support-lead",
        (
            Event(role="user", text="make a short invite", dt_ms=0),
            Event(role="assistant", text="Sure - here's a friendly invite.", dt_ms=700),
            Event(role="user", text="actually make it for the support lead", dt_ms=800),
        ),
        "interject",
        ("I'll adjust it for the support lead.",),
        ("support",),
    ),
    _case(
        "not-formal",
        (
            Event(role="user", text="draft a customer email", dt_ms=0),
            Event(role="assistant", text="Dear customer,", dt_ms=600),
            Event(role="user", text="not formal, make it friendlier", dt_ms=700),
        ),
        "interject",
        ("I'll make it friendlier and less formal.",),
        ("friendlier",),
    ),
    _case(
        "instead-investor-update",
        (
            Event(role="user", text="make a status update", dt_ms=0),
            Event(role="assistant", text="Here's a short internal update:", dt_ms=900),
            Event(role="user", text="instead mention the investor update", dt_ms=650),
        ),
        "interject",
        ("I'll mention the investor update instead.",),
        ("investor",),
    ),
    _case(
        "i-mean-migration",
        (
            Event(role="user", text="write a support reply", dt_ms=0),
            Event(role="assistant", text="Here's a concise reply:", dt_ms=500),
            Event(role="user", text="I mean include the migration deadline", dt_ms=500),
        ),
        "interject",
        ("I'll include the migration deadline.",),
        ("migration",),
    ),
    _case(
        "assistant-continues",
        (
            Event(role="user", text="give me the three steps", dt_ms=0),
            Event(
                role="assistant",
                text="First, define the decision the model has to make.",
                dt_ms=800,
            ),
        ),
        "continue",
        (
            "Second, collect examples where waiting is better than replying immediately.",
        ),
        ("Second",),
    ),
    _case(
        "assistant-second-bullet",
        (
            Event(role="user", text="list the rollout plan", dt_ms=0),
            Event(
                role="assistant", text="First, finish the local verifier.", dt_ms=900
            ),
        ),
        "continue",
        ("Second, run the adapter against the holdout set.",),
        ("Second",),
    ),
    _case(
        "assistant-next-paragraph",
        (
            Event(role="user", text="explain the model timeline", dt_ms=0),
            Event(
                role="assistant",
                text="The first phase proves data and eval plumbing.",
                dt_ms=850,
            ),
        ),
        "continue",
        ("Second, train the neural LoRA on the same contract.",),
        ("Second",),
    ),
    _case(
        "assistant-continues-after-email",
        (
            Event(role="user", text="draft the email in two parts", dt_ms=0),
            Event(role="assistant", text="Subject: Quick update", dt_ms=700),
        ),
        "continue",
        ("Second, add the body copy.",),
        ("Second",),
    ),
    _case(
        "assistant-continues-analysis",
        (
            Event(role="user", text="compare the two options", dt_ms=0),
            Event(role="assistant", text="Option one is faster to test.", dt_ms=650),
        ),
        "continue",
        ("Second, option two gives stronger evidence.",),
        ("Second",),
    ),
    _case(
        "assistant-continues-recipe",
        (
            Event(role="user", text="give me the process", dt_ms=0),
            Event(role="assistant", text="First, collect clean examples.", dt_ms=650),
        ),
        "continue",
        ("Second, train a small adapter.",),
        ("Second",),
    ),
    _case(
        "rapid-backchannel",
        (
            Event(role="user", text="explain the plan", dt_ms=0),
            Event(
                role="assistant",
                text="The first stage is an offline harness.",
                dt_ms=700,
            ),
            Event(role="user", text="yeah", dt_ms=400),
        ),
        "continue",
        ("The second stage is a small LoRA run against fixed double-text scenarios.",),
        ("second",),
    ),
    _case(
        "okay-backchannel",
        (
            Event(role="user", text="walk me through the eval", dt_ms=0),
            Event(
                role="assistant", text="The first check is schema validity.", dt_ms=750
            ),
            Event(role="user", text="okay", dt_ms=500),
        ),
        "continue",
        ("Second, check the action label.",),
        ("Second",),
    ),
    _case(
        "mhm-backchannel",
        (
            Event(role="user", text="tell me the migration plan", dt_ms=0),
            Event(
                role="assistant", text="First, freeze the current artifact.", dt_ms=800
            ),
            Event(role="user", text="mhm", dt_ms=300),
        ),
        "continue",
        ("Second, widen the benchmark.",),
        ("Second",),
    ),
    _case(
        "right-backchannel",
        (
            Event(role="user", text="explain training stages", dt_ms=0),
            Event(role="assistant", text="Stage one is local verification.", dt_ms=700),
            Event(role="user", text="right", dt_ms=450),
        ),
        "continue",
        ("Second, move to a neural LoRA smoke run.",),
        ("Second",),
    ),
    _case(
        "complete-question",
        (Event(role="user", text="what should the first benchmark measure?", dt_ms=0),),
        "respond",
        (
            "It should measure whether the model waits through double texts and incorporates the latest user correction.",
        ),
        ("double", "latest"),
    ),
    _case(
        "summarize-product-update",
        (Event(role="user", text="can you summarize the product update?", dt_ms=0),),
        "respond",
        ("Yes - I can summarize the product update.",),
        ("product",),
    ),
    _case(
        "clear-next-step",
        (Event(role="user", text="what is the clearest next step?", dt_ms=0),),
        "respond",
        ("The clearest next step is to run the verification gate.",),
        ("next",),
    ),
    _case(
        "customer-reply-question",
        (Event(role="user", text="what should the customer reply say?", dt_ms=0),),
        "respond",
        ("The reply should be concise and useful for the customer.",),
        ("customer",),
    ),
)


SEMANTIC_TURN_ACTION_MAP = {
    "start_speaking": "respond",
    "<|start_speaking|>": "respond",
    "continue_listening": "wait",
    "<|continue_listening|>": "wait",
    "start_listening": "interject",
    "<|start_listening|>": "interject",
    "continue_speaking": "continue",
    "<|continue_speaking|>": "continue",
}


def map_semantic_turn_action(action: str) -> str:
    key = action.strip().lower()
    if key not in SEMANTIC_TURN_ACTION_MAP:
        raise ValueError(f"Unsupported semantic turn-taking action: {action!r}")
    return SEMANTIC_TURN_ACTION_MAP[key]
