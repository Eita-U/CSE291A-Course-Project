"""Module A: deterministic workload matrix and prompt construction.

Context length is controlled by padding prompts to a token budget. Response
length is not generated here: each step carries a max_tokens budget that the
agent runner must copy into generation_settings for the backend.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Sequence

ContextClass = Literal["short", "long"]
ResponseClass = Literal["short", "long"]

SHORT_CONTEXT_TOKENS = 128
LONG_CONTEXT_TOKENS = 2048
SHORT_RESPONSE_TOKENS = 32
LONG_RESPONSE_TOKENS = 256
DEFAULT_STEP_COUNTS = (1, 3)
DEFAULT_TOOL_DELAYS_S = (0.0, 0.05)

# Fixed vocabulary so padding is reproducible without a model tokenizer.
_FILLER = (
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
    "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey",
    "xray", "yankee", "zulu", "north", "south", "east", "west", "red", "blue",
)


@dataclass(frozen=True)
class WorkloadProfile:
    name: str
    context_class: ContextClass
    response_class: ResponseClass
    n_steps: int
    tool_delay_s: float
    context_tokens: int
    max_tokens: int

    def __post_init__(self) -> None:
        if self.n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        if self.context_tokens < 1 or self.max_tokens < 1:
            raise ValueError("token budgets must be positive")
        if self.tool_delay_s < 0:
            raise ValueError("tool_delay_s must be non-negative")


@dataclass(frozen=True)
class StepSpec:
    step_id: int
    context_tokens: int
    max_tokens: int
    tool_delay_s: float
    instruction: str


@dataclass(frozen=True)
class AgentTaskSpec:
    task_id: str
    profile_name: str
    seed: int
    profile: WorkloadProfile
    steps: tuple[StepSpec, ...]


def context_token_targets(
    overrides: Mapping[str, int] | None = None,
) -> dict[str, int]:
    values = {"short": SHORT_CONTEXT_TOKENS, "long": LONG_CONTEXT_TOKENS}
    if overrides:
        values.update(overrides)
    return values


def response_token_targets(
    overrides: Mapping[str, int] | None = None,
) -> dict[str, int]:
    values = {"short": SHORT_RESPONSE_TOKENS, "long": LONG_RESPONSE_TOKENS}
    if overrides:
        values.update(overrides)
    return values


def default_matrix(
    *,
    context_tokens: Mapping[str, int] | None = None,
    response_tokens: Mapping[str, int] | None = None,
    step_counts: Sequence[int] = DEFAULT_STEP_COUNTS,
    tool_delays_s: Sequence[float] = DEFAULT_TOOL_DELAYS_S,
) -> tuple[WorkloadProfile, ...]:
    """Cartesian product of context, response, step count, and tool delay."""
    contexts = context_token_targets(context_tokens)
    responses = response_token_targets(response_tokens)
    profiles: list[WorkloadProfile] = []
    for context_class, n_ctx in contexts.items():
        for response_class, n_out in responses.items():
            for n_steps in step_counts:
                for delay in tool_delays_s:
                    delay_label = "tool" if delay > 0 else "notool"
                    name = (
                        f"{context_class}_ctx_{response_class}_resp_"
                        f"{n_steps}step_{delay_label}"
                    )
                    profiles.append(WorkloadProfile(
                        name=name,
                        context_class=context_class,  # type: ignore[arg-type]
                        response_class=response_class,  # type: ignore[arg-type]
                        n_steps=int(n_steps),
                        tool_delay_s=float(delay),
                        context_tokens=int(n_ctx),
                        max_tokens=int(n_out),
                    ))
    return tuple(profiles)


def estimate_tokens(text: str) -> int:
    """Whitespace token count used only to size synthetic prompts."""
    return len(text.split())


def build_prompt(
    instruction: str,
    context_tokens: int,
    extra_text: str = "",
    rng: random.Random | None = None,
) -> str:
    """Build a prompt whose whitespace token count equals context_tokens."""
    if context_tokens < 1:
        raise ValueError("context_tokens must be positive")
    rng = rng or random.Random(0)
    core = " ".join(part for part in (instruction.strip(), extra_text.strip()) if part)
    words = core.split()
    if len(words) > context_tokens:
        return " ".join(words[:context_tokens])
    while len(words) < context_tokens:
        words.append(_FILLER[rng.randrange(len(_FILLER))])
    return " ".join(words)


def _stable_rng(*parts: object) -> random.Random:
    material = ":".join(str(part) for part in parts).encode()
    digest = hashlib.sha256(material).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def generate_task(
    profile: WorkloadProfile,
    task_index: int = 0,
    seed: int = 0,
    task_id_prefix: str = "task",
) -> AgentTaskSpec:
    """Materialize one agent task from a profile. Same inputs always match."""
    if task_index < 0:
        raise ValueError("task_index must be non-negative")
    task_id = f"{task_id_prefix}-{profile.name}-{task_index}"
    steps: list[StepSpec] = []
    for step_id in range(profile.n_steps):
        delay = profile.tool_delay_s if step_id < profile.n_steps - 1 else 0.0
        instruction = (
            f"Task {task_id} step {step_id}: answer using the padded context. "
            f"Produce about {profile.max_tokens} tokens."
        )
        steps.append(StepSpec(
            step_id=step_id,
            context_tokens=profile.context_tokens,
            max_tokens=profile.max_tokens,
            tool_delay_s=delay,
            instruction=instruction,
        ))
    return AgentTaskSpec(
        task_id=task_id,
        profile_name=profile.name,
        seed=seed,
        profile=profile,
        steps=tuple(steps),
    )


def generate_workloads(
    profiles: Iterable[WorkloadProfile] | None = None,
    tasks_per_profile: int = 1,
    seed: int = 0,
    task_id_prefix: str = "task",
) -> tuple[AgentTaskSpec, ...]:
    """Expand the matrix into a deterministic list of agent tasks."""
    if tasks_per_profile < 1:
        raise ValueError("tasks_per_profile must be at least 1")
    selected = tuple(profiles) if profiles is not None else default_matrix()
    tasks: list[AgentTaskSpec] = []
    for profile in selected:
        for task_index in range(tasks_per_profile):
            tasks.append(generate_task(
                profile, task_index=task_index, seed=seed,
                task_id_prefix=task_id_prefix,
            ))
    return tuple(tasks)


def prompt_rng(spec: AgentTaskSpec, step_id: int) -> random.Random:
    return _stable_rng(spec.seed, spec.task_id, step_id)
