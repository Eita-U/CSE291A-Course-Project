"""Small synthetic agent sessions for quick scheduler checks without the trace.

Each agent's context grows every step, like the real traces: step k resends the
whole conversation so far plus a new tool result. Agents share no prefix, so
reuse only comes from keeping an agent's context in its slot. Rows use the
trace columns that replay_trace needs.
"""

import random

_VOCAB = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
    "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
    "xray yankee zulu north south east west river stone cloud field"
).split()


def synthetic_sessions(
    n_agents: int = 8, n_steps: int = 8, words_per_step: int = 1000,
    output_tokens: int = 32, pre_gap_s: float = 0.7, seed: int = 0,
) -> dict[str, list[dict]]:
    """Return {session_id: rows}, one row per step in step order."""
    sessions = {}
    for a in range(n_agents):
        rng = random.Random(f"{seed}-{a}")
        session_id = f"synthetic-agent-{a}"
        messages = [{"role": "system", "content": f"You are agent {a}. Reply in one short sentence."}]
        rows = []
        for step in range(1, n_steps + 1):
            filler = " ".join(rng.choice(_VOCAB) for _ in range(words_per_step))
            messages.append({"role": "user", "content": f"Tool result {step} for agent {a}: {filler}"})
            rows.append({
                "session_id": session_id, "step": step,
                "input": [dict(m) for m in messages],
                "output_length": output_tokens,
                "pre_gap": 0.0 if step == 1 else pre_gap_s,
            })
            messages.append({"role": "assistant", "content": f"Noted step {step}."})
        sessions[session_id] = rows
    return sessions
