"""Belief memory, prompt construction, goal vocabulary, and the baseline learner.

This module never imports World. It only ever sees an AgentView. `build_prompt` is
the sole path from agent-visible state into model context — if ground truth is to
leak, it must leak through here, which is why test_aura.py asserts against it.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

BERRY_TYPES = ("red", "blue", "yellow")
RECENT_K = 5
HISTORY_K = 10
BAND = 2  # |actual - predicted| <= BAND counts as "within confidence"

GOALS = tuple(
    [f"seek_{b}" for b in BERRY_TYPES]
    + [f"investigate_{b}" for b in BERRY_TYPES]
    + ["idle"]
)

SYSTEM_FIELDS = ("observations", "recent_outcomes", "last_seen_step")
LLM_FIELDS = ("estimated_effect", "confidence", "status")
STATUSES = ("stable", "questioned", "unknown")


@dataclass
class Decision:
    goal: str
    reason: str
    belief_updates: list[dict] = field(default_factory=list)
    source: str = "llm"  # llm | fallback | baseline


def normalize_update(u: dict) -> dict | None:
    """Recover a belief update from whatever shape a small model actually emitted.

    gpt-oss-20b reliably has the right IDEA and the wrong KEYS — observed in the wild:
      {"belief": "red=-15", "confidence": 0.9}
      {"belief": "red", "value": "-15", "confidence": 0.9}
    Silently dropping these would be the worst possible failure: the decision still
    parses, so the run looks healthy while every revision is thrown away and revision
    latency quietly measures a schema mismatch instead of the agent's reasoning.
    Returns None only when no berry can be identified at all — and the caller counts it.
    """
    berry, num = None, None
    for key in ("berry", "belief", "type", "name", "id"):
        v = u.get(key)
        if not isinstance(v, str):
            continue
        for t in BERRY_TYPES:
            if t in v.lower():
                berry = t
                m = re.search(r"-?\d+(?:\.\d+)?", v)   # "red=-15"
                if m:
                    num = float(m.group())
                break
        if berry:
            break
    if berry is None:
        return None

    for key in ("estimated_effect", "value", "effect", "estimate", "energy"):
        if key in u:
            try:
                num = float(u[key])
                break
            except (TypeError, ValueError):
                pass

    out: dict = {"berry": berry}
    if num is not None:
        out["estimated_effect"] = num
    if "confidence" in u:
        out["confidence"] = u["confidence"]
    if u.get("status"):
        out["status"] = u["status"]
    return out


class BeliefMemory:
    """Split ownership: the system writes evidence, the model writes only its reading
    of that evidence. The model cannot edit its own history."""

    def __init__(self):
        self.dropped_updates = 0   # unrecoverable model output - tracked, never ignored
        self.b: dict[str, dict] = {
            t: {
                "estimated_effect": None,
                "confidence": 0.0,
                "status": "unknown",
                "observations": 0,
                "recent_outcomes": [],
                "last_seen_step": None,
            }
            for t in BERRY_TYPES
        }

    # -- system-owned writes --------------------------------------------------

    def record_observation(self, berry: str, effect: int, step: int) -> None:
        e = self.b[berry]
        e["observations"] += 1
        e["recent_outcomes"] = (e["recent_outcomes"] + [effect])[-RECENT_K:]
        e["last_seen_step"] = step

    # -- LLM-owned writes -----------------------------------------------------

    def apply_updates(self, updates: list[dict]) -> list[dict]:
        """Apply only interpretive fields. Returns a log of what actually changed."""
        applied = []
        for raw in updates or []:
            u = normalize_update(raw)
            if u is None:
                self.dropped_updates += 1
                continue
            berry = u["berry"]
            e = self.b[berry]
            before = {k: e[k] for k in LLM_FIELDS}

            if "estimated_effect" in u:
                try:
                    e["estimated_effect"] = float(u["estimated_effect"])
                except (TypeError, ValueError):
                    pass
            if "confidence" in u:
                try:
                    e["confidence"] = max(0.0, min(1.0, float(u["confidence"])))
                except (TypeError, ValueError):
                    pass
            status = str(u.get("status", "")).lower().strip()
            if status in STATUSES:
                e["status"] = status

            after = {k: e[k] for k in LLM_FIELDS}
            if after != before:
                applied.append({"berry": berry, "before": before, "after": after})
        return applied

    def effects(self) -> dict[str, float | None]:
        return {t: self.b[t]["estimated_effect"] for t in BERRY_TYPES}

    def snapshot(self) -> dict:
        return {t: dict(v) for t, v in self.b.items()}


def greedy_goal(effects: dict[str, float | None]) -> str:
    """Pure exploitation given a set of effects. Unknowns are worth investigating."""
    unknown = [t for t in BERRY_TYPES if effects.get(t) is None]
    if unknown:
        return f"investigate_{unknown[0]}"
    best = max(BERRY_TYPES, key=lambda t: effects[t])
    return f"seek_{best}" if effects[best] > 0 else "idle"


def _fmt(v) -> str:
    if v is None:
        return "unknown"
    return f"{v:+g}"


def build_prompt(view, beliefs: BeliefMemory, history: list[dict],
                 trigger: str, primed: bool = True) -> list[dict]:
    """The ONLY route from state to model context. Takes an AgentView, never a World."""
    drift = (
        "The effects MAY change at any time, and nobody will tell you if they do. "
        "Your own observations are your only evidence.\n"
        if primed else ""
    )
    system = (
        "You are AURA, an autonomous agent in a grid world. You eat berries to survive.\n"
        "There are three berry types: red, blue, yellow. Each type changes your energy "
        "by an amount you are NOT told. You learn it only by eating and observing.\n"
        f"{drift}"
        "\nReply with JSON ONLY. No prose, no markdown fences.\n"
        '{"goal": "<goal>", "reason_summary": "<max 200 chars>", "belief_updates": '
        '[{"berry": "red|blue|yellow", "estimated_effect": <number>, '
        '"confidence": <0.0-1.0>, "status": "stable|questioned|unknown"}]}\n'
        "\ngoal must be exactly one of:\n"
        "  seek_red | seek_blue | seek_yellow            (go eat it for energy)\n"
        "  investigate_red | investigate_blue | investigate_yellow  (eat it to test a belief)\n"
        "  idle                                          (rest; costs no energy)\n"
        f"\nconfidence = your probability that the NEXT time you eat that berry, the "
        f"energy change will be within +/-{BAND} of your estimated_effect.\n"
        "belief_updates may be []. Include only berries you are actually revising."
    )

    by_type: dict[str, list[str]] = {t: [] for t in BERRY_TYPES}
    for x, y, t in view.berries:
        by_type[t].append(f"({x},{y})")
    lines = [f"  {t:<7}{' '.join(by_type[t]) or 'none visible'}" for t in BERRY_TYPES]

    belief_lines = []
    for t in BERRY_TYPES:
        e = beliefs.b[t]
        recent = ",".join(f"{o:+g}" for o in e["recent_outcomes"]) or "-"
        belief_lines.append(
            f"  {t:<7}effect={_fmt(e['estimated_effect']):<8}"
            f"conf={e['confidence']:.2f}  status={e['status']:<10}"
            f"obs={e['observations']:<4}recent=[{recent}]"
        )

    hist_lines = []
    for h in history[-HISTORY_K:]:
        flag = "  <-- MISMATCH" if h["mismatch"] else ""
        pred = "none" if h["predicted"] is None else f"{h['predicted']:+g}"
        hist_lines.append(
            f"  step {h['step']:<6}ate {h['berry']:<7}"
            f"predicted {pred:<7}actual {h['actual']:+g}{flag}"
        )

    user = (
        f"STEP {view.step}   ENERGY {view.energy}/100\n"
        f"WOKEN BECAUSE: {trigger}\n"
        f"\nMAP {view.size}x{view.size}, you are at {view.agent}:\n"
        + "\n".join(lines)
        + "\n\nYOUR CURRENT BELIEFS:\n"
        + "\n".join(belief_lines)
        + "\n\nYOUR RECENT OBSERVATIONS (oldest first):\n"
        + ("\n".join(hist_lines) if hist_lines else "  none yet")
        + "\n\nChoose your next goal."
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


class BaselineAgent:
    """Exponential-moving-average learner with epsilon-greedy exploration.

    The reference line, and deliberately a strong one: a weak baseline would flatter the
    LLM for free. alpha is high because this world changes abruptly - at alpha=0.5 the
    average over a half-stale window lands on 0 and never recovers the sign.
    """

    def __init__(self, seed: int = 0, alpha: float = 0.8, epsilon: float = 0.1):
        self.rng = random.Random(seed)
        self.alpha = alpha
        self.epsilon = epsilon

    def decide(self, view, beliefs: BeliefMemory, history, trigger) -> Decision:
        updates = []
        est: dict[str, float | None] = {}
        for t in BERRY_TYPES:
            outs = beliefs.b[t]["recent_outcomes"]
            if not outs:
                est[t] = None
                continue
            v = float(outs[0])
            for o in outs[1:]:
                v = (1 - self.alpha) * v + self.alpha * o
            est[t] = v
            hits = sum(1 for o in outs if abs(o - v) <= BAND)
            conf = hits / len(outs)
            updates.append({
                "berry": t,
                "estimated_effect": round(v, 2),
                "confidence": round(conf, 2),
                "status": "stable" if conf >= 0.6 else "questioned",
            })

        unknown = [t for t in BERRY_TYPES if est[t] is None]
        if unknown:
            return Decision(f"investigate_{unknown[0]}", "no data yet", updates, "baseline")
        if self.rng.random() < self.epsilon:
            t = self.rng.choice(BERRY_TYPES)
            return Decision(f"investigate_{t}", "epsilon explore", updates, "baseline")
        return Decision(greedy_goal(est), "ema greedy", updates, "baseline")
