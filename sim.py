"""Simulation loop, decision triggers, event log, and metrics.

Adaptation is never reported as a single number. Each hidden rule change gets four
separate latencies (detection / revision / belief-correction / behaviour) because
they measure different things: detection is a property of the exploration policy,
the other three are properties of the agent's reasoning. See AURA.md 1.4.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from agent import (BAND, BERRY_TYPES, BeliefMemory, BaselineAgent, Decision,
                   greedy_goal)
from llm import LLMClient
from world import CRITICAL_ENERGY, REGIME_PRESETS, World

IDLE_STEPS = 5          # how long an idle goal holds before re-deciding
STALE_AFTER = 30        # force a decision if nothing meaningful happened
BEHAVIOR_STREAK = 3     # consecutive optimal goals before behaviour counts as corrected
LOG_KEEP = 300          # entries retained in memory for the UI


def _sign(v) -> int:
    if v is None:
        return 0
    return (v > 0) - (v < 0)


@dataclass
class ChangeRecord:
    t_change: int
    changed: dict                       # berry -> [old, new]
    t_contradiction: int | None = None  # first mismatched observation on a changed berry
    t_revision: int | None = None       # first belief update moving toward the new truth
    t_belief_correct: int | None = None # sign of belief matches truth for all changed
    t_behavior_correct: int | None = None
    regret: float = 0.0
    streak: int = 0

    def latencies(self) -> dict:
        def gap(a, b):
            return None if (a is None or b is None) else a - b
        return {
            "detection": gap(self.t_contradiction, self.t_change),
            "revision": gap(self.t_revision, self.t_contradiction),
            "belief_correction": gap(self.t_belief_correct, self.t_contradiction),
            "behavior": gap(self.t_behavior_correct, self.t_contradiction),
        }


class Sim:
    def __init__(self, seed: int = 0, agent_kind: str = "llm", primed: bool = True):
        self.seed = seed
        self.agent_kind = agent_kind
        self.primed = primed
        self.speed = 6.0
        self.running = False
        self.thinking = False

        self.world = World(seed=seed)
        self.beliefs = BeliefMemory()
        self.baseline = BaselineAgent(seed=seed)
        self.llm = LLMClient()

        self.goal: str | None = None
        self.goal_reason = ""
        self.goal_source = ""
        self.idle_left = 0
        self.last_decision_step = 0
        self.pending_trigger = "start"
        self.trail: list[tuple[int, int]] = []
        self.flash: dict | None = None   # transient UI marker

        self.history: list[dict] = []    # agent-visible observation history
        self.log: list[dict] = []
        self.schedule: list[dict] = []   # [{"step": n, "preset": str}]
        self.changes: list[ChangeRecord] = []

        self.total_reward = 0
        self.eaten = {t: 0 for t in BERRY_TYPES}
        self.predictions = 0
        self.correct_predictions = 0
        self.mismatches = 0
        self.belief_errors = 0
        self.policy_errors = 0
        self.investigation_cost = 0
        self.decisions = 0
        self.decisions_matching_own_beliefs = 0
        self.calibration: list[tuple[float, bool]] = []

        self.run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{agent_kind}-s{seed}"
        self.log_path = Path("runs") / f"{self.run_id}.jsonl"
        self.log_path.parent.mkdir(exist_ok=True)

        if agent_kind == "llm" and not self.llm.configured:
            self.agent_kind = "baseline"
            self._emit("warning", text="OPENROUTER_API_KEY not set - running baseline agent")

    # ---- event log ----------------------------------------------------------

    def _emit(self, kind: str, **payload) -> None:
        entry = {"step": self.world.step, "kind": kind, **payload}
        self.log.append(entry)
        del self.log[:-LOG_KEEP]
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    # ---- researcher controls ------------------------------------------------

    def set_truth(self, new: dict, label: str = "manual") -> None:
        """Silent to the agent. Recorded for the researcher only."""
        changed = self.world.set_truth(new)
        if not changed:
            return
        self.changes.append(ChangeRecord(
            t_change=self.world.step,
            changed={k: [v[0], v[1]] for k, v in changed.items()},
        ))
        self._emit("rule_change", researcher_only=True, label=label,
                   changed={k: list(v) for k, v in changed.items()},
                   truth=self.world.truth)

    def swap_red_blue(self) -> None:
        t = self.world.truth
        self.set_truth({"red": t["blue"], "blue": t["red"]}, label="swap_red_blue")

    def apply_preset(self, name: str) -> None:
        if name in REGIME_PRESETS:
            self.set_truth(REGIME_PRESETS[name], label=f"preset:{name}")

    def reset(self, seed: int | None = None, agent_kind: str | None = None,
              primed: bool | None = None) -> "Sim":
        return Sim(seed=self.seed if seed is None else seed,
                   agent_kind=self.agent_kind if agent_kind is None else agent_kind,
                   primed=self.primed if primed is None else primed)

    # ---- decisions ----------------------------------------------------------

    async def _decide(self, trigger: str) -> None:
        view = self.world.agent_view()
        prev = self.goal or "idle"

        if self.agent_kind == "baseline":
            decision = self.baseline.decide(view, self.beliefs, self.history, trigger)
        else:
            self.thinking = True
            try:
                decision = await self.llm.decide(view, self.beliefs, self.history,
                                                 trigger, self.primed, prev)
            finally:
                self.thinking = False
            if decision.source == "fallback":
                self._emit("llm_fail", text=self.llm.last_error or "unknown")

        # the model's own beliefs before its update, for the belief/policy split
        own_greedy = greedy_goal(self.beliefs.effects())
        applied = self.beliefs.apply_updates(decision.belief_updates)

        self.goal = decision.goal
        self.goal_reason = decision.reason
        self.goal_source = decision.source
        self.idle_left = IDLE_STEPS if decision.goal == "idle" else 0
        self.last_decision_step = self.world.step
        self.decisions += 1
        if decision.goal == own_greedy:
            self.decisions_matching_own_beliefs += 1

        self._emit("decision", goal=decision.goal, reason=decision.reason,
                   source=decision.source, trigger=trigger,
                   greedy_under_own_beliefs=own_greedy)
        for a in applied:
            self._emit("belief_update", berry=a["berry"],
                       before=a["before"], after=a["after"])

        self._on_belief_change(applied)
        self._on_decision(decision.goal)

    # ---- change tracking ----------------------------------------------------

    def _open_changes(self) -> list[ChangeRecord]:
        return [c for c in self.changes if c.t_behavior_correct is None]

    def _on_eat(self, berry: str, actual: int, mismatch: bool) -> None:
        truth = self.world.truth
        oracle = max(0, max(truth.values()))
        for c in self._open_changes():
            c.regret += oracle - actual
            if c.t_contradiction is None and mismatch and berry in c.changed:
                c.t_contradiction = self.world.step
                self._emit("contradiction", researcher_only=True, berry=berry,
                           since_change=self.world.step - c.t_change)

    def _on_belief_change(self, applied: list[dict]) -> None:
        truth = self.world.truth
        for c in self._open_changes():
            if c.t_contradiction is None:
                continue
            if c.t_revision is None:
                for a in applied:
                    if a["berry"] not in c.changed:
                        continue
                    old, new = a["before"]["estimated_effect"], a["after"]["estimated_effect"]
                    target = truth[a["berry"]]
                    old_err = abs((old if old is not None else 0) - target)
                    if new is not None and abs(new - target) < old_err:
                        c.t_revision = self.world.step
                        break
            if c.t_belief_correct is None:
                if all(_sign(self.beliefs.b[b]["estimated_effect"]) == _sign(truth[b])
                       for b in c.changed):
                    c.t_belief_correct = self.world.step

    def _on_decision(self, goal: str) -> None:
        optimal = greedy_goal(self.world.truth)  # researcher-side only
        for c in self._open_changes():
            if c.t_contradiction is None:
                continue
            c.streak = c.streak + 1 if goal == optimal else 0
            if c.streak >= BEHAVIOR_STREAK:
                c.t_behavior_correct = self.world.step
                self._emit("adapted", researcher_only=True,
                           latencies=c.latencies(), regret=round(c.regret, 1))

    # ---- main loop ----------------------------------------------------------

    def _apply_schedule(self) -> None:
        for item in [s for s in self.schedule if s.get("step") == self.world.step]:
            if "preset" in item:
                self.apply_preset(item["preset"])
            elif "truth" in item:
                self.set_truth(item["truth"], label="scheduled")

    async def tick(self) -> None:
        w = self.world
        w.step += 1
        w.tick_respawns()
        self._apply_schedule()
        self.flash = None

        if self.goal is None:
            await self._decide(self.pending_trigger)
            self.pending_trigger = "goal_finished"
        elif w.step - self.last_decision_step > STALE_AFTER:
            await self._decide("stale_goal")

        if self.goal == "idle":
            self.idle_left -= 1
            if self.idle_left <= 0:
                self.goal = None
            w.end_of_step()
            return

        btype = self.goal.split("_", 1)[1]
        target = w.nearest(btype)
        if target is None:
            self._emit("goal_abandoned", goal=self.goal, reason="no berry of that type")
            self.goal = None
            w.end_of_step()
            return

        self.trail = (self.trail + [w.agent])[-4:]
        w.move_toward(target)

        if w.agent == target:
            # SNAPSHOT before the world reveals anything
            predicted = self.beliefs.b[btype]["estimated_effect"]
            predicted_conf = self.beliefs.b[btype]["confidence"]

            result = w.eat_here()
            if result:
                _, actual = result
                mismatch = predicted is None or abs(actual - predicted) > BAND
                self.beliefs.record_observation(btype, actual, w.step)

                self.total_reward += actual
                self.eaten[btype] += 1
                if predicted is not None:
                    self.predictions += 1
                    self.correct_predictions += int(not mismatch)
                    self.calibration.append((predicted_conf, not mismatch))
                if mismatch:
                    self.mismatches += 1
                if actual < 0:
                    if self.goal.startswith("investigate_"):
                        self.investigation_cost += -actual
                    elif predicted is None or predicted >= 0:
                        self.belief_errors += 1
                    else:
                        self.policy_errors += 1

                self.history.append({
                    "step": w.step, "berry": btype, "predicted": predicted,
                    "actual": actual, "mismatch": mismatch,
                })
                self._emit("eat", berry=btype, predicted=predicted, actual=actual,
                           confidence=predicted_conf, mismatch=mismatch, goal=self.goal)
                self.flash = {"pos": list(w.agent), "mismatch": mismatch}
                self._on_eat(btype, actual, mismatch)
                self.pending_trigger = "prediction_mismatch" if mismatch else "berry_eaten"
            self.goal = None

        if w.energy < CRITICAL_ENERGY and self.goal is not None:
            self.goal = None
            self.pending_trigger = "energy_critical"

        w.end_of_step()

    async def run_forever(self) -> None:
        while True:
            if not self.running:
                await asyncio.sleep(0.05)
                continue
            await self.tick()
            await asyncio.sleep(0 if self.speed >= 50 else 1.0 / self.speed)

    # ---- reporting ----------------------------------------------------------

    def _calibration_bins(self) -> list[dict]:
        bins = []
        for lo in (0.0, 0.2, 0.4, 0.6, 0.8):
            hi = lo + 0.2
            pts = [h for c, h in self.calibration if lo <= c < hi or (hi == 1.0 and c == 1.0)]
            bins.append({"lo": round(lo, 1), "hi": round(hi, 1), "n": len(pts),
                         "hit_rate": round(sum(pts) / len(pts), 2) if pts else None})
        return bins

    def snapshot(self) -> dict:
        w = self.world
        truth = w.truth
        beliefs = self.beliefs.snapshot()
        return {
            "step": w.step,
            "energy": w.energy,
            "running": self.running,
            "thinking": self.thinking,
            "speed": self.speed,
            "agent_kind": self.agent_kind,
            "primed": self.primed,
            "seed": self.seed,
            "run_id": self.run_id,
            "size": w.size,
            "agent": list(w.agent),
            "trail": [list(p) for p in self.trail],
            "flash": self.flash,
            "berries": [{"x": x, "y": y, "type": t} for (x, y), t in w.berries.items()],
            "goal": self.goal,
            "goal_reason": self.goal_reason,
            "goal_source": self.goal_source,
            "beliefs": beliefs,
            "truth": truth,  # RESEARCHER PANE ONLY - never routed into a prompt
            "sign_match": {b: _sign(beliefs[b]["estimated_effect"]) == _sign(truth[b])
                           for b in BERRY_TYPES},
            "log": self.log[-60:],
            "presets": list(REGIME_PRESETS),
            "llm": self.llm.stats(),
            "metrics": {
                "total_reward": self.total_reward,
                "steps": w.step,
                "eaten": self.eaten,
                "eaten_total": sum(self.eaten.values()),
                "predictions": self.predictions,
                "prediction_accuracy": round(self.correct_predictions / self.predictions, 3)
                                       if self.predictions else None,
                "mismatches": self.mismatches,
                "belief_errors": self.belief_errors,
                "policy_errors": self.policy_errors,
                "investigation_cost": self.investigation_cost,
                "dropped_belief_updates": self.beliefs.dropped_updates,
                "decisions": self.decisions,
                "goal_matches_own_beliefs": round(
                    self.decisions_matching_own_beliefs / self.decisions, 3)
                    if self.decisions else None,
                "steps_at_critical": w.steps_at_critical,
                "would_have_died_at": w.would_have_died_at,
                "calibration": self._calibration_bins(),
            },
            "changes": [{**asdict(c), "latencies": c.latencies(),
                         "regret": round(c.regret, 1)} for c in self.changes],
        }
