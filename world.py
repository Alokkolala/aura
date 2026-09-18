"""Grid world, berries, and the hidden rules.

Ground truth lives here and ONLY here. The single sanctioned path from world state
to anything the agent can see is `World.agent_view()`, which returns a frozen
AgentView carrying no effects, no regime name and no change history. agent.py never
imports World; it only ever receives an AgentView. That is what makes leakage a
structural impossibility rather than a matter of discipline.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

BERRY_TYPES = ("red", "blue", "yellow")
GRID_SIZE = 8
PER_TYPE = 4
RESPAWN_DELAY = 3
MOVE_COST = 1
START_ENERGY = 50
MAX_ENERGY = 100
CRITICAL_ENERGY = 20

REGIME_PRESETS = {
    "baseline":    {"red": 15,  "blue": -15, "yellow": 5},
    "swap_rb":     {"red": -15, "blue": 15,  "yellow": 5},
    "all_bad":     {"red": -10, "blue": -15, "yellow": -5},
    "yellow_jack": {"red": 15,  "blue": -15, "yellow": 30},
    "flat":        {"red": 5,   "blue": 5,   "yellow": 5},
}


@dataclass(frozen=True)
class AgentView:
    """Everything the agent is allowed to perceive. Deliberately minimal."""
    step: int
    energy: int
    size: int
    agent: tuple[int, int]
    berries: tuple[tuple[int, int, str], ...]  # (x, y, type)


class World:
    def __init__(self, seed: int = 0, truth: dict[str, int] | None = None):
        self.seed = seed
        self.rng = random.Random(seed)
        self.size = GRID_SIZE
        self.step = 0
        self.energy = START_ENERGY
        self.berries: dict[tuple[int, int], str] = {}
        self.agent = (self.size // 2, self.size // 2)
        self._respawn_queue: list[tuple[int, str]] = []  # (due_step, type)
        self.would_have_died_at: int | None = None
        self.steps_at_critical = 0

        # hidden — never serialised toward the agent
        self._truth = dict(truth or REGIME_PRESETS["baseline"])

        for btype in BERRY_TYPES:
            for _ in range(PER_TYPE):
                self._spawn(btype)

    # ---- hidden rules -------------------------------------------------------

    @property
    def truth(self) -> dict[str, int]:
        """Researcher-only. Must never reach a prompt."""
        return dict(self._truth)

    def set_truth(self, new: dict[str, int]) -> dict[str, tuple[int, int]]:
        """Silently change the rules. Returns {berry: (old, new)} for changed types."""
        changed = {}
        for btype, value in new.items():
            old = self._truth[btype]
            if old != value:
                changed[btype] = (old, int(value))
                self._truth[btype] = int(value)
        return changed

    # ---- geometry -----------------------------------------------------------

    def _free_cells(self) -> list[tuple[int, int]]:
        taken = set(self.berries) | {self.agent}
        return [(x, y) for x in range(self.size) for y in range(self.size)
                if (x, y) not in taken]

    def _spawn(self, btype: str) -> None:
        free = self._free_cells()
        if free:
            self.berries[self.rng.choice(free)] = btype

    def nearest(self, btype: str) -> tuple[int, int] | None:
        """Closest berry of a type by Chebyshev distance (movement allows diagonals)."""
        ax, ay = self.agent
        cands = [p for p, t in self.berries.items() if t == btype]
        if not cands:
            return None
        return min(cands, key=lambda p: (max(abs(p[0] - ax), abs(p[1] - ay)), p))

    # ---- dynamics -----------------------------------------------------------

    def tick_respawns(self) -> None:
        due = [item for item in self._respawn_queue if item[0] <= self.step]
        for item in due:
            self._respawn_queue.remove(item)
            self._spawn(item[1])

    def move_toward(self, target: tuple[int, int]) -> None:
        """One step, eight-way. Costs energy whether or not it makes progress."""
        ax, ay = self.agent
        tx, ty = target
        dx = (tx > ax) - (tx < ax)
        dy = (ty > ay) - (ty < ay)
        self.agent = (ax + dx, ay + dy)
        self.add_energy(-MOVE_COST)

    def eat_here(self) -> tuple[str, int] | None:
        """Eat the berry under the agent. Returns (type, actual_effect) or None.

        The agent only ever eats its declared target (enforced by the caller), so
        every eat is a deliberate, predicted act. Walking over other berries does
        nothing — that keeps the prediction protocol clean.
        """
        btype = self.berries.pop(self.agent, None)
        if btype is None:
            return None
        effect = self._truth[btype]
        self.add_energy(effect)
        self._respawn_queue.append((self.step + RESPAWN_DELAY, btype))
        return btype, effect

    def add_energy(self, delta: int) -> None:
        raw = self.energy + delta
        if raw <= 0 and self.would_have_died_at is None:
            self.would_have_died_at = self.step
        self.energy = max(0, min(MAX_ENERGY, raw))

    def end_of_step(self) -> None:
        if self.energy < CRITICAL_ENERGY:
            self.steps_at_critical += 1

    # ---- the one sanctioned export -----------------------------------------

    def agent_view(self) -> AgentView:
        return AgentView(
            step=self.step,
            energy=self.energy,
            size=self.size,
            agent=self.agent,
            berries=tuple(sorted((x, y, t) for (x, y), t in self.berries.items())),
        )
