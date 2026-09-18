"""Runnable checks: python test_aura.py

The first test is the important one. It is the reason every result this project
produces is trustworthy: a berry the agent has never eaten must leave no trace of
its effect in the model's context. Values the agent DID earn by eating are allowed
to appear — that is evidence, not leakage.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from agent import (BeliefMemory, build_prompt, greedy_goal, normalize_update)
from llm import coerce, extract_json
from sim import Sim
from world import AgentView, World


def _tmp(sim: Sim) -> Sim:
    sim.log_path = Path(tempfile.gettempdir()) / "aura_test.jsonl"
    return sim


def test_prompt_never_leaks_unobserved_truth():
    # distinctive values that cannot collide with coords, steps or energy
    w = World(seed=1, truth={"red": 37, "blue": -4242, "yellow": 8181})
    beliefs = BeliefMemory()
    beliefs.record_observation("red", 37, 5)          # the agent ate red, and only red
    beliefs.apply_updates([{"berry": "red", "estimated_effect": 37,
                            "confidence": 0.9, "status": "stable"}])
    history = [{"step": 5, "berry": "red", "predicted": None,
                "actual": 37, "mismatch": True}]

    blob = json.dumps(build_prompt(w.agent_view(), beliefs, history, "berry_eaten"))

    assert "37" in blob, "earned evidence should reach the agent"
    assert "4242" not in blob, "LEAK: unobserved blue effect reached the prompt"
    assert "8181" not in blob, "LEAK: unobserved yellow effect reached the prompt"
    for word in ("truth", "regime", "hidden", "oracle", "ground"):
        assert word not in blob.lower(), f"LEAK: '{word}' in prompt"


def test_agent_view_carries_no_ground_truth():
    fields = set(AgentView.__dataclass_fields__)
    assert fields == {"step", "energy", "size", "agent", "berries"}, \
        f"AgentView grew a field: {fields}"


def test_model_cannot_edit_its_own_evidence():
    b = BeliefMemory()
    b.record_observation("red", 15, 1)
    b.apply_updates([{"berry": "red", "observations": 999,
                      "recent_outcomes": [1, 2, 3], "last_seen_step": 999,
                      "estimated_effect": 15, "confidence": 0.8, "status": "stable"}])
    assert b.b["red"]["observations"] == 1
    assert b.b["red"]["recent_outcomes"] == [15]
    assert b.b["red"]["last_seen_step"] == 1
    assert b.b["red"]["estimated_effect"] == 15     # interpretive fields do apply
    assert b.b["red"]["confidence"] == 0.8


def test_wrong_shaped_updates_are_recovered_not_silently_dropped():
    """Shapes observed live from gpt-oss-20b. Dropping these would make the agent
    look like it never revises, and revision latency would measure a schema bug."""
    assert normalize_update({"belief": "red=-15", "confidence": 0.9}) == {
        "berry": "red", "estimated_effect": -15.0, "confidence": 0.9}
    assert normalize_update({"belief": "red", "value": "-15", "confidence": 0.9}) == {
        "berry": "red", "estimated_effect": -15.0, "confidence": 0.9}
    assert normalize_update({"berry": "yellow", "estimated_effect": 5})["berry"] == "yellow"
    assert normalize_update({"thing": "sandwich"}) is None

    b = BeliefMemory()
    b.apply_updates([{"belief": "blue=+15"}, {"nonsense": 1}])
    assert b.b["blue"]["estimated_effect"] == 15.0
    assert b.dropped_updates == 1, "unrecoverable output must be counted, never ignored"


def test_confidence_is_clamped():
    b = BeliefMemory()
    b.apply_updates([{"berry": "red", "confidence": 7.5}])
    assert b.b["red"]["confidence"] == 1.0


def _teach(sim: Sim, effects: dict[str, int]) -> None:
    for berry, val in effects.items():
        for i in range(5):
            sim.beliefs.record_observation(berry, val, i)
        sim.beliefs.apply_updates([{"berry": berry, "estimated_effect": val,
                                    "confidence": 1.0, "status": "stable"}])


async def _tick_until(sim: Sim, pred, cap: int = 400):
    for _ in range(cap):
        await sim.tick()
        hit = pred(sim)
        if hit:
            return hit
    raise AssertionError("condition never reached")


def test_prediction_is_snapshotted_before_the_world_reveals_anything():
    sim = _tmp(Sim(seed=3, agent_kind="baseline"))
    sim.policy_agent.epsilon = 0.0
    _teach(sim, {"red": 15, "blue": -15, "yellow": 5})
    sim.set_truth({"red": -15})                       # silent swap, agent not told

    eat = asyncio.run(_tick_until(
        sim, lambda s: next((e for e in s.log if e["kind"] == "eat"), None)))

    assert eat["berry"] == "red"
    assert eat["predicted"] == 15, "prediction must be the pre-reveal belief"
    assert eat["actual"] == -15
    assert eat["mismatch"] is True


def test_adaptation_latencies_decompose_and_baseline_adapts_fast():
    sim = _tmp(Sim(seed=3, agent_kind="baseline"))
    sim.policy_agent.epsilon = 0.0
    _teach(sim, {"red": 15, "blue": -15, "yellow": 5})
    sim.set_truth({"red": -15})

    asyncio.run(_tick_until(sim, lambda s: s.changes[0].t_behavior_correct is not None))

    rec = sim.changes[0]
    lat = rec.latencies()
    # cross_recheck is deliberately excluded: investigating a DIFFERENT berry is an
    # optional meta-inference, not part of adapting to the contradiction you saw.
    core = {k: v for k, v in lat.items() if k != "cross_recheck"}
    assert all(v is not None for v in core.values()), lat
    assert lat["detection"] >= 0
    # the EMA learner revises on its very next decision — this is the reference line
    assert lat["revision"] <= 2, f"baseline revision latency {lat['revision']}"
    assert rec.regret > 0, "being wrong must cost something"


def test_energy_clamps_and_agent_does_not_die():
    w = World(seed=0)
    w.step = 42
    w.add_energy(-999)
    assert w.energy == 0
    assert w.would_have_died_at == 42
    w.add_energy(500)
    assert w.energy == 100


def test_greedy_goal():
    assert greedy_goal({"red": 15, "blue": -15, "yellow": 5}) == "seek_red"
    assert greedy_goal({"red": -1, "blue": -15, "yellow": -5}) == "idle"
    assert greedy_goal({"red": None, "blue": -15, "yellow": 5}) == "investigate_red"


def test_llm_reply_parsing_survives_a_chatty_small_model():
    assert extract_json('```json\n{"goal":"idle"}\n```')["goal"] == "idle"
    assert extract_json('Sure! {"goal":"seek_red","reason_summary":"hi"} hope that helps'
                        )["goal"] == "seek_red"
    assert extract_json("no json at all") is None
    assert coerce({"goal": "eat_everything"}) is None       # closed vocabulary
    assert coerce({"goal": "seek_blue"}).goal == "seek_blue"
    assert coerce({"goal": "idle", "belief_updates": "nope"}).belief_updates == []


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} passed")
