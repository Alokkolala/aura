"""Run several policies through an identical world and an identical silent swap.

    .venv/Scripts/python compare.py
    .venv/Scripts/python compare.py --seeds 0 1 2 --swap-at 40 --steps 400
    .venv/Scripts/python compare.py --agents llm greedy contradiction   (slow: LLM)

The question this exists to answer:

    How can an agent recover from a change in a part of the world that its current
    policy no longer samples?

`xcheck` is the column that matters for the LLM row: steps from the contradiction
to the agent choosing to investigate a DIFFERENT berry than the one that broke.
The hand-coded `contradiction` policy always does this. Whether the LLM does it
without being told is the question this project exists to answer.

Every agent here shares the same belief updater and differs only in when it spends
energy re-checking something it believes it already knows. `greedy` is the control.
"""
from __future__ import annotations

import argparse
import asyncio
import statistics

from agent import POLICIES
from sim import Sim

SWAP = {"red": -15, "blue": 15, "yellow": 5}   # red/blue trade places, silently


async def run_one(agent: str, seed: int, swap_at: int, steps: int) -> dict:
    sim = Sim(seed=seed, agent_kind=agent)
    sim.schedule = [{"step": swap_at, "truth": SWAP}]
    for _ in range(steps):
        await sim.tick()

    c = sim.changes[0] if sim.changes else None
    lat = c.latencies() if c else {}
    return {
        "agent": agent,
        "seed": seed,
        "detection": lat.get("detection"),
        "revision": lat.get("revision"),
        "belief": lat.get("belief_correction"),
        "behaviour": lat.get("behavior"),
        "xcheck": lat.get("cross_recheck"),
        "starved": ",".join(sim.starved(c)) if c else "",
        "starv_steps": sim.starvation_steps,
        "regret": round(c.regret) if c else 0,
        "reward": sim.total_reward,
        "valid": sim.validity()["ok"],
        "cost": sim.llm.stats()["cost_usd"],
        "stopped": sim.llm.stats()["budget_stopped"],
    }


def fmt(v) -> str:
    return "never" if v is None else str(v)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="+", default=list(POLICIES))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--swap-at", type=int, default=60)
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()

    print(f"\nworld seeds {args.seeds} | swap red<->blue at step {args.swap_at} "
          f"| {args.steps} steps\n")
    head = (f"{'agent':<14}{'seed':>5}{'detect':>8}{'revise':>8}{'belief':>8}"
            f"{'behav':>8}{'xcheck':>8}{'starved':>9}{'regret':>8}{'reward':>8}")
    print(head)
    print("-" * len(head))

    rows: dict[str, list[dict]] = {}
    for agent in args.agents:
        # seeds are independent and LLM runs are latency-bound, so fan them out
        rows[agent] = await asyncio.gather(*[
            run_one(agent, s, args.swap_at, args.steps) for s in args.seeds])
        for r in rows[agent]:
            if not r["valid"]:
                print(f"{agent:<14}  RUN INVALID (infrastructure failure) - excluded")
                continue
            print(f"{agent:<14}{r['seed']:>5}{fmt(r['detection']):>8}"
                  f"{fmt(r['revision']):>8}{fmt(r['belief']):>8}"
                  f"{fmt(r['behaviour']):>8}{fmt(r['xcheck']):>8}"
                  f"{r['starved'] or '-':>9}{r['regret']:>8}{r['reward']:>8}")
        print()

    print("median regret, and how often the changed rule was never re-sampled:\n")
    for agent, rs in rows.items():
        ok = [r for r in rs if r["valid"]]
        if not ok:
            continue
        starved = sum(1 for r in ok if r["starved"])
        print(f"  {agent:<14}regret {statistics.median(r['regret'] for r in ok):>7.0f}"
              f"   starved in {starved}/{len(ok)} runs")
    print("\nStarvation is not slow reasoning. It is a belief preventing the "
          "observation\nthat would correct it - the agent never looks, so it never "
          "learns.\n")


if __name__ == "__main__":
    asyncio.run(main())
