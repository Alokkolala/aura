"""Offline analysis of run logs. Costs nothing - reads runs/*.jsonl.

    .venv/Scripts/python analyze.py                  # every LLM run
    .venv/Scripts/python analyze.py --glob '*llm-s0*'

Answers the question the live UI raised and could not settle: the agent chose a goal
that was NOT greedy under its own stated beliefs ~70% of the time. Is that deliberate
exploration, or is it contradicting itself?

Only an LLM run can be asked this. A bandit has no stated reason to contradict.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def classify(ev: dict) -> str:
    goal = ev.get("goal", "")
    greedy = ev.get("greedy_under_own_beliefs")
    if goal == greedy:
        return "coherent"
    if goal.startswith("investigate_"):
        return "deliberate exploration"
    if goal == "idle":
        return "idle despite a positive belief"
    return "chose a worse target than its own belief"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="*llm*.jsonl")
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()

    files = sorted(Path("runs").glob(args.glob))
    if not files:
        print("no matching runs")
        return

    kinds: collections.Counter = collections.Counter()
    reasons: dict[str, list[str]] = collections.defaultdict(list)
    targets: collections.Counter = collections.Counter()
    fallbacks = total = 0

    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("kind") != "decision":
                continue
            total += 1
            if ev.get("source") == "fallback":
                fallbacks += 1
                continue
            k = classify(ev)
            kinds[k] += 1
            if ev["goal"].startswith("investigate_"):
                targets[ev["goal"].split("_", 1)[1]] += 1
            if k != "coherent" and ev.get("reason"):
                reasons[k].append(f"step {ev['step']:>4}  {ev['goal']:<18}"
                                  f"(own greedy: {ev['greedy_under_own_beliefs']})\n"
                                  f'              "{ev["reason"]}"')

    scored = sum(kinds.values())
    print(f"\n{len(files)} run(s), {total} decisions "
          f"({fallbacks} fallback, {scored} scored)\n")
    for k, n in kinds.most_common():
        print(f"  {n:>4}  {100*n/max(1,scored):>5.1f}%  {k}")

    print(f"\ninvestigate targets: "
          f"{dict(targets) or 'none'}")

    for k, rs in reasons.items():
        if k == "deliberate exploration":
            continue          # expected and fine; the others are the interesting ones
        print(f"\n--- {k} ({len(rs)}) ---")
        for r in rs[:args.examples]:
            print(" ", r)

    ex = reasons.get("deliberate exploration", [])
    if ex:
        print(f"\n--- deliberate exploration ({len(ex)}), sample ---")
        for r in ex[:args.examples]:
            print(" ", r)


if __name__ == "__main__":
    main()
