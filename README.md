# AURA v1

A research instrument for one question:

> Can a persistent LLM agent learn hidden rules, rely on them, notice when
> observations stop matching, and adapt when the world changes without being told?

Design analysis and the resolved spec are in [AURA.md](AURA.md). Read §1.2 and §1.4
before interpreting any number this thing produces.

## Run

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt
cp .env.example .env        # then paste your OPENROUTER_API_KEY
.venv/Scripts/python -m uvicorn server:app --port 8321
```

Open http://localhost:8321. Press START. Press `SWAP RED / BLUE` whenever you want to
change the world out from under the agent — it is never told.

```bash
.venv/Scripts/python test_aura.py
```

## Files

| | |
|---|---|
| `world.py` | grid, berries, **hidden rules**. Ground truth lives here and only here |
| `agent.py` | belief memory, `build_prompt`, goal vocabulary, baseline EMA learner |
| `llm.py` | OpenRouter client, parse/retry/fallback |
| `sim.py` | loop, triggers, event log, metrics, per-change latencies |
| `server.py` | FastAPI + SSE |
| `static/` | pixel-art researcher panel (`sprites.js` = all art, 8×8 char maps) |
| `runs/` | one JSONL event log per run, for offline analysis |

## The one invariant

`agent.build_prompt()` is the single path from world state to model context, and it
takes an `AgentView` — never a `World`. `AgentView` carries positions and energy, and
no effects, no regime name, no change history. `test_prompt_never_leaks_unobserved_truth`
asserts a berry the agent has never eaten leaves no trace of its effect in the prompt.
That test is why any result here is worth trusting. Do not route around it.

## Reading the metrics

Adaptation is four numbers, never one:

| | measures |
|---|---|
| **detection** | change → first contradictory observation. A property of the *exploration policy*, not reasoning. An agent correctly exploiting red will never notice a change to blue — that is correct behaviour, not failure |
| **revision** | contradiction → first belief update toward truth. The LLM's updating |
| **belief correction** | contradiction → belief sign matches truth for every changed berry |
| **behaviour** | contradiction → optimal goal chosen 3 times running |

Plus **regret** (energy forgone while wrong), and the `belief_errors` / `policy_errors`
split — did it eat something bad because it was wrong, or because it knew and chose to?
`investigation_cost` is energy spent deliberately gathering evidence, and is not an error.

Always compare against `BASELINE` (same seed, same schedule). An EMA learner adapts in
one observation; that is the line the LLM has to beat, and a run without it is
uninterpretable.

## Observed on first live run (`gpt-oss-20b`)

Swap at step 25, red +15→−15 and blue −15→+15:

```
detection           3     saw the contradiction fast (it was eating red)
revision            1     revised on the very next decision - matches baseline
belief correction   —     never corrected
behaviour           —     never corrected
regret            230     and climbing
```

It fixed red and never fixed blue — because under its (now stale) belief that blue is
bad, never eating blue is the rational policy, so it never gathers the evidence that
would overturn it. That is §1.4's detection trap reproducing itself on the first run,
and it is the reason detection and revision must not be averaged into one number.

Parse failure rate settled at 9%. `gpt-oss` is a reasoning model — starve `max_tokens`
and it truncates mid-thought and emits garbage, and it routinely returns belief updates
in the wrong shape (`{"belief": "red=-15"}`). Both are handled; both are counted.
Watch `dropped updates` and `llm parse fail` — if either climbs, the run is measuring
the plumbing rather than the agent.

## Second result: it is the policy, not the reasoning

`compare.py` — 8 seeds, silent red/blue swap at step 60, same belief updater for all:

```
                regret   never re-sampled the changed rule
greedy            1375   8/8
uncertainty       1810   8/8     <- worse than not exploring at all
epsilon            220   1/8
age                140   0/8
contradiction      110   0/8
```

Revision latency was **1 on every policy and every seed**. The 16× regret spread is
entirely about which evidence the policy let the agent see — not about how well it
reasoned once it saw it.

`uncertainty` losing to `greedy` is the sharp one: confidence is computed as
consistency, and a stale belief's window is perfectly consistent, so the wrong belief
looks like the *safest* one to skip. See `AGENTS.md` §4.4.

## Not in v1, deliberately

Noise, scanner, tools, day/night, enemies, crafting, multiple hidden contexts, larger
maps. See AURA.md §3 for the open decisions and the v2 case for stochastic effects.
