# AGENTS.md — read this before touching anything

You are working on **AURA**, a research instrument. It is not a game and not a demo.
Its output is *measurements*, and measurements can be silently invalidated by changes
that look harmless. This file exists so you don't invalidate them.

Read in this order: this file → `AURA.md` §1.2 and §1.4 → the code.

---

## 1. What we are building

One question, and nothing else:

> Can a persistent LLM agent learn hidden rules from interaction, come to rely on
> them, notice when observations stop matching those beliefs, and adapt when the
> world changes **without ever being told that it changed**?

An agent lives on an 8×8 grid with red, blue and yellow berries. Each type changes
its energy by a hidden amount. The agent is never told those amounts — it can only
learn them by eating. A researcher can silently rewrite the rules mid-run. The agent
receives no notification, no flag, no hint. It only ever sees consequences.

Stripped of the dressing: **a non-stationary 3-armed bandit with an LLM as the
belief-updating mechanism.** Hold that framing. It tells you what is being measured
and where it can break.

---

## 2. The one invariant — do not route around it

`agent.build_prompt()` is the **only** path from world state into model context.
It takes an `AgentView`, never a `World`.

```
World  ──agent_view()──>  AgentView  ──build_prompt()──>  messages  ──> LLM
 │                         (positions, energy, step)
 └── _truth, regime labels, change timestamps  ── NEVER CROSSES THIS LINE
```

`agent.py` does not import `world.py`'s `World`. That is deliberate and load-bearing.
`test_prompt_never_leaks_unobserved_truth` asserts that a berry the agent has never
eaten leaves **no trace of its effect** in the prompt. Values it *did* earn by eating
are allowed through — that is evidence, not leakage.

If you break that test, you have not broken a test. You have invalidated every number
the project has ever produced. Fix the code, never the assertion.

**Things that must never enter a prompt:** true effects, regime id/name, change
timestamps, researcher controls, any metric computed against ground truth, or the
word "changed" in reference to an actual change.

The browser UI *does* receive ground truth. That is correct — the browser is the
instrument panel, not the agent.

---

## 3. Layout

| file | holds |
|---|---|
| `world.py` | grid, berries, respawn, **hidden rules**. Ground truth lives here and only here |
| `agent.py` | `BeliefMemory`, `build_prompt`, goal vocabulary, `normalize_update`, `BaselineAgent` |
| `llm.py` | OpenRouter client, JSON extraction, retry, fallback, failure counters |
| `sim.py` | tick loop, decision triggers, event log, metrics, per-change latencies |
| `server.py` | FastAPI + SSE. Thin — no logic |
| `static/sprites.js` | every pixel asset, as 8×8 char maps |
| `static/index.html` | the researcher panel |
| `compare.py` | runs every policy through an identical world + identical silent swap |
| `runs/*.jsonl` | one event log per run, for offline analysis |

Run: `.venv/Scripts/python -m uvicorn server:app --port 8321` → http://localhost:8321
Test: `.venv/Scripts/python test_aura.py`

---

## 4. Research log — what we learned, and why the code looks like this

Every item below cost something to discover. Do not undo them casually.

### 4.1 From design analysis (before any code)

**Deterministic effects make "confidence" epistemically meaningless.**
If red is *always* exactly +15, one observation is complete evidence and the correct
response to one contradiction is an instant full flip. So v1 does **not** measure
rational belief revision under noise — it measures **anchoring**: how much the model
over-trusts its own stored belief when flipping is correct. This is worth measuring,
but it must be stated, or every latency is compared against an unstated optimum.
Optimal revision latency here is **0 observations after the first contradiction**.
Noise is a v2 change, and it changes what the instrument measures.

**Without a baseline, the numbers are uninterpretable.** "Adaptation latency 7" means
nothing alone. `BaselineAgent` (EMA + ε-greedy) runs in the same harness, same seed,
same schedule. Always compare. A weak baseline would flatter the LLM for free, so the
baseline is deliberately strong.

**Adaptation latency is four numbers, never one.** Collapsing them destroys the
finding — see §5.

**The LLM has two jobs.** It is both the belief updater and the policy chooser. If it
eats poison you cannot attribute the failure unless you separate them. So every
decision logs `greedy_under_own_beliefs` alongside the chosen goal, and every bad
outcome is classified `belief_error` (was wrong) vs `policy_error` (knew, ate anyway)
vs `investigation_cost` (deliberate, and not an error).

**Belief memory ownership must be split or the data corrupts.** The system owns
`observations`, `recent_outcomes`, `last_seen_step`. The model owns only
`estimated_effect`, `confidence`, `status`. The model writes its *reading* of the
evidence and can never edit the evidence itself.

**Confidence needs an operational definition or it is decorative.** Defined as: the
model's probability that the next time it eats this berry, the change is within ±2 of
`estimated_effect`. Every prediction then yields a calibration datum for free.

**Death contradicts persistence.** The spec wanted t=5000+ *and* survival pressure.
An agent dies right after a regime change — exactly the moment of interest. Energy
clamps at 0; `would_have_died_at` and `steps_at_critical` carry the pressure signal
instead. (Confirmed useful: the first live run hit zero at step 40 and kept going.)

### 4.2 From building it (empirical, found the hard way)

**Baseline α=0.5 silently broke belief-correction.** An EMA over a 5-slot ring buffer
holding pre-change data lands on *exactly 0* after one contradiction. `sign(0)` matches
neither `+15` nor `−15`, so `t_belief_correct` never fired and the metric read as "never
corrected" forever. A test caught it. α is now 0.8 because this world changes abruptly.
*Lesson: in a non-stationary world, a slow-averaging reference line is not a
conservative choice — it is a broken one.*

**`gpt-oss-20b` is a reasoning model and will truncate.** At `max_tokens=700` with the
full AURA prompt it burned the budget on `reasoning` and emitted the literal string
`[1]` as content. Parse failure was **50%**. Fix: `max_tokens=2000` +
`reasoning: {effort: "low"}` → **9%**.

**The JSON is often in `reasoning`, not `content`.** `_post` returns both and each is
tried as a parse candidate.

**The worst bug we found — wrong-shaped belief updates.** Live, the model emits:
```json
{"belief": "red=-15", "confidence": 0.9}
{"belief": "red", "value": "-15", "confidence": 0.9}
```
Neither matches the schema. The original `apply_updates` dropped anything without a
`berry` key — **silently**. The decision still parsed, so the run looked healthy while
every revision was thrown away. Revision latency would have measured a schema mismatch
and been reported as LLM anchoring. `normalize_update()` recovers these shapes, and
anything unrecoverable increments `dropped_updates`, which is surfaced in the UI.
*Lesson: in an instrument, silently discarding malformed input is worse than crashing.
It produces confident wrong numbers.*

**A module 404 kills the whole UI silently.** `import from './sprites.js'` resolved to
`/sprites.js` while the file is served at `/static/sprites.js`. The module never ran,
so no grid, no SSE, no errors visible except one 404 — but the static chrome still
rendered, so it *looked* like a data problem.

**"Unknown" is not "diverged."** The Reality-vs-Agent table flagged not-yet-learned
beliefs as divergent, making a fresh run look like a failing one. Three states, not two.

### 4.3 First empirical result — `gpt-oss-20b`, seed 0

Swap at step 25: `red +15→−15`, `blue −15→+15`.

```
detection           3     noticed fast (it was eating red)
revision            1     revised on the very next decision — matches the baseline
belief correction   —     never
behaviour           —     never
regret            230     and climbing
```

It fixed red and **never fixed blue**. Not from stupidity: under its now-stale belief
that blue is bad, never eating blue is the *rational* policy — so it never gathers the
evidence that would overturn it. The agent is stably, rationally wrong.

This is §1.4's detection trap reproducing itself on the very first run, and it is the
single best argument for why detection and revision must never be averaged into one
"adaptation latency."

Other counters that run: parse failure 9%, dropped updates 0, `goal = own greedy` 30%
(it deviates from its own greedy mostly to investigate — the epistemic-behaviour
signal working), investigation cost 135, would-have-died step 40.

### 4.4 Second experiment — observation starvation, isolated

`compare.py`, 8 seeds, red/blue swapped silently at step 60, 400 steps. Every agent
shares the same EMA belief updater and differs **only** in when it spends energy
re-checking something it already believes.

```
                regret   never re-sampled the changed rule
greedy            1375   8/8
uncertainty       1810   8/8      <- worse than not exploring at all
epsilon            220   1/8
age                140   0/8
contradiction      110   0/8
```

**Revision latency was 1 for every policy, on every seed.** Belief revision is not the
bottleneck and never was. The entire 16× spread in regret comes from *which evidence
the policy allowed the agent to see*. This is the finding:

> **Adaptation is limited not only by belief revision, but by which evidence the
> current policy allows the agent to see.**

Call it **observation starvation** (equivalently: policy-induced evidence starvation):

```
wrong belief -> avoid action -> no new evidence -> wrong belief persists
```

It is a distinct failure class from slow reasoning, and `greedy` reproduces it 8/8 —
it is deterministic, not a fluke of one run.

**The surprise: `uncertainty` is worse than `greedy`.** Confidence here is computed as
the consistency of `recent_outcomes`. After the swap, blue's window is still all −15 —
perfectly consistent — so blue becomes the agent's *most* confident belief precisely
because it is stale. Uncertainty-seeking exploration therefore steers **away** from the
one belief that is wrong, spends its budget re-checking things that are genuinely
noisy, and pays the exploration cost for none of the benefit.

That is the confidence-lock worth claiming, and it is narrower than "a confident wrong
belief protects itself": the mechanism is not confidence alone but **confidence used as
an exploration signal, when that confidence has no notion of age**. A consistency-based
confidence in a non-stationary world is not merely uninformative — it is actively
misleading. Note this result is a property of *that* confidence definition; a
recency- or age-weighted confidence would likely invert it, and testing that is next.

**What fixes it.** `age` (re-check anything unseen for K steps) and `contradiction`
(one rule broke, so queue every *other* belief for one re-check) both eliminate
starvation entirely, 0/8. `contradiction` is best and is the cheapest possible
meta-belief about environmental stability:

```
local contradiction -> global suspicion -> targeted re-exploration
```

`epsilon` is a lottery: it recovers on 7 seeds and starves on 1, with regret variance
an order of magnitude wider than either principled policy.

### 4.5 Third experiment — is the LLM coherent with its own beliefs?

`analyze.py`, offline over `runs/*.jsonl`, costs nothing. The live run had flagged
`goal_matches_own_beliefs` at ~30%: the agent picked a goal that was not greedy under
its own stated beliefs 70% of the time. That could be deliberate exploration or plain
incoherence, and nothing distinguished them. Over 90 scored LLM decisions:

```
  66%  deliberate exploration          (investigate_*, not greedy - and fine)
  26%  coherent
   9%  chose a worse target than its own belief
```

So the 70% is mostly **not** a defect. The agent explores on purpose, which is why its
regret should land nearer `epsilon` than `greedy`.

The 9% is the interesting residue, and it is a clean **belief correct + action wrong**
failure — a class a bandit cannot exhibit, because it has no stated reason to
contradict:

```
step 56  seek_red   (own greedy: seek_yellow)
  "Red consistently reduces energy; no evidence of positive effect.
   Need to eat red to survive."
step 58  seek_red   (own greedy: seek_yellow)
  "Red consistently gives -15; blue and yellow are stable; need energy, so eat red."
```

It states the correct belief and takes the opposite action *in the same sentence*.
Both instances occur just after energy hit 0. The working hypothesis is
**stress-induced incoherence**: under survival pressure the action collapses toward
"eat something" while the stated belief stays correct. `steps_at_critical` and
`would_have_died_at` are the covariates to test that against — do not remove them.

This is why `belief_error` and `policy_error` must stay separate counters. Reward
alone would have shown "agent ate poison" and hidden which of the two happened.

---

## 5. How to read the metrics

| metric | measures | do not confuse with |
|---|---|---|
| **detection** | change → first contradictory observation | reasoning. This is the *exploration policy*. An agent correctly exploiting red will never notice a blue change — correct behaviour, not failure |
| **revision** | contradiction → first belief update toward truth | the LLM's updating. Floor is 0 |
| **belief correction** | contradiction → belief sign matches truth for all changed berries | revision (partial ≠ corrected) |
| **behaviour** | contradiction → optimal goal chosen 3× running | belief correction (it can know and not act) |
| **regret** | energy forgone while wrong | total reward |
| **starvation** | changed rule never re-sampled, belief still sign-wrong | slow revision. The agent is not failing to think, it is failing to look |

Four failure classes, which reward alone blends into mush:

| class | meaning |
|---|---|
| `detection failure` | never observed the contradiction |
| `observation starvation` | never observed it *because its own policy avoids that action* |
| `belief error` | observed it, believed the wrong thing, acted on it |
| `policy error` | believed correctly and acted against its own belief anyway |

`investigation_cost` is energy spent deliberately gathering evidence and is **not** a
failure — do not fold it into the others.

**Validity gates.** Before trusting any run, check `llm parse fail` and
`dropped updates`. If either is climbing, you are measuring the plumbing, not the
agent, and the run is void. This is not a nicety — it is the difference between a
finding and an artefact.

---

## 6. Rules for whoever works on this next

1. **Never route around `build_prompt()`.** If you need the model to know something,
   add it to `AgentView` and justify why a real agent would know it.
2. **Never merge the four latencies** into one "adaptation" number.
3. **Never report an LLM run without its baseline run** at the same seed.
4. **Never silently drop model output.** Recover it or count it. Both, ideally.
5. **Keep the goal vocabulary closed**, and keep `investigate_X` distinct from
   `seek_X` — that distinction is the clearest observable of deliberate epistemic
   behaviour in the whole system.
6. **Seed everything.** A run that cannot be replayed cannot be compared.
7. **Do not add v2 features** (noise, scanner, tools, enemies, day/night, crafting,
   bigger maps, multiple hidden contexts). v1 answers one question. See `AURA.md` §16.
8. If you change the model, expect the parse layer to need work. `AURA_MODEL` is an
   env var precisely because the first real finding may be "this model is too small."

---

## 7. Known open ends

- The `unk` (not-yet-known) rendering in Reality-vs-Agent was fixed but **not visually
  re-verified** at step 0 — the reload landed on a run where all beliefs were known.
  Two-line change, no UI test coverage. Verify on a fresh run.
- `compare.py` has not been run with `--agents llm` at scale. That is the open
  question: where does the LLM's self-chosen policy sit against these five? Its
  `investigate_X` goals suggest something between `epsilon` and `contradiction`.
- Test whether an **age-weighted confidence** inverts the `uncertainty` result (§4.4).
  If it does, that is a concrete prescription, not just a diagnosis.
- The natural next step beyond `contradiction`: a real `P(world stable)` meta-belief
  that rises while predictions hold and falls on any surprise, scaling re-exploration
  continuously instead of via a queue.
- Calibration bins are computed and shipped in the snapshot but not yet plotted.
- Regime *schedules* work via `POST /control {"action":"schedule"}` but have no UI.
  Manual buttons only. Schedules are what make runs comparable; wire them up.
- v2's real question, per `AURA.md` §1.2: add stochastic effects, at which point
  confidence becomes epistemically meaningful and "anchoring" becomes "caution."
