# AURA — v1 Design Analysis & Resolved Spec

**What is being tested:** whether a persistent LLM agent, given a structured belief
store and no notification of change, will (a) learn hidden rules from interaction,
(b) rely on them, (c) notice when observations stop matching, and (d) revise.

Stripped of the world dressing, AURA v1 is a **non-stationary 3-armed bandit with an
LLM as the belief-updating mechanism**. That framing is the most useful thing to hold
onto, because it tells you immediately what the measurement is and where it can go
wrong.

---

## Part 1 — Analysis

### 1.1 What the original spec gets right

- **Hidden-rule isolation is the whole experiment.** Correctly identified as the
  single hard constraint. Everything else is negotiable; this is not.
- **Hierarchical control (LLM goals, deterministic movement).** Correct. Per-step LLM
  control would make cost the binding constraint and add motor noise to a measurement
  that is about epistemics, not navigation.
- **Prediction recorded before the outcome is revealed.** This is what turns the
  system from a demo into an instrument. Without it, "mismatch" is reconstructed
  after the fact and therefore unfalsifiable.
- **Persistent memory across regime changes.** Right call. Resetting memory on change
  would delete the exact phenomenon under study.
- **Structured, bounded memory instead of free text.** Prevents the agent from
  writing an unmeasurable narrative and calling it a belief.
- **Scope discipline in §16.** The excluded list is correctly excluded.

### 1.2 The central tension: deterministic effects make confidence meaningless

The spec asks for confidence dynamics (§11: "highlight beliefs whose confidence is
dropping") while also excluding stochastic effects (§16). These are in conflict.

If Red is *always* exactly +15, then a single observation is complete evidence. The
Bayes-optimal confidence after one observation is 1.0, and the Bayes-optimal response
to one contradictory observation is to flip fully, immediately. Optimal adaptation
latency is **one observation**. There is no rational reason for a belief to decay
gradually.

This does not break the experiment — but it changes what the experiment measures.
With deterministic rules you are not measuring *rational* belief revision under
noise. You are measuring **anchoring**: how much the LLM over-trusts its own stored
belief when the correct move is an instant flip.

That is a genuinely worthwhile thing to measure, and the deterministic setting makes
mismatch unambiguous, which is exactly what you want from a first instrument. But it
must be stated explicitly, because otherwise every latency number is silently
compared against an unstated optimum.

**Resolution for v1:** keep effects deterministic. Publish the optimal baseline on
every chart: *revision latency floor = 0 observations after the first contradiction*.
Any positive latency is anchoring, not caution. Add noise in v2, at which point
confidence becomes epistemically real and the measurement changes character.

### 1.3 The missing baseline is the biggest hole

As specified, a run produces numbers with nothing to compare them to. "Adaptation
latency 7 steps" is uninterpretable alone.

A ten-line exponential-moving-average learner with ε-greedy action selection will
solve this world near-optimally and adapt in one observation. It costs almost nothing
to build and it must run in the *same* harness, on the *same* seed, against the
*same* regime schedule.

Then every LLM result reads as: *"the LLM took 14 steps to do what the trivial
learner did in 1"* — or, more interestingly, *"the LLM matched it but explained
itself, and the explanation was wrong."* Without the baseline you cannot tell a
finding from a null result.

**This is the single highest-value addition to the spec.**

### 1.4 "Adaptation latency" is three different quantities

§14 asks for adaptation latency as one number. It is not one number, and collapsing
it destroys the finding. Decompose:

| Quantity | Interval | What it actually measures |
|---|---|---|
| **Detection latency** | rule change → first contradictory observation | the agent's *exploration policy*, not its reasoning |
| **Revision latency** | first contradiction → first belief update toward truth | LLM updating under evidence |
| **Belief-correction latency** | first contradiction → sign of belief matches truth | full correction, not partial |
| **Behavioral latency** | first contradiction → goal choice optimal for *k*=3 consecutive decisions | belief→action coupling |
| **Regret** | energy forgone from change to behavioral correction | the cost of being wrong |

Detection latency is the trap. An agent that correctly believes Red=+15 and
rationally eats only Red will **never observe** a change to Yellow. Slow detection
there is *correct exploitation*, not a failure. Only revision and behavioral latency
are about the LLM's reasoning. A single merged number would credit the LLM for good
exploration or blame it for rational exploitation.

### 1.5 The LLM has two jobs; separate them or results are unattributable

The LLM is both the belief updater and the policy chooser. If the agent eats poison,
you cannot tell whether its beliefs were wrong or its policy was bad given correct
beliefs.

Cheap fix: at every decision, compute the greedy-optimal goal **under the agent's own
stated beliefs** and log whether the LLM's chosen goal matched. That splits every
failure into `belief_error` vs `policy_error`. Two lines of code, and it is the
difference between "the agent failed" and "the agent knew and did it anyway."

### 1.6 Belief-memory ownership is ambiguous and will corrupt the data

§4 says the LLM "may propose updates." §5 shows `belief_updates`. But nothing says
who owns `observations` and `recent_outcomes`. If the LLM writes them, it will drift
and eventually fabricate its own evidence base — at which point the memory is no
longer a record of what happened.

**Hard split:**

| Field | Owner | Why |
|---|---|---|
| `observations` (count) | **system** | mechanical fact |
| `recent_outcomes` (ring buffer, k=5) | **system** | mechanical fact |
| `last_seen_step` | **system** | mechanical fact |
| `estimated_effect` | **LLM** | interpretation |
| `confidence` | **LLM** | interpretation |
| `status` (`stable` / `questioned` / `unknown`) | **LLM** | interpretation |

The system writes the evidence; the LLM writes only the reading of it. The LLM cannot
edit history.

### 1.7 Confidence needs an operational definition or it is decorative

An LLM emitting arbitrary floats gives you a number that cannot be validated, and
"confidence dropping" becomes an aesthetic observation.

Define it so it is checkable against the very next observation:

> **confidence** = the agent's stated probability that the next time it eats this
> berry, the energy change will be `estimated_effect ± 2`.

Now every prediction yields a calibration data point for free, and you can plot
stated confidence against empirical hit rate. Calibration is a far stronger result
than raw accuracy.

### 1.8 Death vs. persistence is an unresolved contradiction

§1 implies death (energy 0–100, "survive"), §14 asks for "survival steps," and §8
demands a single agent running to t=5000+. If the agent dies at step 300 the
long-horizon experiment is over — and it will most plausibly die right after a regime
change, i.e. exactly at the moment of interest.

**Resolution:** energy clamps at 0; the agent does not die. Record
`would_have_died_at_step` and `steps_at_critical_energy` (<20) as the survival-pressure
metrics. You keep the pressure signal and the long horizon. Death is the one
irreversible event the run cannot afford.

### 1.9 Reproducibility is unspecified and required

No seeds are mentioned. Without a seed for berry spawns and a full replay log, the
LLM run and the baseline run inhabit different worlds and cannot be compared. Seed
everything; log every event as JSONL so runs can be re-analyzed offline without
re-running the model.

### 1.10 `gpt-oss-20b` will break JSON, and that must be a measured quantity

A 20B model will not adhere perfectly to a schema over thousands of calls. Required:
schema-constrained output where the provider supports it, one parse-retry, then a
deterministic fallback (hold previous goal) with a logged `llm_parse_failure`.

**Track the parse-failure rate as a first-class metric.** At 30% failures you are
measuring your parser, not the agent, and the result is void. This is also the
strongest argument for the model staying env-swappable: the first real finding may be
that the model is too small for coherent belief maintenance.

### 1.11 Respawn policy is silently load-bearing

If berries respawn uniformly at random, an agent eating only Red will make Red scarce
and the board will fill with the others — or, worse, a type can vanish entirely and
"never learned the swap" becomes a spawn artifact rather than a policy choice.

**Fix:** maintain a constant count per type (e.g. 4 each). When one is eaten, respawn
that same type on a random free cell after a short delay. Availability is then
constant, and any failure to observe a berry type is attributable to the agent's
policy alone.

### 1.12 Smaller things worth deciding now

- **No obstacles on the grid ⇒ no pathfinder.** `sign(dx), sign(dy)` is the whole
  algorithm. The `Pathfinder` module in §15 is one line. It earns its own file again
  the moment walls appear — not before.
- **Regime changes need a schedule, not just buttons.** Manual buttons for
  exploration; a declarative schedule (`{step: 1000, swap: [red, blue]}`) for
  repeatable runs. Both — but the schedule is what makes experiments comparable.
- **`investigate_X` must stay distinct from `seek_X`.** It is the agent explicitly
  spending energy to reduce uncertainty — the clearest observable of deliberate
  epistemic behavior in the whole system. Collapsing the two would erase the
  experiment's most legible signal.
- **LLM call volume ≈ berries eaten.** The "new evidence" trigger fires on every eat.
  At ~1 berry per 6 steps, a 5000-step run is ~800 calls. Cheap on this model, but it
  should be rate-limitable and counted.

---

## Part 2 — Resolved Spec

Decisions from Part 1 applied. Where the original was ambiguous, this picks one.

### 2.1 World

- Grid **8×8**, no obstacles, hard edges (not toroidal).
- Berry types: `red`, `blue`, `yellow`. **4 of each**, held constant.
- Respawn: an eaten berry's type respawns on a random free cell after 3 steps.
- One agent. Energy **0–100**, starts at 50.
- Move cost **1** per step. Idle cost **0**.
- Eating: the agent moves onto a berry cell and eats; energy += hidden effect.
- Energy clamps to [0, 100]. **No death.** `would_have_died_at_step` recorded on the
  first clamp at zero.
- Everything seeded (`--seed`).

### 2.2 Hidden rules

Owned by `HiddenRuleSystem`. Initial regime: `red +15, blue −15, yellow +5`.
Deterministic in v1. Mutable at runtime by the researcher, via button or schedule.
**Never serialized into any prompt.**

### 2.3 Agent architecture

```
LLM  →  goal (closed vocabulary)  →  deterministic executor  →  steps
```

Goal vocabulary (closed set, validated):

```
seek_red | seek_blue | seek_yellow
investigate_red | investigate_blue | investigate_yellow
idle
```

`seek_*` = exploit. `investigate_*` = spend energy to gather evidence. `idle` = rest,
which is the correct answer when every known effect is negative.

Executor: step toward the nearest target of the goal's type via `sign(dx), sign(dy)`;
eat on arrival; goal completes.

**LLM is re-invoked only on:** goal completed · berry eaten · prediction mismatch ·
energy < 20 · no active goal · hard cap of N steps without a decision.
Never per movement step.

### 2.4 Belief memory

```json
{
  "red": {
    "estimated_effect": 15,
    "confidence": 0.91,
    "status": "stable",
    "observations": 8,
    "recent_outcomes": [15, 15, 15, 15, 15],
    "last_seen_step": 402
  }
}
```

First three fields LLM-owned, last three system-owned (§1.6). Persists across regime
changes; never reset. The LLM writes only through `belief_updates`; system fields are
rejected if present in its output.

### 2.5 LLM contract

**Prompt contains only:** step · energy · visible grid (types + positions) · full
belief memory · last 10 self-observations `(berry, predicted, actual, mismatch)` ·
trigger reason · goal vocabulary.

**Prompt never contains:** true effects · regime id or name · change timestamps ·
researcher controls · any ground-truth-derived metric · the word "changed."

Enforced by a single `build_prompt()` seam plus a test asserting no ground-truth value
appears in the serialized prompt. That test is the validity guarantee of every run in
the project.

**Output (schema-constrained):**

```json
{
  "goal": "seek_red",
  "reason_summary": "Red has the highest expected gain.",
  "belief_updates": [
    {"berry": "red", "estimated_effect": 15, "confidence": 0.63, "status": "questioned"}
  ]
}
```

`reason_summary` capped at 200 chars and truncated. Invalid output → one retry →
deterministic fallback (hold previous goal) + logged `llm_parse_failure`.

### 2.6 Prediction protocol

At the instant an eat is committed, **before** the effect is computed:

```
predicted_effect = beliefs[type].estimated_effect    # snapshotted, not recomputed
predicted_conf   = beliefs[type].confidence
```

Then apply the hidden effect and record:

```
actual_effect
mismatch               = abs(actual - predicted) > 2
within_confidence_band = abs(actual - predicted) <= 2   # calibration datum
```

Snapshot at commit time, never reconstructed afterward.

### 2.7 Metrics

**Continuous:** total reward · energy · steps · berries eaten by type · predictions ·
prediction accuracy · mismatches · sign-accuracy of belief vs truth · magnitude error ·
regret vs. an oracle · LLM calls · parse-failure rate · confidence calibration curve ·
`belief_error` vs `policy_error` split (§1.5) · steps at critical energy.

**Per regime change:** `t_change` · `t_contradiction` · `t_revision` ·
`t_belief_correct` · `t_behavior_correct` (sustained k=3) · regret over the gap.
Reported as the four latencies of §1.4, never merged into one.

Change timestamps are recorded for the researcher and are structurally unreachable
from the prompt builder.

### 2.8 Baseline agent

Same harness, same seed, same regime schedule. EMA belief update (α=0.8 — high, because the world changes abruptly) with ε-greedy
selection (ε=0.1). Runs as `--agent=baseline`. Every LLM chart carries its line.

### 2.9 UI

One page, four panes:

- **World** — grid, agent, berries, step, energy, current goal, with exploit vs.
  investigate visually distinct.
- **Agent Mind** — per berry: estimated effect, confidence, observations, status.
  Highlight `questioned` and any confidence that dropped on the last update.
- **Reality vs Agent** — truth ‖ belief ‖ sign-match indicator. The one table that
  makes the whole experiment legible at a glance.
- **Event log** — chronological, mismatches flagged, regime changes marked in the
  researcher's view only.
- **Controls** — start · pause · reset · speed · manual rule edit · `SWAP RED/BLUE` ·
  regime presets · seed · agent selector (llm / baseline).

Researcher pane and agent context are separated by construction, not by discipline:
the prompt builder is a pure function of agent-visible state and cannot reach world
truth.

### 2.10 Architecture (lean cut)

The 12 modules of §15 are the right *seams*, not the right *file count*. For v1:

```
world.py     World · Berry · HiddenRuleSystem · respawn · apply_eat
agent.py     BeliefMemory · build_prompt · goal execution · movement
llm.py       OpenRouter client · schema · retry · fallback
sim.py       loop · triggers · metrics · JSONL event log · regime schedule
server.py    FastAPI + SSE
index.html   the four panes
```

Six files. The one architecturally load-bearing seam — `build_prompt()` as the sole
path from world state to model context — is preserved exactly.

`OPENROUTER_API_KEY` and `AURA_MODEL` (default `openai/gpt-oss-20b`) come from env.
No key in code.

---

## Part 3 — Open decisions (yours)

1. **Stack.** Recommend Python + FastAPI + one HTML page over SSE: researchers will
   want the JSONL event log in pandas, and Python keeps the sim and the analysis in
   one language. Flip to TS/Next if the UI matters more than the analysis.
2. **Noise in v1?** Recommend no — deterministic makes mismatch unambiguous. Accept
   that v1 measures anchoring rather than rational caution (§1.2), and add noise in v2.
3. **Full grid visibility vs. local view?** Recommend full (8×8, no scanner in v1). A
   local view adds a partial-observability confound to an epistemics experiment.
4. **Does the baseline agent ship in v1?** Recommend yes — roughly 40 lines, and
   without it the first run's numbers have no reference line (§1.3).

---

## Part 4 — What counts as a result

**A finding:** the agent holds a high-confidence belief, meets one contradictory
observation, and its revision behavior is measurably different from the EMA baseline —
either it anchors (revision latency > 0, regret accumulates) or it over-corrects on a
single datum. Both are publishable claims about LLM belief maintenance.

**A null result:** the parse-failure rate is high, or belief updates are uncorrelated
with observations, or chosen goals are uncorrelated with the agent's own stated
beliefs. That means the model is too small for the task — which the swappable model
config exists to test, and which is worth knowing early.

**The instrument works if** you can watch the Reality-vs-Agent table diverge after a
silent swap and then converge again — and you can say precisely how many steps each
phase took and what it cost.
