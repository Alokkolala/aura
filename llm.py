"""OpenRouter client: structured decision in, validated Decision out.

A 20B model will not honour a schema perfectly over thousands of calls, so parse
failure is treated as data, not as an exception: one retry, then a deterministic
fallback, and the failure rate is a first-class metric. If it climbs, the run is
measuring the parser rather than the agent.
"""
from __future__ import annotations

import json
import os
import time

import httpx
from dotenv import load_dotenv

from agent import GOALS, Decision, build_prompt

load_dotenv()

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-20b"


def extract_json(text: str | None) -> dict | None:
    """Pull the first balanced object out of a possibly fenced, possibly chatty reply."""
    if not text:
        return None
    t = text.strip()
    if "```" in t:
        for part in t.split("```"):
            part = part.strip()
            if part.lower().startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                t = part
                break
    start = t.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(t[start:i + 1])
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None
    # ponytail: brace counting ignores braces inside strings. Fine for this schema;
    # swap for a streaming parser if reason_summary ever carries JSON.


def coerce(obj: dict | None) -> Decision | None:
    if not obj:
        return None
    goal = str(obj.get("goal", "")).lower().strip()
    if goal not in GOALS:
        return None
    updates = obj.get("belief_updates")
    if not isinstance(updates, list):
        updates = []
    reason = str(obj.get("reason_summary", ""))[:200]
    return Decision(goal=goal, reason=reason,
                    belief_updates=[u for u in updates if isinstance(u, dict)])


class LLMClient:
    def __init__(self):
        self.model = os.environ.get("AURA_MODEL", DEFAULT_MODEL)
        self.key = os.environ.get("OPENROUTER_API_KEY", "")
        self.calls = 0
        self.parse_failures = 0
        self.transport_errors = 0
        self.total_latency = 0.0
        self.last_error: str | None = None
        self._client = httpx.AsyncClient(timeout=60.0)

    @property
    def configured(self) -> bool:
        return bool(self.key)

    async def _post(self, messages: list[dict]) -> list[str]:
        """Returns every field that might hold the answer, best candidate first.

        gpt-oss is a reasoning model: it spends tokens on `reasoning` before emitting
        `content`. Too small a max_tokens truncates it mid-thought and yields garbage
        content (observed: the literal string "[1]"), so the budget is generous and
        reasoning effort is capped. When content still fails to parse, the JSON is
        very often stated verbatim inside the reasoning trace - so try that too.
        """
        r = await self._client.post(
            API_URL,
            headers={"Authorization": f"Bearer {self.key}",
                     "Content-Type": "application/json",
                     "X-Title": "AURA"},
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 2000,
                "reasoning": {"effort": "low"},
                "response_format": {"type": "json_object"},
            },
        )
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        return [c for c in (msg.get("content"), msg.get("reasoning")) if c]

    async def decide(self, view, beliefs, history, trigger, primed=True,
                     prev_goal: str = "idle") -> Decision:
        messages = build_prompt(view, beliefs, history, trigger, primed)

        for attempt in (1, 2):
            self.calls += 1
            t0 = time.monotonic()
            try:
                candidates = await self._post(messages)
            except Exception as exc:  # network, 4xx, 5xx, rate limit
                self.transport_errors += 1
                self.last_error = f"{type(exc).__name__}: {exc}"[:200]
                break
            finally:
                self.total_latency += time.monotonic() - t0

            for raw in candidates:
                decision = coerce(extract_json(raw))
                if decision:
                    return decision

            self.parse_failures += 1
            raw = candidates[0] if candidates else ""
            self.last_error = f"unparseable reply: {str(raw)[:120]}"
            if attempt == 1:
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content":
                     "Invalid. Reply with ONLY the JSON object, no other text."},
                ]

        return Decision(goal=prev_goal, reason="llm unavailable - holding goal",
                        belief_updates=[], source="fallback")

    async def aclose(self) -> None:
        await self._client.aclose()

    def stats(self) -> dict:
        ok = max(1, self.calls)
        return {
            "model": self.model,
            "configured": self.configured,
            "calls": self.calls,
            "parse_failures": self.parse_failures,
            "transport_errors": self.transport_errors,
            "parse_failure_rate": round(self.parse_failures / ok, 3),
            "avg_latency_s": round(self.total_latency / ok, 2),
            "last_error": self.last_error,
        }
