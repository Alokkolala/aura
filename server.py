"""FastAPI host: serves the researcher panel and streams world state over SSE.

Everything this server sends to the browser is the RESEARCHER view, ground truth
included. That is correct — the browser is the instrument panel, not the agent.
The agent's only window on the world is agent.build_prompt().
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from sim import Sim

STATE: dict = {"sim": None, "task": None}


def _restart(new_sim: Sim) -> None:
    if STATE["task"]:
        STATE["task"].cancel()
    STATE["sim"] = new_sim
    STATE["task"] = asyncio.create_task(new_sim.run_forever())


@asynccontextmanager
async def lifespan(app: FastAPI):
    _restart(Sim())
    yield
    if STATE["task"]:
        STATE["task"].cancel()
    await STATE["sim"].llm.aclose()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/events")
async def events():
    async def gen():
        while True:
            yield f"data: {json.dumps(STATE['sim'].snapshot())}\n\n"
            await asyncio.sleep(0.1)
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


@app.post("/control")
async def control(req: Request):
    body = await req.json()
    action = body.get("action")
    sim: Sim = STATE["sim"]

    if action == "start":
        sim.running = True
    elif action == "pause":
        sim.running = False
    elif action == "speed":
        sim.speed = max(1.0, min(50.0, float(body.get("value", 6))))
    elif action == "swap":
        sim.swap_red_blue()
    elif action == "preset":
        sim.apply_preset(body.get("name", ""))
    elif action == "set_rule":
        berry, value = body.get("berry"), int(body.get("value", 0))
        if berry in sim.world.truth:
            sim.set_truth({berry: value})
    elif action == "schedule":
        sim.schedule = body.get("items", [])
    elif action == "reset":
        _restart(Sim(
            seed=int(body.get("seed", sim.seed)),
            agent_kind=body.get("agent", sim.agent_kind),
            primed=bool(body.get("primed", sim.primed)),
        ))
    return {"ok": True}
