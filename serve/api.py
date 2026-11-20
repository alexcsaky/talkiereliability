"""FastAPI app over the scoring engine. Independent of Modal so it can be tested.

Endpoints (build brief, section 4):

* ``POST /score``        {model, prefix, continuations}
* ``POST /score_batch``  {model, requests: [{prefix, continuations}]}
* ``POST /generate``     {model, prompt | messages, max_tokens, top_k, top_p, temperature, seed}
* ``GET  /health``       model commits, dtype, device
"""

from __future__ import annotations

from typing import Callable

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from serve.scoring import ControlMarkerError, Engine, SequenceTooLong


class ScoreReq(BaseModel):
    model: str
    prefix: str
    continuations: list[str] = Field(min_length=1)


class ScoreItem(BaseModel):
    prefix: str
    continuations: list[str] = Field(min_length=1)


class ScoreBatchReq(BaseModel):
    model: str
    requests: list[ScoreItem] = Field(min_length=1)


class GenerateReq(BaseModel):
    model: str
    prompt: str | None = None
    messages: list[dict] | None = None
    max_tokens: int = Field(256, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0)
    top_k: int | None = None
    top_p: float | None = None
    seed: int = 0


def create_app(engines: dict[str, Engine], health: Callable[[], dict],
               generate: Callable[..., dict] | None = None) -> FastAPI:
    app = FastAPI(title="talkie-backtest scoring server")

    def _engine(name: str) -> Engine:
        if name not in engines:
            raise HTTPException(404, f"model {name!r} not loaded; have {sorted(engines)}")
        return engines[name]

    def _run(fn):
        try:
            return fn()
        except ControlMarkerError as e:
            raise HTTPException(422, str(e))
        except SequenceTooLong as e:
            raise HTTPException(413, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get("/health")
    def _health():
        return health()

    @app.post("/score")
    def _score(req: ScoreReq):
        eng = _engine(req.model)
        res = _run(lambda: eng.score(req.prefix, req.continuations))
        return {"model": req.model, "results": [r.to_dict() for r in res]}

    @app.post("/score_batch")
    def _score_batch(req: ScoreBatchReq):
        eng = _engine(req.model)
        res = _run(lambda: eng.score_requests([(r.prefix, r.continuations) for r in req.requests]))
        return {"model": req.model, "results": [[c.to_dict() for c in r] for r in res]}

    @app.post("/generate")
    def _generate(req: GenerateReq):
        if generate is None:
            raise HTTPException(501, "generation not enabled on this server")
        _engine(req.model)
        return _run(lambda: generate(req.model, **req.model_dump(exclude={"model"})))

    return app
