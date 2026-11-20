"""The only route from forecasting code to Talkie.

``TalkieClient`` accepts nothing but ``firewall.Admitted`` strings, so every
prefix and continuation has been through the firewall (rule 1). Backends:

* ``ModalBackend``: the deployed or ephemeral ``TalkieServer`` on Modal.
* ``EngineBackend``: an in-process ``serve.scoring.Engine`` (tests).
"""

from __future__ import annotations

from typing import Callable, Sequence

from firewall import Admitted

Backend = Callable[[str, list[dict]], list[list[dict]]]


class NotAdmitted(TypeError):
    pass


class TalkieClient:
    def __init__(self, backend: Backend, model: str = "talkie-1930-13b-base",
                 chunk: int = 512):
        self.backend = backend
        self.model = model
        self.chunk = chunk

    def score(self, items: Sequence[tuple[Admitted, Sequence[Admitted]]]) -> list[list[dict]]:
        payload = []
        for prefix, conts in items:
            for s in (prefix, *conts):
                if not isinstance(s, Admitted):
                    raise NotAdmitted(f"not admitted by the firewall: {str(s)[:60]!r}")
            payload.append({"prefix": prefix.text, "continuations": [c.text for c in conts]})
        out: list[list[dict]] = []
        for a in range(0, len(payload), self.chunk):
            out.extend(self.backend(self.model, payload[a: a + self.chunk]))
        return out


class EngineBackend:
    def __init__(self, engines: dict):
        self.engines = engines

    def __call__(self, model: str, payload: list[dict]) -> list[list[dict]]:
        res = self.engines[model].score_requests(
            [(p["prefix"], p["continuations"]) for p in payload])
        return [[c.to_dict() for c in r] for r in res]


class ModalBackend:
    """Calls ``TalkieServer.score_batch`` on Modal.

    ``server`` is a ``TalkieServer(model=...)`` instance from serve/modal_app.py,
    or a looked-up deployed class instance.
    """

    def __init__(self, servers: dict, prepend_eot: bool = True):
        self.servers = servers
        self.prepend_eot = prepend_eot

    def __call__(self, model: str, payload: list[dict]) -> list[list[dict]]:
        return self.servers[model].score_batch.remote(payload, prepend_eot=self.prepend_eot)
