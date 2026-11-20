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
                 chunk: int = 512, *, allow_leak_probes: bool = False):
        self.backend = backend
        self.model = model
        self.chunk = chunk
        self.allow_leak_probes = allow_leak_probes

    def score(self, items: Sequence[tuple[Admitted, Sequence[Admitted]]]) -> list[list[dict]]:
        payload = []
        for prefix, conts in items:
            for s in (prefix, *conts):
                if not isinstance(s, Admitted):
                    raise NotAdmitted(f"not admitted by the firewall: {str(s)[:60]!r}")
                if s.leak_probe and not self.allow_leak_probes:
                    raise NotAdmitted(
                        f"leak-probe-exempt string {s.origin!r} refused by confirmatory client"
                    )
            payload.append({"prefix": prefix.text, "continuations": [c.text for c in conts]})
        out: list[list[dict]] = []
        for a in range(0, len(payload), self.chunk):
            out.extend(self.backend(self.model, payload[a: a + self.chunk]))
        return out


class EngineBackend:
    def __init__(self, engines: dict, *, chat_user_prompt: str | None = None):
        self.engines = engines
        self.chat_user_prompt = chat_user_prompt

    def __call__(self, model: str, payload: list[dict]) -> list[list[dict]]:
        res = self.engines[model].score_requests(
            [(p["prefix"], p["continuations"]) for p in payload],
            chat_user_prompt=self.chat_user_prompt,
        )
        return [[c.to_dict() for c in r] for r in res]


class ModalBackend:
    """Calls ``TalkieServer.score_batch`` on Modal.

    ``server`` is a ``TalkieServer(model=...)`` instance from serve/modal_app.py,
    or a looked-up deployed class instance.
    """

    def __init__(self, servers: dict, prepend_eot: bool = True,
                 chat_user_prompt: str | None = None):
        self.servers = servers
        self.prepend_eot = prepend_eot
        self.chat_user_prompt = chat_user_prompt

    def __call__(self, model: str, payload: list[dict]) -> list[list[dict]]:
        return self.servers[model].score_batch.remote(
            payload, prepend_eot=self.prepend_eot,
            chat_user_prompt=self.chat_user_prompt,
        )
