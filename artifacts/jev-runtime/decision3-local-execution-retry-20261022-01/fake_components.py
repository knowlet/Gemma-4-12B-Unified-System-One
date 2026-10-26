"""Explicit CPU fixture using the real frozen request function; no model claims."""

from types import SimpleNamespace


def components(service):
    from time import sleep

    worker = service.b.load_worker()
    # This explicit fixture exercises actual request validation/journaling/fencing
    # with a tiny adapter. Real workers never replace CORE or dependencies.
    worker.CORE = None
    calls = []

    class Adapter:
        evidence_sink = None

        def predict(self, payload):
            if payload.get("fixture_delay"):
                sleep(payload["fixture_delay"])
            calls.append(payload)
            return {"fixture": True, "elements_seen": len(payload.get("elements", []))}

    class Gates:
        def check(self, _phase):
            return None

    def convert(payload, result):
        return {
            "fixture": True,
            "native_calls": len(calls),
            "elements_seen": result["elements_seen"],
            "last_element": (payload.get("elements") or [None])[-1],
        }

    value = SimpleNamespace(
        worker=worker,
        adapter=Adapter(),
        wire=SimpleNamespace(named_browser_response=convert),
        torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda _device: None)),
        gates=Gates(),
        output=service.output,
        startup=None,
        state={"cpu_fixture_only": True, "model_loaded": False, "publisher_warmup_executed": False},
        closed=False,
    )

    def close():
        if value.closed:
            raise ValueError("Fixture double close")
        value.closed = True
        return {"cpu_fixture_only": True}

    value.close = close
    return value
