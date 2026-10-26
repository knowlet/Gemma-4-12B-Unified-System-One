"""Optional generic action outcome feedback; observation and action rules stay intact."""

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps

PROFILE = "typed-stale-action-feedback-v1"
POLICY_URL = "https://api.typesafe.ai/v1/systemone"
MAX_RECENT = 4
_ACTIVE = ContextVar("jev_action_feedback", default=None)


def _identifier(value):
    return value if isinstance(value, str) and len(value) <= 128 else None


def _record(state, phase, decision, before_history, fingerprint):
    """A missing history record does not prove absence of physical browser input."""
    number = state.get("action_feedback_count", 0) + 1
    decision = decision if isinstance(decision, dict) else {}
    item = {
        "attempt": number,
        "outcome": "StalePage",
        "phase": phase,
        "operation": _identifier(decision.get("operation")),
        "target": _identifier(decision.get("target")),
        "choice": _identifier(decision.get("choice")),
        "command_entry_page_fingerprint": _identifier(fingerprint),
        "recorded_action_delta": len(state["history"]) - before_history,
        "physical_input_outcome": "unknown",
    }
    state["action_feedback_count"] = number
    state["action_feedback"] = [*state.get("action_feedback", []), item][-MAX_RECENT:]


@contextmanager
def install(agent_module, model_module):
    """Explicitly wrap selected modules for one sequential benchmark context."""
    agent_class = agent_module.Agent
    original_command = agent_class.command
    original_post = model_module.post_json
    if getattr(original_command, "_jev_feedback_profile", None) is not None:
        raise ValueError("Action feedback is already installed")
    if agent_module.choose is not model_module.choose:
        raise ValueError("Agent and policy model must share their original choose function")

    @wraps(original_command)
    def command(self, name, body=None):
        state = self.state
        before_history = len(state["history"])
        fingerprint = state.get("page", {}).get("fingerprint")
        decision = deepcopy(state.get("decision")) if name == "act" else None
        token = _ACTIVE.set(state)
        try:
            return original_command(self, name, body)
        except agent_module.StalePage:
            # Original tick catches these inner exceptions and re-observes normally.
            if name in {"predict", "act"}:
                _record(state, name, decision, before_history, fingerprint)
            raise
        finally:
            _ACTIVE.reset(token)

    command._jev_feedback_profile = PROFILE

    @wraps(original_post)
    def post(url, key, body):
        state = _ACTIVE.get()
        if url == POLICY_URL and state is not None and state.get("action_feedback"):
            if "action_feedback" in body["state"]:
                raise ValueError("Policy body already contains action feedback")
            body = {
                **body,
                "state": {
                    **body["state"],
                    "action_feedback": deepcopy(state["action_feedback"]),
                },
            }
        return original_post(url, key, body)

    agent_class.command = command
    model_module.post_json = post
    try:
        yield {"agent_protocol": PROFILE, "max_recent": MAX_RECENT}
    finally:
        agent_class.command = original_command
        model_module.post_json = original_post


def wrap_model_adapter(original_adapter):
    """Install inside the existing adapter so recording and routing see feedback."""

    @contextmanager
    @wraps(original_adapter)
    def adapt(agent_module, model_module, **kwargs):
        with original_adapter(agent_module, model_module, **kwargs) as result:
            with install(agent_module, model_module):
                yield result

    return adapt
