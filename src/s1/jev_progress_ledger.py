"""Opt-in, model-neutral signals from caller-supplied browser observations.

The caller supplies a task-independent semantic projection and confirms that an
action actually executed. This module cannot observe a page, infer a goal,
choose an action, modify model answers, or decide task completion. An absent
or explicitly incomplete before/after observation is unknown, never evidence
of no change. A complete observation may contain a known JSON null.

Project relevant values explicitly: control order and timestamps outside that
projection do not affect comparison. Do not omit meaningful counters, URL
parts, form values, or other state merely to make a loop appear. Callers must
provide stable, distinct target identities rather than transient display
indices, and validate page freshness independently before acting.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from dataclasses import dataclass

MAX_BYTES = 64 * 1024


def _validate_json(value):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _validate_json(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _validate_json(item)
        return
    raise ValueError("finite JSON values and string keys required")


def _digest(value):
    try:
        _validate_json(value)
        encoded = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (RecursionError, UnicodeError, TypeError, OverflowError) as error:
        raise ValueError("bounded finite UTF-8 JSON required") from error
    if len(encoded) > MAX_BYTES:
        raise ValueError("semantic projection exceeds size limit")
    return hashlib.sha256(encoded).digest()


def _state_digest(state):
    if state is None:
        return None
    if (
        type(state) is not dict
        or set(state) != {"location", "visible", "controls", "progress", "complete"}
        or type(state["complete"]) is not bool
        or type(state["location"]) is not str
        or type(state["visible"]) is not str
        or type(state["controls"]) is not dict
        or type(state["progress"]) is not dict
    ):
        raise ValueError("explicit semantic state projection required")
    digest = _digest(state)
    return digest if state["complete"] else None


def _action_digest(action):
    if (
        type(action) is not dict
        or set(action) != {"kind", "target", "arguments"}
        or type(action["kind"]) is not str
        or not action["kind"]
        or type(action["target"]) is not str
        or not action["target"]
        or type(action["arguments"]) is not dict
    ):
        raise ValueError("explicit executed action identity required")
    return _digest(action)


@dataclass(frozen=True)
class ProgressSignal:
    sequence: int
    observation: str
    repeated_transition: bool
    alternating_cycle: bool


@dataclass(frozen=True)
class _Transition:
    before: bytes | None
    action: bytes
    after: bytes | None


class ProgressLedger:
    """Keep a bounded in-memory fingerprint ledger; never store raw inputs.

    A repeated transition is the same before/action/after after an unbroken
    chain of known observations. An alternating cycle is A→B→A→B with A != B
    and the same action for both A→B edges. These are advisory signals, not
    proof of failure; deliberate repeated actions and cyclic workflows exist.
    """

    def __init__(self, *, history_limit=32):
        if type(history_limit) is not int or not 3 <= history_limit <= 128:
            raise ValueError("history limit must be an integer between 3 and 128")
        self._history = deque(maxlen=history_limit)
        self._sequence = 0

    def record(self, *, before, action, after, executed):
        if executed is not True:
            raise ValueError("explicitly confirmed executed action required")
        before_hash = _state_digest(before)
        action_hash = _action_digest(action)
        after_hash = _state_digest(after)
        row = _Transition(before_hash, action_hash, after_hash)
        self._history.append(row)
        self._sequence += 1
        if before_hash is None or after_hash is None:
            return ProgressSignal(self._sequence, "unknown", False, False)

        rows = list(self._history)
        repeated = False
        for index in range(len(rows) - 2, -1, -1):
            previous, following = rows[index], rows[index + 1]
            if (
                previous.after is None
                or following.before is None
                or previous.after != following.before
            ):
                break
            if previous == row:
                repeated = True
                break
        cycle = (
            len(rows) >= 3
            and rows[-3].before is not None
            and rows[-3].after is not None
            and rows[-2].before is not None
            and rows[-2].after is not None
            and rows[-3].before != rows[-3].after
            and rows[-3].after == rows[-2].before == row.after
            and rows[-3].before == rows[-2].after == row.before
            and rows[-3].action == row.action
        )
        return ProgressSignal(self._sequence, "known", repeated, cycle)
