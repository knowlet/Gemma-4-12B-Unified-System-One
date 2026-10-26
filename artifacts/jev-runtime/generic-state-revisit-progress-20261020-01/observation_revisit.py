"""Experimental bounded observation metadata; no browser/model integration yet.

The exact key includes the accepted document's page_key, guards and geometry.
The optional content projection is explicitly weaker: it excludes only known
native binding IDs, geometry and guard payloads, retaining ordered descriptors,
all observed editable values and all existing visible text without truncation.
Neither key authorizes an action, proves lack of progress, or changes candidates.

Design reference: browser-use/browser-use ActionLoopDetector (MIT, main read
2026-10-20) provides bounded informational messages rather than blocking actions.
This implementation uses different, conservative observation identities; no
upstream normalization or implementation was copied.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque

PROFILE = "guarded-observation-revisit-experimental-v1"
MAX_WINDOW = 32
MAX_SEQUENCE = 2**53 - 1
MAX_METADATA_BYTES = 4096
_REQUIRED = {
    "observation_profile",
    "url",
    "title",
    "w",
    "h",
    "text",
    "scroll",
    "actions",
    "marker",
    "page_key",
    "guards",
    "omitted_actions",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite_json(value, ancestors=None, depth=0):
    _require(depth <= 64, "Observation JSON nesting exceeds64")
    kind = type(value)
    if kind in (dict, list):
        ancestors = set() if ancestors is None else ancestors
        _require(id(value) not in ancestors, "Cyclic observation JSON")
        ancestors.add(id(value))
        try:
            if kind is dict:
                _require(all(type(k) is str for k in value), "JSON object keys must be strings")
                children = value.values()
            else:
                children = value
            for child in children:
                _finite_json(child, ancestors, depth + 1)
        finally:
            ancestors.remove(id(value))
    else:
        _require(kind in (str, int, float, bool, type(None)), "Finite JSON values required")
        _require(kind is not float or math.isfinite(value), "Finite JSON numbers required")


def _digest(value):
    _finite_json(value)
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(
        ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    try:
        for chunk in encoder.iterencode(value):
            digest.update(chunk.encode("utf-8"))
    except (TypeError, ValueError, UnicodeError, OverflowError) as exc:
        raise ValueError("Finite UTF-8 observation JSON required") from exc
    return digest.hexdigest()


def _validate_page(page):
    _require(
        type(page) is dict and _REQUIRED <= page.keys(), "Complete accepted observation required"
    )
    _require(
        all(type(page[k]) is str for k in ("url", "title", "text"))
        and type(page["observation_profile"]) is dict
        and type(page["observation_profile"].get("id")) is str
        and bool(page["observation_profile"]["id"])
        and type(page["actions"]) is list
        and type(page["guards"]) is dict
        and type(page["marker"]) is list
        and type(page["scroll"]) is dict
        and all(type(page[k]) is int and page[k] > 0 for k in ("w", "h")),
        "Typed native observation fields required",
    )
    key = page["page_key"]
    _require(
        type(key) is list
        and len(key) == 7
        and type(key[6]) is list
        and type(key[0]) in (int, float)
        and not isinstance(key[0], bool)
        and key[0] > 0
        and key[1] == page["url"]
        and type(key[2]) in (int, float)
        and type(key[3]) in (int, float)
        and key[3] == page["scroll"].get("y")
        and type(key[4]) is int
        and type(key[5]) is int
        and key[4] == page["w"]
        and key[5] == page["h"],
        "Complete document/page_key observation required",
    )
    _require(
        all(
            type(row) is list and len(row) == 6 and type(row[0]) is int and row[0] > 0
            for row in key[6]
        )
        and all(
            type(action) is dict
            and type(action.get("id")) is str
            and type(action.get("kind")) is str
            for action in page["actions"]
        ),
        "Native editable rows and ordered action descriptors required",
    )
    if "x" in page["scroll"]:
        _require(page["scroll"]["x"] == key[2], "Contradictory horizontal scroll")


def _project_action(action, position):
    # Native snapshot.js assigns e{position}; it never copies a DOM semantic id
    # into this field. Unrecognized id/name/extra fields are retained verbatim.
    binding = (
        action.get("id") == f"e{position}"
        and action.get("kind") in {"click", "fill", "select"}
        and type(action.get("node")) is int
        and action["node"] > 0
    )
    excluded = set()
    if binding:
        excluded.update(("id", "node", "rect"))
        if "control_node" in action and "control_type" in action:
            excluded.update(("control_node", "control_guard"))
        if action.get("affordance") == "css-hover-entry-v5":
            excluded.add("hover_owner")
    # These are the three appended native control bindings, not semantic IDs.
    controls = {"scroll_down": "scroll", "scroll_up": "scroll", "wait": "wait"}
    if action.get("id") in controls and action.get("kind") == controls[action["id"]]:
        excluded.add("id")
    return {k: v for k, v in action.items() if k not in excluded}


def observation_signatures(page, *, include_projection=False):
    """Compute own keys; never trust or rewrite the weaker native fingerprint."""
    _require(type(include_projection) is bool, "Projection must be explicitly enabled")
    _validate_page(page)
    exact = {k: v for k, v in page.items() if k not in {"fingerprint", "screenshot"}}
    exact_key = _digest(exact)
    projection_key = None
    if include_projection:
        projection = {k: page[k] for k in ("url", "title", "text", "w", "h")}
        projection["observation_profile_id"] = page["observation_profile"]["id"]
        projection.update(
            scroll={**page["scroll"], "x": page["page_key"][2]},
            editable_values=[row[1:] for row in page["page_key"][6]],
            actions=[_project_action(action, i) for i, action in enumerate(page["actions"], 1)],
        )
        projection_key = _digest(projection)
    return exact_key, projection_key


class ObservationRevisitTracker:
    """Caller records accepted observations, not predictions or planned actions.

    A sequence is assigned once by the caller's observation journal. Repeating
    that sequence with identical content is idempotent. No raw observation,
    candidate, goal, history, field text or guard is retained by this tracker.
    """

    def __init__(self, *, window=MAX_WINDOW, include_projection=False):
        _require(type(window) is int and 1 <= window <= MAX_WINDOW, "Window must be1..32")
        _require(type(include_projection) is bool, "Projection must be explicitly enabled")
        self.window, self.include_projection = window, include_projection
        self._entries = deque(maxlen=window)
        self._last = None

    @property
    def retained_count(self):
        return len(self._entries)

    def observe(self, page, *, observation_sequence):
        _require(
            type(observation_sequence) is int and 0 <= observation_sequence <= MAX_SEQUENCE,
            "Observation sequence must be a bounded nonnegative integer",
        )
        exact, projection = observation_signatures(page, include_projection=self.include_projection)
        if self._last is not None:
            sequence, old_exact, old_projection, exact_prior, projection_prior = self._last
            _require(observation_sequence >= sequence, "Observation sequence moved backwards")
            if observation_sequence == sequence:
                _require(
                    (exact, projection) == (old_exact, old_projection), "Sequence content changed"
                )
                return self._metadata(sequence, exact, projection, exact_prior, projection_prior)
        exact_prior = tuple(s for s, e, _ in self._entries if e == exact)
        projection_prior = tuple(
            s for s, _, p in self._entries if projection is not None and p == projection
        )
        result = self._metadata(
            observation_sequence, exact, projection, exact_prior, projection_prior
        )
        self._entries.append((observation_sequence, exact, projection))
        self._last = (observation_sequence, exact, projection, exact_prior, projection_prior)
        return result

    def _metadata(self, sequence, exact, projection, exact_prior, projection_prior):
        notes = []
        if exact_prior:
            notes.append(
                "This document observation exactly matches retained observation sequences "
                + ",".join(map(str, exact_prior))
                + "."
            )
        if projection_prior:
            notes.append(
                "The defined content projection, excluding native binding IDs and geometry, matches retained observation sequences "
                + ",".join(map(str, projection_prior))
                + "."
            )
        result = {
            "profile": PROFILE,
            "observation_sequence": sequence,
            "window_limit": self.window,
            "exact": {
                "sha256": exact,
                "previous_observation_sequences": list(exact_prior),
                "scope": "same document, complete observed page_key/guards/actions/geometry; excludes only screenshot and supplied fingerprint",
            },
            "projection": None
            if projection is None
            else {
                "sha256": projection,
                "previous_observation_sequences": list(projection_prior),
                "scope": "defined observed content/editable values/ordered action descriptors/scroll/viewport; excludes document time, known native bindings, geometry and guards; not complete state or action identity",
            },
            "observation_notes": notes,
        }
        _require(
            len(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode())
            <= MAX_METADATA_BYTES,
            "Bounded observation metadata exceeded",
        )
        return result
