"""Held opt-in advisory for an already observed Jev browser episode.

This helper neither observes nor executes. An owner must bind the actual
snapshot, executed-action history and next policy call to the same episode.
Signals compare a bounded visible projection, not the entire DOM, a task's
success, or proof that nothing happened. Unobserved frames/shadow controls,
omitted actions, text caps, asynchronous work and visually identical but
semantically different pages can defeat or mislead the comparison.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re

from s1.jev_observation_events import _metadata_valid
from s1.jev_progress_ledger import MAX_BYTES, ProgressLedger

SOURCE_ONLY = True
PROFILE = "composed-visible-text-settled-v7"
KEY = "observed_progress_advisory"
SCOPE = "bounded_visible_projection_not_full_dom_or_task_outcome"


def _gate():
    if SOURCE_ONLY is not False:
        raise ValueError("held progress integration requires independent review")


def _encode(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _projection(page, *, cross_document=False):
    """Return a complete projection of the bounded snapshot, or None.

    ``complete`` applies only to the explicitly documented projection. It is
    never a claim that the page, the visible DOM, or a user goal is complete.
    """
    try:
        if (
            type(page) is not dict
            or not _metadata_valid(
                page.get("observation_profile"), exhausted=False, profile=PROFILE
            )
            or type(page.get("url")) is not str
            or type(page.get("title")) is not str
            or type(page.get("text")) is not str
            or len(page["text"]) >= 6000
            or type(page.get("actions")) is not list
            or type(page.get("guards")) is not dict
            or type(page.get("omitted_actions")) is not int
            or page["omitted_actions"] != 0
            or type(page.get("scroll")) is not dict
            or type(page.get("page_key")) is not list
            or not page["page_key"]
            or type(page["page_key"][0]) not in (int, float)
            or not math.isfinite(page["page_key"][0])
            or type(page.get("fingerprint")) is not str
            or not page["fingerprint"]
            or type(page.get("marker")) is not list
            or len(page["marker"]) != 10
            or page["marker"][1] != page["url"]
            or page["marker"][6] != page["title"]
            or page["marker"][7] != page["text"]
        ):
            return None
        control_values = None
        if cross_document:
            key, marker = page["page_key"], page["marker"]
            if (
                len(key) != 7
                or type(marker[0]) is not type(key[0])
                or marker[0] != key[0]
                or key[1] != page["url"]
                or marker[2:6] != key[2:6]
                or marker[4:6] != [page.get("w"), page.get("h")]
                or page["scroll"].get("y") != key[3]
                or type(key[6]) is not list
                or marker[9] != key[6]
                or type(marker[8]) is not list
            ):
                return None
            action_semantics = [
                {k: v for k, v in item.items() if k not in {"id", "rect"}}
                for item in page["actions"]
                if type(item) is dict and item.get("kind") not in {"scroll", "wait"}
            ]
            if marker[8] != action_semantics:
                return None
            control_values, control_nodes = [], set()
            for control in key[6]:
                if (
                    type(control) is not list
                    or len(control) != 6
                    or type(control[0]) is not int
                    or control[0] <= 0
                    or control[0] in control_nodes
                ):
                    return None
                control_nodes.add(control[0])
                # Preserve the native control order and every state field.
                control_values.append(control[1:])
        observations = []
        guard_at_cap = False
        seen_ids = set()
        for item in page["actions"]:
            if (
                type(item) is not dict
                or type(item.get("id")) is not str
                or not item["id"]
                or item["id"] in seen_ids
                or type(item.get("kind")) is not str
                or type(item.get("label")) is not str
            ):
                return None
            seen_ids.add(item["id"])
            fields = {
                key: item[key]
                for key in (
                    "kind",
                    "label",
                    "role",
                    "value",
                    "current_value",
                    "checked",
                    "selected",
                    "expanded",
                    "delta",
                )
                if key in item
            }
            if "node" in item:
                node = item["node"]
                guard = page["guards"].get(str(node))
                if (
                    type(node) is not int
                    or type(guard) is not list
                    or len(guard) != 14
                    or guard[0] != node
                    or type(guard[-1]) is not str
                    or len(guard[-1]) >= 6000
                ):
                    return None
                fields["guard"] = guard[1:]
                guard_at_cap |= len(guard[-1]) == 6000
            elif item["kind"] not in {"scroll", "wait"}:
                return None
            observations.append(fields)
        observations.sort(key=_encode)
        flags = {
            "text_at_cap": len(page["text"]) == 6000,
            "guard_text_at_cap": guard_at_cap,
            "omitted_actions": page["omitted_actions"],
        }
        state = {
            "location": page["url"],
            "visible": page["text"],
            "controls": {"observed_actions": observations},
            "progress": {
                "document_token": page["page_key"][0],
                "title": page["title"],
                "scroll": page["scroll"],
                "viewport": [page.get("w"), page.get("h")],
                "scope_flags": flags,
            },
            "complete": True,
        }
        if cross_document:
            del state["progress"]["document_token"]
            state["progress"]["window"] = page["page_key"][2:6]
            state["controls"]["native_control_state"] = control_values
        if len(_encode(state).encode("utf-8")) > MAX_BYTES:
            return None
        return state, flags
    except (KeyError, TypeError, ValueError, OverflowError, UnicodeError, RecursionError):
        return None


def _semantic_target(page, selected):
    """Resolve a unique target from observed semantics, never a node index."""

    def identity(action):
        kind = action.get("kind")
        if kind in {"scroll", "wait"}:
            return _encode({"kind": kind, "delta": action.get("delta")})
        node = action.get("node")
        guard = page["guards"].get(str(node))
        if type(node) is not int or type(guard) is not list or len(guard) != 14:
            raise ValueError("guarded semantic target required")
        return _encode(
            {
                "kind": kind,
                "label": action.get("label"),
                "role": action.get("role"),
                "value": action.get("value"),
                "current_value": action.get("current_value"),
                "guard": guard[1:],
            }
        )

    target = identity(selected)
    if sum(identity(item) == target for item in page["actions"]) != 1:
        raise ValueError("semantic target is ambiguous")
    return target


def _snapshot_binding(page):
    raw = _encode(page).encode("utf-8")
    if len(raw) > 1024 * 1024:
        raise ValueError("bounded snapshot required")
    return hashlib.sha256(raw).digest()


def _policy_binding(page, history_entry):
    """Fingerprint the upstream model's next-page and last-action view."""
    elements, indices = [], {}
    operations = {"click": "CLICK", "fill": "TYPE_TEXT", "select": "SELECT"}
    for action in page["actions"]:
        kind = action["kind"]
        if kind not in operations:
            continue
        node = action["node"]
        if node not in indices:
            index = str(len(elements) + 1)
            indices[node] = index
            element = {
                key: action[key]
                for key in ("role", "value", "checked", "selected", "expanded")
                if key in action
            }
            element.update(index=index, label=action["label"].split(" → ")[0], operations=[])
            if kind == "select":
                element["value"] = action.get("current_value", "")
                element["options"] = []
            elements.append(element)
        index = indices[node]
        element = elements[int(index) - 1]
        operation = operations[kind]
        if operation not in element["operations"]:
            element["operations"].append(operation)
        if kind == "select":
            element["options"].append(
                {
                    "index": f"{index}:{len(element['options']) + 1}",
                    "label": action["label"],
                    "value": action["value"],
                }
            )
    view = {
        "page": {key: page[key] for key in ("url", "title", "text")},
        "elements": elements,
        "last_action": {
            key: history_entry.get(key) for key in ("action", "kind", "text", "page_changed")
        },
    }
    return hashlib.sha256(_encode(view).encode("utf-8")).digest()


def _episode_binding(episode_id, page, history_entry, execution):
    raw = _encode(
        {
            "episode_id": episode_id,
            "page": page,
            "history_entry": history_entry,
            "execution": execution,
        }
    ).encode("utf-8")
    if len(raw) > 1024 * 1024:
        raise ValueError("bounded snapshot and history required")
    return hashlib.sha256(raw).digest()


class BrowserProgressIntegration:
    """One serial episode; parent must authenticate actual source and events."""

    def __init__(self, *, episode_id, history_limit=32, cross_document=False):
        _gate()
        if type(episode_id) is not str or re.fullmatch(r"[0-9a-f]{32}", episode_id) is None:
            raise ValueError("explicit unique episode identity required")
        if type(cross_document) is not bool:
            raise ValueError("explicit cross-document opt-in required")
        self._episode = episode_id
        self._cross_document = cross_document
        self._limit = history_limit
        self._ledger = ProgressLedger(history_limit=history_limit)
        self._step = None
        self._pending = None
        self._binding = None
        self._episode_binding = None
        self._previous_snapshot = None

    def _unknown(self):
        self._ledger = ProgressLedger(history_limit=self._limit)
        self._step = None
        self._pending = None
        self._binding = None
        self._episode_binding = None
        self._previous_snapshot = None
        return {"observation": "unknown"}

    def record_committed(self, *, episode_id, before, selected_id, after, history_entry, execution):
        """Accept the entry recorded after Browser.act returns, not a prediction.

        The owner supplies an independently verified execution attestation;
        matching fields are consistency checks, not proof of provenance.
        """
        _gate()
        try:
            if (
                episode_id != self._episode
                or type(history_entry) is not dict
                or type(history_entry.get("step")) is not int
                or history_entry["step"] < 1
                or (self._step is not None and history_entry["step"] != self._step + 1)
                or type(selected_id) is not str
                or not selected_id
                or history_entry.get("choice") != selected_id
                or type(before) is not dict
                or type(before.get("actions")) is not list
            ):
                return self._unknown()
            if (
                type(execution) is not dict
                or set(execution)
                != {
                    "episode_id",
                    "step",
                    "choice",
                    "document",
                    "before_fingerprint",
                    "after_fingerprint",
                    "status",
                }
                or execution["episode_id"] != episode_id
                or type(execution["step"]) is not int
                or execution["step"] != history_entry["step"]
                or execution["choice"] != selected_id
                or execution["status"] != "confirmed_executed"
                or execution["before_fingerprint"] != before.get("fingerprint")
                or type(before.get("page_key")) is not list
                or not before["page_key"]
                or type(execution["document"]) is not type(before["page_key"][0])
                or execution["document"] != before["page_key"][0]
                or execution["after_fingerprint"]
                != (after.get("fingerprint") if type(after) is dict else None)
                or ("executed" in history_entry and history_entry["executed"] is not True)
                or history_entry.get("error") is not None
            ):
                return self._unknown()
            matches = [
                row
                for row in before["actions"]
                if type(row) is dict and row.get("id") == selected_id
            ]
            if len(matches) != 1:
                return self._unknown()
            action = matches[0]
            if action.get("kind") != history_entry.get("kind") or action.get(
                "label"
            ) != history_entry.get("action"):
                return self._unknown()
            pre = _projection(before, cross_document=self._cross_document)
            post = (
                _projection(after, cross_document=self._cross_document)
                if after is not None
                else None
            )
            if after is not None:
                if (
                    type(after) is not dict
                    or type(before.get("fingerprint")) is not str
                    or type(after.get("fingerprint")) is not str
                    or type(history_entry.get("page_changed")) is not bool
                    or history_entry["page_changed"]
                    != (after["fingerprint"] != before["fingerprint"])
                    or history_entry.get("url") != after.get("url")
                ):
                    return self._unknown()
            elif history_entry.get("page_changed") is not None:
                return self._unknown()
            if pre is None or post is None:
                return self._unknown()
            if self._cross_document and (
                self._previous_snapshot is not None
                and self._previous_snapshot != _snapshot_binding(before)
            ):
                return self._unknown()
            kind = action["kind"]
            node = action.get("node")
            if self._cross_document:
                target = _semantic_target(before, action)
            elif type(node) is int:
                target = _encode([before["page_key"][0], node])
            elif kind in {"scroll", "wait"}:
                target = _encode([before["page_key"][0], kind, action.get("delta")])
            else:
                return self._unknown()
            arguments = {"value": action.get("value"), "text": history_entry.get("text")}
            signal = self._ledger.record(
                before=pre[0],
                action={"kind": kind, "target": target, "arguments": arguments},
                after=post[0],
                executed=True,
            )
            self._step = history_entry["step"]
            if self._cross_document:
                self._previous_snapshot = _snapshot_binding(after)
            self._pending = None
            self._binding = None
            self._episode_binding = None
            if kind in {"click", "fill", "select"} and (
                signal.repeated_transition or signal.alternating_cycle
            ):
                self._binding = _policy_binding(after, history_entry)
                self._episode_binding = _episode_binding(
                    episode_id, after, history_entry, execution
                )
                self._pending = {
                    "scope": SCOPE,
                    "repeated_transition": signal.repeated_transition,
                    "alternating_cycle": signal.alternating_cycle,
                    "before_limits": pre[1],
                    "after_limits": post[1],
                }
            return {
                "observation": signal.observation,
                "repeated_transition": signal.repeated_transition,
                "alternating_cycle": signal.alternating_cycle,
            }
        except (ValueError, TypeError, KeyError, OverflowError, UnicodeError, RecursionError):
            return self._unknown()

    def annotate_next_policy(
        self,
        body,
        *,
        episode_id=None,
        current_page=None,
        history_entry=None,
        execution=None,
        freshness=None,
    ):
        """Add one advisory after owner-attested freshness, or return untouched.

        The owner must authenticate the current observation and attestation;
        this helper verifies consistency but cannot prove their origin.
        """
        _gate()
        if self._pending is None:
            return body
        if (
            type(body) is not dict
            or type(body.get("state")) is not dict
            or type(body.get("questions")) is not dict
            or KEY in body["state"]
        ):
            self._unknown()
            return body
        try:
            if (
                episode_id != self._episode
                or type(history_entry) is not dict
                or type(history_entry.get("step")) is not int
                or history_entry["step"] != self._step
                or type(current_page) is not dict
                or type(current_page.get("page_key")) is not list
                or not current_page["page_key"]
                or type(freshness) is not dict
                or set(freshness) != {"episode_id", "step", "document", "fingerprint", "status"}
                or freshness["episode_id"] != episode_id
                or type(freshness["step"]) is not int
                or freshness["step"] != self._step
                or type(freshness["document"]) is not type(current_page["page_key"][0])
                or freshness["document"] != current_page["page_key"][0]
                or freshness["fingerprint"] != current_page.get("fingerprint")
                or freshness["status"] != "verified_current"
                or _episode_binding(episode_id, current_page, history_entry, execution)
                != self._episode_binding
            ):
                raise ValueError("owner episode, document or committed step differs")
            state = body["state"]
            recent = state["recent_actions"]
            if type(recent) is not list or not recent:
                raise ValueError("last committed action missing")
            view = {"page": state["page"], "elements": state["elements"], "last_action": recent[-1]}
            if hashlib.sha256(_encode(view).encode("utf-8")).digest() != self._binding:
                raise ValueError("policy state differs from last observed action")
        except (KeyError, TypeError, ValueError, UnicodeError, OverflowError, RecursionError):
            self._unknown()
            return body
        result = copy.deepcopy(body)
        result["state"][KEY] = copy.deepcopy(self._pending)
        self._pending = None
        self._binding = None
        self._episode_binding = None
        return result
