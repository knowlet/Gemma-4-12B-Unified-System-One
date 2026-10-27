"""Held opt-in bridge from Jev's real callbacks to progress advisories.

Install inside the existing runner's model_adapter context, on one Agent only.
This helper does not launch a browser, alter candidates or judge outcomes. Its
caller must authenticate loaded code and run the agent serially. A confirmed
Browser.act result proves only the upstream execution return, not task success.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import os
import re
import secrets
import stat
import threading
from contextlib import contextmanager
from pathlib import Path

SOURCE_ONLY = True
POLICY_URL = "https://api.typesafe.ai/v1/systemone"
MAX_SOURCE = 8 * 1024**2


def _need(value, message):
    if not value:
        raise ValueError(message)


def _gate():
    _need(SOURCE_ONLY is False, "held progress owner requires independent review")


def _pin_file(path, expected):
    path = Path(path)
    _need(path.is_absolute() and path.resolve() == path, "ordinary pinned source path required")
    _need(type(expected) is str and re.fullmatch(r"[0-9a-f]{64}", expected), "source SHA required")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        _need(
            stat.S_ISREG(before.st_mode) and 0 < before.st_size <= MAX_SOURCE,
            "bounded source required",
        )
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(MAX_SOURCE + 1)
        after = os.fstat(fd)
        current = os.stat(path, follow_symlinks=False)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        _need(
            tuple(getattr(before, k) for k in fields)
            == tuple(getattr(after, k) for k in fields)
            == tuple(getattr(current, k) for k in fields)
            and len(raw) == before.st_size
            and hashlib.sha256(raw).hexdigest() == expected,
            "source identity changed",
        )
    finally:
        os.close(fd)


def _verify_pins(pins, objects):
    _need(type(pins) is dict and bool(pins), "explicit owner source pins required")
    for item in objects:
        name = inspect.getsourcefile(item)
        _need(
            type(name) is str and str(Path(name).absolute()) in pins,
            "actual callback source pin required",
        )
    for name, digest in pins.items():
        _need(type(name) is str, "string source paths required")
        _pin_file(name, digest)


def _document(page):
    if (
        type(page) is not dict
        or type(page.get("page_key")) is not list
        or not page["page_key"]
        or type(page.get("marker")) is not list
        or len(page["marker"]) != 10
        or type(page["page_key"][0]) not in (int, float)
        or type(page["marker"][0]) is not type(page["page_key"][0])
        or page["marker"][0] != page["page_key"][0]
        or type(page.get("fingerprint")) is not str
        or not page["fingerprint"]
    ):
        return None
    return page["page_key"][0]


@contextmanager
def attach(
    agent, agent_module, model_module, integration, *, pins, enabled=False, cross_document=False
):
    """Attach one serial episode; disabled is an exact pass-through.

    Call after the runner installs its original post_json callback. Source pin
    checks attest current files, not Python import provenance: the external
    owner must authenticate the loaded modules and the released integration.
    """
    _gate()
    _need(type(enabled) is bool, "explicit opt-in required")
    _need(type(cross_document) is bool, "explicit cross-document opt-in required")
    if not enabled:
        yield
        return
    _need(
        getattr(integration, "SOURCE_ONLY", None) is False, "reviewed integration release required"
    )
    browser = agent.state.get("browser")
    _need(
        browser is not None
        and type(agent) is agent_module.Agent
        and type(browser) is agent_module.Browser
        and agent_module.choose is model_module.choose
        and model_module.choose.__globals__.get("post_json") is model_module.post_json,
        "original agent, browser and policy callbacks required",
    )
    original_command, original_act, original_fresh, original_post = (
        agent.command,
        browser.act,
        browser.fresh,
        model_module.post_json,
    )
    original_choose = model_module.choose
    _need(
        "command" not in agent.__dict__
        and "act" not in browser.__dict__
        and "fresh" not in browser.__dict__
        and getattr(original_command, "__func__", None) is type(agent).command
        and getattr(original_act, "__func__", None) is type(browser).act
        and getattr(original_fresh, "__func__", None) is type(browser).fresh,
        "unwrapped original instance methods required",
    )
    objects = (
        type(agent).command,
        type(browser).act,
        type(browser).fresh,
        model_module.choose,
        original_post,
        integration.BrowserProgressIntegration,
        integration.ProgressLedger,
        integration._metadata_valid,
        attach,
    )
    _verify_pins(pins, objects)
    episode = secrets.token_hex(16)
    tracker = integration.BrowserProgressIntegration(
        episode_id=episode, cross_document=cross_document
    )
    owner_thread = threading.get_ident()
    active = {
        "predict": False,
        "action": False,
        "fresh": None,
        "executed": None,
        "last": None,
        "disabled": False,
        "act_count": 0,
        "selected": None,
    }

    def unknown():
        tracker._unknown()
        active.update(fresh=None, executed=None, last=None)

    def same_thread():
        if (
            threading.get_ident() != owner_thread
            or getattr(agent_module, "Agent", None) is not type(agent)
            or getattr(agent_module, "Browser", None) is not type(browser)
            or getattr(type(agent), "command", None) is not original_command.__func__
            or getattr(type(browser), "act", None) is not original_act.__func__
            or getattr(type(browser), "fresh", None) is not original_fresh.__func__
            or getattr(agent_module, "choose", None) is not original_choose
            or getattr(model_module, "choose", None) is not original_choose
        ):
            active["disabled"] = True
            unknown()
            return False
        return not active["disabled"]

    def fresh(page, action=None):
        try:
            result = original_fresh(page, action=action)
        except BaseException:
            unknown()
            raise
        if same_thread() and active["predict"] and action is None:
            try:
                active["fresh"] = (page, copy.deepcopy(page), result is True)
            except Exception:
                unknown()
        return result

    def act(action, page, text=None):
        if not same_thread() or not active["action"]:
            unknown()
            return original_act(action, page, text=text)
        active["act_count"] += 1
        try:
            before, chosen = copy.deepcopy(page), copy.deepcopy(action)
        except Exception:
            unknown()
            return original_act(action, page, text=text)
        try:
            result = original_act(action, page, text=text)
            if (
                active["act_count"] == 1
                and page is agent.state.get("page")
                and type(chosen.get("id")) is str
                and chosen["id"] == active["selected"]
                and type(result) is dict
                and result == {"executed": chosen["id"]}
            ):
                active["executed"] = (before, chosen, text)
            else:
                unknown()
            return result
        except BaseException:
            unknown()
            raise

    def post(url, key, body):
        if not same_thread() or not active["predict"] or url != POLICY_URL:
            return original_post(url, key, body)
        candidate = body
        try:
            found = active["fresh"]
            last = active["last"]
            current = agent.state.get("page")
            if (
                found is None
                or found[0] is not current
                or found[1] != current
                or not found[2]
                or last is None
                or _document(current) is None
                or agent.state.get("browser") is not browser
            ):
                unknown()
            else:
                history, execution = last
                freshness = {
                    "episode_id": episode,
                    "step": history["step"],
                    "document": _document(current),
                    "fingerprint": current["fingerprint"],
                    "status": "verified_current",
                }
                candidate = tracker.annotate_next_policy(
                    body,
                    episode_id=episode,
                    current_page=current,
                    history_entry=history,
                    execution=execution,
                    freshness=freshness,
                )
                active["last"] = None
        except Exception:
            unknown()
            candidate = body
        return original_post(url, key, candidate)

    def command(name, body=None):
        if not same_thread():
            return original_command(name, body)
        if name == "predict":
            active["predict"] = True
            active["fresh"] = None
            try:
                return original_command(name, body)
            finally:
                active["predict"] = False
                active["fresh"] = None
        if name != "act":
            return original_command(name, body)
        active["action"] = True
        active["executed"] = None
        active["act_count"] = 0
        decision = agent.state.get("decision")
        active["selected"] = decision.get("choice") if type(decision) is dict else None
        try:
            history_before = copy.deepcopy(agent.state.get("history"))
            if type(history_before) is not list:
                raise ValueError("episode history unavailable")
        except Exception:
            unknown()
            active["action"] = False
            active["selected"] = None
            return original_command(name, body)
        try:
            value = original_command(name, body)
        except BaseException:
            unknown()
            active["action"] = False
            active["selected"] = None
            raise
        try:
            captured = active["executed"]
            history = agent.state.get("history")
            after = agent.state.get("page")
            if (
                captured is None
                or type(history) is not list
                or not history
                or len(history) != len(history_before) + 1
                or integration._encode(history[:-1]) != integration._encode(history_before)
                or type(history[-1]) is not dict
                or agent.state.get("browser") is not browser
            ):
                unknown()
                return value
            before, chosen, text = captured
            doc = _document(before)
            after_doc = _document(after)
            if (
                doc is None
                or after_doc is None
                or (not cross_document and (type(after_doc) is not type(doc) or after_doc != doc))
                or chosen.get("id") != history[-1].get("choice")
                or chosen.get("kind") != history[-1].get("kind")
                or chosen.get("label") != history[-1].get("action")
                or text != history[-1].get("text")
            ):
                unknown()
                return value
            row = copy.deepcopy(history[-1])
            execution = {
                "episode_id": episode,
                "step": row["step"],
                "choice": chosen["id"],
                "document": doc,
                "before_fingerprint": before["fingerprint"],
                "after_fingerprint": after["fingerprint"],
                "status": "confirmed_executed",
            }
            result = tracker.record_committed(
                episode_id=episode,
                before=before,
                selected_id=chosen["id"],
                after=after,
                history_entry=row,
                execution=execution,
            )
            active["last"] = (row, execution) if result.get("observation") == "known" else None
            return value
        except Exception:
            unknown()
            return value
        finally:
            active["action"] = False
            active["executed"] = None
            active["selected"] = None

    bindings = (
        (browser, "act", act),
        (browser, "fresh", fresh),
        (agent, "command", command),
        (model_module, "post_json", post),
    )
    installed = []
    primary = None
    try:
        for obj, name, wrapper in bindings:
            try:
                setattr(obj, name, wrapper)
            finally:
                # A custom setter can assign successfully and then raise.
                if obj.__dict__.get(name) is wrapper:
                    installed.append((obj, name, wrapper))
        yield
        _verify_pins(pins, objects)
    except BaseException as error:
        primary = error
        raise
    finally:
        problems = []
        for obj, name, expected in installed:
            if obj.__dict__.get(name) is not expected:
                problems.append(name)
            elif obj is model_module:
                obj.__dict__[name] = original_post
            else:
                del obj.__dict__[name]
        if (
            getattr(agent_module, "Agent", None) is not type(agent)
            or getattr(agent_module, "Browser", None) is not type(browser)
            or getattr(type(agent), "command", None) is not original_command.__func__
            or getattr(type(browser), "act", None) is not original_act.__func__
            or getattr(type(browser), "fresh", None) is not original_fresh.__func__
            or getattr(agent_module, "choose", None) is not original_choose
            or getattr(model_module, "choose", None) is not original_choose
        ):
            problems.append("class_or_policy_callback")
        if problems:
            if primary is not None:
                primary.add_note("progress callback cleanup unconfirmed")
            else:
                raise RuntimeError("progress callback cleanup unconfirmed")
