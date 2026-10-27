"""Opt-in connection of DONE review to the unchanged private capture bootstrap.

The caller owns the provenance of this bridge and the capture bootstrap, as
well as resource supervision. The owner is loaded from pinned, reviewed bytes.
No case, answer, or judge is available to this bridge.
"""

from __future__ import annotations

import sys
import threading
from contextlib import ExitStack
from pathlib import Path

SOURCE_ONLY = True
_LOCK = threading.Lock()


def run(
    bootstrap,
    *,
    enabled=False,
    held_owner_path=None,
    held_owner_sha256=None,
    released_owner_path=None,
    released_owner_sha256=None,
    feedback_enabled=False,
    max_reviews=2,
    event_sink=None,
    **capture_arguments,
):
    """Run the original capture bootstrap, optionally wrapping its runner.

    Disabled delegates directly. Enabled loads only a gate-released owner, then
    wraps the runner as it is loaded by the existing capture bootstrap. Capture
    remains outside the owner adapter and records its extra endpoint calls.
    """
    if type(enabled) is not bool:
        raise ValueError("explicit opt-in required")
    if not enabled:
        return bootstrap.run(**capture_arguments)
    if SOURCE_ONLY is not False:
        raise ValueError("held DONE bootstrap requires independent review")
    if not _LOCK.acquire(blocking=False):
        raise ValueError("serial bootstrap ownership required")
    original_module = None
    owner = None
    owner_name = None
    primary = None
    stack = ExitStack()
    seen = False
    hook_installed = False
    try:
        original_module = bootstrap._module
        if not callable(original_module):
            raise ValueError("original capture bootstrap loader required")
        owner_bytes = bootstrap._capture_source(
            held_owner_path, held_owner_sha256, released_owner_path, released_owner_sha256
        )
        owner_name = "_jev_done_review_" + released_owner_sha256[:16]
        owner = original_module(Path(released_owner_path), owner_bytes, owner_name)
        if owner.SOURCE_ONLY is not False or not callable(owner.attach_to_runner):
            raise ValueError("released DONE review owner required")
        runner_name = "_jev_private_runner_" + capture_arguments["runner_sha256"][:16]

        def load(path, raw, name):
            nonlocal seen
            module = original_module(path, raw, name)
            if name == runner_name:
                if seen:
                    raise ValueError("duplicate runner load refused")
                seen = True
                try:
                    stack.enter_context(
                        owner.attach_to_runner(
                            module,
                            enabled=True,
                            feedback_enabled=feedback_enabled,
                            max_reviews=max_reviews,
                            event_sink=event_sink,
                        )
                    )
                except BaseException:
                    if sys.modules.get(name) is module:
                        del sys.modules[name]
                    raise
            return module

        bootstrap._module = load
        hook_installed = True
        result = bootstrap.run(**capture_arguments)
        if not seen:
            raise ValueError("capture bootstrap did not load the pinned runner")
        return result
    except BaseException as error:
        primary = error
        raise
    finally:
        problems = []
        if hook_installed:
            if bootstrap._module is not load:
                problems.append(ValueError("capture bootstrap loader identity changed"))
            bootstrap._module = original_module
        try:
            stack.close()
        except BaseException as error:
            problems.append(error)
        if owner is not None:
            try:
                if sys.modules.get(owner_name) is not owner:
                    raise ValueError("DONE owner module identity changed")
                del sys.modules[owner_name]
                current = bootstrap._capture_source(
                    held_owner_path, held_owner_sha256, released_owner_path, released_owner_sha256
                )
                if current != owner_bytes:
                    raise ValueError("DONE owner source changed during execution")
            except BaseException as error:
                problems.append(error)
        _LOCK.release()
        if problems:
            if primary is not None:
                for problem in problems:
                    primary.add_note(f"DONE bootstrap cleanup failed: {problem}")
            else:
                raise problems[0]
