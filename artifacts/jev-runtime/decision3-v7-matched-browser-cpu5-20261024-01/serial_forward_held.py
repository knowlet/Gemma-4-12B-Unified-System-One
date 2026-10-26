"""Held cross-process forward serialization for one explicitly owned campaign.

The lock covers actual blocking model work. A failed device synchronization or
evidence write keeps the lease until process exit; HTTP cancellation cannot
authorize another model. This is advisory cooperation, not a no-escape claim.
"""

from __future__ import annotations

import fcntl
import json
import os
import stat
import threading
import time
from pathlib import Path

SOURCE_ONLY = True
MAX_JOURNAL_BYTES = 16 * 1024**2


def require(value, message):
    if not value:
        raise RuntimeError(message)


def ordinary(path):
    path = Path(path).absolute()
    require(".." not in path.parts, "No traversal in campaign serialization paths")
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Ordinary campaign paths")
    return path


class ForwardLease:
    def __init__(self, lock_path, journal_path, expected_stat, *, nonce):
        require(SOURCE_ONLY is False, "SOURCE_ONLY: no serialization IO authorized")
        self.path, self.journal = ordinary(lock_path), ordinary(journal_path)
        self.expected = {key: dict(value) for key, value in expected_stat.items()}
        require(set(self.expected) == {"lock", "journal"}, "Both original inode identities")
        require(
            all(
                set(row) == {"device", "inode"}
                and all(type(number) is int and number >= 0 for number in row.values())
                for row in self.expected.values()
            ),
            "Exact typed lock and journal inode identities",
        )
        require(isinstance(nonce, str) and len(nonce) == 32, "Owned campaign nonce")
        self.nonce, self.fd, self.held = nonce, None, None
        self.fatal = None
        self.local = threading.RLock()

    def _emit(self, event, kind):
        record = {
            "event": event,
            "kind": kind,
            "pid": os.getpid(),
            "thread": threading.get_ident(),
            "monotonic_ns": time.monotonic_ns(),
            "nonce": self.nonce,
        }
        raw = json.dumps(record, separators=(",", ":"), allow_nan=False).encode() + b"\n"
        fd = os.open(self.journal, os.O_APPEND | os.O_WRONLY | os.O_NOFOLLOW)
        primary = None
        try:
            info = os.fstat(fd)
            require(stat.S_ISREG(info.st_mode), "Ordinary existing campaign journal")
            require(
                {"device": info.st_dev, "inode": info.st_ino} == self.expected["journal"],
                "Original shared journal inode required",
            )
            require(info.st_size + len(raw) <= MAX_JOURNAL_BYTES, "Bounded forward journal")
            require(os.write(fd, raw) == len(raw), "Complete forward evidence write")
            os.fsync(fd)
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                os.close(fd)
            except BaseException as secondary:
                if primary is None:
                    raise
                if secondary is not primary:
                    primary.add_note("Forward journal close also failed: " + repr(secondary))

    def begin(self, kind, deadline):
        require(callable(deadline), "Original whole-call deadline callback required")
        require(kind in {"nf4", "decision3"}, "Explicit model family")
        with self.local:
            require(self.fatal is None and self.fd is None, "Prior forward is live or failed")
            deadline()
            fd = os.open(self.path, os.O_RDWR | os.O_NOFOLLOW)
            try:
                info = os.fstat(fd)
                require(
                    stat.S_ISREG(info.st_mode)
                    and {"device": info.st_dev, "inode": info.st_ino} == self.expected["lock"],
                    "Current original shared lock inode required",
                )
                while True:
                    deadline()
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        time.sleep(0.005)
                self.fd = fd
                self.held = (kind, threading.get_ident())
                self._emit("acquired", kind)
                deadline()
            except BaseException as error:
                self.fatal = error
                # Once acquired, retain ownership until the original owner
                # terminates this process. Failed evidence never permits work.
                if self.fd is None:
                    try:
                        os.close(fd)
                    except BaseException as secondary:
                        if secondary is not error:
                            error.add_note(
                                "Unacquired forward lock close also failed: " + repr(secondary)
                            )
                raise

    def end(self, *, synchronize, deadline):
        with self.local:
            require(
                self.fd is not None and self.held[1] == threading.get_ident(),
                "Only the actual forward thread can end its lease",
            )
            try:
                require(self.fatal is None, "A failed lease cannot be released")
                synchronize()
                deadline()
                self._emit("completed", self.held[0])
            except BaseException as error:
                self.fatal = error
                raise
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
                os.close(self.fd)
            except BaseException as error:
                self.fatal = error
                raise
            self.fd, self.held = None, None

    def fail(self, error):
        with self.local:
            if self.fatal is None:
                self.fatal = error
        # Do not close or unlock an unfinished model lease.

    def check(self):
        require(self.fatal is None, "Campaign serialization failure is latched")


class SerialBackend:
    def __init__(self, backend, lease):
        self.backend, self.lease = backend, lease
        self.spec, self.name = backend.spec, backend.name

    def predict(self, request, **kwargs):
        deadline = kwargs["deadline"]
        self.lease.begin("decision3", deadline)
        try:
            result = self.backend.predict(request, **kwargs)
            # The selected Decision3 implementation performs synchronous CPU
            # tensor operations and returns CPU probabilities.
            self.lease.end(synchronize=lambda: None, deadline=deadline)
            return result
        except BaseException as error:
            self.lease.fail(error)
            raise


def install_nf4_hooks(
    model, torch, native, lease, *, resource_gate, campaign_deadline, register_original
):
    """Full CausalLM hooks, including its head and a charged device barrier.

    PEFT.generate delegates to the actual base CausalLM. A decoder-backbone
    hook would release before lm_head/softcap and cannot cover their failures.
    The original resource hook remains independently attached to the backbone.
    """
    require(SOURCE_ONLY is False, "SOURCE_ONLY: no model hooks authorized")
    previous = register_original(model, resource_gate)
    complete_model = model.lm.get_base_model()
    require(complete_model is not model.backbone, "Complete CausalLM must include output head")

    def deadline():
        trace = native.TRACE.get()
        campaign_deadline()
        if trace is not None:
            native.http_deadline(trace["started"])

    def before(_module, _args, _kwargs):
        lease.begin("nf4", deadline)

    def after(_module, _args, _kwargs, output):
        if lease.fd is None:
            return
        if output is None:
            lease.fail(RuntimeError("NF4 forward failed; lease retained until owner exit"))
            return
        lease.end(synchronize=lambda: torch.cuda.synchronize(0), deadline=deadline)

    first = complete_model.register_forward_pre_hook(before, with_kwargs=True)
    last = complete_model.register_forward_hook(after, with_kwargs=True, always_call=True)

    class Hooks:
        def remove(self):
            lease.check()
            require(lease.fd is None, "Actual NF4 forward must complete before hook removal")
            last.remove()
            first.remove()
            previous.remove()

    return Hooks()
