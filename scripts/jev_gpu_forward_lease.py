"""HELD cross-process GPU lease for one explicitly owned browser campaign.

The Decision3 lease surrounds the complete operation-delta provider, including
its witnesses and terminal callbacks. CUDA completion, durable journal writes
and the final deadline check precede unlock. Failures before unlock retain an
acquired lease until process exit. Timers are cooperative; this module does not cancel
HTTP requests, own a service, authenticate Torch origins or qualify a model.
"""

from __future__ import annotations

# ruff: noqa: E402
# Refuse a held CLI before importing dependencies, including platform-only fcntl.
SOURCE_ONLY = True

if SOURCE_ONLY and __name__ == "__main__":
    raise SystemExit("SOURCE_ONLY: held GPU serializer cannot execute")

import fcntl
import json
import os
import stat
import sys
import threading
import time
import types
from pathlib import Path

MAX_JOURNAL_BYTES = 16 * 1024**2
DECISION3_KIND = "decision3-gpu"
KINDS = frozenset({"nf4", DECISION3_KIND})


def require(value, message):
    if not value:
        raise RuntimeError(message)


def release_gate():
    require(SOURCE_ONLY is False, "SOURCE_ONLY: no lease IO, dependencies or hooks authorized")


def ordinary(path):
    release_gate()
    path = Path(path).absolute()
    require(".." not in path.parts, "No traversal in campaign serialization paths")
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Ordinary campaign paths")
    return path


def _note(primary, phase, secondary):
    # Exception subclasses may refuse notes or even fail during repr(). Notes
    # are supplemental; they cannot replace the original latched abort object.
    try:
        if secondary is not primary:
            primary.add_note(phase + " also failed: " + repr(secondary))
    except BaseException:
        pass


def _selected_cuda(torch):
    """Require the caller-selected registered CUDA API, never a fence callback.

    Module registration and these API identities are refusal checks. The owned
    worker separately authenticates actual module origins and selected bytes;
    Python names alone do not establish genuine framework provenance.
    """
    release_gate()
    require(
        isinstance(torch, types.ModuleType)
        and torch.__name__ == "torch"
        and sys.modules.get("torch") is torch,
        "Explicit selected registered Torch module required; no CPU/no-op fence callback",
    )
    cuda = getattr(torch, "cuda", None)
    require(
        isinstance(cuda, types.ModuleType)
        and cuda.__name__ == "torch.cuda"
        and sys.modules.get("torch.cuda") is cuda,
        "Explicit selected registered torch.cuda module required",
    )
    synchronize = getattr(cuda, "synchronize", None)
    require(
        callable(synchronize)
        and getattr(synchronize, "__module__", None) == "torch.cuda"
        and getattr(synchronize, "__name__", None) == "synchronize",
        "Selected torch.cuda.synchronize entry required; generic/no-op callbacks refused",
    )
    for name in ("is_available", "is_initialized", "device_count", "current_device"):
        require(callable(getattr(cuda, name, None)), "Selected CUDA API required: " + name)
    require(
        cuda.is_available() is True
        and cuda.is_initialized() is True
        and type(cuda.device_count()) is int
        and cuda.device_count() == 1
        and type(cuda.current_device()) is int
        and cuda.current_device() == 0,
        "Already initialized single CUDA device0 required; no CPU fallback",
    )
    return cuda, synchronize


class ForwardLease:
    """One shared inode-bound lease for helper and complete D3 GPU calls."""

    def __init__(self, lock_path, journal_path, expected_stat, *, nonce):
        release_gate()
        require(
            type(expected_stat) is dict and set(expected_stat) == {"lock", "journal"},
            "Both original inode identities",
        )
        require(
            all(
                type(row) is dict
                and set(row) == {"device", "inode"}
                and all(type(number) is int and number >= 0 for number in row.values())
                for row in expected_stat.values()
            ),
            "Exact typed lock and journal inode identities",
        )
        require(
            type(nonce) is str and len(nonce) == 32 and all(c in "0123456789abcdef" for c in nonce),
            "Selected campaign nonce must be32 lowercase hexadecimal characters",
        )
        self.path, self.journal = ordinary(lock_path), ordinary(journal_path)
        self.expected = {key: dict(value) for key, value in expected_stat.items()}
        self.nonce, self.fd, self.held = nonce, None, None
        self.fatal = None
        self.local = threading.RLock()

    def _emit(self, event, kind):
        release_gate()
        require(kind in KINDS, "Explicit owned GPU/helper family")
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
                _note(primary, "Forward journal close", secondary)

    def begin(self, kind, deadline):
        release_gate()
        require(callable(deadline), "Original whole-call deadline callback required")
        require(kind in KINDS, "Explicit owned GPU/helper family")
        with self.local:
            require(self.fatal is None and self.fd is None, "Prior forward is live or failed")
            fd = None
            try:
                deadline()
                fd = os.open(self.path, os.O_RDWR | os.O_NOFOLLOW)
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
                # An acquired lease is never released on an error path.
                if self.fd is None and fd is not None:
                    try:
                        os.close(fd)
                    except BaseException as secondary:
                        _note(error, "Unacquired forward lock close", secondary)
                raise

    def end(self, *, synchronize, deadline):
        release_gate()
        require(
            callable(synchronize) and callable(deadline),
            "Explicit completion fence and original deadline required",
        )
        with self.local:
            require(
                self.fd is not None
                and self.held is not None
                and self.held[1] == threading.get_ident(),
                "Only the actual forward thread can end its lease",
            )
            try:
                if self.fatal is not None:
                    raise self.fatal
                synchronize()
                deadline()
                self._emit("completed", self.held[0])
                deadline()  # Includes completed journal write, fsync and close.
            except BaseException as error:
                if self.fatal is None:
                    self.fatal = error
                raise
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            except BaseException as error:
                if self.fatal is None:
                    self.fatal = error
                raise
            # LOCK_UN is the irreversible release point. A later close error
            # still fails/latches this campaign, but must not claim ownership
            # of a lock already released by the kernel.
            released_fd = self.fd
            self.fd, self.held = None, None
            try:
                os.close(released_fd)
            except BaseException as error:
                if self.fatal is None:
                    self.fatal = error
                raise

    def fail(self, error):
        release_gate()
        with self.local:
            if self.fatal is None:
                self.fatal = error
        # No close/unlock: external owned process termination ends this lease.

    def check(self):
        release_gate()
        require(self.fatal is None, "Campaign serialization failure is latched")


class SerialBackend:
    """Serialize the complete composed provider, with a selected CUDA fence.

    The caller places its HTTP-entry DeadlineBackend OUTSIDE this wrapper.
    That callback then covers lock wait, the entire composition call, the GPU
    fence and all durable release evidence. No generic synchronization callback
    is accepted here. The caller owns/guards/closes the supplied provider.
    """

    def __init__(self, backend, lease, *, torch):
        release_gate()
        cuda, synchronize = _selected_cuda(torch)
        require(isinstance(lease, ForwardLease), "This selected shared GPU ForwardLease required")
        require(callable(getattr(backend, "predict", None)), "Complete composed provider required")
        self.backend, self.lease, self.torch = backend, lease, torch
        self.cuda, self.synchronize = cuda, synchronize
        self.spec, self.name = backend.spec, backend.name

    def _fence(self):
        release_gate()
        cuda, synchronize = _selected_cuda(self.torch)
        require(
            cuda is self.cuda and synchronize is self.synchronize,
            "Selected CUDA fence binding changed",
        )
        synchronize(0)

    def predict(self, request, **kwargs):
        release_gate()
        deadline = kwargs.get("deadline")
        require(callable(deadline), "Original whole-call deadline callback required")
        try:
            self.lease.begin(DECISION3_KIND, deadline)
            result = self.backend.predict(request, **kwargs)
            self.lease.end(synchronize=self._fence, deadline=deadline)
            return result
        except BaseException as error:
            try:
                self.lease.fail(error)
            except BaseException as secondary:
                _note(error, "Failed forward lease retention", secondary)
            raise


def install_nf4_hooks(
    model, torch, native, lease, *, resource_gate, campaign_deadline, register_original
):
    """Retain the complete helper CausalLM hook boundary from the old source.

    PEFT.generate delegates to the complete base CausalLM. Its output head and
    softcap stay inside the shared GPU lease. This is a new campaign binding,
    with no inherited CPU proof, resource acceptance or helper qualification.
    """
    release_gate()
    cuda, synchronize = _selected_cuda(torch)
    require(isinstance(lease, ForwardLease), "This selected shared GPU ForwardLease required")
    require(
        callable(campaign_deadline) and callable(register_original),
        "Selected helper deadline and original hook installer required",
    )
    complete_model = model.lm.get_base_model()
    require(complete_model is not model.backbone, "Complete CausalLM must include output head")

    def deadline():
        trace = native.TRACE.get()
        campaign_deadline()
        if trace is not None:
            native.http_deadline(trace["started"])

    def before(_module, _args, _kwargs):
        lease.begin("nf4", deadline)

    def fence():
        current_cuda, current_synchronize = _selected_cuda(torch)
        require(
            current_cuda is cuda and current_synchronize is synchronize,
            "Selected helper CUDA fence binding changed",
        )
        synchronize(0)

    def after(_module, _args, _kwargs, output):
        if lease.fd is None:
            return
        if output is None:
            lease.fail(RuntimeError("NF4 forward failed; lease retained until owner exit"))
            return
        lease.end(synchronize=fence, deadline=deadline)

    previous = register_original(model, resource_gate)
    first = None
    try:
        first = complete_model.register_forward_pre_hook(before, with_kwargs=True)
        last = complete_model.register_forward_hook(after, with_kwargs=True, always_call=True)
    except BaseException as error:
        for label, handle in (
            ("Helper first hook cleanup", first),
            ("Original resource hook cleanup", previous),
        ):
            if handle is not None:
                try:
                    handle.remove()
                except BaseException as secondary:
                    _note(error, label, secondary)
        raise

    class Hooks:
        def remove(self):
            lease.check()
            require(lease.fd is None, "Actual NF4 forward must complete before hook removal")
            primary = None
            for label, handle in (
                ("Helper last hook cleanup", last),
                ("Helper first hook cleanup", first),
                ("Original resource hook cleanup", previous),
            ):
                try:
                    handle.remove()
                except BaseException as error:
                    if primary is None:
                        primary = error
                    else:
                        _note(primary, label, error)
            if primary is not None:
                raise primary

    return Hooks()
