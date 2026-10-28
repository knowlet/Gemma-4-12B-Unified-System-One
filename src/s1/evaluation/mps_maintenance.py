"""Opt-in allocator maintenance after complete MPS inference calls."""

from __future__ import annotations

import copy
import time

MPS_CALL_MAINTENANCE_POLICY = {
    "name": "mps-whole-call-cache-clear-v1",
    "scope": "After each complete dispatched decision or native field helper call, including errors.",
    "operations": ["synchronize", "memory_before", "empty_cache", "synchronize", "memory_after"],
    "latency_scope": "Callback latency includes whole-call cleanup; no warmup calls are discarded.",
    "limits_changed": False,
    "fault_policy": "Preserve the current response/error; a cleanup or trace-write fault blocks all subsequent dispatches.",
    "interpretation": "Opt-in resource policy requiring measurement; no OOM, speed, or Torch 2.8 remediation claim.",
}


class MPSMaintenanceFault(RuntimeError):
    """A prior operational maintenance fault prevents further model dispatch."""


def _error(exc):
    return {"type": type(exc).__name__, "message": str(exc)}


class MPSCallMaintenance:
    """Run cleanup in finally without changing model responses or inference errors.

    The trace contains call identity, memory, timing, and errors only. A failed
    cleanup preserves a successful response but latches a resource fault before
    the next call can touch either the model or MPS.
    """

    def __init__(self, torch, *, device="mps", records=None, save=None):
        if str(device) != "mps":
            raise ValueError("MPS call maintenance requires an MPS model")
        self.torch = torch
        self.records = records if records is not None else []
        self.save = save
        self.fault = None

    def _snapshot(self):
        return {
            "current_allocated_bytes": self.torch.mps.current_allocated_memory(),
            "driver_allocated_bytes": self.torch.mps.driver_allocated_memory(),
            "recommended_max_bytes": self.torch.mps.recommended_max_memory(),
        }

    def _cleanup(self):
        started = time.perf_counter()
        result = {"before": None, "after": None, "error": None}
        try:
            self.torch.mps.synchronize()
            result["before"] = self._snapshot()
            self.torch.mps.empty_cache()
            self.torch.mps.synchronize()
            result["after"] = self._snapshot()
        except Exception as exc:
            result["error"] = _error(exc)
        result["wall_ms"] = (time.perf_counter() - started) * 1000
        return result

    def _publish(self, record):
        self.records.append(record)
        if self.save is not None:
            try:
                self.save(self.records)
            except Exception as exc:
                record["trace_write_error"] = _error(exc)
                self.fault = self.fault or {
                    "attempt": record["attempt"],
                    "stage": "trace_write",
                    "error": record["trace_write_error"],
                }

    def call(self, operation, *, profile, kind):
        started = time.perf_counter()
        record = {
            "attempt": len(self.records) + 1,
            "profile": profile,
            "kind": kind,
            "dispatched": False,
            "inference_error": None,
            "cleanup": None,
            "dispatch_error": None,
            "trace_write_error": None,
        }
        if self.fault is not None:
            exc = MPSMaintenanceFault("Prior MPS call maintenance fault; no model call dispatched")
            record.update(
                dispatch_error=_error(exc),
                prior_fault=copy.deepcopy(self.fault),
                wall_ms_including_cleanup=(time.perf_counter() - started) * 1000,
            )
            self._publish(record)
            raise exc
        record["dispatched"] = True
        try:
            return operation()
        except BaseException as exc:
            record["inference_error"] = _error(exc)
            raise
        finally:
            record["cleanup"] = self._cleanup()
            if record["cleanup"]["error"] is not None:
                self.fault = {
                    "attempt": record["attempt"],
                    "stage": "cleanup",
                    "error": record["cleanup"]["error"],
                }
            record["wall_ms_including_cleanup"] = (time.perf_counter() - started) * 1000
            self._publish(record)

    def operational_errors(self):
        return [
            {
                "attempt": record["attempt"],
                "profile": record["profile"],
                "kind": record["kind"],
                "cleanup_error": (record.get("cleanup") or {}).get("error"),
                "trace_write_error": record["trace_write_error"],
            }
            for record in self.records
            if (record.get("cleanup") or {}).get("error") or record["trace_write_error"]
        ]


class MPSMaintenanceBackend:
    """Transparent whole-predict wrapper with an explicit resource policy."""

    def __init__(self, backend, maintenance, *, profile):
        if str(backend.model.device) != "mps":
            raise ValueError("MPS call maintenance requires an MPS model")
        self.backend = backend
        self.maintenance = maintenance
        self.profile = profile
        self.name = backend.name
        self.model = backend.model
        self.metadata = {
            **copy.deepcopy(backend.metadata),
            "mps_call_maintenance": copy.deepcopy(MPS_CALL_MAINTENANCE_POLICY),
        }

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def predict(self, request):
        return self.maintenance.call(
            lambda: self.backend.predict(request), profile=self.profile, kind="decision"
        )
