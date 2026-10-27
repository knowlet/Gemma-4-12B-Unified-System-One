"""Exercise a declared pair plan with injected synthetic collaborators only.

This source-only harness does not access files, cases, networks, browsers, or
models directly. Injected callbacks are arbitrary code and can access or launch
external resources or share data. This module cannot enforce their isolation or
prove callbacks are trustworthy, a run is real, source pins
are authentic, cases are unseen, or selection was blind. A real controller needs
independent gates, private mapping, original-judge binding, evidence and closure
verification. The synthetic handle intentionally cannot carry an oracle or data.
"""

from copy import deepcopy
from dataclasses import dataclass

from jev_blind_feedback_pair_plan import validate_plan

SOURCE_ONLY = True


class SyntheticHandle:
    """Data-free stand-in; not a real case identifier or case payload."""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class SyntheticLaunch:
    seed: int
    feedback_enabled: bool
    review_enabled: bool
    pins: tuple[tuple[str, str], ...]
    configuration_sha256: str
    budgets: tuple[tuple[str, int], ...]


def run_synthetic_pair(plan, *, selector, launcher):
    """Validate once, then call only supplied synthetic selector/launcher fakes.

    selector(token, seed) returns a fresh SyntheticHandle for the pair. The fake
    launcher receives only that handle and immutable SyntheticLaunch; open()
    returns an object with run() and close(). No goal, judge, oracle, raw case,
    or selector token is delivered to run(). Callback failures stop further
    launches; the full predeclared denominator is still reported. If open()
    raises, ownership of any partially opened resource is unknown. Keyboard
    interrupts propagate after the finally close; no aggregate is claimed then.
    """
    snapshot = deepcopy(plan)
    validated = validate_plan(snapshot)
    total = validated["planned_runs"]
    completed = failed = closure_returned = open_or_close_errors = selector_errors = 0
    handles = []
    stopped = False

    for pair in snapshot["schedule"]:
        try:
            handle = selector(pair["token"], pair["seed"])
            if type(handle) is not SyntheticHandle or any(handle is item for item in handles):
                raise ValueError("fresh synthetic handle required")
            handles.append(handle)
        except Exception:
            selector_errors += 1
            stopped = True
            break

        for arm_name in pair["order"]:
            arm = snapshot["arms"][arm_name]
            spec = SyntheticLaunch(
                seed=pair["seed"],
                feedback_enabled=arm["feedback_enabled"],
                review_enabled=arm["review_enabled"],
                pins=tuple(sorted(arm["pins"].items())),
                configuration_sha256=arm["configuration_sha256"],
                budgets=tuple(sorted(arm["budgets"].items())),
            )
            try:
                session = launcher.open(handle, spec)
            except Exception:
                # open may have acquired a resource before raising.
                failed += 1
                open_or_close_errors += 1
                stopped = True
                break
            run_failed = close_failed = False
            try:
                session.run()
            except Exception:
                run_failed = True
            finally:
                try:
                    session.close()
                except Exception:
                    close_failed = True
            if close_failed:
                open_or_close_errors += 1
            else:
                closure_returned += 1
            if run_failed or close_failed:
                failed += 1
                stopped = True
                break
            completed += 1
        if stopped:
            break

    return {
        "status": "synthetic_callbacks_completed" if not stopped else "unresolved",
        "selected_cases": validated["selected_cases"],
        "planned_runs": total,
        "completed_callbacks": completed,
        "failed_callbacks": failed,
        "not_started": total - completed - failed,
        "selector_errors": selector_errors,
        "close_callbacks_returned": closure_returned,
        "open_or_close_callback_errors": open_or_close_errors,
        "os_closure": "unverified",
        "exposure_status": "unknown",
        "authenticity_and_preregistration": "unverified",
        "runtime_conformance": "unverified",
        "blind_selection": "unverified",
        "callback_trust_and_isolation": "unverified",
    }
