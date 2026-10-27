"""Only data-free synthetic collaborators; no real runner or case access."""

from dataclasses import asdict

import jev_blind_feedback_pair_plan as verifier
import jev_blind_feedback_synthetic_controller as controller
import pytest


def synthetic_plan():
    pins = {name: "a" * 64 for name in verifier._PIN_NAMES}
    budgets = {
        "actions": 16,
        "policy_calls": 16,
        "text_calls": 4,
        "review_calls": 2,
        "time_ms": 60000,
    }

    def arm(feedback):
        return {
            "review_enabled": True,
            "feedback_enabled": feedback,
            "pins": pins.copy(),
            "configuration_sha256": "b" * 64,
            "budgets": budgets.copy(),
        }

    return {
        "schema": "synthetic-feedback-pair-plan-v1",
        "exposure_status": "unknown",
        "population_sha256": "c" * 64,
        "selection_commitment_sha256": "d" * 64,
        "ordering_commitment_sha256": "e" * 64,
        "selected_cases": 2,
        "planned_runs": 4,
        "arms": {"off": arm(False), "on": arm(True)},
        "schedule": [
            {"token": "1" * 32, "seed": 13, "order": ["off", "on"]},
            {"token": "2" * 32, "seed": 29, "order": ["on", "off"]},
        ],
    }


class FakeSession:
    def __init__(self, owner, *, run_error=False, close_error=False):
        self.owner = owner
        self.run_error = run_error
        self.close_error = close_error

    def run(self):
        self.owner.runs += 1
        if self.run_error:
            raise ValueError("ignored; no private message returned")

    def close(self):
        self.owner.closes += 1
        if self.close_error:
            raise ValueError("ignored")


class FakeLauncher:
    def __init__(self, *, fail_open=False, fail_run=False, fail_close=False):
        self.fail_open = fail_open
        self.fail_run = fail_run
        self.fail_close = fail_close
        self.calls = []
        self.runs = self.closes = 0

    def open(self, handle, spec):
        self.calls.append((handle, spec))
        if self.fail_open:
            raise ValueError("ignored")
        return FakeSession(self, run_error=self.fail_run, close_error=self.fail_close)


def test_plan_drives_two_pairs_and_no_case_data_reaches_launch():
    plan = synthetic_plan()
    selected = []

    def selector(token, seed):
        selected.append((token, seed))
        return controller.SyntheticHandle()

    launcher = FakeLauncher()
    result = controller.run_synthetic_pair(plan, selector=selector, launcher=launcher)
    assert controller.SOURCE_ONLY is True
    assert selected == [("1" * 32, 13), ("2" * 32, 29)]
    assert [spec.feedback_enabled for _, spec in launcher.calls] == [False, True, True, False]
    assert [spec.seed for _, spec in launcher.calls] == [13, 13, 29, 29]
    assert launcher.calls[0][0] is launcher.calls[1][0]
    assert launcher.calls[2][0] is launcher.calls[3][0]
    assert launcher.calls[0][0] is not launcher.calls[2][0]
    for handle, spec in launcher.calls:
        assert not hasattr(handle, "__dict__")
        assert set(asdict(spec)) == {
            "seed",
            "feedback_enabled",
            "review_enabled",
            "pins",
            "configuration_sha256",
            "budgets",
        }
        assert spec.review_enabled is True
        assert "1" * 32 not in repr(spec) and "2" * 32 not in repr(spec)
    off = asdict(launcher.calls[0][1])
    on = asdict(launcher.calls[1][1])
    assert off.pop("feedback_enabled") is False
    assert on.pop("feedback_enabled") is True
    assert off == on
    assert result == {
        "status": "synthetic_callbacks_completed",
        "selected_cases": 2,
        "planned_runs": 4,
        "completed_callbacks": 4,
        "failed_callbacks": 0,
        "not_started": 0,
        "selector_errors": 0,
        "close_callbacks_returned": 4,
        "open_or_close_callback_errors": 0,
        "os_closure": "unverified",
        "exposure_status": "unknown",
        "authenticity_and_preregistration": "unverified",
        "runtime_conformance": "unverified",
        "blind_selection": "unverified",
        "callback_trust_and_isolation": "unverified",
    }


@pytest.mark.parametrize("invalid", ["pin", "denominator", "order", "judge", "extra"])
def test_invalid_plan_refused_before_selector_or_launch(invalid):
    plan = synthetic_plan()
    if invalid == "pin":
        plan["arms"]["on"]["pins"]["model"] = "f" * 64
    elif invalid == "denominator":
        plan["planned_runs"] = 2
    elif invalid == "order":
        plan["schedule"][0]["order"] = ["on", "on"]
    elif invalid == "judge":
        plan["arms"]["on"]["pins"]["case_and_judge"] = "f" * 64
    else:
        plan["arms"]["on"]["oracle"] = "not allowed"
    launcher = FakeLauncher()
    selections = []
    with pytest.raises(ValueError):
        controller.run_synthetic_pair(
            plan, selector=lambda *args: selections.append(args), launcher=launcher
        )
    assert selections == [] and launcher.calls == []


def test_callback_cannot_change_validated_launches_by_mutating_original_plan():
    plan = synthetic_plan()

    def selector(token, seed):
        plan["arms"]["on"]["budgets"]["policy_calls"] = 100
        plan["schedule"][0]["order"] = ["on", "on"]
        return controller.SyntheticHandle()

    launcher = FakeLauncher()
    result = controller.run_synthetic_pair(plan, selector=selector, launcher=launcher)
    assert result["completed_callbacks"] == 4
    assert [spec.feedback_enabled for _, spec in launcher.calls] == [False, True, True, False]
    assert all(dict(spec.budgets)["policy_calls"] == 16 for _, spec in launcher.calls)


@pytest.mark.parametrize("mode", ["open", "run", "close"])
def test_failure_stops_remaining_launches_and_preserves_denominator_and_closure(mode):
    launcher = FakeLauncher(
        fail_open=mode == "open", fail_run=mode == "run", fail_close=mode == "close"
    )
    result = controller.run_synthetic_pair(
        synthetic_plan(), selector=lambda *_: controller.SyntheticHandle(), launcher=launcher
    )
    assert result["status"] == "unresolved"
    assert (result["completed_callbacks"], result["failed_callbacks"], result["not_started"]) == (
        0,
        1,
        3,
    )
    assert result["planned_runs"] == 4
    assert launcher.closes == (0 if mode == "open" else 1)
    assert result["open_or_close_callback_errors"] == (1 if mode in ("open", "close") else 0)
    assert result["os_closure"] == "unverified"
    assert result["close_callbacks_returned"] == (1 if mode == "run" else 0)


def test_selector_error_and_reused_handle_fail_closed():
    launcher = FakeLauncher()
    result = controller.run_synthetic_pair(
        synthetic_plan(), selector=lambda *_: object(), launcher=launcher
    )
    assert (result["selector_errors"], result["not_started"], result["planned_runs"]) == (1, 4, 4)
    assert launcher.calls == []
    shared = controller.SyntheticHandle()
    result = controller.run_synthetic_pair(
        synthetic_plan(), selector=lambda *_: shared, launcher=launcher
    )
    assert (result["completed_callbacks"], result["selector_errors"], result["not_started"]) == (
        2,
        1,
        2,
    )
    assert launcher.closes == 2


def test_interruption_propagates_after_close():
    class InterruptSession(FakeSession):
        def run(self):
            raise KeyboardInterrupt

    class InterruptLauncher(FakeLauncher):
        def open(self, handle, spec):
            return InterruptSession(self)

    launcher = InterruptLauncher()
    with pytest.raises(KeyboardInterrupt):
        controller.run_synthetic_pair(
            synthetic_plan(), selector=lambda *_: controller.SyntheticHandle(), launcher=launcher
        )
    assert launcher.closes == 1


def test_later_failure_preserves_completed_and_unstarted_counts():
    class LaterFailureLauncher(FakeLauncher):
        def open(self, handle, spec):
            self.calls.append((handle, spec))
            return FakeSession(self, run_error=len(self.calls) == 2)

    launcher = LaterFailureLauncher()
    result = controller.run_synthetic_pair(
        synthetic_plan(), selector=lambda *_: controller.SyntheticHandle(), launcher=launcher
    )
    assert result["status"] == "unresolved"
    assert (result["completed_callbacks"], result["failed_callbacks"], result["not_started"]) == (
        1,
        1,
        2,
    )
    assert result["close_callbacks_returned"] == 2
    assert launcher.closes == 2


def test_noop_close_does_not_claim_os_closure():
    class NoopSession:
        def run(self):
            pass

        def close(self):
            pass

    class NoopLauncher:
        def open(self, handle, spec):
            return NoopSession()

    result = controller.run_synthetic_pair(
        synthetic_plan(), selector=lambda *_: controller.SyntheticHandle(), launcher=NoopLauncher()
    )
    assert result["close_callbacks_returned"] == 4
    assert result["open_or_close_callback_errors"] == 0
    assert result["os_closure"] == "unverified"
    assert result["callback_trust_and_isolation"] == "unverified"
