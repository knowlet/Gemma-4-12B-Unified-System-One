"""Synthetic availability checks; no historical evidence or runtime access."""

import pytest
from local_evidence import require_local_evidence


@pytest.mark.parametrize("setting", [None, "0", "true"])
def test_missing_inputs_skip_without_strict_replay(tmp_path, monkeypatch, setting):
    if setting is None:
        monkeypatch.delenv("JEV_REQUIRE_LOCAL_EVIDENCE", raising=False)
    else:
        monkeypatch.setenv("JEV_REQUIRE_LOCAL_EVIDENCE", setting)
    present = tmp_path / "present.json"
    present.write_bytes(b"synthetic evidence")
    missing = [tmp_path / "missing-handoff.json", tmp_path / "missing-proof.json"]

    with pytest.raises(pytest.skip.Exception) as exc:
        require_local_evidence(present, *missing)

    reason = str(exc.value)
    assert all(str(path) in reason for path in missing)
    assert str(present) not in reason
    assert "JEV_REQUIRE_LOCAL_EVIDENCE=1" in reason
    assert "Restore" in reason


def test_strict_replay_fails_for_missing_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("JEV_REQUIRE_LOCAL_EVIDENCE", "1")
    missing = tmp_path / "missing-handoff.json"

    with pytest.raises(pytest.fail.Exception) as exc:
        require_local_evidence(missing)

    assert str(missing) in str(exc.value)
    assert "JEV_REQUIRE_LOCAL_EVIDENCE=1" in str(exc.value)


@pytest.mark.parametrize("strict", [False, True])
def test_present_inputs_leave_content_and_type_validation_to_original_guards(
    tmp_path, monkeypatch, strict
):
    monkeypatch.setenv("JEV_REQUIRE_LOCAL_EVIDENCE", "1" if strict else "0")
    invalid_content = tmp_path / "invalid.json"
    invalid_content.write_bytes(b"\xffnot valid JSON or authenticated evidence")
    directory = tmp_path / "directory.json"
    directory.mkdir()

    assert require_local_evidence(invalid_content, directory) is None
    invalid_content.write_bytes(b"mutated")
    assert require_local_evidence(invalid_content, directory) is None


@pytest.mark.parametrize("target_kind", ["missing", "file", "directory"])
@pytest.mark.parametrize("strict", [False, True])
def test_symlinks_are_never_silently_skipped(tmp_path, monkeypatch, target_kind, strict):
    monkeypatch.setenv("JEV_REQUIRE_LOCAL_EVIDENCE", "1" if strict else "0")
    target = tmp_path / "target"
    if target_kind == "file":
        target.write_bytes(b"synthetic evidence")
    elif target_kind == "directory":
        target.mkdir()
    link = tmp_path / "evidence.json"
    link.symlink_to(target, target_is_directory=target_kind == "directory")

    assert require_local_evidence(link) is None


def test_empty_requirement_is_a_noop(monkeypatch):
    monkeypatch.setenv("JEV_REQUIRE_LOCAL_EVIDENCE", "1")
    assert require_local_evidence() is None
