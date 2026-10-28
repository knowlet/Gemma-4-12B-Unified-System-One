"""Synthetic checks only; no benchmark tasks or model execution."""

import math
import os
import sys
from pathlib import Path

import jevbench_heaven_scorer_audit as audit
import pytest


def source_case(tmp_path, suite, version):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    prefix = "jev_decision3_jevbench_" + suite
    suffix = "_release_20261027.py" if version == "legacy" else "_release_v2_20261028.py"
    held = [scripts / (prefix + "_owner.py"), scripts / (prefix + "_supervisor.py")]
    releases = [scripts / (prefix + "_owner" + suffix), scripts / (prefix + "_supervisor" + suffix)]
    held += [scripts / "jev_decision3_jevbench_generic.py", scripts / "jev_gpu_forward_lease.py"]
    releases += [
        scripts / "jev_decision3_jevbench_generic_release_20261027.py",
        scripts / "jev_gpu_forward_lease_release_20261027.py",
    ]
    marker = b"\nSOURCE_ONLY = True\n"
    for source, release in zip(held, releases, strict=True):
        source.write_bytes(b"# synthetic\n" + marker)
        release.write_bytes(source.read_bytes().replace(marker, b"\nSOURCE_ONLY = False\n"))
    names = [
        "jev_decision3_candidate_browser_child.py",
        "jev_decision3_candidate_http_owner.py",
        "jev_stage_memory_profile_v2.py",
    ]
    if suite == "coherence":
        names.append("jevbench_public_request_inventory.py")
    for name in names:
        (scripts / name).write_bytes(b"# synthetic\n")
    files = [*held, *releases, *(scripts / name for name in names)]
    pins = {str(path): audit.sha(path.read_bytes()) for path in files}
    return (
        {
            "source_pins": pins,
            "generic_release": str(releases[2]),
            "transport": {"lease_release": str(releases[3])},
        },
        held,
        releases,
    )


@pytest.mark.parametrize("suite", ["heaven", "coherence"])
@pytest.mark.parametrize("version", ["legacy", "v2"])
def test_exact_release_version_and_gate_only_pairing(tmp_path, suite, version):
    plan, held, releases = source_case(tmp_path, suite, version)
    selected, pair, generic, lease = audit.release_layout(plan, suite, tmp_path)
    assert selected == version
    assert pair == tuple(releases[:2]) and (generic, lease) == tuple(releases[2:])
    for source, release in zip(held, releases, strict=True):
        audit.verify_release_pair(source, release)
    # Even a re-pinned release with an extra change is not a valid gate-only release.
    releases[0].write_bytes(releases[0].read_bytes() + b"# altered\n")
    plan["source_pins"][str(releases[0])] = audit.sha(releases[0].read_bytes())
    with pytest.raises(ValueError, match="release_source_difference"):
        audit.verify_release_pair(held[0], releases[0])


@pytest.mark.parametrize("suite", ["heaven", "coherence"])
@pytest.mark.parametrize(
    "fault", ["mixed", "partial_other", "rogue", "generic", "lease", "symlink"]
)
def test_release_layout_rejects_unapproved_or_mixed_paths_before_opening(
    tmp_path, monkeypatch, suite, fault
):
    plan, _held, releases = source_case(tmp_path, suite, "legacy")
    scripts = tmp_path / "scripts"
    prefix = "jev_decision3_jevbench_" + suite
    if fault in {"mixed", "partial_other"}:
        for component in ["owner", "supervisor"] if fault == "mixed" else ["owner"]:
            path = scripts / (prefix + "_" + component + "_release_v2_20261028.py")
            plan["source_pins"][str(path)] = "0" * 64
    elif fault == "rogue":
        path = scripts / (prefix + "_owner_release_other.py")
        plan["source_pins"][str(path)] = "0" * 64
    elif fault == "generic":
        plan["generic_release"] = str(scripts / "other.py")
    elif fault == "lease":
        plan["transport"]["lease_release"] = str(scripts / "other.py")
    else:
        releases[0].unlink()
        releases[0].symlink_to(releases[1])
    monkeypatch.setattr(Path, "read_bytes", lambda _path: pytest.fail("unexpected read"))
    with pytest.raises(ValueError, match="release_version"):
        audit.release_layout(plan, suite, tmp_path)


@pytest.mark.parametrize("suite", ["heaven", "coherence"])
@pytest.mark.parametrize("version", ["legacy", "v2"])
def test_source_check_hashes_every_selected_release(tmp_path, monkeypatch, suite, version):
    import jevbench_coherence_scorer_audit as coherence

    plan, _held, releases = source_case(tmp_path, suite, version)
    config = tmp_path / "configs" / "synthetic.json"
    config.parent.mkdir()
    config.write_bytes(b"{}")
    plan["source_pins"][str(config)] = audit.sha(config.read_bytes())
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(coherence, "ROOT", tmp_path)
    monkeypatch.setattr(audit.heaven, "CONFIG", config)
    monkeypatch.setattr(coherence.inventory, "COHERENCE_CONFIG", config)
    monkeypatch.setattr(audit, "CONFIG_SHA", audit.sha(config.read_bytes()))
    monkeypatch.setattr(coherence, "CONFIG_SHA", audit.sha(config.read_bytes()))
    for module in (audit.DecisionRequest.__module__, audit.native_payload.__module__):
        plan["source_pins"][str(Path(sys.modules[module].__file__).absolute())] = "external"
    plan["source_pins"][str(Path(audit.heaven.__file__).absolute())] = "external"
    if suite == "heaven":
        result = audit.source_check(plan)
    else:
        result, _ = coherence._source_check(plan)
    assert result.get("release_revision") == ("v2" if version == "v2" else None)
    releases[0].write_bytes(releases[0].read_bytes() + b"# altered\n")
    with pytest.raises(ValueError, match="public_source_pin|source_pin"):
        if suite == "heaven":
            audit.source_check(plan)
        else:
            coherence._source_check(plan)


@pytest.mark.parametrize("suite", ["heaven", "coherence"])
@pytest.mark.parametrize("fault", ["relative", "dotdot", "symlink_file", "symlink_dir"])
def test_source_check_rejects_noncanonical_pin_without_reading_target(
    tmp_path, monkeypatch, suite, fault
):
    import jevbench_coherence_scorer_audit as coherence

    plan, _held, _releases = source_case(tmp_path, suite, "legacy")
    scripts = tmp_path / "scripts"
    target = scripts / "synthetic_target.py"
    target.write_bytes(b"# target must not be read\n")
    if fault == "relative":
        path = Path("scripts/synthetic_target.py")
    elif fault == "dotdot":
        path = scripts / "nested" / ".." / target.name
    elif fault == "symlink_file":
        path = scripts / "alias.py"
        path.symlink_to(target)
    else:
        folder = scripts / "real_dir"
        folder.mkdir()
        target = folder / "nested_target.py"
        target.write_bytes(b"# target must not be read\n")
        alias = scripts / "alias_dir"
        alias.symlink_to(folder, target_is_directory=True)
        path = alias / target.name
    plan["source_pins"][str(path)] = "0" * 64
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(coherence, "ROOT", tmp_path)
    original_read = Path.read_bytes

    def guarded_read(candidate):
        assert candidate != path and candidate != target, "unexpected pin read"
        return original_read(candidate)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    with pytest.raises(ValueError, match="public_source_pin|source_pin"):
        if suite == "heaven":
            audit.source_check(plan)
        else:
            coherence._source_check(plan)


def test_external_pin_is_skipped_without_resolve_or_content_read(tmp_path, monkeypatch):
    allowed = (tmp_path / "scripts", tmp_path / "src/s1", tmp_path / "configs")
    outside = tmp_path / "outside" / "synthetic.py"
    monkeypatch.setattr(
        Path, "resolve", lambda *_args, **_kwargs: pytest.fail("unexpected resolve")
    )
    monkeypatch.setattr(Path, "read_bytes", lambda *_args: pytest.fail("unexpected read"))
    assert audit.public_pin_path(str(outside), allowed, failure="source_pin") is None
    assert (
        audit.public_pin_path(
            str(outside.parent) + "/./synthetic.py", allowed, failure="source_pin"
        )
        is None
    )
    assert (
        audit.public_pin_path(str(outside.parent) + "//synthetic.py", allowed, failure="source_pin")
        is None
    )


@pytest.mark.parametrize("suite", ["heaven", "coherence"])
@pytest.mark.parametrize("alias_separator", ["/./", "//"])
def test_duplicate_noncanonical_pin_is_rejected_before_alias_read(
    tmp_path, monkeypatch, suite, alias_separator
):
    import jevbench_coherence_scorer_audit as coherence

    plan, _held, _releases = source_case(tmp_path, suite, "legacy")
    target = tmp_path / "scripts" / "synthetic_target.py"
    target.write_bytes(b"# synthetic\n")
    digest = audit.sha(target.read_bytes())
    plan["source_pins"][str(target)] = digest
    plan["source_pins"][str(target.parent) + alias_separator + target.name] = digest
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(coherence, "ROOT", tmp_path)
    original_read = Path.read_bytes
    target_reads = []

    def count_read(path):
        if path == target:
            target_reads.append(path)
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", count_read)
    with pytest.raises(ValueError, match="public_source_pin|source_pin"):
        if suite == "heaven":
            audit.source_check(plan)
        else:
            coherence._source_check(plan)
    assert target_reads == [target]


def test_event_encoding_distinguishes_candidate_order_and_rejects_nonfinite():
    first = {"questions": {"synthetic": {"criteria": {"a": "A", "b": "B"}}}}
    reversed_options = {"questions": {"synthetic": {"criteria": {"b": "B", "a": "A"}}}}
    assert audit.sha(audit.encode(first)) != audit.sha(audit.encode(reversed_options))
    with pytest.raises(ValueError):
        audit.encode({"value": math.nan})


def test_evidence_reader_rejects_parent_and_symlink_escape(tmp_path):
    selected = tmp_path / "selected"
    selected.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("synthetic")
    (selected / "link").symlink_to(outside)
    with pytest.raises(ValueError, match="evidence_path"):
        audit.read(selected, "../outside")
    with pytest.raises(ValueError, match="evidence_path"):
        audit.read(selected, "link")


def test_evidence_reader_rejects_nonfinite_json(tmp_path):
    (tmp_path / "synthetic.json").write_text('{"value": NaN}')
    with pytest.raises(ValueError, match="nonfinite"):
        audit.read_json(tmp_path, "synthetic.json")


def test_comparison_preserves_json_types_and_scorer_sorted_keys():
    assert audit.same_json({"b": 2, "a": 1}, {"a": 1, "b": 2})
    assert not audit.same_ordered({"b": 2, "a": 1}, {"a": 1, "b": 2})
    for same_looking in (True, 1.0):
        assert not audit.same_json({"value": 1}, {"value": same_looking})
        assert not audit.same_ordered({"value": 1}, {"value": same_looking})


def journal_plan(path):
    info = path.stat()
    return {
        "transport": {
            "journal": str(path),
            "stat": {"journal": {"device": info.st_dev, "inode": info.st_ino}},
        }
    }


@pytest.mark.parametrize("name", ["journal.dat", "forward.journal", "another-name"])
def test_plan_selected_journal_is_bound_to_inode_and_hash(tmp_path, name):
    journal = tmp_path / name
    raw = b'{"event":"synthetic"}\n'
    journal.write_bytes(raw)
    plan = journal_plan(journal)
    assert audit.read_pinned_journal(tmp_path, plan, audit.sha(raw)) == raw
    with pytest.raises(ValueError, match="lease_digest"):
        audit.read_pinned_journal(tmp_path, plan, "0" * 64)


def test_journal_outside_root_or_symlink_is_rejected(tmp_path):
    selected = tmp_path / "selected"
    selected.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"synthetic")
    with pytest.raises(ValueError, match="evidence_path"):
        audit.read_pinned_journal(selected, journal_plan(outside), audit.sha(b"synthetic"))
    local = selected / "real"
    local.write_bytes(b"synthetic")
    link = selected / "link"
    link.symlink_to(local)
    with pytest.raises(ValueError, match="evidence_path"):
        audit.read_pinned_journal(selected, journal_plan(link), audit.sha(b"synthetic"))


@pytest.mark.parametrize("fault", ["replacement", "bool_inode", "missing_stat"])
def test_journal_changed_or_invalid_plan_identity_fails_closed(tmp_path, fault):
    journal = tmp_path / "forward.journal"
    journal.write_bytes(b"synthetic")
    plan = journal_plan(journal)
    if fault == "replacement":
        replacement = tmp_path / "replacement"
        replacement.write_bytes(b"synthetic")
        replacement.replace(journal)
    elif fault == "bool_inode":
        plan["transport"]["stat"]["journal"]["inode"] = True
    else:
        plan["transport"]["stat"] = None
    with pytest.raises(ValueError, match="journal_identity"):
        audit.read_pinned_journal(tmp_path, plan, audit.sha(b"synthetic"))


def test_journal_mutation_during_read_is_rejected(tmp_path, monkeypatch):
    journal = tmp_path / "forward.journal"
    journal.write_bytes(b"synthetic")
    plan = journal_plan(journal)
    original = os.fstat
    calls = 0

    def concurrent_change(fd):
        nonlocal calls
        info = original(fd)
        if info.st_ino == plan["transport"]["stat"]["journal"]["inode"]:
            calls += 1
            if calls == 2:
                journal.write_bytes(b"modified!")
                os.utime(journal, ns=(1, 1))
        return original(fd)

    monkeypatch.setattr(audit.os, "fstat", concurrent_change)
    with pytest.raises(ValueError, match="journal_identity"):
        audit.read_pinned_journal(tmp_path, plan, audit.sha(b"synthetic"))


@pytest.mark.parametrize("swap_on", ["parent_open", "journal_open"])
def test_parent_symlink_swap_during_open_is_rejected(tmp_path, monkeypatch, swap_on):
    root = tmp_path / "selected"
    root.mkdir()
    parent = root / "nested"
    parent.mkdir()
    journal = parent / "forward.journal"
    journal.write_bytes(b"synthetic")
    plan = journal_plan(journal)
    original = os.open
    moved = tmp_path / "moved-outside"
    swapped = False

    def swap_parent(path, flags, *args, **kwargs):
        nonlocal swapped
        target = "nested" if swap_on == "parent_open" else "forward.journal"
        if not swapped and path == target and kwargs.get("dir_fd") is not None:
            parent.rename(moved)
            parent.symlink_to(moved, target_is_directory=True)
            swapped = True
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(audit.os, "open", swap_parent)
    with pytest.raises(ValueError, match="evidence_path"):
        audit.read_pinned_journal(root, plan, audit.sha(b"synthetic"))
    assert swapped


@pytest.mark.parametrize("swap_on", ["ancestor_open", "root_open", "journal_open"])
def test_run_root_ancestor_symlink_swap_is_rejected(tmp_path, monkeypatch, swap_on):
    ancestor = tmp_path / "ancestor"
    ancestor.mkdir()
    root = ancestor / "selected"
    root.mkdir()
    journal = root / "forward.journal"
    journal.write_bytes(b"synthetic")
    plan = journal_plan(journal)
    original = os.open
    moved = tmp_path / "moved-ancestor"
    swapped = False

    def swap_ancestor(path, flags, *args, **kwargs):
        nonlocal swapped
        target = {
            "ancestor_open": "ancestor",
            "root_open": "selected",
            "journal_open": "forward.journal",
        }[swap_on]
        if not swapped and path == target and kwargs.get("dir_fd") is not None:
            ancestor.rename(moved)
            ancestor.symlink_to(moved, target_is_directory=True)
            swapped = True
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(audit.os, "open", swap_ancestor)
    with pytest.raises(ValueError, match="evidence_path"):
        audit.read_pinned_journal(root, plan, audit.sha(b"synthetic"))
    assert swapped
