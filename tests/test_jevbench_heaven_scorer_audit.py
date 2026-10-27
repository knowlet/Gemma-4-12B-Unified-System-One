"""Synthetic checks only; no benchmark tasks or model execution."""

import math
import os

import jevbench_heaven_scorer_audit as audit
import pytest


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
