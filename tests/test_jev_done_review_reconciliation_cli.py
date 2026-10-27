"""Synthetic files only; no real run evidence."""

import hashlib
import json

import jev_done_review_reconciliation_cli as cli
import pytest
from test_jev_done_review_reconciliation import fixture, wire


def create(tmp_path):
    data = fixture()

    def put(path, value):
        p = tmp_path / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(value if type(value) is bytes else wire(value))

    base = "native/candidate"
    events_dir, capture_dir, service_dir = (
        base + "/done-review-events",
        base + "/client-capture",
        "candidate-service",
    )
    records = base + "/results/attempts/fake-001/model_calls.json"
    put(records, data["runner_records"])
    for i, event in enumerate(data["events"], 1):
        put(f"{events_dir}/event-{i:04}.json", event)
    put(base + "/owned-client-closure.json", data["client_closure"])
    for i, item in enumerate(data["captures"], 1):
        for field in ("record", "request_bytes", "response_bytes"):
            put(
                f"{capture_dir}/attempt-{i:04}/"
                + ("record.json" if field == "record" else field + ".bin"),
                item[field],
            )
    refs = []
    for i, binding in enumerate(data["bindings"], 1):
        raw = wire(binding)
        name = f"binding-{i:04}.json"
        put(service_dir + "/" + name, raw)
        refs.append(
            {"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}
        )
    put(service_dir + "/terminal.json", {"service": {"private_bindings": refs}})
    put(service_dir + "/http-events.jsonl", b"\n".join(wire(e) for e in data["service_events"]))
    return dict(
        runner_records=records,
        events_dir=events_dir,
        capture_dir=capture_dir,
        service_dir=service_dir,
    )


def test_loads_realistic_synthetic_tree(tmp_path):
    args = create(tmp_path)
    result = cli.load(str(tmp_path), **args)
    assert result["matched_events"] == {"review_call": 1, "replan_call": 1}
    assert result["source_origin_verified"] is False
    assert "fake" not in json.dumps(result)


def test_parent_symlink_rejected(tmp_path):
    args = create(tmp_path)
    native = tmp_path / "native"
    native.rename(tmp_path / "real")
    native.symlink_to(tmp_path / "real", target_is_directory=True)
    with pytest.raises(OSError):
        cli.load(str(tmp_path), **args)


def test_symlink_in_root_ancestors_rejected(tmp_path):
    root = tmp_path / "real"
    root.mkdir()
    args = create(root)
    (tmp_path / "link").symlink_to(root, target_is_directory=True)
    with pytest.raises(OSError):
        cli.load(str(tmp_path / "link"), **args)


def test_binding_receipt_mismatch_rejected(tmp_path):
    args = create(tmp_path)
    (tmp_path / "candidate-service/binding-0001.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="binding differs"):
        cli.load(str(tmp_path), **args)


def test_noncontiguous_events_and_multiple_attempts_rejected(tmp_path):
    args = create(tmp_path)
    event = tmp_path / args["events_dir"] / "event-0002.json"
    event.rename(event.parent / "event-0003.json")
    with pytest.raises(ValueError, match="noncontiguous"):
        cli.load(str(tmp_path), **args)
    (event.parent / "event-0003.json").rename(event)
    (tmp_path / "native/candidate/results/attempts/second").mkdir()
    with pytest.raises(ValueError, match="exactly one"):
        cli.load(str(tmp_path), **args)


@pytest.mark.parametrize(
    "status,expected",
    [("supplied_evidence_consistent", 0), ("unresolved", 1), ("no_success_events_observed", 1)],
)
def test_cli_fail_closed_while_printing_aggregate(monkeypatch, capsys, status, expected):
    monkeypatch.setattr(cli, "load", lambda *args, **kwargs: {"status": status})
    monkeypatch.setattr(
        "sys.argv",
        [
            "audit",
            "--run-root",
            "/unused",
            "--runner-records",
            "unused",
            "--events-dir",
            "unused",
            "--capture-dir",
            "unused",
            "--service-dir",
            "unused",
        ],
    )
    assert cli.main() == expected
    assert json.loads(capsys.readouterr().out) == {"status": status}
