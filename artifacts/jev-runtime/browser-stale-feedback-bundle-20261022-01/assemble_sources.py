"""Reproduce explicit source-only successors; never start a model or actor."""

import ast
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LABEL = str(HERE.relative_to(ROOT))
CONTROL_SHA = "73bff770e242b93d5b7b53d24ef933926bd63897a4707d6d2e2a0f06f4438b6d"
PARENT = ROOT / "artifacts/jev-runtime/native-v2-browser-execution-20261022-04"
SPECS = {
    "feedback_native_driver.py": {
        "source": ROOT / "scripts/jev_native_v2_browser.py",
        "sha256": "c513dc5bcb091de4fd0d4bd7a0776078e82be16106328b6de9d8424e32f2ee11",
        "edits": [
            (
                "ROOT = Path(__file__).resolve().parents[1]",
                "ROOT = Path(__file__).resolve().parents[3]",
            ),
            (
                'PROFILE = "jev-current-nf4-v7-browser-v1"',
                'PROFILE = "jev-current-nf4-v7-feedback-browser-v1"',
            ),
            (
                '        "profile": PROFILE,\n        "prompt_format": prompt_format,\n',
                '        "profile": PROFILE,\n        "agent_protocol": "typed-stale-action-feedback-v1",\n        "prompt_format": prompt_format,\n',
            ),
            (
                "    import jev_browser_profile_v7_benchmark as v7\n",
                "    import feedback_bundle as v7\n",
            ),
            (
                '        "scripts/jev_native_v2_browser.py",\n',
                '        "scripts/jev_native_v2_browser.py",\n'
                + "".join(
                    '        "' + LABEL + "/" + name + '",\n'
                    for name in [
                        "feedback_native_driver.py",
                        "feedback_bundle.py",
                        "action_feedback.py",
                    ]
                ),
            ),
        ],
    },
    "launch_browser.py": {
        "source": PARENT / "launch_browser.py",
        "sha256": "44eadeb19d907dc6084e31e5e86aa81bb73f514365ee3722c0baaf274a5fd842",
        "edits": [
            (
                'DRIVER = "scripts/jev_native_v2_browser.py"',
                'DRIVER = "' + LABEL + '/feedback_native_driver.py"',
            ),
            (
                'PROFILE = "jev-current-nf4-v7-browser-owned-v1"',
                'PROFILE = "jev-current-nf4-v7-feedback-browser-owned-v1"',
            ),
            (
                'FORMATS = {"raw": "raw-json-v1", "v2": "shared-elements-v2"}',
                'FORMATS = {"v2-feedback": "shared-elements-v2"}',
            ),
            (
                '    require(plan["schema_id"] == PROFILE and plan["root"] == str(ROOT), "Exact local profile/root")\n',
                '    require(plan["schema_id"] == PROFILE and plan["root"] == str(ROOT), "Exact local profile/root")\n    require(plan.get("agent_protocol") == "typed-stale-action-feedback-v1" and plan.get("matched_control_plan_sha256") == "'
                + CONTROL_SHA
                + '", "Explicit selected feedback protocol/control")\n',
            ),
            (
                "        DRIVER,\n        GUARD_INDEX,\n",
                '        DRIVER,\n        "'
                + LABEL
                + '/feedback_bundle.py",\n        "'
                + LABEL
                + '/action_feedback.py",\n        GUARD_INDEX,\n',
            ),
            (
                '        and receipt["prompt_format"] == plan["prompt_format"],\n',
                '        and receipt["prompt_format"] == plan["prompt_format"]\n        and receipt.get("agent_protocol") == plan["agent_protocol"],\n',
            ),
        ],
    },
    "browser_entry.py": {
        "source": PARENT / "browser_entry.py",
        "sha256": "b3fe4fab7fef0d6042516504cd659eb42fae784d62e0ed66fce5cd36b0ff32c2",
        "edits": [],
    },
    "action_feedback.py": {
        "source": ROOT
        / "artifacts/jev-runtime/browser-stale-feedback-runtime-20261022-01/action_feedback.py",
        "sha256": "79ccf2ccc2d88f9613c1320ee883ce9742b0a88c0535dbfae4de7c97ee428246",
        "edits": [],
    },
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def pin(blob):
    return {"sha256": hashlib.sha256(blob).hexdigest(), "size_bytes": len(blob)}


def projections():
    result = {}
    for name, spec in SPECS.items():
        original = spec["source"].read_bytes()
        need(pin(original)["sha256"] == spec["sha256"], "fixed parent source changed")
        derivative = original
        for before, after in spec["edits"]:
            before, after = before.encode(), after.encode()
            need(derivative.count(before) == 1, "exact unique source edit anchor: " + name)
            derivative = derivative.replace(before, after, 1)
        inverse = derivative
        for before, after in reversed(spec["edits"]):
            before, after = before.encode(), after.encode()
            need(inverse.count(after) == 1, "exact inverse source edit anchor: " + name)
            inverse = inverse.replace(after, before, 1)
        need(
            inverse == original and ast.dump(ast.parse(inverse)) == ast.dump(ast.parse(original)),
            "complete inverse source bytes/AST",
        )
        patch = "".join(
            difflib.unified_diff(
                original.decode().splitlines(keepends=True),
                derivative.decode().splitlines(keepends=True),
                fromfile=str(spec["source"].relative_to(ROOT)),
                tofile=LABEL + "/" + name,
            )
        ).encode()
        result[name] = (
            derivative,
            patch,
            {
                "parent": {"path": str(spec["source"].relative_to(ROOT)), **pin(original)},
                "derivative": {"path": LABEL + "/" + name, **pin(derivative)},
                "literal_substitutions": len(spec["edits"]),
                "inverse_complete_bytes_equal": True,
                "inverse_complete_AST_equal": True,
                "edits": [{"before": before, "after": after} for before, after in spec["edits"]],
            },
        )
    return result


def main():
    rows = []
    for name, (source, patch, row) in projections().items():
        for path, blob in [(HERE / name, source), (HERE / (name + ".patch"), patch)]:
            with path.open("xb") as stream:
                stream.write(blob)
        rows.append(row)
    receipt = {
        "schema_id": "jev-feedback-source-successor-inverse-proof-v1",
        "agent_protocol": "typed-stale-action-feedback-v1",
        "sources": rows,
        "actual_actor_or_model_started": False,
    }
    with (HERE / "source-deltas.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "source_successors_assembled_no_execution", "files": len(rows)}))


if __name__ == "__main__":
    main()
