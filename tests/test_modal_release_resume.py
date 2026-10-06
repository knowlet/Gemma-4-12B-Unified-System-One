"""Exercise the Modal wrapper/dispatcher without initializing Modal resources."""

import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("resume", [False, True])
def test_evaluate_resume_reaches_run_phase(tmp_path, resume):
    source = ROOT / "apps/modal/release.py"
    tree = ast.parse(source.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"evaluate", "main"}
    ]
    for node in functions:
        node.decorator_list = []
    calls = []
    namespace = {
        "Path": Path,
        "json": json,
        "_output": lambda run: None,
        "_phase": lambda *args: calls.append(args) or {"status": "completed"},
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    namespace["evaluate"].remote = namespace["evaluate"]
    namespace["main"](stage="evaluate", run="retry", resume=resume, output=str(tmp_path))
    assert calls == [("evaluate", "retry", resume)]
    assert json.loads((tmp_path / "retry/evaluate-receipt.json").read_text()) == {
        "status": "completed"
    }
