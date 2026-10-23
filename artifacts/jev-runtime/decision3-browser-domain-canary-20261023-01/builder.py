"""Exact-source browser question construction with explicit in-process transport.

No upstream HTTP module/client is imported. Only three unchanged functions and
two literal instruction constants are compiled from authenticated source.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import math
import time
from pathlib import Path
from types import SimpleNamespace

FUNCTIONS = ("action_space", "validate_choice", "choose")
CONSTANTS = ("NEXT_ACTION", "TARGET")


def require(value, message):
    if not value:
        raise RuntimeError(message)


def source_bytes(record):
    path = Path(record["path"])
    raw = path.read_bytes()
    require(
        path.suffix == ".py"
        and len(raw) == record["size_bytes"]
        and len(raw) < 200_000
        and hashlib.sha256(raw).hexdigest() == record["sha256"],
        "selected browser source changed",
    )
    return raw


def load_builder(model_record, questions_record, transport):
    model_raw, question_raw = source_bytes(model_record), source_bytes(questions_record)
    model_tree, question_tree = ast.parse(model_raw), ast.parse(question_raw)
    selected = []
    proof = {}
    for name in FUNCTIONS:
        found = [n for n in model_tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
        require(len(found) == 1, "one unchanged original browser function required")
        node = found[0]
        selected.append(copy.deepcopy(node))
        proof[name] = {
            "line": node.lineno,
            "ast_sha256": hashlib.sha256(ast.dump(node).encode()).hexdigest(),
        }
    constants = {}
    for name in CONSTANTS:
        found = [
            n
            for n in question_tree.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
        ]
        require(len(found) == 1, "one original question instruction required")
        constants[name] = ast.literal_eval(found[0].value)
        require(type(constants[name]) is str, "literal original instruction required")
        proof[name] = {
            "line": found[0].lineno,
            "ast_sha256": hashlib.sha256(ast.dump(found[0]).encode()).hexdigest(),
        }
    module = ast.Module(body=selected, type_ignores=[])
    require(
        all(ast.dump(a) == ast.dump(b) for a, b in zip(selected, module.body, strict=True)),
        "browser function AST changed",
    )
    namespace = {
        "__builtins__": __builtins__,
        "__name__": "_canary_exact_browser_functions",
        "json": __import__("json"),
        "math": math,
        "time": time,
        "os": SimpleNamespace(
            environ={"TYPESAFE_MODEL": "d3-edge", "TYPESAFE_API_KEY": "in-process-canary"}
        ),
        "post_json": transport,
        **constants,
    }
    exec(
        compile(module, model_record["path"], "exec", dont_inherit=True),
        namespace,
    )
    return namespace["choose"], {
        "functions": proof,
        "exact_function_AST": True,
        "compile_dont_inherit": True,
        "upstream_module_or_HTTP_client_imported": False,
        "transport": "explicit in-process callback; no HTTP qualification",
    }


def submitted_request(original, order):
    require(order in ("original", "reversed"), "predeclared option order required")
    result = copy.deepcopy(original)
    if order == "reversed":
        for question in result["questions"].values():
            question["criteria"] = dict(reversed(list(question["criteria"].items())))
    require(
        list(result["questions"]) == list(original["questions"])
        and result["state"] == original["state"]
        and all(
            result["questions"][key]["instructions"] == question["instructions"]
            and result["questions"][key]["criteria"] == question["criteria"]
            and list(result["questions"][key]["criteria"])
            == (
                list(question["criteria"])
                if order == "original"
                else list(reversed(question["criteria"]))
            )
            for key, question in original["questions"].items()
        ),
        "option reversal changed semantic values or question inventory",
    )
    return result


def semantic_diagnostic(decision, expected):
    """Self-authored diagnostic only; never modify or reject a native decision."""
    return {
        "operation": decision["operation"],
        "target": decision["target"],
        "expected_operation": expected["operation"],
        "expected_target": expected["target"],
        "matches_self_authored_predicate": decision["operation"] == expected["operation"]
        and decision["target"] == expected["target"],
        "physical_action_executed": False,
        "helper_called": False,
        "subsequent_state": "not_acted; diagnostic preserves native choice",
        "official_task_or_browser_score": False,
    }


def extract_function(source_record, name, namespace, *, class_name=None):
    tree = ast.parse(source_bytes(source_record))
    pool = tree.body
    if class_name is not None:
        classes = [
            node for node in pool if isinstance(node, ast.ClassDef) and node.name == class_name
        ]
        require(len(classes) == 1, "one original native adapter class required")
        pool = classes[0].body
    found = [node for node in pool if isinstance(node, ast.FunctionDef) and node.name == name]
    require(len(found) == 1, "one unchanged selected native function required")
    node = copy.deepcopy(found[0])
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]),
            source_record["path"],
            "exec",
            dont_inherit=True,
        ),
        namespace,
    )
    return namespace[name], {
        "line": found[0].lineno,
        "ast_sha256": hashlib.sha256(ast.dump(found[0]).encode()).hexdigest(),
        "exact_function_AST": True,
        "compile_dont_inherit": True,
    }


def load_native_predict(source_record, dto, profile, manifest):
    namespace = {
        "__builtins__": __builtins__,
        "copy": copy,
        "hashlib": hashlib,
        "time": time,
        "PROFILE": profile,
        "profile_manifest": manifest,
        **{
            name: getattr(dto, name)
            for name in (
                "_evidence",
                "parse_native_payload",
                "Decision3NativeRequest",
                "BackendTimeoutError",
                "BackendResponseError",
                "normalize_native_response",
            )
        },
    }
    return extract_function(
        source_record, "predict", namespace, class_name="JevDecision3NativeAdapter"
    )


def load_diagnostic_wire(source_record, adapter_profile, wire_profile):
    namespace = {
        "__builtins__": __builtins__,
        "copy": copy,
        "hashlib": hashlib,
        "json": __import__("json"),
        "PROFILE": wire_profile,
        "LEGACY_PROFILE": adapter_profile,
        "METADATA_KEY": "decision3_browser_wire",
    }
    proof = {}
    for name in ("_require", "_json", "named_browser_response"):
        _, proof[name] = extract_function(source_record, name, namespace)
    return SimpleNamespace(named_browser_response=namespace["named_browser_response"]), proof
