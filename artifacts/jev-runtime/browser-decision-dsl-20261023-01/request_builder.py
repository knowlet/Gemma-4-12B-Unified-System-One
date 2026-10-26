"""Capture the pinned upstream request before any transport or answer exists."""

import ast
import copy
import hashlib
import pathlib
import time
import types

REPOSITORY = pathlib.Path(__file__).resolve().parents[3]
SOURCE_PINS = {
    "artifacts/jev-runtime/upstream/jev_ultrafast/model.py": "a85ada8458d23642c5c8a4095acf0565e6268f81dcda26f6308942b653e9c9f3",
    "artifacts/jev-runtime/upstream/jev_ultrafast/questions.py": "66a7054b85040b949e1757311257ea0c5a843c635f9a40c99f6d15ad660fb422",
}
OPERATIONS = ("CLICK", "TYPE_TEXT", "SELECT", "SCROLL_DOWN", "SCROLL_UP", "WAIT", "DONE", "BLOCKED")


class _Captured(BaseException):
    pass


def _source(relative):
    data = (REPOSITORY / relative).read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_PINS[relative]:
        raise ValueError("Pinned upstream source changed")
    return ast.parse(data, filename=relative)


def _exact_nodes(tree, names, kind):
    result = []
    for name in names:
        nodes = [
            n
            for n in tree.body
            if isinstance(n, kind)
            and (
                n.name == name
                if isinstance(n, ast.FunctionDef)
                else any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
            )
        ]
        if len(nodes) != 1:
            raise ValueError("Upstream extraction is not unique: " + name)
        result.extend(nodes)
    return result


def capture_request(page, goal):
    """Execute exact construction functions only; stop at their original POST seam.

    No original module import, client construction, network, model answer, browser,
    task or field-text helper is involved. This does not qualify runtime behavior.
    """
    model = _source(next(iter(SOURCE_PINS)))
    questions = _source(list(SOURCE_PINS)[1])
    captured = []

    def stop(url, key, body):
        if url != "https://api.typesafe.ai/v1/systemone" or key != "capture-only":
            raise ValueError("Unexpected original request seam")
        captured.append(copy.deepcopy(body))
        raise _Captured()

    scope = {
        "__builtins__": __builtins__,
        "os": types.SimpleNamespace(environ={"TYPESAFE_API_KEY": "capture-only"}),
        "time": time,
        "post_json": stop,
    }
    nodes = _exact_nodes(questions, ("NEXT_ACTION", "TARGET"), ast.Assign)
    nodes += _exact_nodes(model, ("action_space", "choose"), ast.FunctionDef)
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, "<exact-pinned-upstream-construction>", "exec", dont_inherit=True), scope)
    try:
        scope["choose"](copy.deepcopy(page), goal, [])
    except _Captured:
        pass
    else:
        raise ValueError("Original request did not stop before answer processing")
    if len(captured) != 1:
        raise ValueError("Exactly one original request construction required")
    request = captured[0]
    if set(request["questions"]["operation"]["criteria"]) != set(OPERATIONS):
        raise ValueError("Exact standard eight-operation contract required")
    _, targets, _ = scope["action_space"](page["actions"])
    return request, targets
