"""Pure, exact-parent source transformation for an unqualified new experiment.

Returns three adjacent runtime package modules in memory. It does not import the
runtime, materialize a browser profile, start an actor, or alter its V8 parents.
The optional information changes model input and needs fresh capacity/quality
evaluation. Original V8 and helper qualifications do not cover this experiment.
"""

from __future__ import annotations

import ast
import hashlib

PROFILE = "guarded-observation-revisit-integration-experimental-v1"
INPUT_PINS = {
    "agent.py": ("aaf8b271cb0ee6643fe7beb525328ab80de8568c7ac044d02b9c6721161e7c8b", 7880),
    "model.py": ("a85ada8458d23642c5c8a4095acf0565e6268f81dcda26f6308942b653e9c9f3", 8469),
    "observation_revisit.py": (
        "80249cdd386619b9121160c94d3a071f586cfa40d75977a7e751138175a6b5d0",
        10461,
    ),
}

WRAPPER = """    def _observe(self, *, screenshot):
        page = self.browser.observe(screenshot=screenshot)
        sequence = self._observation_sequence + 1
        metadata = self._revisit_tracker.observe(page, observation_sequence=sequence)
        self._observation_sequence = sequence
        self._observation_revisit = metadata
        return page

"""

POLICY_HINT = """    if observation_revisit is not None and observation_revisit["observation_notes"]:
        hint = {
            "profile": observation_revisit["profile"],
            "observation_sequence": observation_revisit["observation_sequence"],
            "observation_notes": copy.deepcopy(observation_revisit["observation_notes"]),
        }
        for kind in ("exact", "projection"):
            evidence = observation_revisit[kind]
            if evidence is not None and evidence["previous_observation_sequences"]:
                hint[kind] = {
                    k: copy.deepcopy(evidence[k])
                    for k in ("previous_observation_sequences", "scope")
                }
        body["state"]["observation_revisit"] = hint
"""

# Blocks are narrow literal substitutions against the exact reviewed parent.
# Reversing all blocks must recover every original byte, not just parsed code.
BLOCKS = {
    "agent.py": (
        ("copy-import", "import base64\n", "import base64\nimport copy\n", 1),
        (
            "adjacent-core-import",
            "from .questions import MAX_STEPS\n",
            "from .questions import MAX_STEPS\nfrom .observation_revisit import ObservationRevisitTracker\n",
            1,
        ),
        (
            "per-agent-tracker",
            "        self.pending_text = None\n        self.browser = Browser(url)\n",
            "        self.pending_text = None\n"
            "        self._revisit_tracker = ObservationRevisitTracker(include_projection=True)\n"
            "        self._observation_sequence = -1\n"
            "        self._observation_revisit = None\n"
            "        self.browser = Browser(url)\n",
            1,
        ),
        (
            "initial-observation",
            "page = self.browser.observe(screenshot=self.screenshots)",
            "page = self._observe(screenshot=self.screenshots)",
            1,
        ),
        (
            "observation-wrapper",
            "    def snapshot(self):\n",
            WRAPPER + "    def snapshot(self):\n",
            1,
        ),
        (
            "snapshot-owned-copy",
            '            "elements": action_space(self.state["page"]["actions"])[0],\n',
            '            "elements": action_space(self.state["page"]["actions"])[0],\n'
            '            "observation_revisit": copy.deepcopy(self._observation_revisit),\n',
            1,
        ),
        (
            "later-observations",
            'state["page"] = state["browser"].observe(screenshot=self.screenshots)',
            'state["page"] = self._observe(screenshot=self.screenshots)',
            3,
        ),
        (
            "choose-explicit-metadata",
            'choose(state["page"], state["goal"], state["history"])',
            'choose(state["page"], state["goal"], state["history"], '
            "observation_revisit=self._observation_revisit)",
            1,
        ),
    ),
    "model.py": (
        ("hint-copy-import", "import json\n", "import copy\nimport json\n", 1),
        (
            "choose-keyword",
            "def choose(state, goal, history):\n",
            "def choose(state, goal, history, *, observation_revisit=None):\n",
            1,
        ),
        (
            "state-hint-only-with-notes",
            "    started = time.perf_counter()\n"
            '    result = post_json("https://api.typesafe.ai/v1/systemone",',
            POLICY_HINT
            + "    started = time.perf_counter()\n"
            + '    result = post_json("https://api.typesafe.ai/v1/systemone",',
            1,
        ),
    ),
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _definitions(blob):
    tree = ast.parse(blob)
    definitions = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions[node.name] = ast.dump(node, include_attributes=False)
        elif isinstance(node, ast.ClassDef):
            for method in node.body:
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    definitions[node.name + "." + method.name] = ast.dump(
                        method, include_attributes=False
                    )
    return definitions


def build_runtime_sources(inputs):
    """Validate fixed inputs, return package bytes plus mechanical inverse proof."""
    _require(type(inputs) is dict and set(inputs) == set(INPUT_PINS), "Exact three inputs required")
    for name, (sha, size) in INPUT_PINS.items():
        blob = inputs[name]
        _require(
            type(blob) is bytes and len(blob) == size and hashlib.sha256(blob).hexdigest() == sha,
            "Exact parent bytes required: " + name,
        )
    outputs = dict(inputs)
    records = []
    allowed_changed = {
        "agent.py": {"Agent.__init__", "Agent.snapshot", "Agent.command"},
        "model.py": {"choose"},
    }
    for name, blocks in BLOCKS.items():
        source = inputs[name].decode("utf-8")
        for label, old, new, count in blocks:
            _require(source.count(old) == count, "Exact replacement count: " + label)
            source = source.replace(old, new)
        output = source.encode("utf-8")
        inverse = source
        for label, old, new, count in reversed(blocks):
            _require(inverse.count(new) == count, "Exact inverse count: " + label)
            inverse = inverse.replace(new, old)
        _require(inverse.encode("utf-8") == inputs[name], "Whole byte inverse differs")
        _require(
            ast.dump(ast.parse(inverse), include_attributes=False)
            == ast.dump(ast.parse(inputs[name]), include_attributes=False),
            "Whole inverse AST differs",
        )
        before, after = _definitions(inputs[name]), _definitions(output)
        changed = {key for key in before if before[key] != after.get(key)}
        added = set(after) - set(before)
        _require(changed == allowed_changed[name], "Unexpected changed definition")
        _require(
            added == ({"Agent._observe"} if name == "agent.py" else set()),
            "Unexpected added definition",
        )
        outputs[name] = output
        records.append(
            {
                "module": name,
                "blocks": [{"label": label, "occurrences": count} for label, _, _, count in blocks],
                "byte_inverse_exact": True,
                "whole_inverse_AST_equal": True,
                "changed_definitions": sorted(changed),
                "added_definitions": sorted(added),
                "unchanged_definitions": sorted(set(before) - changed),
            }
        )
    return outputs, {
        "profile": PROFILE,
        "package_modules": {
            name: {"sha256": hashlib.sha256(blob).hexdigest(), "size_bytes": len(blob)}
            for name, blob in outputs.items()
        },
        "core_bytes_unchanged": outputs["observation_revisit.py"]
        == inputs["observation_revisit.py"],
        "transformations": records,
        "runtime_imported": False,
        "materialized_profile": False,
        "capacity_or_browser_qualified": False,
    }
