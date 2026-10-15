"""Pure V3 source/materialization contracts; no browser, model or external requests."""

import hashlib
import inspect
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from s1 import jev_observation_profile as p

HELPER_SHA = hashlib.sha256(Path(p.__file__).read_bytes()).hexdigest()
V2_SHA = "5327d4ab5d615ac0804f4913bb86bb59b383b11ff80f2408eb4aa565243f9dd5"
V3_SHA = "c694f308f29ed5000b7bb60e9374afd9398413d2811e07741cf0f0a6321be552"
V2_DERIVATION = {
    "parent_builder_sha256": "b9739ba962d41bdb20831196ca8b217608e369477d433b39672b17e47abcd46b",
    "parent_v1_snapshot_sha256": "3a4a5e7532a7eb99334b8fd71e181d5d18183af4966757cd42d48e6404cbb197",
    "changes": [
        "generic corrected composed walker retained",
        "profile ID changed to composed-visible-text-v2",
        "bounded structured limit error before snapshot return",
    ],
    "task_specific_rules": False,
}
V2_LIMITS = {
    "text_characters": 6000,
    "visited_nodes": 20000,
    "composed_depth": 128,
    "limit_error_name": "ComposedVisibleTextLimitError",
    "limit_error_prefix": "COMPOSED_VISIBLE_TEXT_LIMIT ",
    "node_or_depth_cap": "throw inside READ_STATE before return; embedded MARKER propagates same error",
}


@pytest.fixture
def canonical():
    configured = os.environ.get("JEV_UPSTREAM_PATH")
    if not configured:
        pytest.skip("set JEV_UPSTREAM_PATH to an existing pinned checkout; never fetches")
    return p.read_canonical_sources(configured)


def build(canonical, profile=p.PRUNED_PROFILE):
    return p.build_profile(canonical, expected_source_sha256=HELPER_SHA, profile=profile)


def materialized(tmp_path, canonical):
    origin = tmp_path / "canonical"
    for label, blob in canonical.items():
        path = origin / label
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    target = tmp_path / "profile"
    result = p.materialize_profile(
        origin, target, expected_source_sha256=HELPER_SHA, profile=p.PRUNED_PROFILE
    )
    return target, result


def verify(target, digest, profile=p.PRUNED_PROFILE):
    return p.verify_materialized_profile(
        target,
        expected_manifest_sha256=digest,
        expected_source_sha256=HELPER_SHA,
        profile=profile,
    )


def test_v3_is_explicit_and_defaults_remain_native():
    assert p.PRUNED_PROFILE == "composed-visible-text-v3"
    assert p.COMPOSED_PROFILE == "composed-visible-text-v2"
    assert len({p.NATIVE_PROFILE, p.COMPOSED_PROFILE, p.PRUNED_PROFILE}) == 3
    for function in (p.build_profile, p.materialize_profile, p.verify_materialized_profile):
        assert inspect.signature(function).parameters["profile"].default == p.NATIVE_PROFILE


@pytest.mark.parametrize(
    "alias",
    [
        "composed-v3",
        "composed-visible-text-v3 ",
        "composed-visible-text-v3-relevant-node-pruning-h02-prototype",
    ],
)
def test_aliases_and_prototype_id_are_not_public_profiles(alias):
    with pytest.raises(ValueError, match="Unknown explicit"):
        p.build_profile({}, expected_source_sha256=HELPER_SHA, profile=alias)


def test_reviewed_v3_snapshot_and_preserved_v2_seven_source_contract(canonical):
    native, v2, v3 = (
        build(canonical, profile)
        for profile in (p.NATIVE_PROFILE, p.COMPOSED_PROFILE, p.PRUNED_PROFILE)
    )
    assert dict(native.runtime_files) == canonical
    assert (
        hashlib.sha256(v2.runtime_files[p.SNAPSHOT]).hexdigest() == p.V2_SNAPSHOT_SHA256 == V2_SHA
    )
    assert (
        hashlib.sha256(v3.runtime_files[p.SNAPSHOT]).hexdigest()
        == p.PRUNED_SNAPSHOT_SHA256
        == V3_SHA
    )
    assert len(v3.runtime_files[p.SNAPSHOT]) == 8869
    assert [key for key in canonical if canonical[key] != v3.runtime_files[key]] == [p.SNAPSHOT]
    assert v3.read_state.count("id:'composed-visible-text-v3'") == 1


def test_canonical_action_result_and_nonempty_text_code_are_exact(canonical):
    v2, v3 = build(canonical, p.COMPOSED_PROFILE).read_state, build(canonical).read_state
    begin, end = "  const words=[], range=", "  const text=words.join"
    assert v2[: v2.index(begin)] == v3[: v3.index(begin)]
    assert v2[v2.index(end) :] == v3[v3.index(end) :]
    text_begin = "    if (node.nodeType===Node.TEXT_NODE) {"
    text_end = "    if (node.nodeType===Node.ELEMENT_NODE &&\n        node.matches"
    assert (
        v2[v2.index(text_begin) : v2.index(text_end)]
        == v3[v3.index(text_begin) : v3.index(text_end)]
    )
    child_filter = "    children=children.filter(child=>child.nodeType!==Node.COMMENT_NODE &&\n      !(child.nodeType===Node.TEXT_NODE && !child.textContent.trim()));\n"
    assert v3.count(child_filter) == 1
    assert (
        v3.index("const assigned=node.assignedNodes({flatten:true})")
        < v3.index(child_filter)
        < v3.index("    for (let i=children.length-1;")
    )
    assert v3.index("depth>textProfile.depth_limit") < v3.index("excludedElement(node)")


def test_v2_manifest_derivation_and_limits_are_unchanged(canonical):
    manifest = build(canonical, p.COMPOSED_PROFILE).manifest()
    assert manifest["derivation"] == V2_DERIVATION
    assert manifest["limits"] == V2_LIMITS
    assert len(manifest) == 15


def test_v3_manifest_describes_relevant_visits_and_preserves_limits(canonical):
    manifest = build(canonical).manifest()
    assert manifest["profile"] == p.PRUNED_PROFILE
    assert manifest["derivation"]["parent_profile"] == p.COMPOSED_PROFILE
    assert manifest["derivation"]["parent_snapshot_sha256"] == V2_SHA
    assert manifest["derivation"]["task_specific_rules"] is False
    limits = manifest["limits"]
    assert {key: limits[key] for key in V2_LIMITS} == V2_LIMITS
    assert "COMMENT and exact trim-empty TEXT" in limits["visited_nodes_semantics"]
    assert "excluded element itself still consumes a visit" in limits["visited_nodes_semantics"]
    assert "not a hard bound" in limits["not_bounded_by_visited_nodes"]
    assert set(manifest) == set(build(canonical, p.COMPOSED_PROFILE).manifest())


def test_v3_complete_materialization_and_verification_use_installed_helper(tmp_path, canonical):
    target, result = materialized(tmp_path, canonical)
    handle = verify(target, result["manifest_sha256"])
    assert handle.bundle.runtime_files[p.SNAPSHOT] == build(canonical).runtime_files[p.SNAPSHOT]
    assert handle.runtime_root == target / "runtime"
    manifest = handle.manifest()
    assert manifest["helper_source_sha256"] == HELPER_SHA
    assert len(manifest["artifact_files"]) == 15 and len(manifest) == 16
    assert (target / manifest["helper_copy"]).read_bytes() == Path(p.__file__).read_bytes()
    assert len([path for path in target.rglob("*") if path.is_file()]) == 16
    assert (
        handle.marker_expression
        == f"(() => {{ const state={handle.read_state}; return state?.marker ?? null; }})()"
    )


@pytest.mark.parametrize("profile", [p.NATIVE_PROFILE, p.COMPOSED_PROFILE])
def test_v3_materialized_tree_cannot_be_claimed_as_another_profile(tmp_path, canonical, profile):
    target, result = materialized(tmp_path, canonical)
    with pytest.raises(ValueError, match="metadata/profile differs"):
        verify(target, result["manifest_sha256"], profile)


@pytest.mark.parametrize("mutation", ["cap", "count_semantics"])
def test_rehashed_v3_manifest_cannot_weaken_or_omit_count_contract(tmp_path, canonical, mutation):
    target, _ = materialized(tmp_path, canonical)
    path = target / "profile-manifest.json"
    manifest = json.loads(path.read_bytes())
    if mutation == "cap":
        manifest["limits"]["visited_nodes"] = 20001
    else:
        del manifest["limits"]["visited_nodes_semantics"]
    blob = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path.write_bytes(blob)
    with pytest.raises(ValueError, match="metadata/profile differs"):
        verify(target, hashlib.sha256(blob).hexdigest())


def test_actual_v3_read_state_and_marker_parse_without_execution(canonical):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node absent; no syntax check claimed")
    bundle = build(canonical)
    code = "const vm=require('vm');for(const text of JSON.parse(require('fs').readFileSync(0,'utf8')))new vm.Script(text);"
    subprocess.run(
        [node, "-e", code],
        input=json.dumps([bundle.read_state, bundle.marker_expression]),
        text=True,
        capture_output=True,
        check=True,
        timeout=5,
    )


def test_helper_cli_forwards_explicit_v3_without_reading_sources(monkeypatch, capsys):
    calls = []

    def materialize(root, output, **kwargs):
        calls.append((root, output, kwargs))
        return {"profile": kwargs["profile"]}

    monkeypatch.setattr(p, "materialize_profile", materialize)
    assert (
        p.main(
            [
                "--canonical-root",
                "/tmp/canonical",
                "--output",
                "/tmp/new-v3",
                "--source-sha256",
                "a" * 64,
                "--profile",
                p.PRUNED_PROFILE,
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["profile"] == p.PRUNED_PROFILE
    assert calls[-1][2] == {"expected_source_sha256": "a" * 64, "profile": p.PRUNED_PROFILE}
