"""V7 link profile contracts, using synthetic DOM and fake model responses."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import jev_observation_profile_v7 as v7
import jev_observation_profile_v7_links as links
import pytest

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "artifacts/jev-runtime/upstream"
SOURCE_SHA = hashlib.sha256(Path(links.__file__).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def bundles():
    canonical = v7.read_canonical_sources(UPSTREAM)
    parent = v7.build_profile(canonical, expected_source_sha256=links.V7_SHA256)
    derived = links.build_profile(canonical, expected_source_sha256=SOURCE_SHA)
    return parent, derived


def test_source_closure_is_reversible_and_only_declared_files_change(bundles):
    parent, derived = bundles
    assert set(parent.runtime_files) == set(derived.runtime_files)
    assert {
        name
        for name in parent.runtime_files
        if parent.runtime_files[name] != derived.runtime_files[name]
    } == {links.MODEL, links.SNAPSHOT}
    assert derived.manifest()["inverse_bytes_equal"] is True
    assert derived.manifest()["profile"] != parent.manifest()["profile"]
    with pytest.raises(ValueError):
        links.build_profile(dict(parent.canonical_files), expected_source_sha256="0" * 64)
    changed = dict(parent.canonical_files)
    changed[links.MODEL] += b"\n"
    with pytest.raises(ValueError):
        links.build_profile(changed, expected_source_sha256=SOURCE_SHA)
    with pytest.raises(ValueError):
        links._derive_model(parent.runtime_files[links.MODEL] + b"def action_space(actions):\n")


NODE_DOM = r"""
const vm=require('vm');
const input=JSON.parse(require('fs').readFileSync(0,'utf8'));
function run(source) {
  const nodes=input.links.map((item,i)=>({
    tagName:item.tag || 'A',type:'',isConnected:true,childNodes:[],parentElement:null,
    checkVisibility:()=>true, closest:()=>null, matches:()=>false,
    hasAttribute:(k)=>k==='href' && item.href!==null,
    getAttribute:(k)=>k==='href'?item.href:(k==='aria-label'?'link '+i:null),
    getBoundingClientRect:()=>({x:5+i,y:5,width:1,height:1}),
  }));
  const body={nodeType:1,childNodes:[],hidden:false,hasAttribute:()=>false,
    matches:()=>false,getAttribute:()=>null,parentElement:null};
  const document={body,baseURI:input.base,documentElement:{scrollHeight:100},title:'synthetic',
    querySelectorAll:s=>s.startsWith('input,')?[]:nodes,
    createRange:()=>({}), getElementById:()=>null};
  const context={document,window:{},location:{href:'https://page.example/current'},
    URL,performance:{timeOrigin:42},scrollX:0,scrollY:0,innerWidth:1000,innerHeight:100,
    getComputedStyle:()=>({display:'block',visibility:'visible',opacity:'1',contentVisibility:'visible'}),
    Node:{ELEMENT_NODE:1,TEXT_NODE:3,COMMENT_NODE:8}};
  return vm.runInNewContext(source,context);
}
const old=run(input.parent), fresh=run(input.derived);
process.stdout.write(JSON.stringify({old,fresh}));
"""


def test_synthetic_snapshot_preserves_actions_and_sanitizes_observed_links(bundles):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for synthetic DOM JavaScript regression")
    parent, derived = bundles
    hrefs = [
        "../path?query=QUERY_SECRET#FRAGMENT_SECRET",
        "https://user:PASSWORD@example.test:8443/a%20b?x=QUERY_SECRET#FRAGMENT_SECRET",
        "javascript:alert(1)",
        "mailto:someone@example.test",
        "https://example.test/" + "a" * 255 + "%C3%A9" + "?QUERY_SECRET",
        "http://[invalid",
        None,
        "/plain",
    ]
    payload = {
        "links": [{"href": href} for href in hrefs] + [{"href": "/not-anchor", "tag": "BUTTON"}],
        "base": "https://base.example/dir/sub/",
        "parent": parent.read_state,
        "derived": derived.read_state,
    }
    result = subprocess.run(
        [node, "-e", NODE_DOM],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    values = json.loads(result.stdout)
    old, fresh = values["old"], values["fresh"]
    assert {k: v for k, v in old.items() if k != "observation_profile"} == {
        k: v for k, v in fresh.items() if k not in {"observation_profile", "link_destinations"}
    }
    destinations = fresh["link_destinations"]
    assert destinations["e1"] == {
        "origin": "https://base.example",
        "path": "/dir/path",
        "truncated": False,
    }
    assert destinations["e2"] == {
        "origin": "https://example.test:8443",
        "path": "/a%20b",
        "truncated": False,
    }
    assert destinations["e5"]["truncated"] is True
    assert len(destinations["e5"]["path"]) <= 256
    assert not destinations["e5"]["path"].endswith("%")
    assert destinations["e8"] == {
        "origin": "https://base.example",
        "path": "/plain",
        "truncated": False,
    }
    assert set(destinations) == {"e1", "e2", "e5", "e8"}
    encoded = json.dumps(destinations)
    for secret in ("PASSWORD", "QUERY_SECRET", "FRAGMENT_SECRET"):
        assert secret not in encoded


def test_model_only_adds_destination_information_and_preserves_choices(bundles, monkeypatch):
    parent, derived = bundles
    package = types.ModuleType("jev_ultrafast")
    package.__path__ = []
    questions = types.ModuleType("jev_ultrafast.questions")
    questions.NEXT_ACTION = "next"
    questions.TARGET = "target"
    questions.TEXT_VALUE = "text"
    monkeypatch.setitem(sys.modules, "jev_ultrafast", package)
    monkeypatch.setitem(sys.modules, "jev_ultrafast.questions", questions)
    modules = []
    for source in (parent.runtime_files[links.MODEL], derived.runtime_files[links.MODEL]):
        module = types.ModuleType("jev_ultrafast.model")
        module.__package__ = "jev_ultrafast"
        exec(compile(source, "model.py", "exec"), module.__dict__)
        modules.append(module)
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-only")
    actions = [
        {"id": "e1", "node": 1, "kind": "click", "role": "link", "label": "A", "value": ""},
        {"id": "e2", "node": 2, "kind": "click", "role": "button", "label": "B", "value": ""},
    ]
    state = {"actions": actions, "url": "https://page.example", "title": "T", "text": "body"}
    destination = {"origin": "https://example.test", "path": "/target", "truncated": False}
    captured = []
    for module in modules:

        def post(_url, _key, body):
            captured.append(body)
            return {
                "model": "fake",
                "answers": {
                    "operation": {
                        "choice": "CLICK",
                        "confidence": 1.0,
                        "probabilities": {
                            k: float(k == "CLICK")
                            for k in body["questions"]["operation"]["criteria"]
                        },
                    },
                    "click_target": {
                        "choice": "1",
                        "confidence": 1.0,
                        "probabilities": {"1": 1.0, "2": 0.0},
                    },
                },
            }

        module.post_json = post
        test_state = dict(state)
        if module is modules[1]:
            test_state["link_destinations"] = {"e1": destination}
        result = module.choose(test_state, "goal", [])
        assert result["choice"] == "e1" and result["target"] == "1"
    old, new = captured
    assert list(old["questions"]) == list(new["questions"])
    assert list(old["questions"]["click_target"]["criteria"]) == list(
        new["questions"]["click_target"]["criteria"]
    )
    assert new["questions"]["click_target"]["criteria"]["1"].pop("link_destination") == destination
    assert new["state"]["elements"][0].pop("link_destination") == destination
    assert old == new
    assert modules[0].action_space(actions) == modules[1].action_space(actions)


def test_materialized_bundle_checks_closed_inventory_and_exact_bytes(tmp_path):
    directory = tmp_path / "profile"
    created = links.materialize_profile(UPSTREAM, directory, expected_source_sha256=SOURCE_SHA)
    verified = links.verify_materialized_profile(
        directory,
        expected_manifest_sha256=created["manifest_sha256"],
        expected_source_sha256=SOURCE_SHA,
    )
    assert verified.read_state == (verified.runtime_root / links.SNAPSHOT).read_text()
    assert verified.manifest()["profile"] == links.PROFILE
    assert "sources/jev_observation_profile_v7_links.py" in verified.manifest()["artifact_files"]
    target = verified.runtime_root / links.MODEL
    target.write_bytes(target.read_bytes() + b"\n# drift\n")
    with pytest.raises(ValueError):
        links.verify_materialized_profile(
            directory,
            expected_manifest_sha256=created["manifest_sha256"],
            expected_source_sha256=SOURCE_SHA,
        )


def test_materialized_bundle_rejects_extra_file(tmp_path):
    directory = tmp_path / "profile"
    created = links.materialize_profile(UPSTREAM, directory, expected_source_sha256=SOURCE_SHA)
    (directory / "extra").write_text("unexpected")
    with pytest.raises(ValueError):
        links.verify_materialized_profile(
            directory,
            expected_manifest_sha256=created["manifest_sha256"],
            expected_source_sha256=SOURCE_SHA,
        )


def test_only_visible_retained_actions_have_destination_entries(bundles):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for synthetic DOM JavaScript regression")
    parent, derived = bundles
    payload = {
        "links": [{"href": f"/path/{index}"} for index in range(260)],
        "base": "https://base.example/",
        "parent": parent.read_state,
        "derived": derived.read_state,
    }
    result = subprocess.run(
        [node, "-e", NODE_DOM],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    snapshots = json.loads(result.stdout)
    old, fresh = snapshots["old"], snapshots["fresh"]
    assert old["actions"] == fresh["actions"]
    assert old["marker"] == fresh["marker"]
    assert fresh["omitted_actions"] == 10
    assert len(fresh["link_destinations"]) == 250
    assert set(fresh["link_destinations"]) == {f"e{i}" for i in range(1, 251)}
