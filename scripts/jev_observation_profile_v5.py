"""Explicit bounded CSS-hover entry discovery on the immutable v4 runtime.

Only a current visible text leaf can become a new CLICK candidate. CSSOM proof
must identify its roleless owner and an interactive popup; no goal, site name or
hidden menu label participates. The inherited v4 executor is byte-identical.
Its hover-induced down/up target changes remain an explicitly known limitation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType

PROFILE = "composed-visible-hover-entries-v5"
PARENT_SOURCE_SHA256 = "f96fb0553c4c8ef1a163633577bfb3ddbb03ddbc1ff6644c36d8fbe2c06d4b43"
PARENT_COPY = "jev_observation_profile_v4.py"
MAX_SOURCE_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 16 * MAX_SOURCE_BYTES
MAX_FILES = 32


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def _encode(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _ordinary_path(path):
    path = Path(path).absolute()
    _require(
        ".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
        "Ordinary canonical paths required",
    )
    return path


def _parent_source():
    path = _ordinary_path(Path(__file__).with_name(PARENT_COPY))
    blob = path.read_bytes()
    _require(
        0 < len(blob) <= MAX_SOURCE_BYTES and _sha(blob) == PARENT_SOURCE_SHA256,
        "Pinned v4 parent source required",
    )
    return path, blob


def _load_parent():
    path, _ = _parent_source()
    name = "_jev_v5_parent_f96fb0553c4c"
    if name in sys.modules:
        module = sys.modules[name]
        _require(
            _sha(Path(module.__file__).read_bytes()) == PARENT_SOURCE_SHA256,
            "Loaded v4 parent changed",
        )
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


_PARENT = _load_parent()
CANONICAL_SHA256 = _PARENT.CANONICAL_SHA256
CANONICAL_REVISION = _PARENT.CANONICAL_REVISION
SNAPSHOT, BROWSER = _PARENT.SNAPSHOT, _PARENT.BROWSER


def _source_pin(expected):
    _require(
        type(expected) is str and re.fullmatch(r"[0-9a-f]{64}", expected),
        "Explicit reviewed v5 helper SHA256 required",
    )
    blob = _ordinary_path(__file__).read_bytes()
    _require(
        0 < len(blob) <= MAX_SOURCE_BYTES and _sha(blob) == expected,
        "Reviewed v5 helper source changed",
    )
    _parent_source()
    return blob


def read_canonical_sources(root):
    return _PARENT.read_canonical_sources(root)


HOVER_DISCOVERY = r"""  // Read actual loaded CSSOM; observation never moves the pointer or opens menus.
  const hoverReport={profile:'css-hover-entry-v5',stylesheets:0,rules:0,hover_selectors:0,
    unsupported_selectors:0,unsupported_groups:0,unreadable_stylesheets:0,pending_stylesheets:0,
    owners:0,ambiguous_owners:0,no_stable_trigger:0,duplicate_bindings:0,actions:0,coverage_complete:true,
    synchronous_only:true,automatic_hover:false,additional_candidate_pruning:false,
    extension_status:'admitted_supported_scope',candidate_batch_committed:true,
    extension_refusal:null,unscanned_hover_candidates:null};
  const hoverLimits={stylesheets:16,rules:512,selector_chars:2048,selectors_per_rule:32,
    hover_selectors:128,declaration_chars:2048,proof_chars:32768,
    depth:4,owners:64,descendants_per_owner:512,label_chars:80,label_raw_chars:256};
  class HoverBudgetError extends Error {
    constructor(stage,scanned,maximum){super('CSS_HOVER_DISCOVERY_LIMIT '+stage);
      this.name='CssHoverDiscoveryLimitError';this.stage=stage;this.scanned=scanned;this.maximum=maximum;}
  }
  const hoverFail=(stage,scanned,maximum)=>{throw new HoverBudgetError(stage,scanned,maximum);};
  cache.hoverBindings=new WeakMap();cache.hoverEpoch=cache.hoverEpoch||0;
  const refuseHoverBudget=(error,report)=>{
    if(!(error instanceof HoverBudgetError))throw error;
    Object.assign(report,{coverage_complete:false,extension_status:'refused_budget',
      candidate_batch_committed:false,actions:0,unscanned_hover_candidates:'unknown',
      extension_refusal:{kind:'css_discovery_budget',stage:error.stage,scanned:error.scanned,maximum:error.maximum}});
    cache.hoverBindings=new WeakMap();cache.hoverEpoch++;
  };
  const compound=/^(?:[a-zA-Z][a-zA-Z0-9_-]*|\*)?(?:[.#][a-zA-Z_][a-zA-Z0-9_-]*)*$/;
  const simple=value=>!!value && compound.test(value);
  const shows=style=>{const d=style.getPropertyValue('display').trim(),
    v=style.getPropertyValue('visibility').trim();
    return (!!d && !['none','inherit','unset','revert','revert-layer'].includes(d)) || v==='visible';};
  const readHoverRules=report=>{
    const found=[], seen=new Set();let sheets=0,rules=0,proofChars=0;
    const visit=(sheet,depth)=>{
      if(seen.has(sheet))return;seen.add(sheet);
      report.stylesheets=++sheets;
      if(sheets>hoverLimits.stylesheets || depth>hoverLimits.depth)hoverFail('stylesheets/depth',
        {stylesheets:sheets,depth},{stylesheets:hoverLimits.stylesheets,depth:hoverLimits.depth});
      if(sheet.disabled)return;
      if(sheet.media?.mediaText && !matchMedia(sheet.media.mediaText).matches)return;
      let values;try{values=sheet.cssRules}catch(error){if(error.name!=='SecurityError')throw error;
        report.unreadable_stylesheets++;return;}
      const walk=(items,level)=>{
        if(level>hoverLimits.depth)hoverFail('rule-depth',level,hoverLimits.depth);
        for(const rule of items){
          report.rules=++rules;if(rules>hoverLimits.rules)hoverFail('rules',rules,hoverLimits.rules);
          if(rule.type===3){if(rule.styleSheet)visit(rule.styleSheet,level+1);else report.pending_stylesheets++;continue;}
          if(rule.type===4){if(matchMedia(rule.conditionText).matches)walk(rule.cssRules,level+1);continue;}
          if(rule.type===12){if(CSS.supports(rule.conditionText))walk(rule.cssRules,level+1);continue;}
          if(rule.type!==1){if(rule.cssRules?.length)report.unsupported_groups++;continue;}
          const text=rule.selectorText;
          if(text.length>hoverLimits.selector_chars)hoverFail('selector-chars',text.length,hoverLimits.selector_chars);
          if(!text.includes(':hover'))continue;
          const parts=text.split(',');if(parts.length>hoverLimits.selectors_per_rule)hoverFail('selector-list',parts.length,hoverLimits.selectors_per_rule);
          for(const raw of parts){const value=raw.trim();if(!value.includes(':hover'))continue;
            if(++report.hover_selectors>hoverLimits.hover_selectors)hoverFail('hover-selectors',report.hover_selectors,hoverLimits.hover_selectors);
            const at=value.indexOf(':hover'), owner=value.slice(0,at).trim(), tail=value.slice(at+6);
            const suffix=tail.trim().replace(/^>\s*/,'');
            if(value.indexOf(':hover',at+1)!==-1 || !simple(owner) || !/^(\s+|\s*>\s*)/.test(tail) ||
                !simple(suffix) || !shows(rule.style)) {report.unsupported_selectors++;continue;}
            const declaration=rule.style.cssText;
            if(declaration.length>hoverLimits.declaration_chars)hoverFail('declaration-chars',declaration.length,hoverLimits.declaration_chars);
            proofChars+=value.length+declaration.length;
            report.proof_chars=proofChars;if(proofChars>hoverLimits.proof_chars)hoverFail('proof-chars',proofChars,hoverLimits.proof_chars);
            found.push({owner,child:suffix,direct:tail.trim().startsWith('>'),proof:[value,declaration]});
          }
        }
      };walk(values,depth);
    };
    for(const link of document.querySelectorAll('link[rel~="stylesheet"]'))if(!link.disabled && !link.sheet)report.pending_stylesheets++;
    for(const sheet of document.styleSheets)visit(sheet,0);
    report.stylesheets=sheets;report.rules=rules;report.proof_chars=proofChars;
    report.coverage_complete=!(report.unsupported_selectors || report.unsupported_groups ||
      report.unreadable_stylesheets || report.pending_stylesheets);
    return found;
  };
  const ownerDescriptions=(rules,report)=>{
    const owners=new Map();
    for(const rule of rules)for(const owner of document.querySelectorAll(rule.owner)){
      let item=owners.get(owner);if(!item){if(owners.size>=hoverLimits.owners)hoverFail('owners',owners.size+1,hoverLimits.owners);
        item={owner,popups:new Set(),proof:[]};owners.set(owner,item);report.owners=owners.size;}
      const nodes=owner.querySelectorAll('*');if(nodes.length>hoverLimits.descendants_per_owner)hoverFail('descendants',nodes.length,hoverLimits.descendants_per_owner);
      let matched=false;
      for(const node of nodes)if(node.matches(rule.child) && (!rule.direct || node.parentElement===owner) &&
          (node.matches(selector) || node.querySelector(selector))){item.popups.add(node);matched=true;}
      if(matched)item.proof.push(rule.proof);
    }
    report.owners=owners.size;return owners;
  };
  const triggerFor=(entry,report)=>{
    const {owner,popups}=entry;
    if(!owner.isConnected || !visible(owner) || !popups.size || owner.matches(selector) ||
        owner.closest('[aria-disabled="true"],[inert]'))return null;
    const nodes=owner.querySelectorAll('*');if(nodes.length>hoverLimits.descendants_per_owner)hoverFail('trigger-descendants',nodes.length,hoverLimits.descendants_per_owner);
    const possible=[];
    for(const node of nodes){
      if(node.children.length || node.tagName==='LABEL' || node.closest(selector) ||
          [...popups].some(popup=>popup.contains(node)) || !hitTarget(node))continue;
      const parts=[];let rawLength=0;
      for(const text of node.childNodes)if(text.nodeType===Node.TEXT_NODE){rawLength+=text.length;
        if(rawLength>hoverLimits.label_raw_chars)break;parts.push(text.textContent);}
      if(rawLength>hoverLimits.label_raw_chars)continue;
      const label=parts.join(' ').replace(/\s+/g,' ').trim();
      if(!label || label.length>hoverLimits.label_chars)continue;
      possible.push({node,label});
    }
    if(possible.length!==1){if(possible.length>1)report.ambiguous_owners++;else report.no_stable_trigger++;return null;}
    return possible[0];
  };
  let firstOwners=new Map(),hoverBuilding=true;
  cache.hoverState=node=>{
    const stored=cache.hoverBindings.get(node);if(!stored)return null;
    const report={stylesheets:0,rules:0,hover_selectors:0,unsupported_selectors:0,unsupported_groups:0,
      unreadable_stylesheets:0,pending_stylesheets:0,owners:0,ambiguous_owners:0,no_stable_trigger:0};
    try{
      const owners=hoverBuilding?firstOwners:ownerDescriptions(readHoverRules(report),report);
      const current=owners.get(stored.owner),trigger=current && triggerFor(current,report);
      if(!trigger || trigger.node!==node || trigger.label!==stored.label)return null;
      return [identity(node),'css-hover-entry-v5',trigger.label,identity(current.owner),
        current.proof,[...current.popups].map(identity),cache.hoverEpoch];
    }catch(error){refuseHoverBudget(error,report);return null;}
  };
  const inheritedHoverGuard=cache.guard;
  cache.guard=node=>cache.hoverBindings.has(node)?cache.hoverState(node):inheritedHoverGuard(node);
  let hoverActions=[];
  try{
    firstOwners=ownerDescriptions(readHoverRules(hoverReport),hoverReport);
    for(const entry of firstOwners.values()){
      const trigger=triggerFor(entry,hoverReport);if(!trigger)continue;
      if(observedActions.some(action=>cache.nodes.get(action.node)===trigger.node))continue;
      if(cache.hoverBindings.has(trigger.node)){hoverReport.duplicate_bindings++;continue;}
      cache.hoverBindings.set(trigger.node,{owner:entry.owner,label:trigger.label});
      const r=trigger.node.getBoundingClientRect();
      hoverActions.push({node:identity(trigger.node),role:'generic',label:trigger.label,
        rect:{x:r.x,y:r.y,w:r.width,h:r.height},kind:'click',value:'',
        affordance:'css-hover-entry-v5',hover_owner:identity(entry.owner)});
    }
  }catch(error){refuseHoverBudget(error,hoverReport);hoverActions=[];firstOwners=new Map();}
  hoverReport.actions=hoverActions.length;
  if(hoverReport.extension_status!=='refused_budget')hoverReport.unscanned_hover_candidates=hoverReport.coverage_complete?0:'unknown';
"""


def _derive_snapshot(v4):
    source = v4.decode()
    source = _PARENT._replace_once(
        source, "  const words=[], range=", HOVER_DISCOVERY + "  const words=[], range="
    )
    source = _PARENT._replace_once(
        source,
        "const observedActions=[...actions,...proxyActions];",
        "let observedActions=[...actions,...proxyActions];",
    )
    source = _PARENT._replace_once(
        source,
        "  const words=[], range=",
        "  observedActions=[...observedActions,...hoverActions];\n  const words=[], range=",
    )
    source = _PARENT._replace_once(
        source,
        "  actions.push(...proxyActions);",
        "  actions.push(...proxyActions,...hoverActions);",
    )
    source = _PARENT._replace_once(source, "id:'" + _PARENT.PROFILE + "'", "id:'" + PROFILE + "'")
    source = _PARENT._replace_once(
        source,
        "  return {observation_profile:textProfile",
        "  hoverBuilding=false;\n  return {observation_profile:textProfile",
    )
    source = _PARENT._replace_once(
        source,
        "label_proxy_actions:proxyActions.length};",
        "label_proxy_actions:proxyActions.length,hover_entry_actions:hoverActions.length,hover_discovery:hoverReport};",
    )
    return source.encode()


@dataclass(frozen=True)
class ProfileBundle:
    canonical_files: object
    runtime_files: object
    _manifest_bytes: bytes
    _helper_source: bytes
    _parent_bundle: object
    _parent_v4_source: bytes

    def manifest(self):
        return json.loads(self._manifest_bytes)

    def _materialized_sources(self):
        parent = self._parent_bundle
        files = {"parent/" + k: v for k, v in parent._materialized_sources().items()}
        files["parent/profile-manifest.json"] = (
            _PARENT._encode(parent.materialized_manifest()) + b"\n"
        )
        files.update({"runtime/" + k: v for k, v in self.runtime_files.items()})
        files["sources/jev_observation_profile_v5.py"] = self._helper_source
        files["sources/jev_observation_profile_v4.py"] = self._parent_v4_source
        files["sources/" + _PARENT.PARENT_COPY] = parent._parent_source
        return files

    def materialized_manifest(self):
        return {
            **self.manifest(),
            "artifact_files": {
                label: {"sha256": _sha(blob), "size_bytes": len(blob)}
                for label, blob in sorted(self._materialized_sources().items())
            },
        }

    @property
    def read_state(self):
        return self.runtime_files[SNAPSHOT].decode()

    @property
    def marker_expression(self):
        return f"(() => {{ const state={self.read_state}; return state?.marker ?? null; }})()"


VerifiedMaterializedProfile = _PARENT.VerifiedMaterializedProfile


def build_profile(canonical_sources, *, expected_source_sha256, profile=PROFILE):
    _require(type(profile) is str and profile == PROFILE, "Explicit v5 profile required")
    helper = _source_pin(expected_source_sha256)
    _, parent_source = _parent_source()
    old = _PARENT.build_profile(
        canonical_sources, expected_source_sha256=PARENT_SOURCE_SHA256, profile=_PARENT.PROFILE
    )
    canonical, runtime = dict(old.canonical_files), dict(old.runtime_files)
    runtime[SNAPSHOT] = _derive_snapshot(runtime[SNAPSHOT])
    _require(
        all(runtime[k] == old.runtime_files[k] for k in runtime if k != SNAPSHOT),
        "Only v5 snapshot may change",
    )
    manifest = {
        "schema": "explicit-observation-runtime-profile-v5",
        "profile": PROFILE,
        "canonical_contract_revision": CANONICAL_REVISION,
        "canonical_Git_verification_performed_by_this_library": False,
        "canonical_Git_prerequisite": "Caller must retain strict benchmark upstream_manifest checks.",
        "canonical_source_sha256": dict(CANONICAL_SHA256),
        "runtime_source_sha256": {k: _sha(v) for k, v in runtime.items()},
        "helper_source_sha256": _sha(helper),
        "parent_helper_source_sha256": PARENT_SOURCE_SHA256,
        "runtime_root": "runtime",
        "canonical_copy_root": "parent/canonical",
        "helper_copy": "sources/jev_observation_profile_v5.py",
        "derivation": {
            "parent_profile": _PARENT.PROFILE,
            "parent_snapshot_sha256": _sha(old.runtime_files[SNAPSHOT]),
            "changes": [
                "append stable visible roleless CSS-hover entry leaves",
                "bind trigger/owner/current CSS proof in freshness guards",
            ],
            "normal_target_order_and_ids_preserved": True,
            "additional_candidate_pruning": False,
            "executor_bytes_equal_to_v4": True,
            "operation_heads_unchanged": True,
            "task_specific_rules": False,
        },
        "limits": {
            **old.manifest()["limits"],
            "hover_stylesheets": 16,
            "hover_css_rules": 512,
            "hover_css_depth": 4,
            "hover_owner_limit": 64,
            "hover_descendants_per_owner": 512,
            "hover_label_characters": 80,
            "hover_raw_label_characters": 256,
            "hover_selector_limit": 128,
            "hover_declaration_characters": 2048,
            "hover_total_proof_characters": 32768,
            "hover_limit_behavior": "known exploration budget exhaustion atomically refuses the whole hover extension; native/v4 candidates continue unchanged",
            "hover_budget_freshness": "clear bindings and invalidate prior hover guard epoch; native guards unchanged",
            "hover_coverage": "simple compound :hover owner with one compound descendant/child; readable active CSSOM only",
            "hover_waits": "none; synchronous observation does not attest delayed reactions",
        },
        "scope": "Interface-exposed diagnostic runtime repair; not blind task evaluation or training qualification.",
        "known_inherited_limitations": [
            "hover can split down/up targets or reveal an overlay; executor unchanged",
            "unsupported selectors/unreadable/pending stylesheets produce explicit incomplete coverage",
        ],
        "not_attested_by_this_library": [
            "browser or model execution",
            "serving qualification",
            "task performance",
            "complete CSS coverage",
        ],
    }
    return ProfileBundle(
        MappingProxyType(canonical),
        MappingProxyType(runtime),
        _encode(manifest),
        helper,
        old,
        parent_source,
    )


def _materialized_labels():
    return {
        *("parent/" + k for k in _PARENT._materialized_labels()),
        *("runtime/" + k for k in CANONICAL_SHA256),
        "sources/jev_observation_profile_v5.py",
        "sources/jev_observation_profile_v4.py",
        "sources/" + _PARENT.PARENT_COPY,
        "profile-manifest.json",
    }


def _write_exclusive(path, blob):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())


def materialize_profile(canonical_root, output, *, expected_source_sha256, profile=PROFILE):
    canonical_root, output = _ordinary_path(canonical_root), _ordinary_path(output)
    _require(
        not output.exists()
        and not output.is_relative_to(canonical_root)
        and not canonical_root.is_relative_to(output),
        "New disjoint profile output required",
    )
    bundle = build_profile(
        read_canonical_sources(canonical_root),
        expected_source_sha256=expected_source_sha256,
        profile=profile,
    )
    output.mkdir(parents=True, exist_ok=False)
    files = bundle._materialized_sources()
    for label, blob in sorted(files.items()):
        _write_exclusive(output / label, blob)
    final = _encode(bundle.materialized_manifest()) + b"\n"
    _write_exclusive(output / ".profile-manifest.pending", final)
    captured, first_files, first_dirs = _capture_tree(
        output, read_contents=True, manifest_name=".profile-manifest.pending"
    )
    _require(
        captured[".profile-manifest.pending"] == final
        and all(captured[k] == v for k, v in files.items()),
        "Prepared profile bytes differ",
    )
    for directory in sorted(
        [output, *(p for p in output.rglob("*") if p.is_dir())],
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    _require(
        _source_pin(expected_source_sha256) == bundle._helper_source,
        "Helper changed during materialization",
    )
    _, last_files, last_dirs = _capture_tree(
        output, read_contents=False, manifest_name=".profile-manifest.pending"
    )
    _require(
        first_files == last_files and first_dirs == last_dirs, "Prepared profile population changed"
    )
    os.replace(output / ".profile-manifest.pending", output / "profile-manifest.json")
    return {
        "output": str(output),
        "profile": PROFILE,
        "manifest_sha256": _sha(final),
        "manifest_size_bytes": len(final),
        "runtime_root": str(output / "runtime"),
        "browser_model_calls": 0,
    }


def _stat_identity(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _capture_tree(root, *, read_contents, manifest_name="profile-manifest.json"):
    labels = _materialized_labels() - {"profile-manifest.json"} | {manifest_name}
    expected_dirs = {
        str(p) for label in labels for p in PurePosixPath(label).parents if str(p) != "."
    }
    files, directories, blobs, total = {}, {}, {}, 0

    def visit(label, descriptor):
        nonlocal total
        before = os.fstat(descriptor)
        _require(stat.S_ISDIR(before.st_mode), "Ordinary profile directory required")
        directories[label] = _stat_identity(before)
        with os.scandir(descriptor) as entries:
            for entry in entries:
                relative = label + "/" + entry.name if label else entry.name
                observed = entry.stat(follow_symlinks=False)
                _require(not stat.S_ISLNK(observed.st_mode), "Profile symlink refused")
                if stat.S_ISDIR(observed.st_mode):
                    _require(relative in expected_dirs, "Unknown profile directory")
                    child = os.open(
                        entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
                    )
                    try:
                        _require(
                            _stat_identity(os.fstat(child)) == _stat_identity(observed),
                            "Directory changed before capture",
                        )
                        visit(relative, child)
                    finally:
                        os.close(child)
                    continue
                _require(
                    stat.S_ISREG(observed.st_mode) and relative in labels,
                    "Unknown or nonordinary profile file",
                )
                _require(0 < observed.st_size <= MAX_SOURCE_BYTES, "Profile file exceeds bound")
                files[relative] = _stat_identity(observed)
                total += observed.st_size
                _require(
                    len(files) <= MAX_FILES and total <= MAX_TOTAL_BYTES,
                    "Complete profile exceeds bound",
                )
                if read_contents:
                    child = os.open(
                        entry.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor
                    )
                    with os.fdopen(child, "rb") as stream:
                        opened = os.fstat(stream.fileno())
                        _require(
                            stat.S_ISREG(opened.st_mode)
                            and _stat_identity(opened) == files[relative],
                            "File changed before capture",
                        )
                        blob = stream.read(MAX_SOURCE_BYTES + 1)
                        _require(
                            len(blob) == opened.st_size
                            and _stat_identity(os.fstat(stream.fileno())) == files[relative],
                            "File changed during capture",
                        )
                        blobs[relative] = blob
        _require(
            _stat_identity(os.fstat(descriptor)) == directories[label],
            "Directory changed during capture",
        )

    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        visit("", descriptor)
        _require(
            _stat_identity(root.stat(follow_symlinks=False)) == directories[""],
            "Profile root changed",
        )
    finally:
        os.close(descriptor)
    _require(
        set(files) == labels and set(directories) == {"", *expected_dirs},
        "Complete profile population differs",
    )
    return blobs, files, directories


def verify_materialized_profile(
    path, *, expected_manifest_sha256, expected_source_sha256, profile=PROFILE
):
    helper = _source_pin(expected_source_sha256)
    _require(
        type(expected_manifest_sha256) is str
        and re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256),
        "Explicit manifest SHA required",
    )
    _require(type(profile) is str and profile == PROFILE, "Explicit v5 profile required")
    root = _ordinary_path(path)
    _require(root.is_dir(), "Materialized profile directory required")
    blobs, first_files, first_dirs = _capture_tree(root, read_contents=True)
    manifest_blob = blobs["profile-manifest.json"]
    _require(_sha(manifest_blob) == expected_manifest_sha256, "Materialized manifest hash differs")
    canonical = {label: blobs["parent/canonical/" + label] for label in CANONICAL_SHA256}
    bundle = build_profile(
        canonical, expected_source_sha256=expected_source_sha256, profile=profile
    )
    _require(
        bundle._helper_source == helper
        and manifest_blob == _encode(bundle.materialized_manifest()) + b"\n",
        "Materialized source/metadata differs",
    )
    _require(
        all(blobs[label] == blob for label, blob in bundle._materialized_sources().items()),
        "Materialized canonical/runtime/helper source differs",
    )
    _, final_files, final_dirs = _capture_tree(root, read_contents=False)
    _require(
        first_files == final_files and first_dirs == final_dirs, "Profile closed inventory changed"
    )
    _require(
        _source_pin(expected_source_sha256) == helper, "Reviewed helper changed during verification"
    )
    return VerifiedMaterializedProfile(root, bundle, _sha(manifest_blob), manifest_blob)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--profile", choices=[PROFILE], default=PROFILE)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            materialize_profile(
                args.canonical_root,
                args.output,
                expected_source_sha256=args.source_sha256,
                profile=args.profile,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
