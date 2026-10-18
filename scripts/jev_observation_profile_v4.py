"""An independent, authenticated Jev runtime for associated control labels.

The pinned v3 builder remains unchanged. This module derives a new snapshot and
browser runtime, preserving original native targets and their existing cap;
additional label targets are appended without further omission. No browser,
model, network client or collector is imported by this source-only builder.
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

PROFILE = "composed-visible-controls-v4"
PARENT_SOURCE_SHA256 = "8d199ac1564af05892e9b21e225d1fdf500f1953840855c902160ed8d56df89d"
PARENT_COPY = "jev_observation_profile_v3_parent.py"
SNAPSHOT = "jev_ultrafast/snapshot.js"
BROWSER = "jev_ultrafast/browser.py"
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


def _parent_path():
    adjacent = Path(__file__).absolute().with_name(PARENT_COPY)
    if adjacent.exists() or adjacent.is_symlink():
        return _ordinary_path(adjacent)
    return _ordinary_path(
        Path(__file__).absolute().parents[1] / "src/s1/jev_observation_profile.py"
    )


def _parent_source():
    path = _parent_path()
    _require(
        path.is_file() and path.stat().st_size <= MAX_SOURCE_BYTES, "Pinned v3 parent file required"
    )
    blob = path.read_bytes()
    _require(_sha(blob) == PARENT_SOURCE_SHA256, "Pinned v3 parent source changed")
    return path, blob


def _load_parent():
    path, _ = _parent_source()
    name = "_jev_observation_profile_v4_parent_8d199ac1564a"
    existing = sys.modules.get(name)
    if existing is not None:
        _require(
            _sha(Path(existing.__file__).read_bytes()) == PARENT_SOURCE_SHA256,
            "Loaded parent source changed",
        )
        return existing
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


def _source_pin(expected):
    _require(
        type(expected) is str and re.fullmatch(r"[0-9a-f]{64}", expected),
        "Explicit reviewed v4 helper SHA256 required",
    )
    blob = _ordinary_path(__file__).read_bytes()
    _require(
        0 < len(blob) <= MAX_SOURCE_BYTES and _sha(blob) == expected,
        "Reviewed v4 helper source changed",
    )
    _parent_source()
    return blob


def read_canonical_sources(root):
    return _PARENT.read_canonical_sources(root)


PROXY_DISCOVERY = r"""  // A rendered label is a real native click target for its semantic control.
  cache.labelOwners = new WeakMap();
  const enabledOwner=e=>e?.isConnected && e.tagName==='INPUT' &&
    ['checkbox','radio'].includes(e.type) && !e.matches(':disabled') &&
    !e.closest('[aria-disabled="true"],[aria-hidden="true"],[inert]');
  const hitTarget=e=>{
    if (!e?.isConnected || !visible(e) || e.closest('[aria-disabled="true"],[inert]')) return false;
    const r=e.getBoundingClientRect(),x=r.x+r.width/2,y=r.y+r.height/2;
    if (!(r.width>0 && r.height>0 && x>=0 && y>=0 && x<innerWidth && y<innerHeight)) return false;
    const hit=document.elementFromPoint(x,y);
    if (!e.contains(hit)) return false;
    const interactive=hit.closest(selector);
    return e.tagName!=='LABEL' || !interactive || interactive===e.control;
  };
  cache.proxyState=label=>{
    const owner=cache.labelOwners.get(label);
    if (!enabledOwner(owner) || label?.tagName!=='LABEL' || label.control!==owner ||
        !hitTarget(label)) return null;
    return [identity(owner),owner.type,name(owner),String(owner.checked),owner.id,
      label.htmlFor,owner.getAttribute('aria-checked'),owner.getAttribute('aria-disabled'),
      String(owner.name),owner.form ? identity(owner.form) : null];
  };
  const normalGuard=cache.guard;
  cache.guard=e=>{
    const ordinary=normalGuard(e);
    if (!ordinary || !cache.labelOwners.has(e)) return ordinary;
    const binding=cache.proxyState(e);
    return binding ? [...ordinary,'associated-label-control-v4',binding] : null;
  };
  const proxyActions=[];
  for (const owner of document.querySelectorAll('input[type="checkbox"],input[type="radio"]')) {
    if (!enabledOwner(owner) || hitTarget(owner)) continue;
    // Native label order and one target per semantic owner, never a site selector.
    const label=[...owner.labels].find(label=>label.control===owner && hitTarget(label) &&
      !actions.some(action=>action.kind==='click' && cache.nodes.get(action.node)===label));
    if (!label) continue;
    cache.labelOwners.set(label,owner);
    const r=label.getBoundingClientRect();
    proxyActions.push({node:identity(label),role:owner.type,label:name(owner)||owner.type,
      rect:{x:r.x,y:r.y,w:r.width,h:r.height},kind:'click',value:String(owner.value),
      checked:String(owner.checked),control_node:identity(owner),control_type:owner.type,
      control_guard:cache.proxyState(label)});
  }
  const observedActions=[...actions,...proxyActions];
"""

PROXY_EXECUTOR = """              const c=window.__jevFast, proxy=c?.labelOwners?.get(e);
              if (proxy || action.control_node!==undefined) {
                const owner=c?.nodes.get(action.control_node);
                if (action.kind!=='click' || e.tagName!=='LABEL' || e.control!==owner ||
                    proxy!==owner || !['checkbox','radio'].includes(action.control_type) ||
                    owner.type!==action.control_type || String(owner.checked)!==action.checked ||
                    !Array.isArray(action.control_guard) ||
                    JSON.stringify(c.proxyState(e))!==JSON.stringify(action.control_guard)) return null;
              }
"""


def _replace_once(source, before, after):
    _require(source.count(before) == 1, "Pinned derivation anchor changed")
    return source.replace(before, after, 1)


def _derive_snapshot(v3):
    source = v3.decode()
    source = _replace_once(
        source, "  const words=[], range=", PROXY_DISCOVERY + "  const words=[], range="
    )
    source = _replace_once(
        source,
        "for (const a of actions) if (!(a.node in guards))",
        "for (const a of observedActions) if (!(a.node in guards))",
    )
    source = _replace_once(
        source, "const semantics=actions.map", "const semantics=observedActions.map"
    )
    source = _replace_once(
        source, "  actions.splice(250);", "  actions.splice(250);\n  actions.push(...proxyActions);"
    )
    source = _replace_once(source, "id:'composed-visible-text-v3'", "id:'" + PROFILE + "'")
    source = _replace_once(
        source,
        "page_key,guards,omitted_actions};",
        "page_key,guards,omitted_actions,label_proxy_actions:proxyActions.length};",
    )
    return source.encode()


def _derive_browser(canonical):
    source = canonical.decode()
    anchor = "              if (action.kind==='fill' && (e.readOnly || e.getAttribute('aria-readonly')==='true')) return null;\n"
    return _replace_once(source, anchor, PROXY_EXECUTOR + anchor).encode()


@dataclass(frozen=True)
class ProfileBundle:
    canonical_files: object
    runtime_files: object
    _manifest_bytes: bytes
    _helper_source: bytes
    _parent_source: bytes

    def manifest(self):
        return json.loads(self._manifest_bytes)

    def _materialized_sources(self):
        files = {"canonical/" + k: v for k, v in self.canonical_files.items()}
        files.update({"runtime/" + k: v for k, v in self.runtime_files.items()})
        files["sources/jev_observation_profile_v4.py"] = self._helper_source
        files["sources/" + PARENT_COPY] = self._parent_source
        return files

    def materialized_manifest(self):
        manifest = self.manifest()
        manifest["artifact_files"] = {
            label: {"sha256": _sha(blob), "size_bytes": len(blob)}
            for label, blob in sorted(self._materialized_sources().items())
        }
        return manifest

    @property
    def read_state(self):
        return self.runtime_files[SNAPSHOT].decode()

    @property
    def marker_expression(self):
        return f"(() => {{ const state={self.read_state}; return state?.marker ?? null; }})()"


@dataclass(frozen=True)
class VerifiedMaterializedProfile:
    path: Path
    bundle: ProfileBundle
    manifest_sha256: str
    _manifest_bytes: bytes

    def manifest(self):
        return json.loads(self._manifest_bytes)

    @property
    def runtime_root(self):
        return self.path / "runtime"

    @property
    def read_state(self):
        return self.bundle.read_state

    @property
    def marker_expression(self):
        return self.bundle.marker_expression


def build_profile(canonical_sources, *, expected_source_sha256, profile=PROFILE):
    _require(profile == PROFILE and type(profile) is str, "Explicit v4 profile required")
    helper = _source_pin(expected_source_sha256)
    _, parent_source = _parent_source()
    v3 = _PARENT.build_profile(
        canonical_sources,
        expected_source_sha256=PARENT_SOURCE_SHA256,
        profile=_PARENT.PRUNED_PROFILE,
    )
    canonical, runtime = dict(v3.canonical_files), dict(v3.runtime_files)
    runtime[SNAPSHOT] = _derive_snapshot(runtime[SNAPSHOT])
    runtime[BROWSER] = _derive_browser(canonical[BROWSER])
    _require(
        all(runtime[k] == canonical[k] for k in canonical if k not in {SNAPSHOT, BROWSER}),
        "Unrelated canonical source changed",
    )
    manifest = {
        "schema": "explicit-observation-runtime-profile-v4",
        "profile": PROFILE,
        "canonical_contract_revision": CANONICAL_REVISION,
        "canonical_Git_verification_performed_by_this_library": False,
        "canonical_Git_prerequisite": "Caller must retain strict benchmark upstream_manifest checks.",
        "canonical_source_sha256": dict(CANONICAL_SHA256),
        "runtime_source_sha256": {label: _sha(blob) for label, blob in runtime.items()},
        "helper_source_sha256": _sha(helper),
        "parent_helper_source_sha256": PARENT_SOURCE_SHA256,
        "runtime_root": "runtime",
        "canonical_copy_root": "canonical",
        "helper_copy": "sources/jev_observation_profile_v4.py",
        "parent_helper_copy": "sources/" + PARENT_COPY,
        "derivation": {
            "parent_profile": _PARENT.PRUNED_PROFILE,
            "parent_snapshot_sha256": _sha(v3.runtime_files[SNAPSHOT]),
            "changes": [
                "new explicit profile identity",
                "append visible associated-label checkbox/radio targets",
                "bind owner identity and state in freshness guards",
                "recheck owner binding immediately before native mouse input",
            ],
            "task_specific_rules": False,
            "normal_target_order_and_ids_preserved": True,
            "proxy_candidate_pruning": False,
            "executor_checkbox_state_assignment": False,
        },
        "limits": {
            **v3.manifest()["limits"],
            "native_action_limit": 250,
            "label_proxy_actions_limit": None,
            "native_omitted_actions_semantics": "Canonical native discovery cap unchanged; no additional proxy omission. Serving capacity refusals remain separate.",
        },
        "scope": "Case005 interface-exposed runtime repair; not blind to that diagnostic or score/training qualification.",
        "not_attested_by_this_library": [
            "browser or model execution",
            "prepared token capacity",
            "serving qualification",
            "training or task performance",
        ],
    }
    return ProfileBundle(
        MappingProxyType(canonical),
        MappingProxyType(runtime),
        _encode(manifest),
        helper,
        parent_source,
    )


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


def _materialized_labels():
    return {
        *("canonical/" + k for k in CANONICAL_SHA256),
        *("runtime/" + k for k in CANONICAL_SHA256),
        "sources/jev_observation_profile_v4.py",
        "sources/" + PARENT_COPY,
        "profile-manifest.json",
    }


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
    _require(type(profile) is str and profile == PROFILE, "Explicit v4 profile required")
    root = _ordinary_path(path)
    _require(root.is_dir(), "Materialized profile directory required")
    blobs, first_files, first_dirs = _capture_tree(root, read_contents=True)
    manifest_blob = blobs["profile-manifest.json"]
    _require(_sha(manifest_blob) == expected_manifest_sha256, "Materialized manifest hash differs")
    canonical = {label: blobs["canonical/" + label] for label in CANONICAL_SHA256}
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
