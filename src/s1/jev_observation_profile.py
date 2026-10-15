"""Build explicit, source-authenticated Jev observation profiles.

The default returns seven canonical source files byte for byte. V2 changes
generic visible-text traversal, its metadata and its fail-closed limit error.
Explicit V3 additionally skips excluded subtrees and zero-contribution leaves.
This library never imports a browser, model, collector or artifact Python code.
Canonical Git verification remains the caller's separate strict prerequisite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType

NATIVE_PROFILE = "native-1231850"
COMPOSED_PROFILE = "composed-visible-text-v2"
PRUNED_PROFILE = "composed-visible-text-v3"
CANONICAL_REVISION = "1231850a0bf1a0c0341fe408ef1668dbbfdfac46"
CANONICAL_SHA256 = MappingProxyType(
    {
        "jev_ultrafast/__init__.py": "60f025b8d24465e2c0f7a3547d43b8020fd8843de7b6cb3f892ef63f5548cf63",
        "jev_ultrafast/snapshot.js": "e50473501c8fb8e70f3b21866d987393e3f2315c639d638bd477d170e81ed78d",
        "jev_ultrafast/agent.py": "aaf8b271cb0ee6643fe7beb525328ab80de8568c7ac044d02b9c6721161e7c8b",
        "jev_ultrafast/browser.py": "ee08dafa93d731bd292aa0263a578a29c8001cf22eef3c59fe44a159649155ee",
        "jev_ultrafast/model.py": "a85ada8458d23642c5c8a4095acf0565e6268f81dcda26f6308942b653e9c9f3",
        "jev_ultrafast/questions.py": "66a7054b85040b949e1757311257ea0c5a843c635f9a40c99f6d15ad660fb422",
        "jev_ultrafast/static/fixture.html": "e48e7d7b6225b215060692e0a28c3d0b59e465e4b94c1bc510d5f37ff0b8bfbe",
    }
)
SNAPSHOT = "jev_ultrafast/snapshot.js"
PARENT_BUILDER_SHA256 = "b9739ba962d41bdb20831196ca8b217608e369477d433b39672b17e47abcd46b"
PARENT_V1_SNAPSHOT_SHA256 = "3a4a5e7532a7eb99334b8fd71e181d5d18183af4966757cd42d48e6404cbb197"
V2_SNAPSHOT_SHA256 = "5327d4ab5d615ac0804f4913bb86bb59b383b11ff80f2408eb4aa565243f9dd5"
PRUNED_SNAPSHOT_SHA256 = "c694f308f29ed5000b7bb60e9374afd9398413d2811e07741cf0f0a6321be552"
MAX_SOURCE_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 4 * 1024 * 1024
MAX_MATERIALIZED_FILES = 16
MAX_MATERIALIZED_BYTES = MAX_MATERIALIZED_FILES * MAX_SOURCE_BYTES

REPLACEMENT = r"""  const words=[], range=document.createRange(); let length=0;
  const textProfile={id:'composed-visible-text-v2',visited_nodes:0,
    node_limit:20000,depth_limit:128,node_limit_reached:false,depth_limit_reached:false};
  const composedParent=e=>e?.assignedSlot || e?.parentElement || e?.getRootNode?.()?.host || null;
  const composedVisible=e=>{
    let depth=0;
    for (let current=e; current; current=composedParent(current)) {
      if (++depth>128) {textProfile.depth_limit_reached=true;return false;}
      const style=getComputedStyle(current);
      if (current.hidden || current.hasAttribute('inert') ||
          current.getAttribute('aria-hidden')==='true' || style.display==='none' ||
          style.visibility==='hidden' || style.visibility==='collapse' ||
          Number(style.opacity)===0 || style.contentVisibility==='hidden') return false;
    }
    return true;
  };
  const seen=new WeakSet(), stack=[{node:document.body,depth:0}];
  while (stack.length && length<6000) {
    const {node,depth}=stack.pop();
    if (seen.has(node)) continue;
    seen.add(node);
    if (textProfile.visited_nodes>=textProfile.node_limit) {
      textProfile.node_limit_reached=true;break;
    }
    textProfile.visited_nodes++;
    if (depth>textProfile.depth_limit) {textProfile.depth_limit_reached=true;continue;}
    if (node.nodeType===Node.TEXT_NODE) {
      const value=node.textContent.trim(), parent=node.assignedSlot || node.parentElement || node.getRootNode()?.host;
      if (!value || !parent || parent.closest('script,style,noscript,template') ||
          !composedVisible(parent)) continue;
      range.selectNodeContents(node);const r=range.getBoundingClientRect();
      if (r.width>0 && r.height>0 && r.bottom>0 && r.top<innerHeight &&
          r.right>0 && r.left<innerWidth) {
        words.push(value);length+=value.length;
      }
      continue;
    }
    if (node.nodeType===Node.ELEMENT_NODE &&
        node.matches('script,style,noscript,template')) continue;
    let children;
    if (node.nodeType===Node.ELEMENT_NODE && node.tagName==='SLOT') {
      const assigned=node.assignedNodes({flatten:true});
      children=assigned.length ? assigned : [...node.childNodes];
    } else if (node.nodeType===Node.ELEMENT_NODE && node.shadowRoot?.mode==='open') {
      children=[...node.shadowRoot.childNodes];
    } else children=[...node.childNodes];
    for (let i=children.length-1;i>=0;i--) stack.push({node:children[i],depth:depth+1});
  }
  if (textProfile.node_limit_reached || textProfile.depth_limit_reached) {
    const error=new Error('COMPOSED_VISIBLE_TEXT_LIMIT '+JSON.stringify(textProfile));
    error.name='ComposedVisibleTextLimitError';throw error;
  }
"""


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


def _source_pin(expected):
    _require(
        type(expected) is str and re.fullmatch(r"[0-9a-f]{64}", expected),
        "Explicit reviewed helper SHA256 required",
    )
    source = Path(__file__).read_bytes()
    _require(_sha(source) == expected, "Reviewed helper source changed")
    return source


def read_canonical_sources(root):
    """Read exactly seven bounded, nonsymlinked files; no Git/process execution."""
    root = _ordinary_path(root)
    _require(root.is_dir(), "Canonical source directory required")
    result = {}
    for label in sorted(CANONICAL_SHA256):
        path = _ordinary_path(root / label)
        _require(
            path.is_file() and stat.S_ISREG(path.stat().st_mode), "Canonical ordinary file required"
        )
        with path.open("rb") as stream:
            result[label] = stream.read(MAX_SOURCE_BYTES + 1)
    return result


def _derive_v2(native):
    text = native.decode("utf-8")
    start_anchor, end_anchor = "  const words=[], walker=", "  const text=words.join"
    _require(
        text.count(start_anchor) == text.count(end_anchor) == 1, "Canonical text anchors changed"
    )
    start, end = text.index(start_anchor), text.index(end_anchor)
    result = text[:start] + REPLACEMENT + text[end:]
    original = "  return {url:location.href,title:document.title,w:innerWidth,h:innerHeight,text,"
    _require(result.count(original) == 1, "Canonical result anchor changed")
    result = result.replace(
        original,
        "  return {observation_profile:textProfile,url:location.href,title:document.title,w:innerWidth,h:innerHeight,text,",
    )
    return result.encode("utf-8")


def _derive_v3(native):
    """Derive the reviewed generic pruning delta without changing V2 bytes."""
    source = _derive_v2(native).decode("utf-8")
    old_check = """      const style=getComputedStyle(current);
      if (current.hidden || current.hasAttribute('inert') ||
          current.getAttribute('aria-hidden')==='true' || style.display==='none' ||
          style.visibility==='hidden' || style.visibility==='collapse' ||
          Number(style.opacity)===0 || style.contentVisibility==='hidden') return false;
"""
    excluded_element = """  const excludedElement=e=>{
    const style=getComputedStyle(e);
    return e.hidden || e.hasAttribute('inert') ||
      e.getAttribute('aria-hidden')==='true' || style.display==='none' ||
      style.visibility==='hidden' || style.visibility==='collapse' ||
      Number(style.opacity)===0 || style.contentVisibility==='hidden';
  };
"""
    push = (
        "    for (let i=children.length-1;i>=0;i--) stack.push({node:children[i],depth:depth+1});"
    )
    for anchor in (old_check, "  const composedVisible=e=>{", "    let children;", push):
        _require(source.count(anchor) == 1, "Reviewed V3 traversal anchors changed")
    source = source.replace(
        "  const composedVisible=e=>{", excluded_element + "  const composedVisible=e=>{"
    )
    source = source.replace(old_check, "      if (excludedElement(current)) return false;\n")
    source = source.replace(
        "    let children;",
        "    if (node.nodeType===Node.ELEMENT_NODE && excludedElement(node)) continue;\n"
        "    let children;",
    )
    source = source.replace(
        push,
        "    children=children.filter(child=>child.nodeType!==Node.COMMENT_NODE &&\n"
        "      !(child.nodeType===Node.TEXT_NODE && !child.textContent.trim()));\n" + push,
    )
    source = source.replace(f"id:'{COMPOSED_PROFILE}'", f"id:'{PRUNED_PROFILE}'")
    result = source.encode("utf-8")
    _require(_sha(result) == PRUNED_SNAPSHOT_SHA256, "Reviewed V3 snapshot derivation differs")
    return result


@dataclass(frozen=True)
class ProfileBundle:
    canonical_files: object
    runtime_files: object
    _manifest_bytes: bytes
    _helper_source: bytes

    def manifest(self):
        """Return the pure source-bundle contract without a disk inventory."""
        return json.loads(self._manifest_bytes)

    def _materialized_sources(self):
        files = {"canonical/" + k: v for k, v in self.canonical_files.items()}
        files.update({"runtime/" + k: v for k, v in self.runtime_files.items()})
        files["sources/jev_observation_profile.py"] = self._helper_source
        return files

    def materialized_manifest(self):
        """Return the deterministic complete disk contract; no I/O or mutation."""
        manifest = self.manifest()
        manifest["artifact_files"] = {
            label: {"sha256": _sha(blob), "size_bytes": len(blob)}
            for label, blob in sorted(self._materialized_sources().items())
        }
        return manifest

    @property
    def read_state(self):
        return self.runtime_files[SNAPSHOT].decode("utf-8")

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


def build_profile(canonical_sources, *, expected_source_sha256, profile=NATIVE_PROFILE):
    """Return detached immutable files; no alias, fallback or installation."""
    _require(
        type(profile) is str and profile in {NATIVE_PROFILE, COMPOSED_PROFILE, PRUNED_PROFILE},
        "Unknown explicit snapshot profile",
    )
    helper = _source_pin(expected_source_sha256)
    _require(
        set(canonical_sources) == set(CANONICAL_SHA256),
        "Exactly seven canonical source files required",
    )
    canonical = {}
    for label in sorted(CANONICAL_SHA256):
        blob = canonical_sources[label]
        _require(
            type(blob) is bytes and 0 < len(blob) <= MAX_SOURCE_BYTES,
            "Bounded immutable source bytes required",
        )
        _require(_sha(blob) == CANONICAL_SHA256[label], "Canonical source mismatch: " + label)
        canonical[label] = blob
    _require(
        sum(map(len, canonical.values())) <= MAX_TOTAL_BYTES,
        "Canonical source bundle exceeds bound",
    )
    runtime = dict(canonical)
    if profile == COMPOSED_PROFILE:
        runtime[SNAPSHOT] = _derive_v2(canonical[SNAPSHOT])
    elif profile == PRUNED_PROFILE:
        runtime[SNAPSHOT] = _derive_v3(canonical[SNAPSHOT])
    manifest = {
        "schema": "explicit-snapshot-profile-v1",
        "profile": profile,
        "canonical_contract_revision": CANONICAL_REVISION,
        "canonical_Git_verification_performed_by_this_library": False,
        "canonical_Git_prerequisite": "Caller must retain existing strict upstream_manifest checks; content-authentication here is not a relaxed Git verifier.",
        "canonical_source_sha256": dict(CANONICAL_SHA256),
        "runtime_source_sha256": {label: _sha(blob) for label, blob in runtime.items()},
        "helper_source_sha256": _sha(helper),
        "runtime_root": "runtime",
        "canonical_copy_root": "canonical",
        "helper_copy": "sources/jev_observation_profile.py",
        "derivation": None
        if profile == NATIVE_PROFILE
        else {
            "parent_builder_sha256": PARENT_BUILDER_SHA256,
            "parent_v1_snapshot_sha256": PARENT_V1_SNAPSHOT_SHA256,
            "changes": [
                "generic corrected composed walker retained",
                "profile ID changed to composed-visible-text-v2",
                "bounded structured limit error before snapshot return",
            ],
            "task_specific_rules": False,
        },
        "limits": None
        if profile == NATIVE_PROFILE
        else {
            "text_characters": 6000,
            "visited_nodes": 20000,
            "composed_depth": 128,
            "limit_error_name": "ComposedVisibleTextLimitError",
            "limit_error_prefix": "COMPOSED_VISIBLE_TEXT_LIMIT ",
            "node_or_depth_cap": "throw inside READ_STATE before return; embedded MARKER propagates same error",
        },
        "semantics": "Agent/browser/model/questions/action discovery/guards source unchanged. Composed text also changes native marker values; it is a separate observation profile, not historical native score parity.",
        "not_attested_by_this_library": [
            "runtime installation",
            "collector/validator dispatch",
            "browser/model execution",
            "corpus migration",
            "score/training qualification",
        ],
    }
    if profile == PRUNED_PROFILE:
        manifest["derivation"] = {
            "parent_profile": COMPOSED_PROFILE,
            "parent_snapshot_sha256": V2_SNAPSHOT_SHA256,
            "changes": [
                "profile ID changed to composed-visible-text-v3",
                "skip descendants excluded by the existing composed ancestor predicate",
                "filter COMMENT and exact trim-empty TEXT children before reverse push",
                "remaining order, nonempty text processing, action code and limits unchanged",
            ],
            "task_specific_rules": False,
        }
        manifest["limits"]["visited_nodes_semantics"] = (
            "Counts visited nodes after COMMENT and exact trim-empty TEXT children are filtered "
            "and descendants of excluded elements are skipped. The excluded element itself "
            "still consumes a visit before its children are skipped."
        )
        manifest["limits"]["not_bounded_by_visited_nodes"] = (
            "This is not a hard bound on all DOM reads, child-list creation/filtering, "
            "CSS/ancestor checks, text trimming or elapsed time."
        )
    _require(
        all(runtime[label] == canonical[label] for label in canonical if label != SNAPSHOT),
        "Non-snapshot source changed",
    )
    return ProfileBundle(
        MappingProxyType(canonical), MappingProxyType(runtime), _encode(manifest), helper
    )


def _write_exclusive(path, blob):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())


def materialize_profile(canonical_root, output, *, expected_source_sha256, profile=NATIVE_PROFILE):
    """Write one NEW source tree, never alter the canonical checkout or globals."""
    canonical_root, output = _ordinary_path(canonical_root), _ordinary_path(output)
    _require(not output.exists(), "New output required; never overwrite")
    _require(
        not output.is_relative_to(canonical_root) and not canonical_root.is_relative_to(output),
        "Canonical and profile output must not overlap",
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
    _require(
        {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()} == set(files),
        "Materialized source population changed",
    )
    for label, blob in files.items():
        _require((output / label).read_bytes() == blob, "Materialized source bytes changed")
    _require(
        Path(__file__).read_bytes() == bundle._helper_source,
        "Helper changed during materialization",
    )
    manifest = bundle.materialized_manifest()
    final = _encode(manifest) + b"\n"
    pending = output / ".profile-manifest.pending"
    _write_exclusive(pending, final)
    for directory in sorted(
        [output, *(p for p in output.rglob("*") if p.is_dir())],
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    result = {
        "output": str(output),
        "profile": profile,
        "manifest_sha256": _sha(final),
        "manifest_size_bytes": len(final),
        "runtime_root": str(output / "runtime"),
        "browser_model_calls": 0,
    }
    # Final atomic publication. Losing the rename on a crash can leave pending;
    # no semantic validation follows a successful manifest publication.
    os.replace(pending, output / "profile-manifest.json")
    return result


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
        *("canonical/" + label for label in CANONICAL_SHA256),
        *("runtime/" + label for label in CANONICAL_SHA256),
        "sources/jev_observation_profile.py",
        "profile-manifest.json",
    }


def _capture_materialized_tree(root, *, read_contents):
    """Capture one fixed ordinary tree using bounded no-follow descriptor reads."""
    expected_files = _materialized_labels()
    expected_dirs = {
        str(parent)
        for label in expected_files
        for parent in PurePosixPath(label).parents
        if str(parent) != "."
    }
    files, directories, blobs = {}, {}, {}
    total = 0

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
                    _require(relative in expected_dirs, "Unknown nested profile directory")
                    child = os.open(
                        entry.name,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=descriptor,
                    )
                    try:
                        _require(
                            _stat_identity(os.fstat(child)) == _stat_identity(observed),
                            "Profile directory changed before read",
                        )
                        visit(relative, child)
                    finally:
                        os.close(child)
                    continue
                _require(stat.S_ISREG(observed.st_mode), "Ordinary profile file required")
                _require(relative in expected_files, "Unknown extra profile file")
                _require(0 < observed.st_size <= MAX_SOURCE_BYTES, "Profile file exceeds bound")
                files[relative] = _stat_identity(observed)
                total += observed.st_size
                _require(
                    len(files) <= MAX_MATERIALIZED_FILES and total <= MAX_MATERIALIZED_BYTES,
                    "Complete materialized profile exceeds bound",
                )
                if read_contents:
                    fd = os.open(
                        entry.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor
                    )
                    with os.fdopen(fd, "rb") as stream:
                        opened = os.fstat(stream.fileno())
                        _require(
                            stat.S_ISREG(opened.st_mode)
                            and _stat_identity(opened) == files[relative],
                            "Profile file changed before read",
                        )
                        blob = stream.read(MAX_SOURCE_BYTES + 1)
                        _require(
                            len(blob) == opened.st_size
                            and len(blob) <= MAX_SOURCE_BYTES
                            and _stat_identity(os.fstat(stream.fileno())) == files[relative],
                            "Profile file changed during bounded read",
                        )
                    blobs[relative] = blob
        _require(
            _stat_identity(os.fstat(descriptor)) == directories[label],
            "Profile directory changed during capture",
        )

    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        visit("", root_fd)
        _require(
            _stat_identity(root.stat(follow_symlinks=False)) == directories[""],
            "Profile root changed during capture",
        )
    finally:
        os.close(root_fd)
    _require(set(files) == expected_files, "Missing materialized profile file")
    _require(set(directories) == {"", *expected_dirs}, "Missing materialized profile directory")
    return blobs, files, directories


def verify_materialized_profile(
    path, *, expected_manifest_sha256, expected_source_sha256, profile=NATIVE_PROFILE
):
    """Authenticate disk bytes and return a detached immutable identity handle.

    This is not a Git verifier, runtime installer, filesystem lock, browser
    execution or fit admission. Callers retain strict canonical Git checks and
    must reverify before later importing live files and at lifecycle boundaries.
    Captured helper/runtime Python is always data and is never executed here.
    """
    helper = _source_pin(expected_source_sha256)
    _require(
        type(expected_manifest_sha256) is str
        and re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256),
        "Explicit reviewed materialized manifest SHA256 required",
    )
    _require(
        type(profile) is str and profile in {NATIVE_PROFILE, COMPOSED_PROFILE, PRUNED_PROFILE},
        "Unknown explicit snapshot profile",
    )
    root = _ordinary_path(path)
    _require(root.is_dir(), "Materialized profile directory required")
    blobs, first_files, first_dirs = _capture_materialized_tree(root, read_contents=True)
    manifest_blob = blobs["profile-manifest.json"]
    _require(_sha(manifest_blob) == expected_manifest_sha256, "Materialized manifest hash differs")
    canonical = {label: blobs["canonical/" + label] for label in CANONICAL_SHA256}
    bundle = build_profile(
        canonical, expected_source_sha256=expected_source_sha256, profile=profile
    )
    _require(bundle._helper_source == helper, "Reviewed helper changed during verification")
    expected_manifest = _encode(bundle.materialized_manifest()) + b"\n"
    _require(manifest_blob == expected_manifest, "Materialized manifest metadata/profile differs")
    expected_sources = bundle._materialized_sources()
    _require(
        all(blobs[label] == blob for label, blob in expected_sources.items()),
        "Materialized canonical/runtime/helper bytes differ",
    )
    _, final_files, final_dirs = _capture_materialized_tree(root, read_contents=False)
    _require(
        first_files == final_files and first_dirs == final_dirs,
        "Materialized profile closed inventory changed during verification",
    )
    _require(
        _ordinary_path(root) == root and _source_pin(expected_source_sha256) == helper,
        "Reviewed source/path changed during verification",
    )
    return VerifiedMaterializedProfile(root, bundle, _sha(manifest_blob), manifest_blob)


def main(argv=None):
    """Materialize a new profile source tree; never launch a browser or model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument(
        "--profile",
        choices=(NATIVE_PROFILE, COMPOSED_PROFILE, PRUNED_PROFILE),
        default=NATIVE_PROFILE,
    )
    args = parser.parse_args(argv)
    result = materialize_profile(
        args.canonical_root,
        args.output,
        expected_source_sha256=args.source_sha256,
        profile=args.profile,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
