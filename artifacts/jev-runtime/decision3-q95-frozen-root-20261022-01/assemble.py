"""Copy saved Q95 source/metadata/cache bytes only; never open tensor payloads."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/albert_hsieh/Gemma-4-12B-Unified-System-One')
HERE = Path(__file__).resolve().parent
QUALIFICATION = ROOT / 'artifacts/jev-runtime/public-field-syntax-helper-20261018-01/qualify-01'
SOURCE = QUALIFICATION / 'sources'
E_SHA = '95f2eab0be9646da18694b18c872e044db09ba2fb1c09708077b01b24d58ae44'
R_SHA = '702f7ff3cf194d1db56f86474e48ffc88cdfa8c9c6d3930467e3c12ea6492016'
Q_SHA = 'd33bfcac49f90c47b626c3915baf3d5fcbfb2e32522bfaccbbfec59dce35d8db'
F_SHA = 'a66125c589b7408337685d06daf4d0eb80265b07e7e9a55786442562389b87d3'
CACHE_SHA = '6e5ac130fa1f5b6bb315f71d17cb8d3d1b8f2552eef319a1e695c06ebf5b0dc6'
REVISION = 'a66f836b56605039fe040f330180e336d19b3362'
HUB = Path('artifacts/jev-local-hf/hub')
SNAPSHOT = HUB / 'models--knowlet--Gemma-4-12B-Unified-System-One/snapshots' / REVISION
GRAMMAR = Path('artifacts/jev-runtime/field-json-grammar-20261018-01/full-compile-01/cache')
FREEZE = Path('artifacts/jev-runtime/frozen-public-observation-36/freeze-manifest.json')
BASE_PLAN = Path('artifacts/jev-runtime/public-native-action-bf16-qualification-plan-20261016-01/plan-01.json')
MODEL_PIN = {'sha256': '11144621122123135eb79c9cfd0466a2c51f230109124aefe401cdbec8c71a53', 'size_bytes': 23919549408}


def require(value, message):
    if not value:
        raise ValueError(message)


def encode(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode() + b'\n'


def ordinary(path):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'Ordinary path required')
    return path


def small_bytes(path):
    path = ordinary(path)
    require(path.is_file() and path.stat().st_size <= 64 * 1024**2, 'Small non-tensor file only')
    require(path.suffix != '.safetensors', 'Tensor payload reads forbidden during assembly')
    return path.read_bytes()


def pin_bytes(blob):
    return {'sha256': hashlib.sha256(blob).hexdigest(), 'size_bytes': len(blob)}


def small_pin(path):
    return pin_bytes(small_bytes(path))


def ref(path):
    return {'path': str(path.relative_to(ROOT)), **small_pin(path)}


def copy(source, destination, expected=None):
    blob = small_bytes(source)
    if expected is not None:
        require(pin_bytes(blob) == expected, 'Exact saved bytes differ: ' + str(source))
    destination = ordinary(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        stream.write(blob)
    require(small_pin(destination) == pin_bytes(blob), 'Copied bytes differ')
    return pin_bytes(blob)


def write(name, value):
    with (HERE / name).open('xb') as stream:
        stream.write(encode(value))


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--assemble', action='store_true')
    args = parser.parse_args(argv)
    if not args.assemble:
        print('DRY_NO_MODEL_OR_WEIGHT_READ; explicit --assemble copies small saved bytes only')
        return 0
    execution_blob = small_bytes(QUALIFICATION / 'execution.json')
    runtime_blob = small_bytes(QUALIFICATION / 'runtime.json')
    require(pin_bytes(execution_blob)['sha256'] == E_SHA, 'Selected execution95')
    require(pin_bytes(runtime_blob)['sha256'] == R_SHA, 'Selected runtime348')
    require(small_pin(QUALIFICATION / 'qualification.json')['sha256'] == Q_SHA, 'Selected historical Q95')
    execution, runtime = json.loads(execution_blob), json.loads(runtime_blob)
    require(len(execution['files']) == 95 and len(runtime['files']) == 348, 'Exact source/runtime counts')
    copied = {}
    for label, expected in execution['files'].items():
        require(not Path(label).is_absolute() and '..' not in Path(label).parts, 'Confined source label')
        copied[label] = copy(SOURCE / label, HERE / label, expected)
    require(small_pin(ROOT / FREEZE)['sha256'] == F_SHA, 'Frozen84 authority')
    copied[str(FREEZE)] = copy(ROOT / FREEZE, HERE / FREEZE)
    design = json.loads(small_bytes(QUALIFICATION / 'plan-02.json'))
    base = {k: design['base_plan'][k] for k in ['sha256', 'size_bytes']}
    copied[str(BASE_PLAN)] = copy(QUALIFICATION / 'plan-01.json', HERE / BASE_PLAN, base)
    for source, name in [(QUALIFICATION / 'execution.json', 'execution95.json'),
                         (QUALIFICATION / 'runtime.json', 'runtime348.json'),
                         (QUALIFICATION / 'plan-02.json', 'design-wrapper.json'),
                         (QUALIFICATION / 'inputs.json', 'inputs.json')]:
        copied['support/' + name] = copy(source, HERE / 'support' / name)
    manifest_blob = small_bytes(ROOT / GRAMMAR / 'manifest.json')
    require(pin_bytes(manifest_blob)['sha256'] == CACHE_SHA, 'Selected grammar cache authority')
    grammar = {'manifest.json': pin_bytes(manifest_blob), **json.loads(manifest_blob)['files']}
    require(len(grammar) == 8, 'Eight exact grammar cache files')
    for label, expected in grammar.items():
        copied[str(GRAMMAR / label)] = copy(ROOT / GRAMMAR / label, HERE / GRAMMAR / label, expected)
    snapshot_metadata = {}
    for source in sorted((ROOT / SNAPSHOT).iterdir()):
        if source.name == 'model.safetensors':
            continue
        resolved = source.resolve(strict=True)
        require(resolved.is_relative_to(ROOT / HUB), 'Original HF payload confined to original hub')
        expected = runtime['base_metadata'].get(source.name)
        snapshot_metadata[source.name] = copy(resolved, HERE / SNAPSHOT / source.name, expected)
        copied[str(SNAPSHOT / source.name)] = snapshot_metadata[source.name]
    require(set(runtime['base_metadata']) <= set(snapshot_metadata), 'Selected eight HF metadata files')
    # A tensor hardlink would change the original shared inode ctime/nlink.
    # Parent explicitly holds that operation while the native N3 unit is active.
    original = (ROOT / SNAPSHOT / 'model.safetensors').resolve(strict=True)
    require(original.is_relative_to(ROOT / HUB) and original.stat().st_size == MODEL_PIN['size_bytes'],
            'Confined original tensor metadata')
    s = original.stat()
    pending = {'schema_id': 'decision3-q95-pending-tensor-link-v1', 'status': 'held_until_nativeN3_external_closure',
               'source': str(original), 'destination': str(HERE / SNAPSHOT / 'model.safetensors'),
               'expected_payload_pin': MODEL_PIN, 'payload_read_or_rehashed': False,
               'hardlink_created': False, 'source_stat_before_link': {
                   'device': s.st_dev, 'inode': s.st_ino, 'size_bytes': s.st_size,
                   'mtime_ns': s.st_mtime_ns, 'ctime_ns': s.st_ctime_ns, 'nlink': s.st_nlink},
               'reason': 'Hardlink modifies original sharedinode nlink/ctime; wait for active N3 closure.'}
    write('pending-tensor-link.json', pending)
    old = json.loads(small_bytes(ROOT / 'artifacts/jev-runtime/public-field-syntax-helper-20261018-01/root-semantic-candidate-serve-launch-01.json'))
    argv = list(old['serve_argv'])
    argv[3] = str(HERE / 'scripts/jev_qualify_public_field_syntax_bf16.py')
    for flag, value in [('--execution-path', HERE / 'support/execution95.json'),
                        ('--runtime-path', HERE / 'support/runtime348.json'),
                        ('--plan-path', HERE / 'support/design-wrapper.json'),
                        ('--inputs-path', HERE / 'support/inputs.json'),
                        ('--output', HERE / 'serve-01')]:
        argv[argv.index(flag) + 1] = str(value)
    env = dict(old['environment_updates'])
    env.update(HF_HOME=str(HERE / 'artifacts/jev-local-hf'), HF_HUB_CACHE=str(HERE / HUB),
               PYTHONPATH=':'.join([str(ROOT / 'artifacts/jev-runtime/overlay'), str(HERE / 'src'), str(HERE / 'scripts')]),
               HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_TOKEN_PATH='/nonexistent-q95-public-token')
    plan = {'schema_id': 'decision3-q95-isolated-frozen-serve-plan-v1', 'source_root': str(HERE),
            'status': 'prepared_unexecuted_tensor_link_and_live_owner_admission_pending', 'launch_ready': False,
            'cwd': str(HERE), 'argv': argv, 'environment_updates': env,
            'environment_unset': old['environment_unset'], 'source_count': 95, 'runtime_file_count': 348,
            'helper_profile': 'field-syntax-preserving-base-helper-v1',
            'qualification': ref(QUALIFICATION / 'qualification.json'),
            'execution': ref(HERE / 'support/execution95.json'), 'runtime': ref(HERE / 'support/runtime348.json'),
            'design': ref(HERE / 'support/design-wrapper.json'), 'inputs': ref(HERE / 'support/inputs.json'),
            'source_files': copied, 'pending_tensor_link': ref(HERE / 'pending-tensor-link.json'),
            'future_private_config_policy': 'Fresh ownUID/nlink1/mode0600 controller-config generated by original qualified controller; never copy old private config/token.',
            'fresh_live_requirements': ['nativeN3 external closure before tensor hardlink',
                'fresh sameboot host/rawGPU identity and unchanged full independent Q95 memory floors',
                'owned kernel process boundary and exact live process/listener/private-credential binding',
                'actual selected native runtime/model/qualification/source/cache checks by unchanged serve code'],
            'execution_so_far': {'framework_imports': 0, 'model_loads': 0, 'GPU': 0, 'units': 0, 'HTTP': 0,
                                 'tensor_reads': 0, 'tensor_hardlinks': 0},
            'historical_qualification_is_fresh_live_qualification': False}
    write('serve-plan-01.json', plan)
    write('assembly-inventory.json', {'schema_id': 'decision3-q95-frozen-small-assembly-v1',
        'files': copied, 'file_count': len(copied), 'total_bytes': sum(v['size_bytes'] for v in copied.values()),
        'source95_identity_preserved': True, 'symlinks_created': 0, 'hardlinks_created': 0,
        'tensor_payload_reads': 0, 'weight_link_pending': True})
    print(json.dumps({'copied_files': len(copied), 'copied_bytes': sum(v['size_bytes'] for v in copied.values()),
                      'sources': 95, 'runtime_selected': 348, 'grammar_cache': 8, 'snapshot_metadata': len(snapshot_metadata),
                      'tensor_link': 'held', 'serve_plan': ref(HERE / 'serve-plan-01.json')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
