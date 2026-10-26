"""Execute only saved-byte Q95 metadata verifiers; never launch or load a model."""
from __future__ import annotations

import builtins
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = Path('/home/albert_hsieh/Gemma-4-12B-Unified-System-One')
QUALIFIER = 'scripts/jev_qualify_public_field_syntax_bf16.py'
ARTIFACT = 'scripts/jev_public_action_artifact.py'
E_SHA = '95f2eab0be9646da18694b18c872e044db09ba2fb1c09708077b01b24d58ae44'
R_SHA = '702f7ff3cf194d1db56f86474e48ffc88cdfa8c9c6d3930467e3c12ea6492016'
Q_SHA = 'd33bfcac49f90c47b626c3915baf3d5fcbfb2e32522bfaccbbfec59dce35d8db'


def require(value, message):
    if not value:
        raise ValueError(message)


def pin(path):
    require(path.suffix != '.safetensors' and path.stat().st_size <= 64 * 1024**2,
            'Metadata/source/cache only; no tensor payload read')
    blob = path.read_bytes()
    return {'sha256': hashlib.sha256(blob).hexdigest(), 'size_bytes': len(blob)}


def ref(path):
    return {'path': str(path.relative_to(PROJECT)), **pin(path)}


def main():
    started = time.monotonic()
    require(sys.flags.no_user_site and sys.dont_write_bytecode, 'Selected -B -s flags required')
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if name.split('.')[0] in {'torch', 'transformers', 'peft', 'tokenizers', 'safetensors', 'modal', 'ctypes'}:
            raise AssertionError('Framework/native import forbidden in metadata preflight: ' + name)
        return original_import(name, *args, **kwargs)

    builtins.__import__ = checked_import

    def no_actors(event, _args):
        if event in {'socket.connect', 'socket.bind', 'socket.sendto', 'subprocess.Popen',
                     'os.system', 'os.fork', 'os.posix_spawn'}:
            raise AssertionError('Actors/network forbidden in metadata preflight: ' + event)

    sys.addaudithook(no_actors)
    plan = json.loads((HERE / 'serve-plan-02.json').read_bytes())
    execution_path = HERE / 'support/execution95.json'
    runtime_path = HERE / 'support/runtime348.json'
    execution_blob, runtime_blob = execution_path.read_bytes(), runtime_path.read_bytes()
    require(hashlib.sha256(execution_blob).hexdigest() == E_SHA, 'Selected95 manifest')
    require(hashlib.sha256(runtime_blob).hexdigest() == R_SHA, 'Selected348 runtime')
    execution = json.loads(execution_blob)
    sys.path[:0] = [str(HERE / 'scripts'), str(HERE / 'src')]

    def load(name, label):
        path = HERE / label
        blob = path.read_bytes()
        require(pin(path) == execution['files'][label], 'Saved module pin')
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        exec(compile(blob, str(path), 'exec', dont_inherit=True), module.__dict__)
        return module

    load('jev_public_action_artifact', ARTIFACT)
    q = load('jev_frozen_q95_metadata_subject', QUALIFIER)
    require(q.ROOT == HERE, 'Natural copied qualifier ROOT')
    artifact_path = Path(plan['argv'][plan['argv'].index('--artifact-path') + 1])
    handoff = (artifact_path / 'source-handoff.json').read_bytes()
    verified_sources = q.verify_execution_sources(execution_blob, expected_sha256=E_SHA,
        expected_training_sha256=q.TRAINING90_SHA, training_blob=handoff)
    verified_sources.check()
    require(verified_sources.manifest()['files'] == execution['files'], 'Exact saved95 closure')
    runtime = q.verify_runtime_closure(runtime_blob, expected_sha256=R_SHA)
    q._verify_cached(runtime)  # Exactly eight metadata files; no model.safetensors read.
    manifest, grammar_blobs = q._field_cache_bytes(path=q.FIELD_CACHE)
    require(len(grammar_blobs) == 8, 'Exact grammar cache eight')
    design_blob = (HERE / 'support/design-wrapper.json').read_bytes()
    design = json.loads(design_blob)
    base_blob = (HERE / design['base_plan']['path']).read_bytes()
    q.apply_design_delta(base_blob, design_blob)
    for flag, expected in [('--manifest-sha256', plan['argv'][plan['argv'].index('--manifest-sha256') + 1]),
                           ('--receipt-sha256', plan['argv'][plan['argv'].index('--receipt-sha256') + 1]),
                           ('--training-handoff-sha256', q.TRAINING90_SHA)]:
        name = {'--manifest-sha256': 'adapter/manifest.json', '--receipt-sha256': 'receipt.json',
                '--training-handoff-sha256': 'source-handoff.json'}[flag]
        require(pin(artifact_path / name)['sha256'] == expected, 'Existing public artifact metadata')
    qualification = Path(plan['argv'][plan['argv'].index('--qualification-path') + 1])
    require(pin(qualification / 'qualification.json')['sha256'] == Q_SHA, 'Exact historical Q95')
    pending = json.loads((HERE / 'pending-tensor-link.json').read_bytes())
    link = json.loads((HERE / 'tensor-link-01.json').read_bytes())
    target = Path(pending['destination'])
    source = Path(pending['source'])
    before, actual = source.stat(), target.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_nlink)
            == (actual.st_dev, actual.st_ino, actual.st_size, actual.st_mtime_ns, actual.st_ctime_ns, actual.st_nlink),
            'Exact linked inode metadata')
    require(actual.st_size == pending['expected_payload_pin']['size_bytes']
            and actual.st_ino == link['target_after']['inode']
            and actual.st_ctime_ns == link['target_after']['ctime_ns'], 'Recorded link metadata unchanged')
    require(sys.modules['jev_public_action_artifact'].resolve_base_checkpoint(q._cache()) == target,
            'Original resolver admits exact confined ordinary target')
    actual = {str(p.relative_to(HERE)): pin(p) for folder in ['src', 'scripts']
              for p in (HERE / folder).rglob('*.py')}
    require(actual == execution['files'], 'Exact95 copied project Python population')
    require(not list(HERE.rglob('*.pyc')) and not any(p.is_symlink() for p in HERE.rglob('*')),
            'No bytecode caches or escaping symlinks')
    result = {'schema_id': 'decision3-q95-frozen-saved-byte-preflight-v1',
        'status': 'saved_metadata_source_runtime_cache_and_confined_tensor_link_preflight_passed',
        'launch_ready': False, 'qualifier_source': ref(HERE / QUALIFIER),
        'serve_plan': ref(HERE / 'serve-plan-02.json'), 'source95_full_bytes_verified': True,
        'source95_count': 95, 'frozen84_and_training90_verified': True,
        'base_plan_and_design_semantics_verified': True, 'installed_runtime_files_verified': len(runtime['files']),
        'installed_runtime_file_bytes': sum(v['size_bytes'] for v in runtime['files'].values()),
        'selected_distributions_verified': len(runtime['distributions']), 'base_metadata_verified': len(runtime['base_metadata']),
        'grammar_cache_files_verified': len(grammar_blobs), 'grammar_cache_bytes': sum(len(v) for v in grammar_blobs.values()),
        'historical_qualification': ref(qualification / 'qualification.json'),
        'historical_qualification_full_tree_reverified': False, 'historical_qualification_reused_as_fresh_live_proof': False,
        'tensor_payload_reads_or_rehashes': 0, 'tensor_hardlinks': 1, 'no_framework_imports': True,
        'no_models_GPU_browser_units_services_or_Comfy_calls': True, 'live_helper_identity_or_coexistence_qualified': False,
        'documented_runtime_scope': 'Selected source bytes/METADATA/version/origin observations, accepted with actual runtime/CUDA/kernel/geometry at launch. No generalized whole-execution/indirectDSO authentication claimed.',
        'tensor_link': ref(HERE / 'tensor-link-01.json'),
        'remaining': [
                      'Actual unchanged serve must perform full public/qualification/model/runtime checks under fresh admission.',
                      'Independent live process/listener/credential/co-resident checker and kernel owner proof required.'],
        'elapsed_seconds': time.monotonic() - started,
        'preflight_source': ref(Path(__file__))}
    with (HERE / 'preflight-linked-01.json').open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')
    print(json.dumps({'result': ref(HERE / 'preflight-linked-01.json'), 'sources': 95, 'runtime': len(runtime['files']),
                      'grammar': len(grammar_blobs), 'tensor_link': 'exact_confined_hardlink_metadata_verified', 'elapsed_seconds': time.monotonic() - started}))


if __name__ == '__main__':
    main()
