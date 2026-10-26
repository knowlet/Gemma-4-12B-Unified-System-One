from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
import types

root = Path('/home/albert_hsieh/Gemma-4-12B-Unified-System-One')
runtime = root / 'artifacts/jev-runtime'
owner_dir = runtime / 'llama-cpp-owner-20261023-06'
directory = runtime / 'decision3-publisher-batch4-cpu-both-20261023-04'
controller = owner_dir / 'controller.py'
spec_path = directory / 'root-edge-both-command-spec-01.json'
review_path = directory / 'independent-root-selection-review-01.json'
expected_spec = os.environ['JEV_FULL04_SPEC_SHA256']
expected_review = os.environ['JEV_FULL04_REVIEW_SHA256']
assert expected_spec == '59396734477f90fa1a3f24bec60633796de06f84215597a4e59a5b5d9a274d49'
assert expected_review == '1730df03a1e79b2f0a8acb09b01f24e61be3036ed5df688e693eca72214ba6bb'
expected_controller = 'aa2af9911fde9fd06526c3fd5fe45f6521e80103bbccf9c29b98c7f3bba5952d'
output = owner_dir / 'actual-07'
intent_path = owner_dir / 'root-launch-full04-edge-both-07.json'
terminal_path = owner_dir / 'root-execution-full04-edge-both-07.json'
gate_index = os.environ.get('JEV_FULL04_GATE_INDEX', '07')
assert gate_index == '07'
gate_path = directory / ('root-full04-initial-gate-' + gate_index + '.json')

def ref(path):
    blob = path.read_bytes()
    assert len(blob) <= 1048576
    return {'path': str(path.relative_to(root)), 'sha256': hashlib.sha256(blob).hexdigest(), 'size_bytes': len(blob)}

def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())

assert ref(controller)['sha256'] == expected_controller
assert ref(spec_path)['sha256'] == expected_spec
assert ref(review_path)['sha256'] == expected_review
assert not any(path.exists() for path in (output, intent_path, terminal_path, gate_path))
spec = json.loads(spec_path.read_text())
for row in spec['sources']:
    assert ref(root / row['path']) == row
guard = types.ModuleType('_root_native_full04_original_guard')
guard.__file__ = str(controller)
sys.modules[guard.__name__] = guard
exec(compile(controller.read_bytes(), str(controller), 'exec'), vars(guard))
base = [guard.ref(controller), guard.ref(spec_path), *(guard.ref(root / label) for label, _ in guard.FIXED.values())]
guard.command_selection(spec_path, expected_spec, base)
boundary, observer = guard.load_fixed('boundary'), guard.load_fixed('observer')
old_path = runtime / 'llama-cpp-owner-20261023-03/actual-05/external-boundary-closure.json'
old = json.loads(old_path.read_text())
assert ref(old_path)['sha256'] == '2458fc5ea1b5a41617a7c2fd9c6266440c15e727cc8469d9004d620fe2b13790'
selected = boundary.Selection(**old['selection'])
assert selected.boot_id == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
subject = types.SimpleNamespace(selection=selected, actor=old['actor'], tracked={(row['pid'], row['start_ticks']): row for row in old['tracked_cohort']})
samples = []
for index in range(2):
    before = boundary.status(selected)
    scan = boundary.UnitBoundary.scan_absence(subject)
    after = boundary.status(selected)
    absent = not (boundary.CGROUP_ROOT / selected.cgroup.lstrip('/')).exists()
    samples.append({'status_before': before, 'original_method_process_scan': scan, 'status_after': after, 'cgroup_absent': absent})
    assert before['LoadState'] == after['LoadState'] == 'not-found'
    assert before['MainPID'] == after['MainPID'] == '0' and absent
    assert not scan['matches'] and not scan['unknown']
    if index == 0:
        time.sleep(0.1)
gate = {'profile': 'full04-Edge-original-readonly-initial-gate-v1', 'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'controller_source': ref(controller), 'old_original_closure': ref(old_path), 'prior_catalog05_successfully_authenticated_closed': True, 'later_original_boundary_samples': samples, 'no_service_or_cache_actions': True}
native_closure_path = runtime / 'llama-cpp-owner-20261023-06/actual-06/external-boundary-closure.json'
native_closure = json.loads(native_closure_path.read_text())
assert ref(native_closure_path)['sha256'] == 'e7a01835feb8cad14d9373af8545abdc0702dff27b42e9cf8fe905487ada0584'
supplemental_path = runtime / 'decision3-browser-domain-canary-20261023-01/root-supplemental-actual06-closure-02.json'
assert ref(supplemental_path)['sha256'] == 'ade10418a4c4dfec49907643ba11da3ac45cad86b07f1dc714b2c178ba3df3cd'
supplemental = json.loads(supplemental_path.read_text())
assert supplemental['original_closure'] == ref(native_closure_path)
assert supplemental['original_whole_unit_exit_confirmed_preserved'] is False
assert supplemental['later_two_original_empty_scans_and_cgroup_absence'] is True
assert native_closure['whole_unit_exit_confirmed'] is False
assert native_closure['post_stop_terminal_verified'] is True
assert native_closure['cgroup_absent'] is True and native_closure['sources_unchanged'] is True
native_selection = boundary.Selection(**native_closure['selection'])
assert native_selection.boot_id == selected.boot_id
native_subject = types.SimpleNamespace(selection=native_selection, actor=native_closure['actor'], tracked={(row['pid'], row['start_ticks']): row for row in native_closure['tracked_cohort']})
native_samples = []
for index in range(2):
    before = boundary.status(native_selection)
    scan = boundary.UnitBoundary.scan_absence(native_subject)
    after = boundary.status(native_selection)
    absent = not (boundary.CGROUP_ROOT / native_selection.cgroup.lstrip('/')).exists()
    native_samples.append({'status_before': before, 'original_method_process_scan': scan, 'status_after': after, 'cgroup_absent': absent})
    assert before['LoadState'] == after['LoadState'] == 'not-found'
    assert before['MainPID'] == after['MainPID'] == '0' and absent and not scan['matches'] and not scan['unknown']
    if index == 0:
        time.sleep(0.1)
gate['prior_native06_original_closure'] = ref(native_closure_path)
gate['prior_native06_later_original_absence'] = ref(supplemental_path)
gate['original_native06_whole_unit_exit_confirmed_preserved'] = native_closure['whole_unit_exit_confirmed']
gate['later_native06_original_boundary_samples'] = native_samples
try:
    gate['gate'] = guard.host(observer, selected.boot_id, 130595864576, initial=True)
    gate['passed'] = True
except BaseException as error:
    gate['passed'] = False
    gate['error'] = guard.failure(error)
save(gate_path, gate)
if not gate['passed']:
    print(json.dumps({'status': 'NO_LAUNCH_ORIGINAL_GATE_FAILED', 'gate_ref': ref(gate_path), 'error': gate['error']}, ensure_ascii=False))
    raise SystemExit(2)
argv = [str(root / '.venv/bin/python'), '-B', '-s', str(controller), '--run', '--expected-entry-sha256', expected_controller, '--selection', str(spec_path), '--selection-sha256', expected_spec, '--output', str(output)]
started = time.monotonic()
utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
process = subprocess.Popen(argv, cwd=root, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
identity = {'pid': process.pid}
try:
    raw = Path(f'/proc/{process.pid}/stat').read_text()
    identity.update(starttime_ticks=int(raw[raw.rfind(')') + 2:].split()[19]), exe=os.readlink(f'/proc/{process.pid}/exe'), cwd=os.readlink(f'/proc/{process.pid}/cwd'))
except (FileNotFoundError, ProcessLookupError):
    identity['exited_before_identity_sample'] = True
save(intent_path, {'profile': 'root-selected-full04-Edge-both-launch-v1', 'created_at_utc': utc, 'prelaunch_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(), 'boot_id': selected.boot_id, 'argv': argv, 'controller_identity': identity, 'controller_sha256': expected_controller, 'selection_sha256': expected_spec, 'independent_source_selection_review_sha256': expected_review, 'root_gate': ref(gate_path), 'all_original_1479_callbacks_231_then1248_and180_10000_10800_required': True})
stdout, stderr = process.communicate()
result = {'profile': 'root-selected-full04-Edge-both-terminal-v1', 'started_at_utc': utc, 'finished_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'argv': argv, 'controller_identity': identity, 'exit_code': process.returncode, 'wall_seconds': time.monotonic() - started, 'stdout': stdout, 'stderr': stderr, 'selection_sha256': expected_spec, 'independent_source_selection_review_sha256': expected_review, 'scope': 'Root terminal only. Require original owner completed/final-authentication/closure and actual full benchmark global and per-stage authentication, continuation or terminal plus original current closure required. No automatic score or browser qualification.'}
save(terminal_path, result)
print(json.dumps({'exit_code': process.returncode, 'wall_seconds': result['wall_seconds'], 'root_terminal_record': str(terminal_path)}))
raise SystemExit(process.returncode)
