from pathlib import Path
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import types

root = Path('/home/albert_hsieh/Gemma-4-12B-Unified-System-One')
runtime = root / 'artifacts/jev-runtime'
owner_dir = runtime / 'llama-cpp-owner-20261024-07'
directory = runtime / 'decision3-publisher-batch4-cpu-both-20261024-05'
controller = owner_dir / 'controller.py'
spec_path = directory / 'root-edge-both-command-spec-01.json'
review_path = directory / 'independent-root-selection-review-01.json'
expected_spec = os.environ['JEV_FULL05_SPEC_SHA256']
expected_review = os.environ['JEV_FULL05_REVIEW_SHA256']
expected_smoke_closure = '5f5d05cb4d0ac54d7a9c9df07ba0eb8ba8934c5e660235197620532965dfd914'
assert re.fullmatch(r'[0-9a-f]{64}', expected_smoke_closure), 'Exact accepted owner07 smoke closure SHA256 required'
assert re.fullmatch(r'[0-9a-f]{64}', expected_spec)
assert re.fullmatch(r'[0-9a-f]{64}', expected_review)
expected_controller = '685a9f809f429f4e9658083dc8530b563740625f9ba7816908fdc58002de256d'
output = owner_dir / 'actual-02'
intent_path = owner_dir / 'root-launch-full05-edge-both-02.json'
terminal_path = owner_dir / 'root-execution-full05-edge-both-02.json'
gate_index = os.environ.get('JEV_FULL05_GATE_INDEX', '01')
assert gate_index == '01'
gate_path = directory / ('root-full05-initial-gate-' + gate_index + '.json')

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
assert not any(path.exists() or path.is_symlink() for path in (output, intent_path, terminal_path, gate_path))
spec = json.loads(spec_path.read_text())
for row in spec['sources']:
    assert ref(root / row['path']) == row
guard = types.ModuleType('_root_full05_coexist_guard')
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
gate = {'profile': 'full05-Edge-coexist-readonly-initial-gate-v1', 'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'controller_source': ref(controller), 'old_original_closure': ref(old_path), 'prior_catalog05_successfully_authenticated_closed': True, 'later_original_boundary_samples': samples, 'no_service_or_cache_actions': True}
native_closure_path = runtime / 'llama-cpp-owner-20261023-06/actual-08/external-boundary-closure.json'
native_closure = json.loads(native_closure_path.read_text())
assert ref(native_closure_path)['sha256'] == '1e96d9c32870238e857368ffa307e2ba6fad4e4d579c6d4fce2a63788bc57bc4'
assert native_closure['closure_mode'] == 'authenticated_stop'
assert native_closure['primary_error'] is None and native_closure['closure_errors'] == []
assert native_closure['whole_unit_exit_confirmed'] is True
assert native_closure['observed_cgroup_cohort_exit_confirmed'] is True
assert native_closure['post_stop_terminal_verified'] is True
assert native_closure['cgroup_absent'] is True and native_closure['sources_unchanged'] is True
assert len(native_closure['two_process_scans']) == 2
assert all(not row['matches'] and not row['unknown'] for row in native_closure['two_process_scans'])
native_selection = boundary.Selection(**native_closure['selection'])
assert native_selection.boot_id == selected.boot_id
assert native_selection.unit == 'jev-d3-llama-cpp-cpu4-2g-long-20261023-06.service' and native_selection.memory_max_bytes == 2147483648
assert native_selection.runtime_seconds == 10800 and native_selection.stop_seconds == 3
assert native_closure['before_stop']['Description'] == native_selection.nonce
assert native_closure['before_stop']['InvocationID'] == native_closure['invocation_id']
assert native_closure['before_stop']['MainPID'] == str(native_closure['actor']['pid'])
assert native_closure['before_stop']['ControlGroup'] == native_closure['actor']['cgroup'] == native_selection.cgroup
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
gate['prior_replay08_original_closure'] = ref(native_closure_path)
gate['original_replay08_whole_unit_exit_confirmed'] = native_closure['whole_unit_exit_confirmed']
gate['later_replay08_original_boundary_samples'] = native_samples
assert guard.UNIT == 'jev-d3-llama-cpp-cpu4-2g-coexist-20261024-07.service'
assert guard.PROFILE == 'llama-cpp-cpu4-2g-coexist-command-owned-v1'
assert guard.HOST_OBSERVATION_POLICY == 'comfy-observed-cpu-coexist-v1'
smoke_closure_path = owner_dir / 'actual-01/external-boundary-closure.json'
assert ref(smoke_closure_path)['sha256'] == expected_smoke_closure
smoke_closure = json.loads(smoke_closure_path.read_text())
assert smoke_closure['closure_mode'] == 'authenticated_stop'
assert smoke_closure['primary_error'] is None and smoke_closure['closure_errors'] == []
assert smoke_closure['whole_unit_exit_confirmed'] is True
assert smoke_closure['observed_cgroup_cohort_exit_confirmed'] is True
assert smoke_closure['post_stop_terminal_verified'] is True
assert smoke_closure['cgroup_absent'] is True and smoke_closure['sources_unchanged'] is True
assert len(smoke_closure['two_process_scans']) == 2
assert all(not row['matches'] and not row['unknown'] for row in smoke_closure['two_process_scans'])
fresh_selection = boundary.Selection(**smoke_closure['selection'])
assert fresh_selection.boot_id == selected.boot_id
assert fresh_selection.unit == guard.UNIT and fresh_selection.memory_max_bytes == 2147483648
assert fresh_selection.runtime_seconds == 10800 and fresh_selection.stop_seconds == 3
assert smoke_closure['before_stop']['Description'] == fresh_selection.nonce
assert smoke_closure['before_stop']['InvocationID'] == smoke_closure['invocation_id']
assert smoke_closure['before_stop']['MainPID'] == str(smoke_closure['actor']['pid'])
assert smoke_closure['before_stop']['ControlGroup'] == smoke_closure['actor']['cgroup'] == fresh_selection.cgroup
fresh_subject = types.SimpleNamespace(selection=fresh_selection, actor=smoke_closure['actor'], tracked={(row['pid'], row['start_ticks']): row for row in smoke_closure['tracked_cohort']})
fresh_samples = []
for index in range(2):
    before = boundary.status(fresh_selection)
    scan = boundary.UnitBoundary.scan_absence(fresh_subject)
    after = boundary.status(fresh_selection)
    absent = not (boundary.CGROUP_ROOT / fresh_selection.cgroup.lstrip('/')).exists()
    fresh_samples.append({'status_before': before, 'original_method_process_scan': scan, 'status_after': after, 'cgroup_absent': absent})
    assert before['LoadState'] == after['LoadState'] == 'not-found'
    assert before['MainPID'] == after['MainPID'] == '0' and absent and not scan['matches'] and not scan['unknown']
    if index == 0:
        time.sleep(0.1)
gate['prior_native03_owner07_original_closure'] = ref(smoke_closure_path)
gate['original_native03_owner07_whole_unit_exit_confirmed'] = smoke_closure['whole_unit_exit_confirmed']
gate['later_native03_owner07_original_boundary_samples'] = fresh_samples
gate['absence_selection_nonce_scope'] = 'Actual prior smoke nonce/cohort checked for current absence; new owner creates a fresh random launch nonce'
gate['host_observation_policy'] = guard.HOST_OBSERVATION_POLICY
try:
    gate['gate'] = guard.host(observer, selected.boot_id, 130595864576, initial=True)
    gate['passed'] = True
except BaseException as error:
    gate['passed'] = False
    gate['error'] = guard.failure(error)
save(gate_path, gate)
if not gate['passed']:
    print(json.dumps({'status': 'NO_LAUNCH_NEW_PROFILE_GATE_FAILED', 'gate_ref': ref(gate_path), 'error': gate['error']}, ensure_ascii=False))
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
save(intent_path, {'profile': 'root-selected-full05-Edge-coexist-launch-v1', 'created_at_utc': utc, 'prelaunch_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(), 'boot_id': selected.boot_id, 'argv': argv, 'controller_identity': identity, 'controller_sha256': expected_controller, 'selection_sha256': expected_spec, 'independent_source_selection_review_sha256': expected_review, 'root_gate': ref(gate_path), 'all_original_1479_callbacks_231_then1248_and180_10000_10800_required': True, 'host_observation_policy': guard.HOST_OBSERVATION_POLICY})
stdout, stderr = process.communicate()
result = {'profile': 'root-selected-full05-Edge-coexist-terminal-v1', 'started_at_utc': utc, 'finished_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'argv': argv, 'controller_identity': identity, 'exit_code': process.returncode, 'wall_seconds': time.monotonic() - started, 'stdout': stdout, 'stderr': stderr, 'selection_sha256': expected_spec, 'independent_source_selection_review_sha256': expected_review, 'scope': 'Root terminal only. Require owner07 completed/final full authentication/current original closure and independently audited full1479 campaign and typed stage evidence. No automatic score or browser qualification; no old profile fit transfer.'}
save(terminal_path, result)
print(json.dumps({'exit_code': process.returncode, 'wall_seconds': result['wall_seconds'], 'root_terminal_record': str(terminal_path)}))
raise SystemExit(process.returncode)
