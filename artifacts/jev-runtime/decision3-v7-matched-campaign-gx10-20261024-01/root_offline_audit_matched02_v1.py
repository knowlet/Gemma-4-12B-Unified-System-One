"""Read-only, stdlib-only audit for root-selected matched-02 after terminal.

SOURCE AUTHORING ONLY: this file must not be executed while the campaign is live.
It imports no project modules, starts no process, changes no files, and prints
only aggregate metadata and fixed diagnostic codes. Actual bodies, judge inputs,
labels, probabilities, states and raw exception messages never reach stdout.
"""

from __future__ import annotations

import hashlib
import json
import copy
import os
import re
import stat
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path('/home/albert_hsieh/Gemma-4-12B-Unified-System-One')
BASE = ROOT / 'artifacts/jev-runtime/decision3-v7-matched-campaign-gx10-20261024-01'
RUN = BASE / 'matched-02'
OUT = RUN / 'campaign'
CPU_OWNER = ROOT / 'artifacts/jev-runtime/llama-cpp-owner-20261024-09'
PLAN_SHA = 'f7406956166c12f052ae16435556e7cca6f2dac45d70f0d1adc4a71f1fe4d52d'
OWNER_SHA = '635b24c059361eb780bb9c89cd5cf0b0975ff72662b5f270c92834a32e24bdc5'
CAPABILITY_SHA = '4dbbb9470d011931854f6473c0840ef7f0746c4194c719cde6ab454c3e79d463'
V7_SHA = 'd9f4236d1d46c9472c3d6c1551342639ecb870e6c0dd419246927645410ea868'
BENCHMARK_SHA = 'fe1410675e4decb06a8491a5a83d2ec965629c99f8912b797446c4cb2c155f6c'
LAUNCHER_SHA = '31450f9002ad216c32b77751997eb27748c491d7bc757c4410294d3704fdfb9e'
ALIAS = 'd3-edge-dsl01-operation-delta-cpu5-candidate'
PROFILE = 'composed-visible-text-settled-v7'
CASES = (
    'public-arxiv-title-filter-results-v1',
    'public-arxiv-original-paper-open-v1',
    'public-python-counter-most-common-v1',
    'public-nasa-mars-facts-moons-v1',
) + tuple(f'heldout-public-{i:03d}' for i in range(6))
SHA_RE = re.compile(r'[0-9a-f]{64}')
RID_RE = re.compile(r'[0-9a-f]{32}')
JSON_MAX = 64 * 1024**2
FILE_MAX = 4 * 1024**3
TOTAL_MAX = 8 * 1024**3
FILE_COUNT_MAX = 20000
_cache = {}


class AuditRequired(Exception):
    pass


def need(condition, code):
    if not condition:
        raise AuditRequired(code)


def ordinary(value):
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    need('..' not in path.parts and path.is_relative_to(ROOT), 'path.outside_repo')
    need(not any(p.is_symlink() for p in (path, *path.parents)), 'path.symlink')
    return path


def signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def pin(value, maximum=FILE_MAX):
    path = ordinary(value)
    info = path.stat()
    need(stat.S_ISREG(info.st_mode) and 0 <= info.st_size <= maximum, 'file.bounds')
    key = (str(path), signature(info))
    if key in _cache:
        return _cache[key]
    digest, count = hashlib.sha256(), 0
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, 'rb') as stream:
        before = os.fstat(stream.fileno())
        while True:
            block = stream.read(1024**2)
            if not block:
                break
            count += len(block)
            need(count <= maximum, 'file.streaming_bounds')
            digest.update(block)
        after = os.fstat(stream.fileno())
    need(signature(info) == signature(before) == signature(after) == signature(path.stat()),
         'file.changed_during_hash')
    need(count == info.st_size, 'file.incomplete_hash')
    result = {'sha256': digest.hexdigest(), 'size_bytes': count}
    _cache[key] = result
    return result


def raw(value, maximum=JSON_MAX):
    path = ordinary(value)
    before = pin(path, maximum)
    with path.open('rb') as stream:
        blob = stream.read(maximum + 1)
    need(len(blob) == before['size_bytes'] <= maximum
         and hashlib.sha256(blob).hexdigest() == before['sha256']
         and pin(path, maximum) == before, 'file.changed_during_read')
    return blob


def decode(blob):
    def pairs(rows):
        result = {}
        for key, value in rows:
            need(key not in result, 'json.duplicate_key')
            result[key] = value
        return result
    return json.loads(blob, object_pairs_hook=pairs,
                      parse_constant=lambda _x: need(False, 'json.nonfinite'))


def load(value):
    return decode(raw(value))


def auth(record, expected=None):
    need(type(record) is dict and set(record) in (
        {'path', 'sha256'}, {'path', 'sha256', 'size_bytes'}), 'ref.shape')
    need(type(record['sha256']) is str and SHA_RE.fullmatch(record['sha256']), 'ref.sha')
    path = ordinary(record['path'])
    need(expected is None or path == ordinary(expected), 'ref.expected_path')
    actual = pin(path)
    need(actual['sha256'] == record['sha256'], 'ref.hash_mismatch')
    if 'size_bytes' in record:
        need(type(record['size_bytes']) is int
             and record['size_bytes'] == actual['size_bytes'], 'ref.size_mismatch')
    return path


def auth_json(record, expected=None):
    return load(auth(record, expected))


def inventory(directory):
    directory = ordinary(directory)
    index = load(directory / 'output-index.json')
    files = index['files']
    need(type(files) is dict and len(files) == index['file_count'] <= FILE_COUNT_MAX,
         'inventory.count')
    actual = set()
    for path in directory.rglob('*'):
        ordinary(path)
        need(not path.is_symlink(), 'inventory.symlink')
        if path.is_dir():
            continue
        need(path.is_file(), 'inventory.nonregular')
        actual.add(path.relative_to(directory).as_posix())
        need(len(actual) <= FILE_COUNT_MAX + 1, 'inventory.population_bound')
    need(actual == set(files) | {'output-index.json'}, 'inventory.population_mismatch')
    total = 0
    for name, record in files.items():
        need(type(name) is str and name not in {'', 'output-index.json'}
             and not Path(name).is_absolute() and '..' not in Path(name).parts,
             'inventory.member_path')
        need(type(record) is dict and set(record) == {'sha256', 'size_bytes'},
             'inventory.ref_shape')
        actual_pin = pin(directory / name)
        need(actual_pin == record, 'inventory.member_mismatch')
        total += actual_pin['size_bytes']
        need(total <= TOTAL_MAX, 'inventory.total_bound')
    need(total == index['total_bytes'], 'inventory.byte_total')
    return {'file_count': len(files), 'total_bytes': total}


def stable(row):
    return {key: row[key] for key in ('pid', 'start_ticks', 'pgid', 'sid', 'uid', 'cgroup')}


def process_absent(identity):
    need(type(identity) is dict and all(type(identity.get(k)) is int and identity[k] > 0
         for k in ('pid', 'pgid', 'sid', 'start_ticks')), 'process.identity_shape')
    path = Path('/proc') / str(identity['pid']) / 'stat'
    try:
        text = path.read_text()
    except FileNotFoundError:
        return
    fields = text[text.rfind(')') + 2:].split()
    need(int(fields[19]) != identity['start_ticks'], 'process.original_still_live')


def binding(value, *, schema, root_sha, index, case_id, service_sha=None, nonce=None):
    need(value.get('schema') == schema and value.get('root_plan_sha256') == root_sha
         and value.get('case_index') == index and value.get('case_id') == case_id,
         'binding.case_root')
    if service_sha is not None:
        need(value.get('plan_sha256') == service_sha, 'binding.service_plan')
    if nonce is not None:
        need(value.get('nonce') == nonce, 'binding.nonce')


def audit_outer(plan, receipt):
    terminal = load(BASE / 'root-execution-matched-02.json')
    need(terminal['exit_code'] == 0, 'terminal.nonzero_or_incomplete')
    need(terminal['selected_plan']['sha256'] == PLAN_SHA
         and terminal['selected_owner']['sha256'] == OWNER_SHA, 'terminal.selection')
    auth(terminal['selected_plan'], BASE / 'root-matched-plan-03.json')
    auth(terminal['selected_owner'], BASE / 'owner_released_v2.py')
    auth(terminal['root_launcher_source'], BASE / 'root_launch_matched_02.py')
    need(type(terminal['executed_git_commit']) is str
         and re.fullmatch(r'[0-9a-f]{40}', terminal['executed_git_commit']), 'terminal.git_commit')
    selected = raw(RUN / 'selected-plan.json')
    need(hashlib.sha256(selected).hexdigest() == PLAN_SHA and decode(selected) == plan,
         'outer.selected_plan')
    result = load(RUN / 'root-external-closure.json')
    need(result['plan_sha256'] == PLAN_SHA and result['reason'] == 'terminal'
         and result['unit_readback_verified'] is True and result['primary_error'] is None
         and result['candidate_pump_closure_error'] is None, 'outer.terminal_status')
    term, actor = result['terminal_before_stop'], result['actor']
    need(all(term.get(k) == v for k, v in {
        'Result': 'success', 'ExecMainCode': '1', 'ExecMainStatus': '0', 'MainPID': '0'
    }.items()) and term['InvocationID'] == result['invocation_id']
         and RID_RE.fullmatch(result['invocation_id'])
         and term['ExecMainPID'] == str(actor['pid']), 'outer.unit_terminal')
    need(actor['pid'] == actor['pgid'] == actor['sid'], 'outer.session_identity')
    need(result['cgroup_absent_after_stop'] is True and result['after_stop']['MainPID'] == '0'
         and type(result['stop_result']) is str and result['changed_sources'] == [],
         'outer.stop_sources')
    scans = result['two_proc_scans']
    need(len(scans) == 2 and all(row['matches'] == row['unreadable'] == [] for row in scans),
         'outer.process_scans')
    need(result['model_cache_aliases_before']
         and result['model_cache_aliases_after'] == result['model_cache_aliases_before'],
         'outer.model_alias_closure')
    host = result['host_after']
    need(host['boot_id'] == plan['boot_id']
         and host['mem_total_bytes'] == result['selected_memtotal_bytes']
         and host['comfy_state'] in {'http_idle', 'http_busy', 'absent_confirmed'},
         'outer.final_host')
    need(receipt['schema'] == 'jev-nf4-d3-matched-v7-campaign-v1'
         and receipt['root_plan_sha256'] == PLAN_SHA
         and receipt['stage'] == 'complete_pending_external_closure'
         and receipt['error'] is None, 'campaign.stage')
    intent = result['actor_read']
    need(intent['actor'] == actor and intent['plan_sha256'] == PLAN_SHA
         and intent['unit'] == plan['unit'] and intent['nonce'] == plan['nonce'],
         'outer.actor_intent')
    process_absent(actor)
    need(not (Path('/sys/fs/cgroup') / plan['cgroup'].lstrip('/')).exists(),
         'outer.cgroup_still_present')
    return result


def audit_bundle(record, directory, index):
    manifest = auth_json(record['bundle_manifest_ref'], directory / 'runtime-bundle/bundle-manifest.json')
    need(manifest['schema_id'] == 'jev-v7-isolated-runner-bundle-v1'
         and manifest['profile'] == PROFILE and manifest['profile_helper_sha256'] == V7_SHA
         and manifest['case_module'] == ('original' if index < 4 else 'heldout'),
         'bundle.profile')
    for name, value in manifest['files'].items():
        need('/' not in name and name not in {'', '.', '..'}, 'bundle.member_path')
        need(pin(directory / 'runtime-bundle' / name) == value, 'bundle.member_hash')
    files = manifest['files']
    need(files['original_jev_browser_benchmark.py']['sha256'] == BENCHMARK_SHA
         and files['launcher.py']['sha256'] == LAUNCHER_SHA
         and files['jev_observation_profile_v7.py']['sha256'] == V7_SHA
         and files['jev_public_web_cases.py']['sha256']
         == '0ec9826d334cb2fbed24ade8780fa785291017d904b18233419ced760750bca3',
         'bundle.source_pins')
    if index >= 4:
        need(files['jev_heldout_public_cases.py']['sha256']
             == '339cddf596a25ab433ca66e23924631b6261e62efc06f351878d4ef5c3f6f8f9',
             'bundle.heldout_judge_source_pin')
    delta = manifest['source_delta']
    need(delta['inverse_bytes_equal'] is True and delta['inverse_AST_equal'] is True
         and delta['tracker_inverse_bytes_equal'] is True, 'bundle.declared_inverse')
    # Reproduce only the literal, source-pinned V7 substitutions; do not import
    # or execute the builder, runner, observation helper or judge.
    original = raw(directory / 'runtime-bundle/original_jev_browser_benchmark.py')
    replacements = [
        (b'from s1 import jev_observation_profile as profile',
         b'import jev_observation_profile_v7 as profile', 3),
        (b'PRUNED_OBSERVATION_PROFILE = "composed-visible-text-v3"',
         b'PRUNED_OBSERVATION_PROFILE = "composed-visible-text-settled-v7"', 1),
        (b'from s1.jev_observation_events import ObservationProfileTracker',
         b'from jev_observation_events_v7 import ObservationProfileTracker', 1),
        (b'                        snapshots = []\n',
         b'                        snapshots = []\n'
         b'                        _settling_primary = None\n'
         b'                        _settling_cursor = profile.settling_cursor(agent_module.Browser)\n', 1),
        (b'                        except Exception as exc:\n                            error = f"{type(exc).__name__}: {exc}"\n',
         b'                        except Exception as exc:\n                            _settling_primary = exc\n'
         b'                            error = f"{type(exc).__name__}: {exc}"\n', 1),
        (b'                            _write_json(attempt / "final_state.json", final_state, secrets)\n',
         b'                            _write_json(attempt / "final_state.json", final_state, secrets)\n'
         b'                            profile.write_settling_sidecar(agent_module.Browser, _settling_cursor,\n'
         b'                                attempt / "navigation-settling.json", _write_json, secrets,\n'
         b'                                _settling_primary if _settling_primary is not None else sys.exception(), _redact)\n', 1),
    ]
    if index >= 4:
        replacements.append((b'        from jev_public_web_cases import (\n',
                             b'        from jev_heldout_public_cases import (\n', 1))
    expected = original
    for before, after, count in replacements:
        need(expected.count(before) == count, 'bundle.literal_substitution_anchor')
        expected = expected.replace(before, after)
    need(raw(directory / 'runtime-bundle/runner.py') == expected,
         'bundle.exact_allowed_runner_bytes')


def audit_case(record, arm, index):
    case_id, group = CASES[index], 'known4' if index < 4 else 'public6'
    directory = OUT / arm / case_id
    need(record['schema'] == 'jev-v7-matched-single-case-v1'
         and record['case_id'] == case_id and record['group'] == group
         and record['case_denominator'] == 1 and record['suite_denominator'] == 10
         and record['infrastructure_complete'] is True and record['error'] is None
         and type(record['task_success']) is bool, 'case.identity_completion')
    need(all(record.get(k) == v for k, v in {
        'profile': PROFILE, 'profile_helper_sha256': V7_SHA, 'seed': 20261014,
        'repeats': 1, 'max_steps': 16, 'action_profile': 'standard',
        'text_model': 'generated-actions-nf4', 'text_model_reasoning': 'none',
        'capability_sha256': CAPABILITY_SHA,
        'policy_model': 'generated-actions-nf4' if arm == 'baseline' else ALIAS,
    }.items()), 'case.fixed_protocol')
    caller = auth_json(record['receipt_ref'], directory / 'caller-receipt.json')
    need(caller == {k: v for k, v in record.items() if k != 'receipt_ref'},
         'case.immutable_caller_receipt')
    need(load(OUT / f'{arm}-case-{index + 1:02d}.json') == record, 'case.progress_join')
    summary = auth_json(record['summary_ref'], directory / 'results/summary.json')
    need(summary['attempts'] == 1 and len(summary['outcomes']) == 1
         and not summary.get('observation_profile_failed'), 'case.original_summary')
    configuration = summary['configuration']
    need(load(directory / 'results/configuration.json') == configuration,
         'case.configuration_file_join')
    judge_sha = ('0ec9826d334cb2fbed24ade8780fa785291017d904b18233419ced760750bca3'
                 if index < 4 else
                 '339cddf596a25ab433ca66e23924631b6261e62efc06f351878d4ef5c3f6f8f9')
    need(configuration['public_judge_sha256'] == judge_sha
         and all(configuration.get(k) == v for k, v in {
             'seed': 20261014, 'repeats': 1, 'max_steps': 16,
             'suite_split': 'development', 'model_requested': record['policy_model'],
             'endpoint': record['policy_endpoint'],
         }.items()), 'case.original_judge_and_configuration')
    profile = configuration['observation_profile']
    need(profile['id'] == PROFILE and profile['helper_source_sha256'] == V7_SHA,
         'case.original_observation_profile_identity')
    outcome = summary['outcomes'][0]
    attempt = directory / 'results/attempts' / (case_id + '-000')
    need(load(attempt / 'outcome.json') == outcome, 'case.original_outcome_join')
    final_state = load(attempt / 'final_state.json')
    need(outcome['case_id'] == case_id + '-000' and outcome['task_id'] == case_id
         and type(outcome['success']) is bool, 'case.judge_identity')
    # Exact _task_success from pinned scripts/jev_browser_benchmark.py.
    judged = bool(outcome['verification'].get('success') and final_state
                  and final_state['status'] == 'done' and outcome['error'] is None)
    need(outcome['success'] == judged == record['task_success']
         and summary['successes'] == int(judged)
         and summary['success_rate'] == int(judged), 'case.original_judge_success')
    expected_goal = outcome['verification']['success']
    need(outcome['goal_satisfied'] == (None if expected_goal is None else bool(expected_goal)),
         'case.goal_verification_join')
    calls = load(attempt / 'model_calls.json')
    need(type(calls) is list, 'case.model_calls_shape')
    policy_count = sum(row['kind'] == 'policy' for row in calls)
    need(outcome['policy_calls'] == summary['policy_calls'] == policy_count
         and outcome['text_calls'] == summary['text_calls'] == len(calls) - policy_count,
         'case.original_call_counts')
    audit_bundle(record, directory, index)
    trace = auth_json(record['http_trace_ref'], directory / 'http/http-attempts.json')
    need(trace['schema'] == 'jev-v7-matched-client-http-attempts-v1'
         and trace['capture_complete'] is True and trace['transport_behavior_changed'] is False
         and trace['http_attempts'] == len(trace['attempts']), 'case.http_trace_complete')
    rows = trace['attempts']
    need(len(rows) >= len(calls), 'case.http_attempt_coverage')
    for number, row in enumerate(rows, 1):
        target = directory / 'http' / f'attempt-{number:04d}'
        need(row['index'] == number and row['capture_complete'] is True, 'http.sequence_capture')
        metadata = auth_json(row['metadata_ref'], target / 'metadata.json')
        need(metadata == {k: v for k, v in row.items() if k != 'metadata_ref'},
             'http.metadata_join')
        auth(row['request'], target / 'request.bin')
        need(row['request']['sha256'] == row['request_sha256'], 'http.request_hash')
        path = urlsplit(row['endpoint']).path
        need(row['endpoint'] in {record['policy_endpoint'], record['text_endpoint'] + '/chat/completions'},
             'http.selected_endpoint')
        need(path in {'/v1/systemone', '/chat/completions'}, 'http.route')
        if row['response'] is None:
            need(row.get('transport_error') and row['http_status'] is None,
                 'http.transport_failure_evidence')
        else:
            auth(row['response'], target / 'response.bin')
            need(row['response']['sha256'] == row['response_sha256']
                 and type(row['http_status']) is int and 100 <= row['http_status'] <= 599,
                 'http.response_hash_status')
    return {'success': judged, 'policy_calls': policy_count,
            'helper_calls': len(calls) - policy_count, 'http': rows}


def audit_child(record, template_case, template, actor, index):
    case_id, number = CASES[index], index + 1
    target = RUN / 'handoff' / f'case-{number:02d}'
    output = CPU_OWNER / f'actual-{index + 31:02d}'
    need(template_case['owner_output'] == str(output)
         and record['owner_output'] == str(output), 'cpu.original_output_selection')
    service = auth_json(record['service_plan_ref'], target / 'service-plan.json')
    service_sha, nonce = record['service_plan_ref']['sha256'], template_case['nonce']
    binding(record, schema='jev-v7-matched-case-closed-v1', root_sha=PLAN_SHA,
            index=number, case_id=case_id, service_sha=service_sha, nonce=nonce)
    need(record['parent_identity'] == actor and record['controller_exit_code'] == 0
         and record['owner_completed'] is True
         and record['original_scoped_closure_positive'] is True
         and record['interrupted_for_cleanup'] is False, 'cpu.closed_status')
    need(load(target / 'closed.json') == record, 'cpu.handoff_closed_join')
    for key in ('case_index', 'case_id', 'root_plan_sha256', 'parent_identity', 'owner_output'):
        need(service[key] == record[key], 'cpu.service_case_join')
    need(service['handoff']['nonce'] == nonce
         and service['serialization'] == load(RUN / 'serialization.json'), 'cpu.shared_serialization')
    auth(service['serialization_ref'], RUN / 'serialization.json')
    spec = auth_json(record['command_spec_ref'], target / 'command-spec.json')
    # Independently reproduce the selected pump's pure runtime expansion.
    descriptor_ref = service['serialization_ref']
    expected_service = copy.deepcopy(template['service_plan'])
    expected_service.update(
        case_index=number, case_id=case_id, root_plan_sha256=PLAN_SHA,
        parent_identity=copy.deepcopy(actor), owner_output=str(output),
        serialization=load(RUN / 'serialization.json'),
        serialization_ref=copy.deepcopy(descriptor_ref),
        handoff={'ready_path': str(target / 'service-ready.json'),
                 'stop_path': str(target / 'service-stop.json'),
                 'closed_path': str(target / 'service-closed.json'), 'nonce': nonce})
    need(service == expected_service, 'cpu.exact_selected_service_expansion')
    expected_spec = copy.deepcopy(template['command_spec'])
    substitutions = {'{SERVICE_PLAN}': str(target / 'service-plan.json'),
                     '{SERVICE_PLAN_SHA256}': service_sha}
    for command in expected_spec['commands']:
        command['argv'] = [substitutions.get(value, value) for value in command['argv']]
    def relative_ref(value):
        path = auth(value)
        return {'path': path.relative_to(ROOT).as_posix(),
                'sha256': value['sha256'], 'size_bytes': pin(path)['size_bytes']}
    expected_spec['sources'] += [relative_ref(record['service_plan_ref']), relative_ref(descriptor_ref)]
    need(spec == expected_spec, 'cpu.exact_selected_command_expansion')
    cfg = auth_json(record['sealed_config_ref'], output / 'sealed-config.json')
    owner_receipt = auth_json(record['owner_receipt_ref'], output / 'receipt.json')
    closure = auth_json(record['closure_ref'], output / 'external-boundary-closure.json')
    terminal = auth_json(record['owner_terminal_ref'], target / 'controller-terminal.json')
    admission = auth_json(record['admission_ref'], output / 'admission.json')
    worker = auth_json(record['worker_receipt_ref'], output / 'v7-full-choice-service/worker-receipt.json')
    local = auth_json(record['service_closed_ref'], target / 'service-closed.json')
    need(cfg == owner_receipt['selected_config'] and cfg['output'] == str(output)
         and cfg['runtime_seconds'] == spec['runtime_seconds'] == 7200
         and cfg['selection'] == record['command_spec_ref']['path']
         and cfg['selection_sha256'] == record['command_spec_ref']['sha256']
         and owner_receipt['closure'] == closure and owner_receipt['completed'] is True
         and owner_receipt['primary_error'] is None and owner_receipt['secondary_errors'] == []
         and owner_receipt['sources_unchanged'] is True, 'cpu.original_owner_receipt')
    need(cfg['profile'] == owner_receipt['profile']
         == 'llama-cpp-cpu5-fast-same-cluster-4g-coexist-command-owned-v1'
         and cfg['unit'] == 'jev-d3-llama-cpp-cpu5-fast-4g-coexist-20261024-09.service',
         'cpu.original_owner_identity')
    need(terminal['schema'] == 'jev-v7-matched-controller-terminal-v1'
         and terminal['exit_code'] == 0 and terminal['controller_identity'] == record['controller_identity']
         and terminal['root_plan_sha256'] == PLAN_SHA and terminal['case_index'] == number
         and terminal['case_id'] == case_id, 'cpu.controller_terminal')
    expected_argv = [str(ROOT / '.venv/bin/python'), '-B', '-s',
                     str(CPU_OWNER / 'controller.py'), '--run', '--expected-entry-sha256',
                     template['controller_source']['sha256'], '--selection',
                     record['command_spec_ref']['path'], '--selection-sha256',
                     record['command_spec_ref']['sha256'], '--output', str(output)]
    need(terminal['argv'] == expected_argv, 'cpu.controller_exact_argv')
    selected = closure['selection']
    need(all(selected.get(k) == v for k, v in {
        'unit': cfg['unit'], 'nonce': cfg['nonce'], 'boot_id': cfg['boot_id'],
        'memory_max_bytes': 4 * 1024**3, 'runtime_seconds': 7200, 'stop_seconds': 3,
    }.items()) and closure['closure_mode'] == 'authenticated_stop'
         and closure['closure_errors'] == [], 'cpu.original_closure_binding')
    need(all(closure[k] is True for k in (
        'whole_unit_exit_confirmed', 'observed_cgroup_cohort_exit_confirmed',
        'post_stop_terminal_verified', 'cgroup_absent', 'sources_unchanged')),
         'cpu.original_positive_closure')
    need(all(closure[k] is False for k in (
        'no_escape_enforced', 'whole_tree_exit_confirmed', 'continuation_authorized', 'model_qualified')),
         'cpu.no_scope_inflation')
    scans = closure['two_process_scans']
    need(len(scans) == 2 and all(row['matches'] == row['unknown'] == [] for row in scans),
         'cpu.process_scans')
    need(closure['before_stop']['Description'] == cfg['nonce']
         and closure['before_stop']['InvocationID'] == closure['invocation_id']
         and RID_RE.fullmatch(closure['invocation_id'])
         and closure['before_stop']['MainPID'] == str(closure['actor']['pid']),
         'cpu.actual_invocation_actor')
    need(admission['config_sha256'] == record['sealed_config_ref']['sha256']
         and admission['nonce'] == cfg['nonce']
         and admission['invocation_id'] == closure['invocation_id']
         and stable(admission['actor']) == stable(closure['actor']), 'cpu.actual_admission')
    need(cfg['sources'] == cfg['base_sources'] + spec['sources']
         and len(cfg['sources']) <= 64 and len(spec['sources']) <= 58, 'cpu.source_union')
    for source in cfg['sources']:
        auth(source)
    binding(worker, schema='jev-v7-full-choice-service-receipt-v1', root_sha=PLAN_SHA,
            index=number, case_id=case_id, service_sha=service_sha, nonce=nonce)
    binding(local, schema='jev-v7-full-choice-service-closed-v1', root_sha=PLAN_SHA,
            index=number, case_id=case_id, service_sha=service_sha, nonce=nonce)
    for value in (worker, local):
        need(value['owner_output'] == str(output) and value['service_success'] is True
             and value['server_thread_drained'] is True
             and value['final_native_artifact_authentication'] is True
             and value['sources_unchanged'] is True, 'cpu.service_positive_closure')
    need(worker['stage'] == 'service_complete_pending_external_owner_closure'
         and worker['cleanup_errors'] == [] and worker['pending_native_or_server'] is False
         and worker['helper_model_loaded'] is False and worker['helper_model_calls'] == 0
         and worker['native_state']['model_loads'] == 1, 'cpu.one_genuine_model_no_helper')
    command_results = owner_receipt['state']['actor_exit']['command_results']
    need(len(command_results) == 1 and command_results[0]['returncode'] == 0
         and command_results[0]['argv'] == [v.replace('{OUTPUT}', str(output))
                                           for v in spec['commands'][0]['argv']],
         'cpu.actual_worker_command')
    observed, actual_worker = command_results[0]['observed_identity'], worker['worker']
    need(stable(observed['process']) == stable(actual_worker['process'])
         and all(observed[k] == actual_worker[k] for k in ('exe', 'argv', 'cwd')),
         'cpu.actual_worker_identity')
    process_absent(record['controller_identity'])
    process_absent(actual_worker['process'])
    return {'worker_pid': actual_worker['process']['pid'], 'output': output,
            'service': service, 'spec': spec}


def audit_serialization(actor, children):
    descriptor = load(RUN / 'serialization.json')
    need(set(descriptor) == {'lock_path', 'journal_path', 'expected_stat', 'nonce'}
         and RID_RE.fullmatch(descriptor['nonce']), 'serialization.descriptor')
    for key, name in (('lock_path', 'lock'), ('journal_path', 'journal')):
        path = ordinary(descriptor[key])
        need(path == RUN / ('forward.lock' if name == 'lock' else 'forward-journal.jsonl'),
             'serialization.selected_paths')
        info = path.stat()
        need({'device': info.st_dev, 'inode': info.st_ino} == descriptor['expected_stat'][name],
             'serialization.original_inodes')
    blob = raw(RUN / 'forward-journal.jsonl', 16 * 1024**2)
    need(not blob or blob.endswith(b'\n'), 'serialization.partial_jsonl')
    active, last_time = None, -1
    counts = Counter()
    allowed = {'nf4': {actor['pid']}, 'decision3': {row['worker_pid'] for row in children}}
    for line in blob.splitlines():
        event = decode(line)
        need(set(event) == {'event', 'kind', 'pid', 'thread', 'monotonic_ns', 'nonce'}
             and event['nonce'] == descriptor['nonce']
             and event['kind'] in allowed and event['pid'] in allowed[event['kind']]
             and all(type(event[k]) is int and event[k] > 0
                     for k in ('pid', 'thread', 'monotonic_ns'))
             and event['monotonic_ns'] >= last_time, 'serialization.event_binding')
        last_time = event['monotonic_ns']
        identity = (event['kind'], event['pid'], event['thread'])
        if event['event'] == 'acquired':
            need(active is None, 'serialization.overlapping_forward')
            active = identity
        elif event['event'] == 'completed':
            need(active == identity, 'serialization.unmatched_completion')
            active = None
            counts[(event['kind'], event['pid'])] += 1
        else:
            need(False, 'serialization.unknown_event')
    need(active is None, 'serialization.unfinished_forward')
    return counts


def audit_http_and_forwards(cases, children, counts, actor):
    native = OUT / 'native'
    nf4_exchanges = {}
    for path in (native / 'http').glob('*/exchange.json'):
        exchange = load(path)
        rid = exchange.get('api_request_id') or exchange['request_id']
        need(RID_RE.fullmatch(rid) and rid not in nf4_exchanges, 'nf4.exchange_id')
        for key, filename in (('request', 'request.bin'), ('response', 'response.bin')):
            need(exchange[key] == pin(path.parent / filename), 'nf4.exchange_bytes')
        nf4_exchanges[rid] = (exchange, path.parent)
    server = load(native / 'server-index.json')
    need(server['all_requests_sealed'] is True and server['pending_request_ids'] == []
         and server['errors'] == [] and server['local_service_thread_drained'] is True,
         'nf4.server_evidence_closed')
    for item in server['requests']:
        need(item['sealed'] is True and item['write_failed'] is False, 'nf4.request_sealed')
        path = native / 'server' / item['path']
        need(pin(path) == {'sha256': item['sha256'], 'size_bytes': item['bytes']},
             'nf4.server_request_hash')
        events = [decode(line) for line in raw(path).splitlines()]
        need(len(events) == item['events'] and events[-1]['event'] == item['last_event']
             and item['last_event'] in {'server_returned', 'server_failed'}
             and all(row['request_id'] == item['request_id'] for row in events)
             and all(events[i]['sequence'] < events[i + 1]['sequence']
                     for i in range(len(events) - 1)), 'nf4.server_event_sequence')
    successful_nf4 = 0
    policy_http, helper_http, transport_failures, non200 = 0, 0, 0, 0
    for arm in ('baseline', 'candidate'):
        for index, case in enumerate(cases[arm]):
            for row in case['http']:
                path = urlsplit(row['endpoint']).path
                policy = path == '/v1/systemone'
                policy_http += int(policy)
                helper_http += int(not policy)
                if row['response'] is None:
                    transport_failures += 1
                    continue
                non200 += int(row['http_status'] != 200)
                rid = row['server_request_id']
                need(type(rid) is str and RID_RE.fullmatch(rid), 'http.response_server_id')
                if arm == 'candidate' and policy:
                    server_path = children[index]['output'] / 'v7-full-choice-service/server-evidence' / rid
                    exchange = load(server_path / 'exchange.json')
                    need(exchange['request_id'] == rid and exchange['status'] == row['http_status']
                         and exchange['captured_body_sha256'] == row['request_sha256']
                         and exchange['response_sha256'] == row['response_sha256']
                         and pin(server_path / 'request.bin')['sha256'] == row['request_sha256']
                         and pin(server_path / 'response.bin')['sha256'] == row['response_sha256'],
                         'cpu.client_server_bytes_join')
                    if row['http_status'] == 200:
                        need(exchange['observer_errors'] == [] and exchange['primary_error'] is None
                             and exchange['candidate_constructed'] is True
                             and exchange['candidate_retained'] is True, 'cpu.authorized_200')
                else:
                    need(rid in nf4_exchanges, 'nf4.client_exchange_missing')
                    exchange, _directory = nf4_exchanges[rid]
                    need(exchange['status'] == row['http_status']
                         and exchange['request']['sha256'] == row['request_sha256']
                         and exchange['response']['sha256'] == row['response_sha256'],
                         'nf4.client_server_bytes_join')
                    successful_nf4 += int(row['http_status'] == 200)
    candidate_predicts, candidate_encoder_forwards = 0, 0
    for child in children:
        predicts = 0
        evidence_root = child['output'] / 'v7-full-choice-service/server-evidence'
        for directory in evidence_root.iterdir():
            if directory.name == '.gitignore':
                need(raw(directory) == b'*\n', 'cpu.evidence_ignore_file')
                continue
            need(directory.is_dir() and RID_RE.fullmatch(directory.name), 'cpu.evidence_directory')
            chunks, completed = [], False
            events = sorted(directory.glob('event-*.json'))
            need(len(events) <= 10000, 'cpu.event_population_bound')
            for path in events:
                event = load(path)
                if event.get('event') == 'chunk_returned':
                    chunk = event['data']
                    rows, width = chunk['rows'], chunk['padded_width']
                    need(type(rows) is int and 1 <= rows <= 4 and type(width) is int
                         and 1 <= width <= 16384 and chunk['status'] == 'projected'
                         and chunk['first_layer_events'] == [[rows, width, 1024]]
                         and chunk['final_norm_events'] == [[rows, width, 1024]]
                         and chunk['projection']['target_logits_bitwise_unchanged'] is True,
                         'cpu.genuine_encoder_shape_projection')
                    chunks.append(chunk)
                if event.get('event') == 'completed':
                    data = event['data']
                    if data.get('status') == 'completed' and data.get('backend_predict_calls') == 1:
                        completed = True
            exchange = load(directory / 'exchange.json')
            if exchange['status'] == 200:
                need(chunks and completed, 'cpu.successful_response_without_native_forward')
            if chunks:
                need(completed, 'cpu.forward_without_adapter_completion')
                predicts += 1
                candidate_encoder_forwards += len(chunks)
        need(counts[('decision3', child['worker_pid'])] == predicts,
             'cpu.actual_predicts_serialization_join')
        candidate_predicts += predicts
    need(counts[('nf4', actor['pid'])] >= successful_nf4,
         'nf4.successful_http_without_complete_causal_lm_forward')
    return {'policy_http_attempts': policy_http, 'helper_http_attempts': helper_http,
            'transport_failure_attempts': transport_failures, 'non200_response_attempts': non200,
            'nf4_completed_forward_leases': counts[('nf4', actor['pid'])],
            'candidate_completed_predict_leases': candidate_predicts,
            'candidate_encoder_forwards': candidate_encoder_forwards,
            'nf4_forward_attribution': 'authenticated actor PID aggregate; per-case client/server bytes joined',
            'forward_overlap_observed': False, 'unfinished_forward_lease': False,
            'helper_runtime_qualification_claimed': False}


def main():
    report = {'schema': 'root-matched-v7-terminal02-independent-audit-v1',
              'target_attempt': 'matched-02', 'root_plan_sha256': PLAN_SHA,
              'status': 'audit_required', 'complete_score': None, 'audit_required': [],
              'scope': 'original known4+public6; public development cases; fresh browsers; no official JevBench score',
              'new_unseen_evaluation_claim': False, 'competitor_win_claim': False,
              'official_jevbench_heaven_score': None, 'official_jevbench_coherence_score': None,
              'raw_task_data_in_report': False, 'final_zero_oom_claim': False}
    stage = 'selection'
    try:
        need(pin(BASE / 'root-matched-plan-03.json')['sha256'] == PLAN_SHA, 'selection.plan_hash')
        need(pin(BASE / 'owner_released_v2.py')['sha256'] == OWNER_SHA, 'selection.owner_hash')
        plan = load(BASE / 'root-matched-plan-03.json')
        need(plan['attempt'] == '02' and ROOT / plan['run_directory'] == RUN
             and ROOT / plan['output'] == OUT, 'selection.output_paths')
        stage = 'late_failure_markers'
        for name in ('root-finalization-error.json', 'finalization-failure.json'):
            need(not any(RUN.rglob(name)), 'terminal.late_failure_marker')
        receipt = load(OUT / 'receipt.json')
        stage = 'outer_closure'
        result = audit_outer(plan, receipt)
        stage = 'output_inventories'
        report['inventories'] = {'campaign': inventory(OUT), 'native': inventory(OUT / 'native')}
        stage = 'original_nf4_receipt'
        native = auth_json(receipt['native_receipt_ref'], OUT / 'native/receipt.json')
        need(native['stage'] == 'complete_pending_external_closure'
             and native['attempts'] == native['planned_denominator'] == 10
             and native['prompt_format'] == 'shared-elements-v2'
             and native['model_execution'] is True and native['metadata_authenticated'] is True
             and native['server_thread_drained'] is True
             and native['source_verification']['success'] is True, 'nf4.original_receipt')
        auth(receipt['helper_prepared_ref'], OUT / 'native/helper-prepared.json')
        loaded = auth_json(receipt['nf4_model_loaded_ref'], OUT / 'native/model-loaded.json')
        need(loaded['revision'] == 'a66f836b56605039fe040f330180e336d19b3362'
             and loaded['quantization'] == 'nf4' and loaded['max_context'] == 8192,
             'nf4.actual_loaded_model_selection')
        stage = 'original_twenty_cases'
        arms = receipt['arms']
        need([row['arm'] for row in arms] == ['baseline', 'candidate'], 'arms.identities')
        cases, scores = {}, {}
        helper_endpoints = set()
        for arm in arms:
            name, records = arm['arm'], arm['records']
            need(arm['planned_denominator'] == arm['attempted_cases'] == arm['denominator'] == 10
                 and arm['known4_attempts'] == 4 and arm['public6_attempts'] == 6
                 and arm['infrastructure_complete'] is True and len(records) == 10
                 and [row['case_id'] for row in records] == list(CASES), 'arms.exact_denominator')
            cases[name] = [audit_case(row, name, i) for i, row in enumerate(records)]
            successes = sum(row['success'] for row in cases[name])
            need(type(arm['successes']) is int and arm['successes'] == successes
                 and arm['success_rate'] == successes / 10
                 and arm['completed_cases'] == 10 and arm['failed_tasks'] == 10 - successes,
                 'arms.original_score_recomputation')
            scores[name] = {'successes': successes, 'attempts': 10, 'success_rate': successes / 10,
                            'known4_successes': sum(row['success'] for row in cases[name][:4]),
                            'public6_successes': sum(row['success'] for row in cases[name][4:])}
            helper_endpoints.update(row['text_endpoint'] for row in records)
        need(len(helper_endpoints) == 1, 'arms.same_resident_helper_endpoint')
        need(native['successes'] == scores['baseline']['successes'], 'nf4.baseline_success_join')
        need([(row['group'], row['denominator'], row['summary']['attempts']) for row in native['groups']]
             == [('known4', 4, 4), ('public6', 6, 6)], 'nf4.baseline_group_denominators')
        native_outcomes = [outcome for group in native['groups'] for outcome in group['summary']['outcomes']]
        for i, outcome in enumerate(native_outcomes):
            summary = auth_json(arms[0]['records'][i]['summary_ref'])
            need(summary['outcomes'][0] == outcome, 'nf4.exact_original_group_outcomes')
        need(len(native_outcomes) == 10, 'nf4.original_group_population')
        stage = 'ten_original_cpu_closures'
        template = auth_json(plan['matched']['cpu_template'])
        need(len(template['cases']) == 10, 'cpu.template_denominator')
        pump, records = result['candidate_pump_closure'], arms[1]['children']
        need(pump['children'] == records and len(records) == 10
             and pump['active_controller_absent'] is True and pump['independently_closed'] is True
             and pump['complete_ten_cases'] is True and pump['failure'] is None,
             'cpu.independent_pump_closure')
        children = [audit_child(record, template['cases'][i], template, result['actor'], i)
                    for i, record in enumerate(records)]
        stage = 'actual_forward_serialization'
        counts = audit_serialization(result['actor'], children)
        stage = 'actual_http_forward_joins'
        report['execution'] = audit_http_and_forwards(cases, children, counts, result['actor'])
        report['complete_score'] = scores
        report['independently_closed_cpu_cases'] = 10
        report['status'] = 'root_audited_complete_development_comparison'
    except AuditRequired as error:
        report['audit_required'].append({'stage': stage, 'code': str(error)})
    except BaseException as error:
        # Never print exception messages: third-party field names, paths and
        # actual payload values are not an approved public diagnostic channel.
        report['audit_required'].append({'stage': stage,
                                         'code': 'schema_or_read_not_established',
                                         'exception_type': type(error).__name__})
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report['complete_score'] is not None else 2


if __name__ == '__main__':
    raise SystemExit(main())
