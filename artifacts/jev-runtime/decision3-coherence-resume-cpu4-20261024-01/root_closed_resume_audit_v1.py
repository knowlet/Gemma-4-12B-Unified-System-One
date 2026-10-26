"""Root offline audit: original CPU4 resumed coherence, never model inference."""
import hashlib
import importlib.util
import json
import math
import os
import resource
import sys
from pathlib import Path

resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
resource.setrlimit(resource.RLIMIT_CPU, (110, 110))
ROOT = Path('/home/albert_hsieh/Gemma-4-12B-Unified-System-One')
BASE = ROOT / 'artifacts/jev-runtime/decision3-coherence-resume-cpu4-20261024-01'
ACTUAL = ROOT / 'artifacts/jev-runtime/llama-cpp-owner-20261024-07/actual-06'
OUTPUT = ACTUAL / 'coherence-resume'


def load(path):
    return json.loads(Path(path).read_text())


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def ref(path):
    path = Path(path)
    blob = path.read_bytes()
    return {'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(blob).hexdigest(),
            'size_bytes': len(blob)}


def auth(row):
    path = Path(row['path'])
    path = path if path.is_absolute() else ROOT / path
    assert path.is_relative_to(ROOT) and not path.is_symlink()
    actual = ref(path)
    assert actual['sha256'] == row['sha256'] and actual['size_bytes'] == row['size_bytes']
    return path


def save(path, value):
    with path.open('x') as stream:
        os.chmod(path, 0o600)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def main():
    terminal = load(BASE / 'root-execution-02.json')
    assert terminal['exit_code'] == 0
    selection = load(auth(terminal['selection']))
    spec = load(auth(terminal['selected_spec']))
    assert selection['actual_output'] == str(ACTUAL) and len(spec['sources']) == 56
    cfg, owner = load(ACTUAL / 'sealed-config.json'), load(ACTUAL / 'receipt.json')
    closure = load(ACTUAL / 'external-boundary-closure.json')
    assert owner['completed'] and owner['primary_error'] is None and owner['secondary_errors'] == []
    assert owner['sources_unchanged'] is True and owner['selected_config'] == cfg and owner['closure'] == closure
    assert cfg['profile'] == 'llama-cpp-cpu4-2g-coexist-command-owned-v1'
    assert cfg['unit'] == 'jev-d3-llama-cpp-cpu4-2g-coexist-20261024-07.service'
    assert cfg['runtime_seconds'] == 10800 and len(cfg['sources']) == 61
    for row in cfg['sources']:
        auth(row)
    assert closure['closure_mode'] == 'authenticated_stop' and closure['closure_errors'] == []
    assert all(closure[k] is True for k in ('whole_unit_exit_confirmed', 'observed_cgroup_cohort_exit_confirmed',
        'post_stop_terminal_verified', 'cgroup_absent', 'sources_unchanged'))
    assert all(closure[k] is False for k in ('model_qualified', 'continuation_authorized', 'whole_tree_exit_confirmed'))
    assert len(closure['two_process_scans']) == 2
    assert all(not row['matches'] and not row['unknown'] for row in closure['two_process_scans'])
    commands = owner['state']['actor_exit']['command_results']
    assert len(commands) == 1 and commands[0]['returncode'] == 0
    assert commands[0]['argv'] == [arg.replace('{OUTPUT}', str(ACTUAL)) for arg in spec['commands'][0]['argv']]
    assert closure['selection']['memory_max_bytes'] == 2147483648
    state = load(OUTPUT / 'state.json')
    assert state['stage'] == 'complete_pending_original_owner07_external_closure'
    assert state['model_loads'] == 1 and state['accepted_prior_count'] == 1234 and state['new_completed_count'] == 14
    assert state['request_population'] == 1248 and state['cuda_initialized'] is False
    assert state['cross_invocation_resume'] is True and state['original_full06_remains_failed'] is True
    assert state['fresh_full06_success'] is False and state['final_full_native_and_input_authentication'] is True
    assert state['native_provenance']['model_sha256'] == '6e00d04abbdbdf6d35c9e12065769d9edaacc3bfe9404583891797718d239dc8'
    assert state['actual_environment']['OMP_NUM_THREADS'] == '4'
    assert state['actual_environment']['CUDA_VISIBLE_DEVICES'] == ''
    inputs = load(auth(state['manifest_ref']))
    assert inputs['accepted_prior_count'] == 1234
    request_rows = load(auth(inputs['request_manifest_ref']))['requests']
    assert len(request_rows) == 1248
    inventory = load(OUTPUT / 'cache-inventory.json')['requests']
    assert len(inventory) == 1248 and len(set(row['path'] for row in inventory)) == 1248
    assert [Path(row['path']).stem for row in inventory] == [row['request_fingerprint'] for row in request_rows]
    for row in inventory:
        path = auth(row)
        assert path.parent == OUTPUT / 'coherence-cache'
    assert {p.name for p in (OUTPUT / 'coherence-cache').iterdir()} == {row['request_fingerprint'] + '.json' for row in request_rows}
    prefix_count = 0
    for shard_ref in inputs['accepted_entry_shards']:
        shard = load(auth(shard_ref))
        for entry in shard['entries']:
            old = auth(entry['cache_ref'])
            new = OUTPUT / 'coherence-cache' / (entry['request_fingerprint'] + '.json')
            assert old.read_bytes() == new.read_bytes()
            for key in ('raw_return_ref', 'validated_ref', 'prepared_ref'):
                auth(entry[key])
            prefix_count += 1
    assert prefix_count == 1234
    progress = load(OUTPUT / 'native-evidence/progress.json')
    assert progress['started_calls'] == progress['completed_calls'] == progress['raw_native_returns'] == 14
    # Original TimedPredictor durably writes progress before its finally clears
    # the in-memory active pointer; the final durable entry is completed call13.
    assert progress['stage'] == 'jevbench-mini'
    assert progress['active_call']['index'] == 13 and progress['active_call']['status'] == 'complete'
    for index, row in enumerate(request_rows[1234:]):
        evidence = OUTPUT / 'native-evidence'
        prepared = load(evidence / f'{index:05d}-prepared.json')
        raw = load(evidence / f'{index:05d}-raw_returned_before_validation.json')
        validated = load(evidence / f'{index:05d}-validated.json')
        assert not (evidence / f'{index:05d}-failed.json').exists()
        assert all(v['index'] == index and v['stage'] == 'jevbench-mini' for v in (prepared, raw, validated))
        answer = load(OUTPUT / 'coherence-cache' / (row['request_fingerprint'] + '.json'))
        assert compact(raw['value']['answers']) == compact(answer)
        lengths = prepared['value']['lengths']
        assert prepared['value']['original_question_order'] == list(answer) and set(lengths) == set(answer)
        assert all(type(n) is int and 0 < n <= 16384 for n in lengths.values())
        assert raw['value']['usage'] == {'input_tokens': sum(lengths.values()), 'output_tokens': 0}
        assert raw['value']['model'] == 'd3-edge' and validated['status'] == 'validated_pending_durable_evidence'
        assert 0 <= validated['validation_seconds'] <= 180
        for key in ('before_owned_counters', 'after_owned_counters'):
            sample = validated[key]
            assert sample['cgroup'] == '0::' + closure['actor']['cgroup']
            assert all(sample['counters']['memory.events'][k] == 0 for k in ('oom', 'oom_kill', 'oom_group_kill'))
    report = load(OUTPUT / 'coherence-mini.json')
    assert report['result']['requests'] == 0 and report['scores']['overall'] == state['official_score']
    assert report['scores']['n_cases'] == 240 and report['scores']['n_checks'] == 50
    assert math.isfinite(state['official_score']) and 0 <= state['official_score'] <= 1
    sys.path.insert(0, str(ROOT / 'scripts'))
    import summarize_jevbench_campaign as audit
    source = ROOT / 'artifacts/jevbench/20261022-local-score-01/upstream/coherence'
    replay = audit.replay_coherence(report, OUTPUT / 'coherence-cache', source)
    assert not {'torch', 'transformers', 'd3_runtime'}.intersection(sys.modules)
    private = {'schema': 'root-closed-resumed-coherence-actual06-full-audit-v1',
               'terminal_ref': ref(BASE / 'root-execution-02.json'), 'owner_receipt_ref': ref(ACTUAL / 'receipt.json'),
               'closure_ref': ref(ACTUAL / 'external-boundary-closure.json'), 'state_ref': ref(OUTPUT / 'state.json'),
               'report_ref': ref(OUTPUT / 'coherence-mini.json'), 'cache_inventory_ref': ref(OUTPUT / 'cache-inventory.json'),
               'independent_official_replay': replay, 'original_1234_byte_identical': True,
               'new14_native_returns_equal_cache': True, 'original_failed_campaign_remains_failed': True}
    private_path = BASE / 'root-closed-resumed-coherence-actual06-full-review-01.json'
    save(private_path, private)
    public = {'schema': 'root-accepted-resumed-coherence-actual06-v1',
        'status': 'ACCEPTED_COMPLETE_ORIGINAL_CPU4_CROSS_INVOCATION_RESUME',
        'private_full_audit_ref': ref(private_path), 'executed_git_commit': terminal['executed_git_commit'],
        'complete_request_population': 1248, 'authenticated_unchanged_prior_answers': 1234,
        'new_native_callbacks': 14, 'new_native_raw_returns': 14, 'actual_constructor_count': 1,
        'new_owner_completed': True, 'new_authenticated_original_scope_closure_positive': True,
        'new_original_source_union_unchanged': True, 'CPU_threads': 4, 'native_batch': 4,
        'memory_max_bytes': 2147483648, 'cuda_initialized': False, 'callback_deadline_seconds': 180,
        'new_invocation_wall_seconds': terminal['wall_seconds'], 'cross_invocation_resume': True,
        'original_failed_full06_requalified': False, 'full_fresh_1248_execution_claim': False,
        'coherence_score_0_to_1': state['official_score'], 'coherence_score_0_to_100': 100 * state['official_score'],
        'official_interval_0_to_1': state['official_interval'], 'cases': 240, 'relations': 50,
        'planned_tests': 1200, 'independent_official_replay_passed': True,
        'replay_raw_cache_files': replay['raw_cache_files'],
        'raw_cache_inventory_sha256': replay['raw_cache_inventory_sha256'],
        'official_outcomes_comparison': replay['deterministic_outcome_comparison'],
        'Heaven_accuracy_reported_separately': True, 'browser_score': None,
        'cross_model_win': False, 'raw_payloads_or_probabilities_in_this_record': False}
    save(BASE / 'root-accepted-coherence-actual06-summary-01.json', public)
    print(json.dumps(public, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
