"""Root-only offline authentication of a failed run's valid cache prefix."""

import datetime
import hashlib
import json
import math
import os
import resource
import sys
from pathlib import Path

resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
ROOT = Path('/home/albert_hsieh/Gemma-4-12B-Unified-System-One')
OLD = ROOT / 'artifacts/jev-runtime/decision3-publisher-batch4-cpu-both-20261024-06'
PRIOR = ROOT / 'artifacts/jev-runtime/llama-cpp-owner-20261024-07/actual-04'
WORK = PRIOR / 'edge-both'
NEW = ROOT / 'artifacts/jev-runtime/decision3-coherence-resume-cpu4-20261024-01'
DEST = NEW / 'root-prefix-inputs-01'
PROFILE = 'decision3-edge-gx10-publisher-cpu2g-coexist-batch4-both-stage-completions-ece-schema-20261024-v1'
BENCH_SHA = 'b5caa794d864917a637034bd1cebe0487c9eec0e5c3c423835d24fca4f397021'


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(compact(value)).hexdigest()


def pin(path):
    path = Path(path)
    assert path.resolve().is_relative_to(ROOT)
    before = path.stat()
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(1024**2):
            h.update(block)
    after = path.stat()
    assert (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    return {'path': str(path), 'sha256': h.hexdigest(), 'size_bytes': after.st_size}


def load(path):
    path = Path(path)
    assert path.stat().st_size < 4 * 1024**2
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite')))


def check(record):
    path = Path(record['path'])
    path = path if path.is_absolute() else ROOT / path
    actual = pin(path)
    assert actual['sha256'] == record['sha256']
    assert actual['size_bytes'] == record.get('size_bytes', record.get('size', actual['size_bytes']))
    return path


def put(name, value):
    path = DEST / name
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode() + b'\n'
    assert len(data) < 1024**2
    with path.open('xb') as f:
        os.chmod(path, 0o600)
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    return pin(path)


def main():
    assert not DEST.exists()
    plan_path = OLD / 'edge-plan.json'
    plan = load(plan_path)
    assert plan['profile'] == PROFILE and plan['threads'] == 4
    assert plan['cpu_affinity'] == [16, 17, 18, 19] and plan['batch_size'] == 4
    assert plan['model']['model_sha256'] == '6e00d04abbdbdf6d35c9e12065769d9edaacc3bfe9404583891797718d239dc8'
    benchmark_path = ROOT / 'artifacts/jev-runtime/decision3-publisher-lite-cpu-20261023-01/benchmark-source-plan.json'
    assert pin(benchmark_path)['sha256'] == BENCH_SHA
    benchmark = load(benchmark_path)
    for row in benchmark['source_files'] + benchmark['existing_dataset_metadata']:
        check(row)
    cfg = load(PRIOR / 'sealed-config.json')
    receipt = load(PRIOR / 'receipt.json')
    closure = load(PRIOR / 'external-boundary-closure.json')
    assert receipt['selected_config'] == cfg and receipt['closure'] == closure
    assert receipt['completed'] is False and receipt['sources_unchanged'] is True
    assert cfg['profile'] == 'llama-cpp-cpu4-2g-coexist-command-owned-v1'
    assert closure['closure_mode'] == 'authenticated_stop' and closure['closure_errors'] == []
    assert all(closure[k] is True for k in ('whole_unit_exit_confirmed', 'observed_cgroup_cohort_exit_confirmed',
        'post_stop_terminal_verified', 'cgroup_absent', 'sources_unchanged'))
    assert all(closure[k] is False for k in ('model_qualified', 'continuation_authorized', 'whole_tree_exit_confirmed'))
    assert len(closure['two_process_scans']) == 2
    assert all(not row['matches'] and not row['unknown'] for row in closure['two_process_scans'])
    assert len(cfg['sources']) == 59
    for row in cfg['sources']:
        check(row)
    coherence = Path(benchmark['coherence']['root'])
    sys.path.insert(0, str(coherence / 'src'))
    import jevbench as jb
    from jevbench.backends import _check_answers
    from jevbench.plan import decode_requests
    from jevbench.taxonomy import scored
    assert Path(jb.__file__).resolve() == (coherence / 'src/jevbench/__init__.py').resolve()
    suite = jb.load_suite('jevbench-mini')
    assert len(suite.cases) == 240 and len(suite.plan) == 1200
    assert suite.plan.seed == 0 and suite.plan.trials == 1
    requests = {}

    def add(state, questions):
        fp = digest({'b': PROFILE, 's': state, 'q': questions, 'r': 0})
        payload_hash = digest({'state': state, 'questions': questions})
        if fp in requests:
            assert requests[fp]['payload_sha256'] == payload_hash
        else:
            requests[fp] = {'request_fingerprint': fp, 'payload_sha256': payload_hash,
                            'state': state, 'questions': questions}

    for case in suite.cases:
        add(case.state, case.questions)
    for case in suite.cases:
        for relation in scored():
            assert suite.plan.covers(case.id, relation)
            for row in suite.plan.rows(case.id, relation):
                assert 'states' not in row['params']
                for questions in decode_requests(case, row['requests']).values():
                    add(case.state, questions)
    assert len(requests) == 1248
    cache = sorted((WORK / 'coherence-cache').iterdir())
    assert all(p.is_file() and not p.is_symlink() and p.suffix == '.json' and len(p.stem) == 64 for p in cache)
    n = len(cache)
    assert n == 1234 and {p.stem for p in cache} == set(list(requests)[:n])
    progress = load(WORK / 'native-evidence/progress.json')
    assert progress['completed_calls'] == progress['raw_native_returns'] == n + 231
    assert progress['started_calls'] == n + 232
    entries = []
    for ordinal, (fp, request) in enumerate(list(requests.items())[:n]):
        index = ordinal + 231
        evidence = WORK / 'native-evidence'
        prepared_path = evidence / f'{index:05d}-prepared.json'
        raw_path = evidence / f'{index:05d}-raw_returned_before_validation.json'
        validated_path = evidence / f'{index:05d}-validated.json'
        assert not (evidence / f'{index:05d}-failed.json').exists()
        prepared, raw, validated = map(load, (prepared_path, raw_path, validated_path))
        assert all(v['index'] == index and v['stage'] == 'jevbench-mini' for v in (prepared, raw, validated))
        assert prepared['event'] == 'prepared'
        assert prepared['value']['original_question_order'] == list(request['questions'])
        lengths = prepared['value']['lengths']
        assert set(lengths) == set(request['questions'])
        assert all(type(x) is int and 0 < x <= 16384 for x in lengths.values())
        assert raw['event'] == 'raw_returned_before_validation' and raw['value']['model'] == 'd3-edge'
        assert validated['status'] == 'validated_pending_durable_evidence'
        assert math.isfinite(validated['validation_seconds']) and 0 <= validated['validation_seconds'] <= 180
        before, after = validated['before_owned_counters'], validated['after_owned_counters']
        assert before['cgroup'] == after['cgroup'] == '0::' + closure['actor']['cgroup']
        assert before['observed_monotonic'] <= after['observed_monotonic']
        assert all(after['counters']['memory.events'][key] == 0 for key in ('oom', 'oom_kill', 'oom_group_kill'))
        cache_path = WORK / 'coherence-cache' / (fp + '.json')
        answer = load(cache_path)
        assert compact(raw['value']['answers']) == compact(answer)
        _check_answers(answer, request['questions'], PROFILE)
        probabilities = {qid: value['probabilities'] if 'probabilities' in value else value['noul']
                         for qid, value in answer.items()}
        entries.append({'request_fingerprint': fp, 'payload_sha256': request['payload_sha256'],
                        'answers_sha256': digest(answer), 'probabilities_sha256': digest(probabilities),
                        'cache_ref': pin(cache_path), 'raw_return_ref': pin(raw_path),
                        'validated_ref': pin(validated_path), 'prepared_ref': pin(prepared_path),
                        'original_call_index': index})
    DEST.mkdir(parents=True, mode=0o700)
    request_ref = put('request-manifest.json', {'schema': 'decision3-original-coherence1248-request-manifest-v1',
        'requests': [{key: row[key] for key in ('request_fingerprint', 'payload_sha256')} for row in requests.values()]})
    shards = [put(f'accepted-{start//256+1:02d}.json', {'schema': 'root-accepted-decision3-coherence-cache-shard-v1',
                'entries': entries[start:start+256]}) for start in range(0, n, 256)]
    union_ref = put('prior-source-union.json', {'schema': 'original-cpu4-selected-source-union-v1',
        'sources': cfg['sources'], 'sealed_config_ref': pin(PRIOR/'sealed-config.json')})
    audit_ref = put('root-prefix-audit.json', {'schema': 'root-original-cpu4-cache-prefix-audit-v1',
        'status': 'ROOT_ACCEPTED_CACHE_PREFIX_ONLY', 'accepted_prior_count': n, 'planned_requests': 1248,
        'missing_requests': 1248-n, 'cache_matches_exact_original_request_prefix': True,
        'every_cached_answer_equals_original_raw_native_answer': True,
        'every_cached_call_has_original_prepared_validated_and_counter_evidence': True,
        'original_selected_source_union_unchanged': True, 'original_scope_closure_positive': True,
        'original_campaign_success': False, 'original_model_qualified': False,
        'new_resume_complete_score': None, 'raw_payloads_or_probabilities_in_this_audit': False,
        'original_receipt_ref': pin(PRIOR/'receipt.json'), 'progress_ref': pin(WORK/'native-evidence/progress.json')})
    top = {'schema': 'root-accepted-decision3-cpu4-coherence-resume-inputs-v1',
        'status': 'ROOT_ACCEPTED_ORIGINAL_CPU4_CACHE_PREFIX', 'original_backend_identity': PROFILE,
        'accepted_prior_count': n, 'request_manifest_ref': request_ref, 'accepted_entry_shards': shards,
        'root_prefix_audit_ref': audit_ref, 'prior_closure_ref': pin(PRIOR/'external-boundary-closure.json'),
        'prior_source_union_ref': union_ref, 'original_full06_plan_ref': pin(plan_path),
        'benchmark_source_plan_ref': pin(benchmark_path),
        'settings': {'suite': 'jevbench-mini', 'seed': 0, 'repeats': 1, 'trials': 1, 'concurrency': 1,
                     'tolerance': 0.05, 'min_n': 5, 'bootstrap': 1000, 'diagnostics': False}}
    top_ref = put('root-accepted-inputs.json', top)
    print(json.dumps({'schema': 'root-coherence-prefix-audit-summary-v1', 'accepted_prefix_count': n,
        'missing_requests': 1248-n, 'planned_requests': 1248, 'root_accepted_input_ref': top_ref,
        'original_campaign_remains_failed': True, 'full_coherence_score': None}))


if __name__ == '__main__':
    main()
