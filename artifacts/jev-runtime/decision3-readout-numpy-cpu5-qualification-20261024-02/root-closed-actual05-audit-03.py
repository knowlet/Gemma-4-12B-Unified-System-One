"""Read-only root audit of the closed CPU5 smoke; no model/scorer execution."""

import argparse
import array
import ast
import struct
import sys
from datetime import datetime
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
HERE = Path(__file__).resolve().parent
OWNER = ROOT / "artifacts/jev-runtime/llama-cpp-owner-20261024-09"
OUT = OWNER / "actual-05"
UNIT = "jev-d3-llama-cpp-cpu5-fast-4g-coexist-20261024-09.service"
PROFILE = "llama-cpp-cpu5-fast-same-cluster-4g-coexist-command-owned-v1"
PLAN_SHA = "39fb916a3adeb093d8900d3d6efa56a772147e65b35ae8edc4260a34759a27b2"
SPEC_SHA = "e0982d1489e8b4d6778c32a964c5c0b61490d9b9a64860582e448f483007b7ab"
CONTROLLER_SHA = "81846efe2ee0357da07283c1a77a606283a920cbddef668c8f62def291057dee"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ref(path):
    return {
        "path": str(path.relative_to(ROOT)),
        "size_bytes": path.stat().st_size,
        "sha256": sha(path),
    }


def read(path):
    require(
        path.is_file() and not path.is_symlink() and path.stat().st_size <= 1048576,
        "Bounded original JSON required",
    )
    return json.loads(path.read_text())


def authenticate(row):
    p = Path(row["path"])
    p = p if p.is_absolute() else ROOT / p
    info = p.stat()
    size = row.get("size_bytes", row.get("size"))
    require(size is None or size == info.st_size, "Selected file size differs")
    if "stat" in row:
        require(
            all(type(v) is int and getattr(info, k) == v for k, v in row["stat"].items()),
            "Selected original stat identity differs",
        )
    if "sha256" in row:
        require(sha(p) == row["sha256"], "Selected file hash differs")
    else:
        require(bool(row.get("stat")), "Original stat-only scope required")


def execute():
    terminal_path = OWNER / "root-execution-np5-05.json"
    require(
        terminal_path.is_file() and (OUT / "receipt.json").is_file(),
        "Original root and owner must be terminal first",
    )
    terminal, receipt = read(terminal_path), read(OUT / "receipt.json")
    closure = read(OUT / "external-boundary-closure.json")
    require(
        terminal["exit_code"] == 0
        and receipt["profile"] == PROFILE
        and receipt["completed"] is True
        and receipt["primary_error"] is None
        and receipt["secondary_errors"] == []
        and receipt["sources_unchanged"] is True
        and receipt["comfy_service_modified"] is False,
        "Original successful owner and unchanged sources required",
    )
    require(
        closure["closure_mode"] == "authenticated_stop"
        and closure["closure_errors"] == []
        and all(
            closure[k] is True
            for k in (
                "whole_unit_exit_confirmed",
                "observed_cgroup_cohort_exit_confirmed",
                "post_stop_terminal_verified",
                "cgroup_absent",
                "sources_unchanged",
            )
        )
        and len(closure["two_process_scans"]) == 2
        and all(not row["matches"] and not row["unknown"] for row in closure["two_process_scans"]),
        "Original current authenticated scoped closure required",
    )
    for k in (
        "no_escape_enforced",
        "whole_tree_exit_confirmed",
        "continuation_authorized",
        "model_qualified",
    ):
        require(closure[k] is False, "Original universal nonqualification flags required")
    config = receipt["selected_config"]
    require(
        config == read(OUT / "sealed-config.json")
        and config["profile"] == PROFILE
        and config["unit"] == UNIT
        and config["output"] == str(OUT)
        and config["selection_sha256"] == SPEC_SHA,
        "Exact current selected config required",
    )
    admission = read(OUT / "admission.json")
    require(
        admission["nonce"] == config["nonce"] == closure["selection"]["nonce"]
        and admission["invocation_id"] == closure["invocation_id"]
        and closure["selection"]["unit"] == UNIT
        and all(
            admission["actor"][k] == closure["actor"][k]
            for k in ("pid", "start_ticks", "pgid", "sid", "uid", "cgroup")
        )
        and admission["config_sha256"] == sha(OUT / "sealed-config.json"),
        "Current admission/config/closure invocation differs",
    )
    spec_path, plan_path = HERE / "root-command-spec-01.json", HERE / "selected-plan-01.json"
    require(
        sha(spec_path) == SPEC_SHA
        and sha(plan_path) == PLAN_SHA
        and sha(OWNER / "controller.py") == CONTROLLER_SHA,
        "Selected source/plan pins differ",
    )
    spec, plan = read(spec_path), read(plan_path)
    base = [
        OWNER / "controller.py",
        spec_path,
        ROOT / "artifacts/jev-runtime/decision3-local-owner-terminal-20261022-01/owner_boundary.py",
        ROOT / "artifacts/jev-runtime/observer-system-services-20261022-01/comfy_observer.py",
        ROOT / "artifacts/jev-runtime/observer-system-services-20261022-01/system-services.json",
    ]
    expected = {r["path"]: r for r in [*spec["sources"], *(ref(p) for p in base)]}
    require(
        len(spec["sources"]) == 53
        and len(expected) == 58
        and {r["path"]: r for r in config["sources"]} == expected,
        "Exact53 plus5 original source union required",
    )
    for row in config["sources"] + spec["binary_stats"]:
        authenticate(row)
    native_plan = read(Path(plan["native_plan_path"]))
    for row in [
        *native_plan["model_files"],
        *native_plan["runtime_files"],
        native_plan["manifest"],
        *native_plan["owner_sources"],
    ]:
        authenticate(row)
    require(
        len(native_plan["model_files"]) == 20 and len(native_plan["runtime_files"]) == 28,
        "Original model/runtime inventory differs",
    )
    expected_argv = [
        str(ROOT / ".venv/bin/python"),
        "-B",
        "-s",
        str(OWNER / "controller.py"),
        "--run",
        "--expected-entry-sha256",
        CONTROLLER_SHA,
        "--selection",
        str(spec_path),
        "--selection-sha256",
        SPEC_SHA,
        "--output",
        str(OUT),
    ]
    require(
        terminal["argv"] == expected_argv
        and terminal["profile"] == "root-selected-np5-cpu5-zero-delta-terminal-v1",
        "Exact root CPU5 invocation differs",
    )
    command = read(OUT / "command-01-output.json")["result"]
    started = read(OUT / "command-01-started.json")
    intent = read(OUT / "command-01-intent.json")
    require(
        command["returncode"] == 0
        and command["pid"] == started["pid"]
        and command["argv"]
        == [v.replace("{OUTPUT}", str(OUT)) for v in spec["commands"][0]["argv"]],
        "Original successful selected child command required",
    )
    observed = started["observed_identity"]
    child, wrapper = observed["process"], admission["actor"]
    require(
        child["pid"] == started["pid"] == command["pid"]
        and child["pid"] != wrapper["pid"]
        and all(child[k] == wrapper[k] for k in ("pgid", "sid", "uid", "cgroup"))
        and all(
            child[k] == command["observed_identity"]["process"][k]
            for k in ("pid", "start_ticks", "pgid", "sid", "uid", "cgroup")
        )
        and all(observed[k] == command["observed_identity"][k] for k in ("exe", "argv", "cwd"))
        and observed["argv"] == intent["argv"] == command["argv"]
        and observed["cwd"]
        == intent["cwd"]
        == command["cwd"]
        == spec["commands"][0]["cwd"].replace("{OUTPUT}", str(OUT))
        and observed["exe"]
        == intent["resolved_executable"]
        == command["resolved_executable"]
        == str(Path(command["argv"][0]).resolve(strict=True)),
        "Current admitted wrapper cohort and exact child identity/exe/argv/cwd required",
    )
    state_path = OUT / 'np5-qualification/state.json'
    s = read(state_path)
    q = s['calls'][0]
    require(s['profile'] == plan['profile'] and s['plan_sha256'] == PLAN_SHA
            and s['completed'] is True and s['model_loads'] == s['native_request_calls'] == 1
            and len(s['calls']) == 1 and q['completed'] is True
            and s['initial_full_model_manifest_runtime_sources_authenticated'] is True
            and s['final_full_model_manifest_runtime_source_identities_unchanged'] is True
            and s['cuda_initialized_final'] is False, 'Actual one-load/native-call authentication')
    proc = s['process']
    require(proc['pid'] == child['pid'] and proc['starttime_ticks'] == child['start_ticks']
            and all(proc[k] == child[k] for k in ('pgid','sid'))
            and proc['cgroup'].strip() == '0::'+child['cgroup']
            and proc['affinity'] == list(range(15,20)), 'Actual state child binding')
    require(s['owned_limits'] == {'memory.max':'4294967296','memory.swap.max':'0','cpu.max':'500000 100000'}, 'Original owned limits')
    require(s['host_startup_admission']['required_available_bytes'] == 30414140212
            and s['host_startup_admission']['MemAvailable'] >= 30414140212, 'Startup memory admission')
    require(s['training_eligible'] is False and s['feature_commit_status'] == 'PENDING_NONTRAINING'
            and s['benchmark_calls'] == s['browser_calls'] == 0 and s['benchmark_scores'] is None
            and all(s[k] is False for k in ('http_qualified','browser_qualified','model_qualified','no_escape_audited','whole_tree_audited','continuation_qualified')), 'Diagnostic permanently nontraining')
    provenance = s['native_provenance']
    require(provenance['model_sha256'] == plan['model']['model_sha256']
            and provenance['backbone_dtype'] == 'bfloat16' and provenance['readout_dtype'] == 'float32'
            and provenance['temperature'] == 1.0 and provenance['batch_size'] == 4
            and provenance['attention_mode'] == 'noncausal_full_attention'
            and s['native_runtime_info']['device'] == 'cpu'
            and s['actual_environment']['OMP_NUM_THREADS'] == '5'
            and s['actual_environment']['CUDA_VISIBLE_DEVICES'] == '', 'Actual native CPU/BF16/FP32 provenance')
    require(q['first_layer_events'] == q['final_norm_events'] == [[4,188,1024]]
            and q['extra_encoder_passes'] == q['extra_final_norm_passes'] == 0
            and q['baseline_tensor_identity_and_versions_unchanged'] is True, 'Same-forward hooks/baseline')
    require(0 < q['complete_prediction_seconds'] <= 20
            and 0 < s['elapsed_seconds'] < command['elapsed_seconds'] < terminal['wall_seconds'] < 120
            and abs((datetime.fromisoformat(terminal['finished_at_utc'])-datetime.fromisoformat(terminal['started_at_utc'])).total_seconds()-terminal['wall_seconds']) < .01,
            'Original complete-call, command and root bounds/order')
    fixture = read(HERE/'fixture.json')
    request = q['full_source_fixture_request']
    require(request == fixture['request'], 'Actual fixture exact original source')
    raw_path = OUT/'np5-qualification/raw-native-response.json'
    raw_envelope = read(raw_path)
    raw = raw_envelope['raw_response']
    require(raw_envelope['feature_commit_status'] == 'PENDING_NONTRAINING'
            and list(raw['answers']) == list(request['questions']), 'Actual complete raw answer order')
    counts = [8,1,2,3]
    for (key, question), count in zip(request['questions'].items(), counts):
        answer = raw['answers'][key]
        require(answer['type'] == 'choice' and list(answer['probabilities']) == list(question['criteria'])
                and len(answer['probabilities']) == count, 'All actual legal ordered probabilities')
        values = list(answer['probabilities'].values())
        require(all(type(v) in (int,float) and math.isfinite(v) and 0 <= v <= 1 for v in values)
                and abs(sum(values)-1) <= 1e-6
                and answer['choice'] == list(question['criteria'])[max(range(count),key=lambda i:values[i])], 'Normalized actual ordered argmax')
        confidence = 1.0 if count < 2 else 1+sum(v*math.log(v) for v in values if v > 0)/math.log(count)
        require(abs(answer['confidence']-confidence) < 1e-6, 'Original entropy confidence including singleton')
    numeric = []
    for name in ('replay','numpy_zero_delta'):
        result = q[name]
        require(result['absolute_tolerance'] == 1e-6 and result['relative_tolerance'] == 1e-5
                and result['all_legal_probabilities_matched'] is True
                and result['all_ordered_argmaxes_matched'] is True
                and result['semantic_correctness_evaluated'] is False
                and [(v['head'],v['legal_options']) for v in result['rows']] == list(zip(request['questions'],counts))
                and all(math.isfinite(v['max_absolute_error']) and 0 <= v['max_absolute_error'] <= 1.1e-5 for v in result['rows']), 'Fixed mixed-tolerance replay proof')
        numeric.append({'kind':name, 'all_legal_probabilities_matched':True, 'all_ordered_argmaxes_matched':True,
                        'max_absolute_errors':[v['max_absolute_error'] for v in result['rows']]})
    require(q['numpy_zero_delta']['FP32_Torch_NumPy_copy_exact'] is True
            and q['numpy_zero_delta']['illegal_probabilities_exact_zero'] is True
            and q['numpy_zero_delta']['zero_delta_shape'] == [255,1024]
            and q['numpy_zero_delta']['feature_shape'] == [4,1024], 'Exact FP32 copies/shape and illegal-zero evidence')
    payload_path = OUT/'np5-qualification/payloads-pending.json'
    payload = read(payload_path)
    require(payload['status'] == 'PENDING_NONTRAINING' and payload['labels'] is None
            and payload['training_eligible'] is False and payload['ordered_heads'] == list(request['questions'])
            and payload['legal_counts'] == counts and len(payload['payloads']) == 2, 'No labels or training eligibility')
    arrays, payload_refs = [], []
    for i, row in enumerate(payload['payloads']):
        authenticate(row)
        path = Path(row['path'])
        require(path.parent == OUT/'np5-qualification' and row['training_eligible'] is False
                and row['status'] == 'PENDING_NONTRAINING' and row['dtype'] == 'float32'
                and row['format'] == 'npy_no_pickle' and row['shape'] == ([4,1024] if i == 0 else [255,1024]), 'Bounded permanently diagnostic payload')
        content = path.read_bytes()
        require(len(content) < 2*1024**2 and content[:8] == b'\x93NUMPY\x01\x00', 'NPY1 ordinary bounded payload')
        header_size = struct.unpack('<H',content[8:10])[0]
        header = ast.literal_eval(content[10:10+header_size].decode('latin1'))
        require(header == {'descr':'<f4','fortran_order':False,'shape':tuple(row['shape'])}, 'Exact FP32 C-order no-pickle header')
        data = content[10+header_size:]
        require(len(data) == math.prod(row['shape'])*4
                and hashlib.sha256(data).hexdigest() == payload['baseline_feature_FP32_sha256'][i], 'FP32 source-copy bytes digest')
        values = array.array('f'); values.frombytes(data)
        if sys.byteorder != 'little': values.byteswap()
        require(all(math.isfinite(v) for v in values), 'Finite copied FP32 payload')
        arrays.append(values); payload_refs.append(ref(path))
    # Separate stdlib float64 accumulation witness on every legal slot. The actual
    # contract remains FP32 Torch/NumPy and is checked by the pinned producer.
    h,w = arrays; witness = []
    for row_index,(key,question) in enumerate(request['questions'].items()):
        count = counts[row_index]
        logits = [math.fsum(h[row_index*1024+j]*w[k*1024+j] for j in range(1024)) for k in range(count)]
        exps = [math.exp(v-max(logits)) for v in logits]; total = math.fsum(exps)
        probs = [v/total for v in exps]; original = list(raw['answers'][key]['probabilities'].values())
        require(all(abs(a-b) <= 1e-6+1e-5*abs(b) for a,b in zip(probs,original))
                and max(range(count),key=lambda i:probs[i]) == max(range(count),key=lambda i:original[i]), 'Separate all-legal stdlib replay witness')
        witness.append({'head':key,'legal_count':count,'max_absolute_error':max(abs(a-b) for a,b in zip(probs,original))})
    catalog_path = OUT/'np5-qualification/catalog-actual-pending.json'
    catalog = read(catalog_path)
    require(catalog['status'] == 'PENDING_CLOSED_ROOT_ACCEPTANCE' and catalog['training_eligible'] is False, 'Closed pending catalog scope')
    before,after = catalog['before'],catalog['after']
    for item in (before,after):
        rows = item['critical_files_and_mapped_math_candidates']
        gpu = item['mapped_gpu_math_candidates_excluded_from_CPU_projection']
        require(item['schema'] == 'owned-ML313-numpy-discovery-critical-projection-v1'
                and item['numpy_version'] == '2.5.2' and item['python']['version'] == [3,13,14]
                and item['python']['resolved_path'] == command['resolved_executable']
                and item['actual_discovery_after_native_load'] is True
                and item['preselected_catalog'] is False and item['full_DSO_or_transitive_closure_qualified'] is False
                and 0 < len(rows)+len(gpu) <= 24 and sum(r['size_bytes'] for r in rows) <= 512*1024**2
                and all(0 < r['size_bytes'] <= 256*1024**2 for r in rows), 'Original bounded actual NumPy projection')
        for row in rows:
            require(type(row.get('sha256')) is str and re.fullmatch(r'[0-9a-f]{64}', row['sha256'])
                    and set(row['stat']) == {'st_dev','st_ino','st_size','st_mtime_ns'}
                    and row['size_bytes'] == row['stat']['st_size']
                    and str(Path(row['path']).resolve(strict=True)) == row['resolved_path'], 'CPU hash/stat/resolved identity required')
            authenticate(row)
        for row in gpu:
            require('sha256' not in row and set(row['stat']) == {'st_dev','st_ino','st_size','st_mtime_ns'}
                    and row['size_bytes'] == row['stat']['st_size'] and row['hashed'] is False and row['included_in_CPU_projection'] is False
                    and row['execution_or_symbol_binding_proven'] is False
                    and Path(row['path']).name in ('libcublas.so.13','libcublasLt.so.13'), 'GPU stat-only candidates scope')
            authenticate(row)
        require(set(item['mapped_math_candidates']).issubset({r['resolved_path'] for r in rows})
                and all('cublas' not in Path(r['path']).name.lower() for r in rows), 'CPU hash/GPU stat projection separation')
    require(all(after.get(k) == before.get(k) for k in ('numpy_version','python','numpy_build_configuration'))
            and all(row in after['critical_files_and_mapped_math_candidates'] for row in before['critical_files_and_mapped_math_candidates']), 'Before/after runtime catalog unchanged')
    counters=s['after_final_auth_owned_counters']['counters']
    require(type(counters['memory.peak']) is int and counters['memory.peak'] >= 0 and all(type(v) is int and v >= 0 for v in counters['memory.events'].values()), 'Typed original memory diagnostics')
    checks = dict.fromkeys(('root_exit0','owner_completed','sources_unchanged','current_closure_positive','full_authentication','all_legal_probabilities_matched','all_ordered_argmaxes_matched','FP32_copy_exact'),True)
    result = {
        'schema':'root-closed-np5-actual05-numeric-audit-v1','status':'passed','authority':'root',
        'separate_reviewer_actual_bytes_read':False,'independent_actual_review_claimed':False,
        'checks':checks,'source_union':58,'model_files_authenticated':20,'runtime_files_authenticated':28,'runtime_sha256_files':22,'runtime_stat_only_files':6,
        'root_execution':ref(terminal_path),'owner_receipt':ref(OUT/'receipt.json'),
        'current_closure':ref(OUT/'external-boundary-closure.json'),
        'selected_plan':ref(plan_path),'command_spec':ref(spec_path),'actor_state':ref(state_path),
        'catalog':ref(catalog_path),'raw_native_response':ref(raw_path),'payload_manifest':ref(payload_path),'diagnostic_payload_refs':payload_refs,
        'model_loads':1,'native_request_calls':1,'first_layer_events':q['first_layer_events'],
        'final_norm_events':q['final_norm_events'],'legal_counts':counts,'replay':numeric,
        'additional_stdlib_float64_accumulation_witness':witness,
        'tolerance':{'absolute':1e-6,'relative':1e-5,'comparison':'abs(replay-native) <= 1e-6 + 1e-5*abs(native)'},
        'call_seconds':q['complete_prediction_seconds'],'actor_seconds':s['elapsed_seconds'],
        'root_seconds':terminal['wall_seconds'],'owned_peak_bytes':counters['memory.peak'],'memory_events_diagnostic':counters['memory.events'],
        'numpy_version':after['numpy_version'],'critical_files':len(after['critical_files_and_mapped_math_candidates']),
        'critical_bytes':sum(r['size_bytes'] for r in after['critical_files_and_mapped_math_candidates']),
        'excluded_GPU_stat_only_count':len(after['mapped_gpu_math_candidates_excluded_from_CPU_projection']),
        'diagnostic_arrays_permanently_nontraining':True,'training_eligible':False,'semantic_correctness_evaluated':False,
        'benchmark_scores':None,'full_DSO_or_transitive_closure_qualified':False,
        'scope':'Root checked original terminal/owner/current scoped closure, all selected source/model hashes and original28 runtime identities (22 hashes and6 stat-only), raw ordered distributions, copied finite FP32 files and a separate legal-slot stdlib replay. Producer pinned-source actual FP32 replay retained. No separate actual reader, fit or semantic/browser/benchmark pass.'}
    return result


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=execute()
    with args.output.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps({'status':result['status'],'source_union':result['source_union'],'root_seconds':result['root_seconds'],'numeric_heads':len(result['additional_stdlib_float64_accumulation_witness'])}))

if __name__ == '__main__':
    main()
