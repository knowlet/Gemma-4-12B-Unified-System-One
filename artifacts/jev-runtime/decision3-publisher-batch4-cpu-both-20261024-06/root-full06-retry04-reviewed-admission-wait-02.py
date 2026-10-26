"""Read-only owner07 CPU-coexist admission wait; --launch permits the reviewed retry04 launcher.

The mandatory 60s condition means consecutive passing samples spanning 60s, never
a proof of continuous-time availability or harmless coexistence. It strengthens the selected gate, which
the launcher and unchanged owner must both execute again before any model work.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import types

ROOT = Path('/home/albert_hsieh/Gemma-4-12B-Unified-System-One')
RUNTIME = ROOT / 'artifacts/jev-runtime'
OWNER = RUNTIME / 'llama-cpp-owner-20261024-07'
CAMPAIGN = RUNTIME / 'decision3-publisher-batch4-cpu-both-20261024-06'
CONTROLLER = OWNER / 'controller.py'
CONTROLLER_SHA = '685a9f809f429f4e9658083dc8530b563740625f9ba7816908fdc58002de256d'
SPEC_SHA = 'fbaa2fbe1fb2bcd95236a4ccf6f71e69b8b6a4feae78f98429ff04d6c22ecfee'
REVIEW_SHA = '5d090a58eaea8145aec9f76fd63203a0258d217110b7e0c063d5a5edd1ab3cb3'
LAUNCHER = CAMPAIGN / 'root-full06-retry04-reviewed-launcher.py'
BOOT_ID = '90eeafaa-7c2f-45f0-8b45-f74ef2ee1005'
MEM_TOTAL = 130595864576
REQUIRED_INITIAL_BYTES = 28266656564
INTERVAL_SECONDS = 5.0
# A long scheduling/read delay restarts the mandatory window, instead of silently
# crediting an unobserved interval as passing samples. This is still sampled.
MAX_BETWEEN_SAMPLE_SECONDS = 7.5
UNUSED = (
    OWNER / 'actual-04',
    OWNER / 'root-launch-full06-edge-both-04.json',
    OWNER / 'root-execution-full06-edge-both-04.json',
    CAMPAIGN / 'root-full06-initial-gate-02.json',
)


def emit(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def fresh_outputs():
    if any(path.exists() or path.is_symlink() for path in UNUSED):
        raise ValueError('Fresh actual04/intent/terminal/gate02 paths required')


def launcher_ref(expected=None):
    if LAUNCHER.is_symlink() or not LAUNCHER.is_file():
        raise ValueError('Ordinary reviewed launcher source required')
    with LAUNCHER.open('rb') as stream:
        blob = stream.read(1048577)
    if not 0 < len(blob) <= 1048576:
        raise ValueError('Bounded launcher source required')
    result = {'path': str(LAUNCHER), 'size_bytes': len(blob),
              'sha256': hashlib.sha256(blob).hexdigest()}
    if expected is not None and result['sha256'] != expected:
        raise ValueError('Independently selected full06 retry04 launcher changed')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-seconds', type=int, choices=(180, 600), default=600)
    parser.add_argument('--quiet-seconds', type=int, choices=(60,), default=60)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--launcher-sha256')
    args = parser.parse_args(argv)
    if args.launch and (type(args.launcher_sha256) is not str
                        or re.fullmatch(r'[0-9a-f]{64}', args.launcher_sha256) is None):
        parser.error('--launch requires an independently reviewed --launcher-sha256')
    if args.launcher_sha256 is not None and re.fullmatch(r'[0-9a-f]{64}', args.launcher_sha256) is None:
        parser.error('Malformed launcher source SHA256')
    fresh_outputs()
    selected_launcher = launcher_ref(args.launcher_sha256)
    source = CONTROLLER.read_bytes()
    if hashlib.sha256(source).hexdigest() != CONTROLLER_SHA:
        raise ValueError('Selected owner07 controller source changed')
    guard = types.ModuleType('_root_full06_retry04_coexist_admission_wait')
    guard.__file__ = str(CONTROLLER)
    sys.modules[guard.__name__] = guard
    exec(compile(source, str(CONTROLLER), 'exec'), vars(guard))
    assert guard.UNIT == 'jev-d3-llama-cpp-cpu4-2g-coexist-20261024-07.service'
    assert guard.PROFILE == 'llama-cpp-cpu4-2g-coexist-command-owned-v1'
    assert guard.HOST_OBSERVATION_POLICY == 'comfy-observed-cpu-coexist-v1'
    assert guard.MEMORY_MAX_BYTES == guard.INITIAL_HOST_EXTRA_BYTES == 2147483648
    observer = guard.load_fixed('observer')
    started = time.monotonic()
    deadline = started + args.wait_seconds
    checks = consecutive = resets = 0
    first_pass = last_pass = None
    last_error = None
    emit({'status': 'READONLY_SELECTED_CPU_COEXIST_ADMISSION_WAIT',
          'wait_seconds': args.wait_seconds, 'poll_seconds': INTERVAL_SECONDS,
          'quiet_seconds_required': args.quiet_seconds,
          'quiet_scope': 'Consecutive full-reserve strict known-HTTP passing samples; not continuous-time availability or harmless-coexistence proof',
          'initial_R_plus_entire2GiB_bytes': REQUIRED_INITIAL_BYTES,
          'host_observation_policy': guard.HOST_OBSERVATION_POLICY,
          'explicit_launch_requested': args.launch,
          'launcher': selected_launcher, 'model_launched': False})
    while time.monotonic() < deadline:
        checks += 1
        try:
            gate = guard.host(observer, BOOT_ID, MEM_TOTAL, initial=True)
            if gate['required_available_bytes'] != REQUIRED_INITIAL_BYTES:
                raise ValueError('Original fullR+entire2GiB initial admission differs')
            if (gate['host_observation_policy'] != guard.HOST_OBSERVATION_POLICY
                    or gate['reserve_bytes'] != 26119172916 or gate['initial'] is not True):
                raise ValueError('Exact selected host policy/reserve/initial gate differs')
            if (gate['observation']['comfy_state'] not in {'http_idle', 'http_busy'}
                    or gate['observation']['comfy_evidence']['http_status'] != 200):
                raise ValueError('Explicit strict known HTTP200 observation required by retry wait')
        except Exception as error:
            last_error = guard.failure(error)
            if first_pass is not None:
                resets += 1
                emit({'status': 'SAMPLED_QUIET_WINDOW_RESET', 'checks': checks,
                      'consecutive_passing_samples': consecutive,
                      'error': last_error, 'model_launched': False})
            elif checks == 1:
                emit({'status': 'FIRST_SELECTED_GATE_FAILURE', 'error': last_error,
                      'model_launched': False})
            first_pass = last_pass = None
            consecutive = 0
        else:
            finished = time.monotonic()
            if finished >= deadline:
                break
            if last_pass is not None and finished - last_pass > MAX_BETWEEN_SAMPLE_SECONDS:
                resets += 1
                first_pass = None
                consecutive = 0
            if first_pass is None:
                first_pass = finished
            last_pass = finished
            consecutive += 1
            span = finished - first_pass
            if span >= args.quiet_seconds:
                fresh_outputs()
                launcher_ref(args.launcher_sha256)
                if time.monotonic() >= deadline:
                    break
                emit({'status': 'PRELIMINARY_SAMPLED_ADMISSION_PASSED',
                      'checks': checks, 'consecutive_passing_samples': consecutive,
                      'sampled_quiet_span_seconds': span, 'window_resets': resets,
                      'elapsed_wait_seconds': time.monotonic() - started,
                      'quiet_scope': 'Sampled reserve and known-HTTP policy observations only; no reservation or continuous-availability guarantee',
                      'launcher_must_repeat_original_two_absence_scans_and_full_gate': True,
                      'model_launched': False})
                if not args.launch:
                    emit({'status': 'ADMITTED_READONLY_NO_LAUNCH_REQUESTED',
                          'model_launched': False})
                    return 0
                environment = dict(os.environ)
                environment.update(JEV_FULL06_GATE_INDEX='02',
                                   JEV_FULL06_SPEC_SHA256=SPEC_SHA,
                                   JEV_FULL06_REVIEW_SHA256=REVIEW_SHA)
                result = subprocess.run([str(ROOT / '.venv/bin/python'), '-B', '-s',
                                         str(LAUNCHER)], cwd=ROOT, env=environment,
                                        stdin=subprocess.DEVNULL, check=False)
                return result.returncode
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(INTERVAL_SECONDS, remaining))
    emit({'status': 'NO_ADMISSION_NO_MODEL_LAUNCHED', 'checks': checks,
          'wait_seconds': args.wait_seconds, 'window_resets': resets,
          'last_error': last_error, 'model_launched': False,
          'scope': 'Outer wait elapsed; no launcher/unit/model was started'})
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
