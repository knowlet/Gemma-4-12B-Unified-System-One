"""Read-only original admission wait; an explicit --launch permits the reviewed launcher.

The optional60s condition means consecutive passing samples spanning60s, never
a proof of continuous-time idle. It only strengthens the original gate, which
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
OWNER = RUNTIME / 'llama-cpp-owner-20261023-06'
CAMPAIGN = RUNTIME / 'decision3-readout-feature-replay-20261023-01'
CONTROLLER = OWNER / 'controller.py'
CONTROLLER_SHA = 'aa2af9911fde9fd06526c3fd5fe45f6521e80103bbccf9c29b98c7f3bba5952d'
SPEC_SHA = '138c836f7416c5437a6f9abe6fbba519d4e2da4ddfcd2a1d110c51bc6d86a3cb'
REVIEW_SHA = 'dddcb62f977222c2ce2c36e78f54ed4beafff929a92f0107e38f2a24656adf1a'
LAUNCHER = Path('/tmp/jev_replay_actual08_launch.py')
BOOT_ID = '90eeafaa-7c2f-45f0-8b45-f74ef2ee1005'
MEM_TOTAL = 130595864576
REQUIRED_INITIAL_BYTES = 28266656564
INTERVAL_SECONDS = 5.0
# A long scheduling/read delay restarts the optional window, instead of silently
# crediting an unobserved interval as passing samples. This is still sampled.
MAX_BETWEEN_SAMPLE_SECONDS = 7.5
UNUSED = (
    OWNER / 'actual-08',
    OWNER / 'root-launch-readout-replay-08.json',
    OWNER / 'root-execution-readout-replay-08.json',
    CAMPAIGN / 'root-replay-initial-gate-08.json',
)


def emit(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def fresh_outputs():
    if any(path.exists() or path.is_symlink() for path in UNUSED):
        raise ValueError('Fresh actual08/intent/terminal/gate07 paths required')


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
        raise ValueError('Independently selected full07 launcher changed')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-seconds', type=int, choices=(180, 600), default=600)
    parser.add_argument('--quiet-seconds', type=int, choices=(0, 60), default=60)
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
        raise ValueError('Original owner06 controller source changed')
    guard = types.ModuleType('_root_replay08_original_admission_wait')
    guard.__file__ = str(CONTROLLER)
    sys.modules[guard.__name__] = guard
    exec(compile(source, str(CONTROLLER), 'exec'), vars(guard))
    observer = guard.load_fixed('observer')
    started = time.monotonic()
    deadline = started + args.wait_seconds
    checks = consecutive = resets = 0
    first_pass = last_pass = None
    last_error = None
    emit({'status': 'READONLY_ORIGINAL_ADMISSION_WAIT',
          'wait_seconds': args.wait_seconds, 'poll_seconds': INTERVAL_SECONDS,
          'quiet_seconds_required': args.quiet_seconds,
          'quiet_scope': 'Consecutive original passing samples, not continuous-time proof',
          'initial_R_plus_entire2GiB_bytes': REQUIRED_INITIAL_BYTES,
          'explicit_launch_requested': args.launch,
          'launcher': selected_launcher, 'model_launched': False})
    while time.monotonic() < deadline:
        check_started = time.monotonic()
        checks += 1
        try:
            gate = guard.host(observer, BOOT_ID, MEM_TOTAL, initial=True)
            if gate['required_available_bytes'] != REQUIRED_INITIAL_BYTES:
                raise ValueError('Original fullR+entire2GiB initial admission differs')
        except Exception as error:
            last_error = guard.failure(error)
            if first_pass is not None:
                resets += 1
                emit({'status': 'SAMPLED_QUIET_WINDOW_RESET', 'checks': checks,
                      'consecutive_passing_samples': consecutive,
                      'error': last_error, 'model_launched': False})
            elif checks == 1:
                emit({'status': 'FIRST_ORIGINAL_GATE_FAILURE', 'error': last_error,
                      'model_launched': False})
            first_pass = last_pass = None
            consecutive = 0
        else:
            finished = time.monotonic()
            if finished >= deadline:
                break
            if last_pass is not None and check_started - last_pass > MAX_BETWEEN_SAMPLE_SECONDS:
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
                      'quiet_scope': 'Sampled observations only; no reservation/continuous-idle guarantee',
                      'launcher_must_repeat_original_two_absence_scans_and_full_gate': True,
                      'model_launched': False})
                if not args.launch:
                    emit({'status': 'ADMITTED_READONLY_NO_LAUNCH_REQUESTED',
                          'model_launched': False})
                    return 0
                environment = dict(os.environ)
                environment.update(JEV_REPLAY_GATE_INDEX='08',
                                   JEV_REPLAY_SPEC_SHA256=SPEC_SHA,
                                   JEV_REPLAY_REVIEW_SHA256=REVIEW_SHA)
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
