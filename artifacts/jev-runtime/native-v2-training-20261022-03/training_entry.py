"""Keep the fresh fitter in the owned main process and session."""

import json
import os
import runpy
import signal
import sys
from pathlib import Path

HERE = Path(__file__).absolute().parent
ROOT = HERE.parents[2]


def main():
    if os.environ.get("JEV_NATIVE_TRAIN_EXECUTE") != "native-v2-142-gx10-v1":
        print("DRY_NO_MODEL_OR_TRAINING")
        return
    if os.getsid(0) != os.getpid() or os.getpgrp() != os.getpid():
        os.setsid()
    raw = (Path("/proc") / str(os.getpid()) / "stat").read_text()
    fields = raw[raw.rfind(")") + 2 :].split()
    actor = {
        "pid": os.getpid(),
        "pgid": os.getpgrp(),
        "sid": os.getsid(0),
        "start_ticks": int(fields[19]),
    }
    assert actor["pid"] == actor["pgid"] == actor["sid"]
    with (HERE / "actor-intent-01.json").open("x") as stream:
        json.dump(
            {"actor": actor, "argv": sys.argv, "unit": os.environ["JEV_NATIVE_TRAIN_UNIT"]}, stream
        )

    def expired(_signum, _frame):
        raise TimeoutError("Owned native training cooperative10740-second deadline")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(10740)
    sys.argv = [str(ROOT / "scripts/jev_browser_fit_native_actions.py"), *sys.argv[1:]]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
