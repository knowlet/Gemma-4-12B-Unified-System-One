"""Unchanged selected matched-control driver with explicit new local components."""

from pathlib import Path

from checked_source import module

selected = module(
    "artifacts/jev-runtime/native-v2-q95-browser-control-20261022-01/control_driver.py",
    "00295d7110522dd346320e598298f923bda56fa4a73dba427035371801c73543",
    "_native_q95_pin_selected_driver",
)
selected.HERE = Path(__file__).absolute().parent


def __getattr__(name):
    return getattr(selected, name)


def main(argv=None):
    return selected.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
