"""The unchanged checked actor entry, importing this selected local owner."""

from checked_source import module

selected = module(
    "artifacts/jev-runtime/native-v2-q95-browser-control-20261022-01/browser_entry.py",
    "ea0f70680691e1467523fb3a0a8fe34f661d8bb3e2076d00dcecd8a0488aa28d",
    "_native_q95_pin_selected_entry",
)


def main(argv=None):
    return selected.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
