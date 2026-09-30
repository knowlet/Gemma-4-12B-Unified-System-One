"""Load local TOML without importing inference runtimes or contacting services."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .contracts import (
    ModelRegistry,
    ModelSpec,
    ProfileRegistry,
    ProfileSpec,
    SuiteRegistry,
    SuiteSpec,
)


def _index(items, kind):
    result = {item.id: item for item in items}
    if len(result) != len(items):
        raise ValueError(f"duplicate {kind} ids")
    return result


def _read(path, schema):
    return schema.model_validate(tomllib.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class Registry:
    models: dict[str, ModelSpec]
    suites: dict[str, SuiteSpec]
    profiles: dict[str, ProfileSpec]
    suite_directory: Path
    baseline_revision: str

    @classmethod
    def load(cls, models, suites, profiles):
        model_file = _read(models, ModelRegistry)
        model_index = _index(model_file.models, "model")
        suite_index = _index(_read(suites, SuiteRegistry).suites, "suite")
        profile_index = _index(_read(profiles, ProfileRegistry).profiles, "profile")
        for profile in profile_index.values():
            if profile.suite not in suite_index:
                raise ValueError(f"profile {profile.id}: unknown suite {profile.suite}")
            missing = set(profile.models) - model_index.keys()
            if missing:
                raise ValueError(f"profile {profile.id}: unknown models {sorted(missing)}")
        return cls(
            model_index,
            suite_index,
            profile_index,
            Path(suites).resolve().parent,
            model_file.baseline_revision,
        )

    def dataset_path(self, suite: SuiteSpec) -> Path:
        # Paths are relative to suites.toml, never the caller's working directory.
        return (self.suite_directory / suite.dataset).resolve()
