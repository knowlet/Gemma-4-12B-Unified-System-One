"""Load optional calibration metadata alongside a model checkpoint."""

from __future__ import annotations

import json
import math
from pathlib import Path


def load_temperature(name: str | Path, *, revision: str | None = None) -> float:
    """Read a positive, finite temperature from local or Hub ``s1_config.json``.

    Call after loading the model, passing its ``config._commit_hash`` when
    available, otherwise the requested revision. A resolved commit must take
    precedence over a branch or tag so calibration matches the loaded weights.

    Only an absent optional file defaults to 1.0. Access, network, cache, JSON,
    and invalid calibration errors propagate. Hub dependencies are imported
    only for remote checkpoints.
    """
    model_path = Path(name)
    if model_path.is_dir():
        config_path = model_path / "s1_config.json"
        try:
            # lstat distinguishes an absent file from a broken symlink. Reading
            # an existing but unusable file must fail, not reset calibration.
            config_path.lstat()
        except FileNotFoundError:
            return 1.0
    else:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import RemoteEntryNotFoundError

        try:
            config_path = Path(hf_hub_download(str(name), "s1_config.json", revision=revision))
        except RemoteEntryNotFoundError:
            # LocalEntryNotFoundError also covers offline/cache misses; those
            # do not establish that the optional file is absent on the Hub.
            return 1.0

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "temperature" not in config:
        raise ValueError(f"{config_path}: expected an object containing temperature")
    temperature = config["temperature"]
    message = f"{config_path}: temperature must be a positive, finite number"
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError(message)
    try:
        temperature = float(temperature)
    except OverflowError as exc:
        raise ValueError(message) from exc
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(message)
    return temperature
