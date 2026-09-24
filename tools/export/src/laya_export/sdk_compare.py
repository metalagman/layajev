"""Local-only comparison with a separately supplied verified SDK."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any

from .observations import build_observations
from .provenance import verify_sdk


def compare_sdk(sdk_root: Path, sdk_python: Path, probe: Path) -> dict[str, str]:
    """Verify SDK identity, run its pure semantic probe, and compare observations."""

    identity = verify_sdk(sdk_root)
    python = sdk_python.resolve(strict=True)
    probe_path = probe.resolve(strict=True)
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "NO_PROXY": "*",
        "no_proxy": "*",
    }
    result = subprocess.run(
        [str(python), "-I", str(probe_path), str(sdk_root.resolve(strict=True))],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(sdk_root.resolve(strict=True)),
        env=environment,
    )
    observed = json.loads(result.stdout)
    expected = build_observations()
    mismatch = first_mismatch(expected, observed)
    if mismatch:
        raise ValueError(f"pinned SDK semantic mismatch at {mismatch}")
    return identity


def first_mismatch(expected: Any, observed: Any, path: str = "$") -> str | None:
    """Return the first deterministic structural or numeric mismatch path."""

    if isinstance(expected, float) and isinstance(observed, (float, int)):
        return None if math.isclose(expected, float(observed), rel_tol=1e-12, abs_tol=1e-12) else path
    if type(expected) is not type(observed):
        return path
    if isinstance(expected, dict):
        if expected.keys() != observed.keys():
            return path
        for key in sorted(expected):
            mismatch = first_mismatch(expected[key], observed[key], f"{path}.{key}")
            if mismatch:
                return mismatch
        return None
    if isinstance(expected, list):
        if len(expected) != len(observed):
            return path
        for index, (want, got) in enumerate(zip(expected, observed, strict=True)):
            mismatch = first_mismatch(want, got, f"{path}[{index}]")
            if mismatch:
                return mismatch
        return None
    return None if expected == observed else path
