"""Canonical fixture envelope generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .observations import build_observations
from .provenance import EXPECTED_SDK_REVISION, EXPECTED_SOURCE_REVISION


def build_fixture(profile_path: Path) -> dict[str, Any]:
    """Build deterministic synthetic reference metadata and observations."""

    profile_bytes = profile_path.read_bytes()
    return {
        "schema_version": 1,
        "source_revision": EXPECTED_SOURCE_REVISION,
        "sdk_revision": EXPECTED_SDK_REVISION,
        "profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
        "preprocessing_version": 1,
        "synthetic_only": True,
        "observations": build_observations(),
    }


def canonical_json(value: Any) -> str:
    """Serialize one fixture deterministically as UTF-8 JSON text."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
