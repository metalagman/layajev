"""Offline verification for pinned official source and SDK inputs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any


EXPECTED_PROFILE_ID = "laya-multilingual-v1"
EXPECTED_SOURCE_ID = "convaiinnovations/laya-multilingual"
EXPECTED_SOURCE_REVISION = "052592a15d198d9ad47da779604259b10b47b7aa"
EXPECTED_SDK_REPOSITORY = "NandhaKishorM/laya"
EXPECTED_SDK_REVISION = "573e5b62696ba441230cd6be71d593331b5d23af"
EXPECTED_SDK_VERSION = "0.3.5"
EXPECTED_SDK_FILES = {
    "LICENSE": "a6cba85bc92e0cff7a450b1d873c0eaa2e9fc96bf472df0247a26bec77bf3ff9",
    "laya/__init__.py": "43e8c7d02969b37381915850cb4970e62b6ea0a328ac26c1efc9fb3fca0916d7",
    "laya/agent.py": "128567096446c5d39af8e4a3a7c4dd9e32a134a1b099ce5a5eed383beeff1b89",
    "laya/common.py": "f231d42fcec84da203222fcaa89c083b22776e00341e66e118183d754e1dcabf",
}
FORBIDDEN_CHECKPOINT_SUFFIXES = {".bin", ".pkl", ".pickle", ".pt", ".pth", ".py"}


def load_profile(path: Path) -> dict[str, Any]:
    """Load and validate the one supported export profile."""

    profile = json.loads(path.read_text(encoding="utf-8"))
    if profile.get("profile_version") != 1 or profile.get("id") != EXPECTED_PROFILE_ID:
        raise ValueError("unsupported export profile")
    source = profile.get("source_model", {})
    if source.get("id") != EXPECTED_SOURCE_ID or source.get("revision") != EXPECTED_SOURCE_REVISION:
        raise ValueError("export profile does not bind the pinned official source")
    sdk = profile.get("reference_sdk", {})
    if (
        sdk.get("repository") != EXPECTED_SDK_REPOSITORY
        or sdk.get("revision") != EXPECTED_SDK_REVISION
        or sdk.get("version") != EXPECTED_SDK_VERSION
    ):
        raise ValueError("export profile does not bind the pinned official SDK")
    return profile


def verify_source(profile: dict[str, Any], source_root: Path) -> list[dict[str, Any]]:
    """Verify every official checkpoint input without importing or modifying it."""

    root = source_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("source root must be a directory")
    expected_paths: set[str] = set()
    for item in profile["source_files"]:
        relative = item["path"]
        if not isinstance(relative, str):
            raise ValueError(f"invalid source profile path: {relative!r}")
        pure_path = PurePosixPath(relative)
        if (
            not relative
            or "\\" in relative
            or pure_path.is_absolute()
            or any(part in ("", ".", "..") for part in pure_path.parts)
            or pure_path.as_posix() != relative
        ):
            raise ValueError(f"invalid source profile path: {relative!r}")
        if relative in expected_paths:
            raise ValueError(f"duplicate source profile path: {relative}")
        expected_paths.add(relative)
    observations: list[dict[str, Any]] = []
    for entry in sorted(root.rglob("*")):
        relative = entry.relative_to(root).as_posix()
        if entry.is_symlink():
            raise ValueError(f"source contains a symlink: {relative}")
        if entry.is_file() and entry.suffix.lower() in FORBIDDEN_CHECKPOINT_SUFFIXES:
            raise ValueError(f"source contains executable or pickle-capable content: {relative}")
        if entry.is_file() and relative not in expected_paths:
            raise ValueError(f"source contains an undeclared file: {relative}")
    for expected in profile["source_files"]:
        relative = expected["path"]
        candidate = root.joinpath(*relative.split("/"))
        stat = candidate.stat(follow_symlinks=False)
        if not candidate.is_file() or candidate.is_symlink():
            raise ValueError(f"source artifact is not a regular file: {relative}")
        if stat.st_size != expected["size"]:
            raise ValueError(f"source size mismatch: {relative}")
        digest = sha256_file(candidate)
        if digest != expected["sha256"]:
            raise ValueError(f"source SHA-256 mismatch: {relative}")
        observations.append({"path": relative, "size": stat.st_size, "sha256": digest})
    return observations


def verify_sdk(sdk_root: Path) -> dict[str, str]:
    """Verify an explicit local SDK worktree without importing it."""

    root = sdk_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("SDK root must be a directory")
    if (root / ".git").exists():
        revision = run_git(root, "rev-parse", "HEAD")
        if revision != EXPECTED_SDK_REVISION:
            raise ValueError(f"SDK revision mismatch: {revision}")
        status = run_git(root, "status", "--porcelain", "--untracked-files=no")
        if status:
            raise ValueError("SDK worktree has tracked modifications")
    else:
        revision = EXPECTED_SDK_REVISION
    for relative, expected_digest in EXPECTED_SDK_FILES.items():
        candidate = root.joinpath(*relative.split("/"))
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(f"SDK source file is missing or not regular: {relative}")
        if sha256_file(candidate) != expected_digest:
            raise ValueError(f"SDK source digest mismatch: {relative}")
    init_path = root / "laya" / "__init__.py"
    source = init_path.read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', source, re.MULTILINE)
    if not match or match.group(1) != EXPECTED_SDK_VERSION:
        raise ValueError("SDK package version mismatch")
    return {
        "repository": EXPECTED_SDK_REPOSITORY,
        "revision": revision,
        "version": match.group(1),
    }


def sha256_file(path: Path) -> str:
    """Hash a file through a fixed-size buffer."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def run_git(root: Path, *arguments: str) -> str:
    """Run one local, non-shell Git inspection command."""

    environment = {
        "PATH": os.environ.get("PATH", ""),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
        env=environment,
    )
    return result.stdout.strip()
