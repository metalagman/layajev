"""Atomic, offline export of the pinned official checkpoint."""

from __future__ import annotations

from datetime import datetime, UTC
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

from .architecture import build_model, load_exact_weights
from .onnx_export import export_onnx, validate_onnx
from .provenance import load_profile, sha256_file, verify_sdk, verify_source


def export_bundle(
    profile_path: Path,
    source_root: Path,
    sdk_root: Path,
    output: Path,
    epoch: int,
    repository_root: Path,
    verifier: Path,
) -> dict[str, Any]:
    """Build and atomically publish one fully verified bundle."""

    def emit_stage(message: str) -> None:
        print(f"convert: {message}", file=sys.stderr, flush=True)

    emit_stage("verifying source and SDK")
    profile = load_profile(profile_path)
    verify_source(profile, source_root)
    verify_sdk(sdk_root)
    source = source_root.resolve(strict=True)
    sdk = sdk_root.resolve(strict=True)
    destination = output.resolve(strict=False)
    repo = repository_root.resolve(strict=True)
    verifier_path = verifier.resolve(strict=True)
    reject_overlap(source, destination)
    reject_overlap(sdk, destination)
    if destination.exists():
        raise ValueError("output already exists; ready bundles are immutable")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        emit_stage("building and loading model")
        configuration = json.loads((source / "rl_agent_config.json").read_text(encoding="utf-8"))
        model = build_model(configuration, source / "encoder")
        weight_observation = load_exact_weights(model, source / "model.safetensors")
        emit_stage("exporting ONNX graph")
        export_onnx(model, stage / "model.onnx")
        emit_stage("validating graph and runtime parity")
        graph_observation = validate_onnx(stage / "model.onnx")
        parity = compare_runtime(model, stage / "model.onnx")
        emit_stage("packaging and verifying bundle")
        copy_artifacts(source, sdk, stage)
        manifest = build_manifest(
            profile_path,
            profile,
            stage,
            epoch,
            repo,
        )
        (stage / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        subprocess.run(
            [str(verifier_path), str(stage)],
            check=True,
            timeout=180,
            env={"PATH": os.environ.get("PATH", "")},
        )
        manifest_sha256 = sha256_file(stage / "manifest.json")
        result = {
            "bundle": str(destination),
            "weights": weight_observation,
            "graph": graph_observation,
            "parity": parity,
            "manifest_sha256": manifest_sha256,
        }
        emit_stage("publishing verified bundle")
        atomic_publish(stage, destination)
        return result
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def reject_overlap(input_root: Path, output: Path) -> None:
    """Reject either direction of input/output containment."""

    if input_root == output or input_root in output.parents or output in input_root.parents:
        raise ValueError("input and output paths must not overlap")


def atomic_publish(stage: Path, destination: Path) -> None:
    """Atomically rename a directory without replacing a concurrent target."""

    rename_at2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if rename_at2 is None:
        raise RuntimeError("atomic no-replace directory publication is unavailable")
    rename_at2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    rename_at2.restype = ctypes.c_int
    at_current_working_directory = -100
    rename_no_replace = 1
    result = rename_at2(
        at_current_working_directory,
        os.fsencode(stage),
        at_current_working_directory,
        os.fsencode(destination),
        rename_no_replace,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(error_number, "output already exists", destination)
        raise OSError(error_number, os.strerror(error_number), destination)


def copy_artifacts(source: Path, sdk: Path, stage: Path) -> None:
    """Copy only contract-declared runtime, license, and notice artifacts."""

    (stage / "tokenizer").mkdir()
    shutil.copyfile(source / "tokenizer" / "tokenizer.json", stage / "tokenizer" / "tokenizer.json")
    shutil.copyfile(source / "tokenizer" / "tokenizer_config.json", stage / "tokenizer" / "tokenizer_config.json")
    shutil.copyfile(source / "rl_agent_config.json", stage / "rl_agent_config.json")
    license_bytes = (sdk / "LICENSE").read_bytes()
    (stage / "LICENSE.model").write_bytes(license_bytes)
    (stage / "LICENSE.sdk").write_bytes(license_bytes)
    (stage / "NOTICE.model").write_text(
        "Laya multilingual model\nCopyright Convai Innovations\nLicensed under Apache-2.0.\n",
        encoding="utf-8",
        newline="\n",
    )
    (stage / "NOTICE.sdk").write_text(
        "Exporter decision-head semantics adapted from NandhaKishorM/laya v0.3.5.\nLicensed under Apache-2.0.\n",
        encoding="utf-8",
        newline="\n",
    )


def compare_runtime(model, graph: Path) -> dict[str, float]:
    """Compare both raw outputs before rounding at the frozen FP32 tolerance."""

    import numpy as np
    import onnxruntime as ort
    import torch

    input_ids = torch.tensor([[2, 10, 3, 4, 20, 4, 21, 3, 30, 31, 3]], dtype=torch.int64)
    attention_mask = torch.ones_like(input_ids)
    marker_pos = torch.tensor([[3, 5]], dtype=torch.int64)
    marker_mask = torch.tensor([[True, True]], dtype=torch.bool)
    qtype = torch.tensor([0], dtype=torch.int64)
    with torch.inference_mode():
        expected = [value.numpy() for value in model(input_ids, attention_mask, marker_pos, marker_mask, qtype)]
    session = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
    observed = session.run(
        None,
        {
            "input_ids": input_ids.numpy(),
            "attention_mask": attention_mask.numpy(),
            "marker_pos": marker_pos.numpy(),
            "marker_mask": marker_mask.numpy(),
            "qtype": qtype.numpy(),
        },
    )
    names = ("logits", "act_logits")
    result: dict[str, float] = {}
    for name, want, got in zip(names, expected, observed, strict=True):
        if want.shape != got.shape or not np.isfinite(got).all():
            raise ValueError(f"invalid {name} output")
        if not np.allclose(want, got, atol=1e-4, rtol=1e-4):
            raise ValueError(f"{name} parity exceeds frozen tolerance")
        result[name + "_max_abs"] = float(np.max(np.abs(want - got)))
    return result


def exporter_git_revision(repository_root: Path) -> str:
    """Return the last committed exporter revision, independent of later docs."""

    paths = ("tools/export/src/laya_export", "tools/export/sdk_probe.py")
    command = ["git", "-C", str(repository_root)]
    status = subprocess.run(
        [*command, "status", "--porcelain", "--untracked-files=all", "--", *paths],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout
    if status:
        raise ValueError("exporter sources must be committed before export")
    revision = subprocess.run(
        [*command, "log", "-1", "--format=%H", "--", *paths],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout.strip()
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("exporter sources have no committed Git revision")
    return revision


def build_manifest(
    profile_path: Path,
    profile: dict[str, Any],
    stage: Path,
    epoch: int,
    repository_root: Path,
) -> dict[str, Any]:
    """Construct a canonical bundle-v1 manifest from staged artifact identities."""

    external_data = sorted(path.name for path in stage.glob("model.onnx.*"))
    roles = {
        "LICENSE.model": "license",
        "LICENSE.sdk": "license",
        "NOTICE.model": "notice",
        "NOTICE.sdk": "notice",
        "model.onnx": "model",
        "rl_agent_config.json": "agent-config",
        "tokenizer/tokenizer.json": "tokenizer",
        "tokenizer/tokenizer_config.json": "tokenizer-config",
    }
    for name in external_data:
        roles[name] = "model-external-data"
    files = []
    for relative in sorted(roles):
        artifact = stage / relative
        files.append(
            {
                "role": roles[relative],
                "path": relative,
                "size": artifact.stat().st_size,
                "sha256": sha256_file(artifact),
            }
        )
    exporter_files = [
        path.relative_to(repository_root).as_posix()
        for path in (repository_root / "tools/export/src/laya_export").glob("*.py")
    ]
    exporter_files.append("tools/export/sdk_probe.py")
    exporter_digest = framed_digest(repository_root, exporter_files)
    git_revision = exporter_git_revision(repository_root)
    created = datetime.fromtimestamp(epoch, UTC).isoformat().replace("+00:00", "Z")
    return {
        "schema_version": 1,
        "bundle": {"id": "laya-multilingual", "version": "0.1.0"},
        "provenance": {
            "source_model": {
                "id": profile["source_model"]["id"],
                "revision": profile["source_model"]["revision"],
            },
            "sdk": {
                "repository": profile["reference_sdk"]["repository"],
                "revision": profile["reference_sdk"]["revision"],
                "version": profile["reference_sdk"]["version"],
            },
            "profile": {"id": profile["id"], "sha256": sha256_file(profile_path)},
            "exporter": {
                "name": "laya-go-export",
                "version": "1.0.0",
                "source_sha256": exporter_digest,
                "git_revision": git_revision,
            },
            "lock": {
                "path": "tools/export/uv.lock",
                "sha256": sha256_file(repository_root / "tools/export/uv.lock"),
            },
            "created_at": created,
            "source_date_epoch": epoch,
            "redistribution": "approved",
            "licenses": [
                {"component": "source-model", "spdx": "Apache-2.0", "path": "LICENSE.model"},
                {"component": "sdk-derived-exporter-code", "spdx": "Apache-2.0", "path": "LICENSE.sdk"},
            ],
        },
        "preprocessing": profile["preprocessing"],
        "model": {
            "path": "model.onnx",
            "format": "onnx",
            "opset": 18,
            "precision": "fp32",
            "external_data": external_data,
            "inputs": profile["export"]["inputs"],
            "outputs": profile["export"]["outputs"],
        },
        "tokenizer": {
            "json_path": "tokenizer/tokenizer.json",
            "config_path": "tokenizer/tokenizer_config.json",
            "special_tokens": {"cls": 2, "sep": 3, "mask": 4, "pad": 0},
        },
        "calibration": {
            "config_path": "rl_agent_config.json",
            "temperature_min": 0.5,
            "temperature_max": 5.0,
            "invalid_temperature": 1.0,
            "selection_order": ["option_bucket", "qtype", "default"],
            "actions": [{"index": 0, "id": "escalate"}],
        },
        "runtime": {
            "engine": "onnxruntime",
            "minimum_version": "1.20.0",
            "maximum_exclusive": "2.0.0",
            "execution_providers": ["CPUExecutionProvider"],
        },
        "files": files,
    }


def framed_digest(root: Path, relative_paths: list[str]) -> str:
    """Digest ordered source paths and bytes without concatenation ambiguity."""

    digest = hashlib.sha256()
    for relative in sorted(relative_paths):
        data = (root / relative).read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative.encode("utf-8"))
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()
