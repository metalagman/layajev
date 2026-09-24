"""Generate the frozen official-reference and ONNX parity corpus."""

from __future__ import annotations

from collections import OrderedDict
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .architecture import build_model, load_exact_weights
from .exporter import framed_digest
from .provenance import load_profile, sha256_file, verify_sdk, verify_source
from .reference import (
    Batch,
    SequenceItem,
    build_sequence,
    clamp_temperature,
    collate,
    confidence_from_probabilities,
    format_answer,
    render_options,
    select_temperature,
    softmax,
)


ATOL = 1e-4
RTOL = 1e-4


def write_corpus(path: Path, corpus: dict[str, Any]) -> None:
    """Atomically write canonical corpus JSON, replacing only the selected file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            json.dump(
                corpus,
                output,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class BundleTokenizer:
    """Adapt the pinned tokenizer.json to the narrow reference protocol."""

    def __init__(self, path: Path, special_tokens: dict[str, int]):
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(str(path))
        self.cls_token_id = special_tokens["cls"]
        self.sep_token_id = special_tokens["sep"]
        self.mask_token_id = special_tokens["mask"]
        self.pad_token_id = special_tokens["pad"]
        self.mask_token = "<mask>"

    def encode(self, text: str) -> list[int]:
        """Encode without tokenizer-added special tokens."""

        return self._tokenizer.encode(text, add_special_tokens=False).ids


def generate_corpus(
    profile_path: Path,
    source_root: Path,
    sdk_root: Path,
    bundle_root: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Generate all synthetic observations from verified pinned inputs."""

    profile = load_profile(profile_path)
    verify_source(profile, source_root)
    verify_sdk(sdk_root)
    source = source_root.resolve(strict=True)
    bundle = bundle_root.resolve(strict=True)
    repository = repository_root.resolve(strict=True)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_bundle_identity(manifest, bundle, profile_path, profile, repository)

    tokenizer = BundleTokenizer(
        bundle / manifest["tokenizer"]["json_path"],
        manifest["tokenizer"]["special_tokens"],
    )
    config = json.loads((source / "rl_agent_config.json").read_text(encoding="utf-8"))
    model = build_model(config, source / "encoder")
    weights = load_exact_weights(model, source / "model.safetensors")
    specifications, negative_cases = case_specifications(
        tokenizer,
        profile["preprocessing"]["max_len"],
        profile["preprocessing"]["head_max_len"],
    )
    runs = (
        ("ordered-mixed-batch", specifications[:3]),
        ("coverage-short-batch", specifications[3:12]),
        ("boundary-batch", specifications[12:]),
    )

    import numpy as np
    import onnxruntime as ort
    import torch

    session = ort.InferenceSession(
        str(bundle / manifest["model"]["path"]),
        providers=["CPUExecutionProvider"],
    )
    cases: list[dict[str, Any]] = []
    run_observations: list[dict[str, Any]] = []
    for run_id, run_specs in runs:
        items = [
            build_sequence(
                tokenizer,
                specification["state"]["value"],
                specification["question"],
                max_len=profile["preprocessing"]["max_len"],
                head_max_len=profile["preprocessing"]["head_max_len"],
            )
            for specification in run_specs
        ]
        batch = collate(items, tokenizer.pad_token_id)
        inputs = batch_arrays(batch)
        with torch.inference_mode():
            reference_outputs = model(
                *(torch.from_numpy(inputs[name]) for name in input_names())
            )
        reference = [value.numpy() for value in reference_outputs]
        onnx = session.run(None, inputs)
        maximum_differences: dict[str, float] = {}
        for name, expected, observed in zip(
            ("logits", "act_logits"), reference, onnx, strict=True
        ):
            if expected.shape != observed.shape or not np.isfinite(observed).all():
                raise ValueError(f"{run_id}: invalid {name} output")
            if not np.allclose(expected, observed, atol=ATOL, rtol=RTOL):
                raise ValueError(f"{run_id}: {name} parity exceeds frozen tolerance")
            maximum_differences[name] = float(np.max(np.abs(expected - observed)))
        usage = {
            "input_tokens": int(inputs["attention_mask"].sum()),
            "output_tokens": 0,
        }
        run_observations.append(
            {
                "id": run_id,
                "case_ids": [specification["id"] for specification in run_specs],
                "tensors": {name: inputs[name].tolist() for name in input_names()},
                "reference_outputs": {
                    "logits": reference[0].tolist(),
                    "act_logits": reference[1].tolist(),
                },
                "onnx_outputs": {
                    "logits": onnx[0].tolist(),
                    "act_logits": onnx[1].tolist(),
                },
                "maximum_absolute_difference": maximum_differences,
                "usage": usage,
            }
        )
        for row, (specification, item) in enumerate(zip(run_specs, items, strict=True)):
            cases.append(
                build_case_observation(
                    tokenizer,
                    specification,
                    item,
                    run_id,
                    row,
                    reference[0][row],
                    reference[1][row],
                    config,
                    usage,
                    profile["preprocessing"]["max_len"],
                    profile["preprocessing"]["head_max_len"],
                )
            )

    categories: dict[str, list[str]] = {}
    for case in cases:
        for category in case["categories"]:
            categories.setdefault(category, []).append(case["id"])
    for case in negative_cases:
        for category in case["categories"]:
            categories.setdefault(category, []).append(case["id"])
    manifest_digest = sha256_file(manifest_path)
    return {
        "schema_version": 1,
        "metadata": {
            "synthetic_only": True,
            "source_model": manifest["provenance"]["source_model"],
            "sdk": manifest["provenance"]["sdk"],
            "profile": manifest["provenance"]["profile"],
            "exporter": manifest["provenance"]["exporter"],
            "lock": manifest["provenance"]["lock"],
            "preprocessing_version": manifest["preprocessing"]["version"],
            "bundle_id": "sha256:" + manifest_digest,
            "manifest_sha256": manifest_digest,
            "bundle_files": manifest["files"],
            "tolerance": {"atol": ATOL, "rtol": RTOL},
            "weights": weights,
        },
        "inventory": {
            "categories": {key: categories[key] for key in sorted(categories)},
            "case_ids": [case["id"] for case in cases],
            "negative_case_ids": [case["id"] for case in negative_cases],
            "run_ids": [run["id"] for run in run_observations],
        },
        "cases": cases,
        "negative_cases": negative_cases,
        "runs": run_observations,
    }


def verify_bundle_identity(
    manifest: dict[str, Any],
    bundle: Path,
    profile_path: Path,
    profile: dict[str, Any],
    repository: Path,
) -> None:
    """Bind generation to the current exporter and every declared bundle file."""

    provenance = manifest.get("provenance", {})
    if provenance.get("source_model") != {
        "id": profile["source_model"]["id"],
        "revision": profile["source_model"]["revision"],
    }:
        raise ValueError("bundle source identity does not match the profile")
    expected_profile = {"id": profile["id"], "sha256": sha256_file(profile_path)}
    if provenance.get("profile") != expected_profile:
        raise ValueError("bundle profile identity does not match")
    exporter_files = [
        path.relative_to(repository).as_posix()
        for path in (repository / "tools/export/src/laya_export").glob("*.py")
    ]
    exporter_files.append("tools/export/sdk_probe.py")
    if provenance.get("exporter", {}).get("source_sha256") != framed_digest(
        repository, exporter_files
    ):
        raise ValueError("bundle exporter source identity does not match")
    if provenance.get("lock", {}).get("sha256") != sha256_file(
        repository / "tools/export/uv.lock"
    ):
        raise ValueError("bundle lock identity does not match")
    declared = {item["path"] for item in manifest.get("files", [])}
    actual = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if declared != actual:
        raise ValueError("bundle file inventory mismatch")
    for item in manifest["files"]:
        path = bundle / item["path"]
        if path.stat().st_size != item["size"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"bundle file identity mismatch: {item['path']}")


def input_names() -> tuple[str, ...]:
    return ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")


def batch_arrays(batch: Batch) -> dict[str, Any]:
    """Convert an immutable reference batch to exact ONNX input dtypes."""

    import numpy as np

    return {
        "input_ids": np.asarray(batch.input_ids, dtype=np.int64),
        "attention_mask": np.asarray(batch.attention_mask, dtype=np.int64),
        "marker_pos": np.asarray(batch.marker_pos, dtype=np.int64),
        "marker_mask": np.asarray(batch.marker_mask, dtype=np.bool_),
        "qtype": np.asarray(batch.qtype, dtype=np.int64),
    }


def build_case_observation(
    tokenizer: BundleTokenizer,
    specification: dict[str, Any],
    item: SequenceItem,
    run_id: str,
    row: int,
    logits_row: Any,
    action_logits_row: Any,
    config: dict[str, Any],
    usage: dict[str, int],
    max_len: int,
    head_max_len: int,
) -> dict[str, Any]:
    """Record exact preprocessing, raw, derived, and typed observations."""

    question = specification["question"]
    option_count = len(render_options(question))
    logits = [float(value) for value in logits_row[:option_count]]
    action_logits = [float(value) for value in action_logits_row]
    temperatures = config["temperature"]
    by_options = config["temperature_by_options"]
    raw_temperature, effective_temperature = select_temperature(
        item.qtype, option_count, temperatures, by_options
    )
    probabilities = softmax(logits, effective_temperature)
    action_probabilities = softmax(action_logits, 1.0)
    full = build_sequence(
        tokenizer,
        specification["state"]["value"],
        question,
        max_len=1_000_000,
        head_max_len=head_max_len,
    )
    derived: dict[str, Any] = {
        "probabilities": probabilities,
        "action_probabilities": action_probabilities,
        "entropy_confidence": confidence_from_probabilities(probabilities),
        "selected_index": max(range(option_count), key=probabilities.__getitem__),
    }
    if question["t"] == "score":
        derived["expected_score"] = sum(
            index * value for index, value in enumerate(probabilities)
        )
    if question["t"] == "noul":
        derived["true_probability"] = probabilities[1]
    return {
        "id": specification["id"],
        "categories": specification["categories"],
        "state": specification["state"],
        "question": question,
        "rendered_options": render_options(question),
        "run": {"id": run_id, "row": row},
        "sequence": {
            "input_ids": list(item.input_ids),
            "marker_positions": list(item.marker_positions),
            "qtype": item.qtype,
            "original_input_tokens": len(full.input_ids),
            "effective_input_tokens": len(item.input_ids),
            "truncated": len(full.input_ids) > max_len,
        },
        "raw": {"logits": logits, "act_logits": action_logits},
        "calibration": {
            "raw_temperature": raw_temperature,
            "effective_temperature": clamp_temperature(raw_temperature),
        },
        "derived": derived,
        "typed_result": format_answer(
            question, logits, action_logits, temperatures, by_options
        ),
        "usage": usage,
    }


def case_specifications(
    tokenizer: BundleTokenizer, max_len: int, head_max_len: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return the reviewed, stable synthetic coverage inventory."""

    shared_state = '{"emoji":"🧭","text":"mixed Русский + English","n":1}'
    specifications = [
        case(
            "english-choice",
            ["english", "choice", "batch", "ordered-options"],
            "text",
            shared_state,
            choice("Choose the safest route", [("safe", "lowest risk"), ("fast", "lowest latency"), ("cheap", "lowest cost")]),
        ),
        case(
            "russian-score",
            ["russian", "score", "batch", "ordered-options"],
            "text",
            shared_state,
            score("Оцени срочность", ["не срочно", "обычно", "срочно", "критично"]),
        ),
        case(
            "mixed-mask-noul",
            ["mixed-language", "unicode", "mask-replacement", "noul", "batch"],
            "text",
            shared_state + " <mask>",
            {"t": "noul", "ins": "Is <mask> маршрут acceptable?", "crit": {}},
        ),
        case(
            "unicode-choice",
            ["unicode", "choice", "ordered-options"],
            "text",
            "Café ☕ — مرحبا — 中文 — 👩🏽‍💻",
            choice("Select exact Unicode", [("combining", "Café"), ("rtl", "مرحبا"), ("cjk", "中文"), ("emoji", "👩🏽‍💻")]),
        ),
        case(
            "two-options-choice",
            ["option-count-2", "choice"],
            "text",
            "binary route",
            choice("Pick one", [("left", "go left"), ("right", "go right")]),
        ),
        case(
            "six-options-choice",
            ["option-count-6", "choice"],
            "text",
            "six routes",
            choice("Pick one", [(f"route-{index}", f"route number {index}") for index in range(6)]),
        ),
        case(
            "eleven-options-choice",
            ["option-count-11", "choice"],
            "text",
            "eleven routes",
            choice("Pick one", [(f"route-{index:02d}", f"route number {index}") for index in range(11)]),
        ),
        case("json-null-noul", ["json-primitive", "noul"], "json", "null", {"t": "noul", "ins": "Is null present?", "crit": {}}),
        case("json-bool-choice", ["json-primitive", "choice"], "json", "true", choice("Read boolean", [("false", "false value"), ("true", "true value")])),
        case("json-number-score", ["json-primitive", "score"], "json", "42", score("Rate the number", ["low", "medium", "high"])),
        case("json-string-noul", ["json-primitive", "unicode", "noul"], "json", '"строка"', {"t": "noul", "ins": "Is this a string?", "crit": {"false": "нет", "true": "да"}}),
        case(
            "long-options-choice",
            ["long-options", "choice", "head-budget"],
            "text",
            "option body cap",
            choice("Choose despite long descriptions", [(f"long-{index}", ("detail " * 90) + str(index)) for index in range(4)]),
        ),
    ]
    boundary_question = choice(
        "Boundary input",
        [("keep", "retain the state prefix"), ("drop", "reject silent overflow")],
    )
    empty = build_sequence(
        tokenizer, "", boundary_question, max_len=max_len, head_max_len=head_max_len
    )
    state_token_target = max_len - len(empty.input_ids)
    exact_state = "x " * (state_token_target - 1)
    if len(tokenizer.encode(exact_state)) != state_token_target:
        raise ValueError("cannot construct deterministic exact-fit state")
    specifications.extend(
        [
            case(
                "exact-fit-choice",
                ["boundary", "exact-fit", "choice"],
                "text",
                exact_state,
                boundary_question,
            ),
            case(
                "right-truncated-choice",
                ["boundary", "overflow", "right-truncation", "choice"],
                "text",
                exact_state + "overflow suffix " * 24,
                boundary_question,
            ),
        ]
    )
    impossible_question = choice(
        "Too many options",
        [(f"option-{index:03d}", f"value {index}") for index in range(300)],
    )
    impossible = build_sequence(
        tokenizer,
        "marker safety",
        impossible_question,
        max_len=max_len,
        head_max_len=head_max_len,
    )
    negative_cases = [
        {
            "id": "overflow-default-error",
            "categories": ["boundary", "overflow", "default-rejection"],
            "source_case_id": "right-truncated-choice",
            "expected_error": "input_too_long",
        },
        {
            "id": "impossible-marker-layout-error",
            "categories": ["boundary", "impossible-markers", "head-budget"],
            "question_option_count": 300,
            "retained_marker_count": len(impossible.marker_positions),
            "expected_error": "input_too_long",
        },
    ]
    return specifications, negative_cases


def case(
    identifier: str,
    categories: list[str],
    state_kind: str,
    state_value: str,
    question: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": identifier,
        "categories": categories,
        "state": {"kind": state_kind, "value": state_value},
        "question": question,
    }


def choice(instructions: str, criteria: list[tuple[str, str]]) -> dict[str, Any]:
    return {
        "t": "choice",
        "ins": instructions,
        "crit": OrderedDict(criteria),
    }


def score(instructions: str, levels: list[str]) -> dict[str, Any]:
    return {"t": "score", "ins": instructions, "crit": levels}
