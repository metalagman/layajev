"""Pinned Laya v0.3.5 preprocessing and postprocessing semantics.

This module intentionally uses only the Python standard library. Tokenization
is supplied through a narrow protocol so hermetic tests need no model runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Mapping, Protocol, Sequence


QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {value: key for key, value in QTYPES.items()}
TEMP_MIN = 0.5
TEMP_MAX = 5.0


class Tokenizer(Protocol):
    """Minimal tokenizer surface needed by the pinned preprocessing path."""

    cls_token_id: int
    sep_token_id: int
    mask_token_id: int
    mask_token: str
    pad_token_id: int

    def encode(self, text: str) -> list[int]:
        """Encode text without adding special tokens."""


@dataclass(frozen=True)
class SequenceItem:
    """One exact model sequence and its option markers."""

    input_ids: tuple[int, ...]
    marker_positions: tuple[int, ...]
    qtype: int


@dataclass(frozen=True)
class Batch:
    """Five complete, row-major model inputs represented as immutable tuples."""

    input_ids: tuple[tuple[int, ...], ...]
    attention_mask: tuple[tuple[int, ...], ...]
    marker_pos: tuple[tuple[int, ...], ...]
    marker_mask: tuple[tuple[bool, ...], ...]
    qtype: tuple[int, ...]


def render_criterion(value: Any) -> str:
    """Render a criterion exactly like pinned upstream common.py."""

    if isinstance(value, str):
        return value
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(", ", ": "),
        default=str,
    )


def render_options(question: Mapping[str, Any]) -> list[str]:
    """Render option texts in input order; noul is always false then true."""

    question_type = require_question_type(question)
    criteria = question.get("crit")
    if question_type == "choice":
        if not isinstance(criteria, Mapping) or not criteria:
            raise ValueError("choice criteria must be a non-empty mapping")
        return [
            str(key)
            if value is None or value == ""
            else f"{key}: {render_criterion(value)}"
            for key, value in criteria.items()
        ]
    if question_type == "score":
        if not isinstance(criteria, Sequence) or isinstance(criteria, (str, bytes)) or not criteria:
            raise ValueError("score criteria must be a non-empty sequence")
        return [
            f"level {index}: {render_criterion(value)}"
            for index, value in enumerate(criteria)
        ]

    noul_criteria = criteria or {}
    if not isinstance(noul_criteria, Mapping):
        raise ValueError("noul criteria must be a mapping when present")
    false_criterion = noul_criteria.get("false")
    true_criterion = noul_criteria.get("true")
    return [
        "false: "
        + (
            render_criterion(false_criterion)
            if false_criterion not in (None, "")
            else "no, the statement does not hold"
        ),
        "true: "
        + (
            render_criterion(true_criterion)
            if true_criterion not in (None, "")
            else "yes, the statement holds"
        ),
    ]


def build_sequence(
    tokenizer: Tokenizer,
    state: str,
    question: Mapping[str, Any],
    max_len: int = 1024,
    head_max_len: int = 256,
    option_order: Sequence[int] | None = None,
    truncate_left: bool = False,
) -> SequenceItem:
    """Build the exact pinned sequence without parsing the State string."""

    if not isinstance(state, str):
        raise TypeError("state must be the exact text or compact JSON string")
    if max_len <= 0 or head_max_len <= 0:
        raise ValueError("sequence budgets must be positive")
    question_type = require_question_type(question)
    instructions = str(question["ins"]).replace(tokenizer.mask_token, " ")
    head_ids = tokenizer.encode(f"{question_type} question: {instructions}")
    options = render_options(question)
    order = list(option_order) if option_order is not None else list(range(len(options)))
    if sorted(order) != list(range(len(options))):
        raise ValueError("option_order must be a permutation of option indexes")
    option_ids = [
        [tokenizer.mask_token_id]
        + tokenizer.encode(" " + options[index].replace(tokenizer.mask_token, " "))[:48]
        for index in order
    ]
    option_budget = head_max_len - sum(len(option) for option in option_ids)
    if option_budget < 16:
        per_option = max(4, (head_max_len - 16) // max(1, len(option_ids)))
        option_ids = [option[:per_option] for option in option_ids]
        option_budget = head_max_len - sum(len(option) for option in option_ids)
    head_ids = head_ids[: max(8, option_budget)]

    input_ids = [tokenizer.cls_token_id, *head_ids, tokenizer.sep_token_id]
    markers: list[int] = []
    for option in option_ids:
        markers.append(len(input_ids))
        input_ids.extend(option)
    input_ids.append(tokenizer.sep_token_id)
    room = max(0, max_len - len(input_ids) - 1)
    state_ids = tokenizer.encode(state.replace(tokenizer.mask_token, " "))
    state_ids = state_ids[-room:] if truncate_left else state_ids[:room]
    input_ids.extend(state_ids)
    input_ids.append(tokenizer.sep_token_id)
    input_ids = input_ids[:max_len]
    markers = [marker for marker in markers if marker < max_len]
    return SequenceItem(tuple(input_ids), tuple(markers), QTYPES[question_type])


def collate(items: Sequence[SequenceItem], pad_id: int) -> Batch:
    """Pad sequence and marker axes independently in input order."""

    if not items:
        raise ValueError("at least one sequence item is required")
    sequence_len = max(len(item.input_ids) for item in items)
    marker_len = max(len(item.marker_positions) for item in items)
    input_ids: list[tuple[int, ...]] = []
    attention: list[tuple[int, ...]] = []
    marker_pos: list[tuple[int, ...]] = []
    marker_mask: list[tuple[bool, ...]] = []
    for item in items:
        padding = sequence_len - len(item.input_ids)
        marker_padding = marker_len - len(item.marker_positions)
        input_ids.append(item.input_ids + (pad_id,) * padding)
        attention.append((1,) * len(item.input_ids) + (0,) * padding)
        marker_pos.append(item.marker_positions + (0,) * marker_padding)
        marker_mask.append((True,) * len(item.marker_positions) + (False,) * marker_padding)
    return Batch(
        tuple(input_ids),
        tuple(attention),
        tuple(marker_pos),
        tuple(marker_mask),
        tuple(item.qtype for item in items),
    )


def clamp_temperature(value: Any, low: float = TEMP_MIN, high: float = TEMP_MAX) -> float:
    """Clamp a finite numeric temperature, falling back to 1.0 otherwise."""

    try:
        temperature = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(temperature):
        return 1.0
    return min(high, max(low, temperature))


def temperature_bucket(qtype: int, option_count: int) -> str:
    """Return the pinned qtype/cardinality calibration bucket."""

    if qtype not in QTYPE_NAMES:
        raise ValueError(f"unknown qtype: {qtype}")
    size = "2" if option_count <= 2 else "3-5" if option_count <= 5 else "6-10" if option_count <= 10 else "11+"
    return f"{QTYPE_NAMES[qtype]}:{size}"


def softmax(logits: Sequence[float], temperature: float) -> list[float]:
    """Compute stable softmax after temperature scaling."""

    if not logits:
        raise ValueError("at least one logit is required")
    if not all(math.isfinite(value) for value in logits):
        raise ValueError("logits must be finite")
    scale = clamp_temperature(temperature)
    scaled = [value / scale for value in logits]
    maximum = max(scaled)
    exponents = [math.exp(value - maximum) for value in scaled]
    total = sum(exponents)
    return [value / total for value in exponents]


def confidence_from_probabilities(probabilities: Sequence[float]) -> float:
    """Return one minus normalized Shannon entropy."""

    count = len(probabilities)
    if count < 2:
        return 1.0
    entropy = -sum(value * math.log(max(1e-12, min(1.0, value))) for value in probabilities)
    return min(1.0, max(0.0, 1.0 - entropy / math.log(count)))


def select_temperature(
    qtype: int,
    option_count: int,
    temperatures: Sequence[Any],
    temperatures_by_options: Mapping[str, Any],
) -> tuple[Any, float]:
    """Return raw and effective temperature using bucket-before-qtype order."""

    if len(temperatures) != len(QTYPES):
        raise ValueError("temperature must contain exactly three qtype values")
    raw = temperatures_by_options.get(
        temperature_bucket(qtype, option_count), temperatures[qtype]
    )
    return raw, clamp_temperature(raw)


def format_answer(
    question: Mapping[str, Any],
    logits: Sequence[float],
    action_logits: Sequence[float],
    temperatures: Sequence[Any],
    temperatures_by_options: Mapping[str, Any],
) -> dict[str, Any]:
    """Format one upstream-compatible typed answer from raw graph outputs."""

    question_type = require_question_type(question)
    options = render_options(question)
    if len(logits) != len(options):
        raise ValueError("logit count must equal option count")
    raw_temperature, effective_temperature = select_temperature(
        QTYPES[question_type], len(options), temperatures, temperatures_by_options
    )
    probabilities = softmax(logits, effective_temperature)
    action_probabilities = softmax(action_logits, 1.0)
    confidence = round(confidence_from_probabilities(probabilities), 4)
    action = {"act_probability": round(action_probabilities[0], 4)}
    common = {
        "confidence": confidence,
        "action": action,
        "calibration": {
            "raw_temperature": raw_temperature,
            "effective_temperature": effective_temperature,
        },
    }
    if question_type == "choice":
        criteria = question["crit"]
        keys = [str(key) for key in criteria]
        best = max(range(len(probabilities)), key=probabilities.__getitem__)
        return {
            "type": "choice",
            "choice": keys[best],
            "probabilities": {
                key: round(value, 4) for key, value in zip(keys, probabilities, strict=True)
            },
            **common,
        }
    if question_type == "score":
        criteria = question["crit"]
        score = sum(index * value for index, value in enumerate(probabilities))
        return {
            "type": "score",
            "score": round(score, 4),
            "legend": {str(index): value for index, value in enumerate(criteria)},
            "probabilities": {
                str(index): round(value, 4) for index, value in enumerate(probabilities)
            },
            **common,
        }
    probability_true = probabilities[1]
    return {
        "type": "noul",
        "noul": round(probability_true, 4),
        "confidence": round(max(probability_true, 1.0 - probability_true), 4),
        "action": action,
        "calibration": common["calibration"],
    }


def require_question_type(question: Mapping[str, Any]) -> str:
    """Validate and return a supported question type."""

    question_type = question.get("t")
    if question_type not in QTYPES:
        raise ValueError(f"unsupported question type: {question_type!r}")
    if "ins" not in question:
        raise ValueError("question instructions are required")
    return str(question_type)


class CodepointTokenizer:
    """Deterministic tokenizer used only for SDK semantic comparison tests."""

    cls_token_id = 2
    sep_token_id = 3
    mask_token_id = 4
    mask_token = "<mask>"
    pad_token_id = 0

    def encode(self, text: str) -> list[int]:
        """Map each Unicode code point to a stable non-special token ID."""

        return [ord(character) + 10 for character in text]

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        if add_special_tokens:
            raise ValueError("reference comparison never adds tokenizer special tokens")
        return {"input_ids": self.encode(text)}
