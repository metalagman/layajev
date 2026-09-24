"""Deterministic synthetic observations for pinned-SDK comparisons."""

from __future__ import annotations

from typing import Any

from .reference import (
    CodepointTokenizer,
    build_sequence,
    clamp_temperature,
    collate,
    confidence_from_probabilities,
    render_options,
    temperature_bucket,
)


QUESTIONS: tuple[dict[str, Any], ...] = (
    {
        "t": "choice",
        "ins": "Choose <mask> route: привет",
        "crit": {"alpha": "first", "beta": {"rank": 2}},
    },
    {
        "t": "score",
        "ins": "Rate urgency",
        "crit": ["low", "medium", "high"],
    },
    {
        "t": "noul",
        "ins": "Does it hold?",
        "crit": {"false": "нет", "true": "да"},
    },
)
STATE = '{"emoji":"🧭","text":"mixed Русский + English","n":1}'


def build_observations() -> dict[str, Any]:
    """Build JSON-compatible observations without model or network access."""

    tokenizer = CodepointTokenizer()
    items = [
        build_sequence(tokenizer, STATE, question, max_len=1024, head_max_len=256)
        for question in QUESTIONS
    ]
    batch = collate(items, tokenizer.pad_token_id)
    return {
        "rendered_options": [render_options(question) for question in QUESTIONS],
        "sequences": [
            {
                "input_ids": list(item.input_ids),
                "marker_positions": list(item.marker_positions),
                "qtype": item.qtype,
            }
            for item in items
        ],
        "batch": {
            "input_ids": [list(row) for row in batch.input_ids],
            "attention_mask": [list(row) for row in batch.attention_mask],
            "marker_pos": [list(row) for row in batch.marker_pos],
            "marker_mask": [list(row) for row in batch.marker_mask],
            "qtype": list(batch.qtype),
        },
        "temperatures": [
            clamp_temperature(value)
            for value in (None, "bad", float("nan"), float("inf"), 0.1, 1.25, 9.0)
        ],
        "temperature_buckets": [
            temperature_bucket(qtype, count)
            for qtype in range(3)
            for count in (2, 3, 6, 11)
        ],
        "confidences": [
            confidence_from_probabilities(values)
            for values in ([1.0], [0.5, 0.5], [0.8, 0.2], [0.7, 0.2, 0.1])
        ],
    }
