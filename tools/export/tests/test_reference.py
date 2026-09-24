from __future__ import annotations

import math
import unittest

from laya_export.reference import (
    CodepointTokenizer,
    build_sequence,
    clamp_temperature,
    collate,
    confidence_from_probabilities,
    format_answer,
    render_options,
    select_temperature,
    softmax,
    temperature_bucket,
)


class TrackingTokenizer(CodepointTokenizer):
    def __init__(self) -> None:
        self.texts: list[str] = []

    def encode(self, text: str) -> list[int]:
        self.texts.append(text)
        return super().encode(text)


class ReferenceTest(unittest.TestCase):
    def test_render_options_preserves_order_and_upstream_defaults(self) -> None:
        self.assertEqual(
            render_options(
                {
                    "t": "choice",
                    "ins": "choose",
                    "crit": {"a": None, "b": "second", "c": False, "d": 0},
                }
            ),
            ["a", "b: second", "c: false", "d: 0"],
        )
        self.assertEqual(
            render_options({"t": "score", "ins": "rate", "crit": ["low", {"high": True}]}),
            ["level 0: low", 'level 1: {"high": true}'],
        )
        self.assertEqual(
            render_options({"t": "noul", "ins": "statement"}),
            [
                "false: no, the statement does not hold",
                "true: yes, the statement holds",
            ],
        )

    def test_sequence_keeps_exact_compact_json_string_and_replaces_mask(self) -> None:
        tokenizer = TrackingTokenizer()
        state = '{"b":2,"a":"<mask>"}'
        item = build_sequence(
            tokenizer,
            state,
            {
                "t": "choice",
                "ins": "pick <mask> now",
                "crit": {"first": "one <mask>", "second": "two"},
            },
        )
        self.assertEqual(item.qtype, 0)
        self.assertEqual(len(item.marker_positions), 2)
        self.assertIn('{"b":2,"a":" "}', tokenizer.texts)
        self.assertNotIn('{"a":" ","b":2}', tokenizer.texts)
        self.assertTrue(all("<mask>" not in text for text in tokenizer.texts))
        self.assertEqual(item.input_ids[0], tokenizer.cls_token_id)
        self.assertEqual(item.input_ids[-1], tokenizer.sep_token_id)

    def test_head_allocation_caps_options_without_losing_markers(self) -> None:
        tokenizer = CodepointTokenizer()
        question = {
            "t": "choice",
            "ins": "instructions" * 100,
            "crit": {f"option-{index}": "description" * 100 for index in range(12)},
        }
        item = build_sequence(tokenizer, "state", question, max_len=1024, head_max_len=64)
        self.assertEqual(len(item.marker_positions), 12)
        self.assertEqual(list(item.marker_positions), sorted(item.marker_positions))
        self.assertLessEqual(len(item.input_ids), 1024)

    def test_sequence_exposes_marker_loss_at_impossible_budget(self) -> None:
        tokenizer = CodepointTokenizer()
        question = {
            "t": "choice",
            "ins": "x",
            "crit": {str(index): "y" for index in range(10)},
        }
        item = build_sequence(tokenizer, "state", question, max_len=8, head_max_len=16)
        self.assertLess(len(item.marker_positions), len(render_options(question)))

    def test_collate_pads_sequence_and_marker_axes_independently(self) -> None:
        tokenizer = CodepointTokenizer()
        first = build_sequence(tokenizer, "short", {"t": "noul", "ins": "a"})
        second = build_sequence(
            tokenizer,
            "longer",
            {"t": "choice", "ins": "b", "crit": {"x": "x", "y": "y", "z": "z"}},
        )
        batch = collate([first, second], tokenizer.pad_token_id)
        self.assertEqual(len(batch.input_ids[0]), len(batch.input_ids[1]))
        self.assertEqual(len(batch.marker_pos[0]), 3)
        self.assertEqual(batch.marker_pos[0][-1], 0)
        self.assertFalse(batch.marker_mask[0][-1])
        self.assertEqual(batch.qtype, (2, 0))

    def test_temperature_selection_and_stable_softmax(self) -> None:
        self.assertEqual(clamp_temperature(None), 1.0)
        self.assertEqual(clamp_temperature(math.nan), 1.0)
        self.assertEqual(clamp_temperature(0.1), 0.5)
        self.assertEqual(clamp_temperature(9), 5.0)
        self.assertEqual(temperature_bucket(0, 11), "choice:11+")
        raw, effective = select_temperature(0, 2, [1.2, 1.3, 1.4], {"choice:2": 0.1})
        self.assertEqual(raw, 0.1)
        self.assertEqual(effective, 0.5)
        probabilities = softmax([10000.0, 9999.0], 1.0)
        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertGreater(probabilities[0], probabilities[1])
        self.assertEqual(confidence_from_probabilities([1.0]), 1.0)

    def test_softmax_properties_hold_across_synthetic_ranges(self) -> None:
        for count in range(1, 17):
            logits = [float(index - count // 2) * 13.5 for index in range(count)]
            for temperature in (0.5, 1.0, 5.0):
                probabilities = softmax(logits, temperature)
                shifted = softmax([value + 10000.0 for value in logits], temperature)
                self.assertAlmostEqual(sum(probabilities), 1.0, places=14)
                self.assertTrue(all(0.0 <= value <= 1.0 for value in probabilities))
                for original, translated in zip(probabilities, shifted, strict=True):
                    self.assertTrue(math.isclose(original, translated, rel_tol=1e-13, abs_tol=1e-13))

    def test_format_answers_keeps_answer_and_action_confidence_separate(self) -> None:
        answer = format_answer(
            {"t": "choice", "ins": "pick", "crit": {"a": "first", "b": "second"}},
            [2.0, 0.0],
            [-2.0, 2.0],
            [1.0, 1.0, 1.0],
            {},
        )
        self.assertEqual(answer["choice"], "a")
        self.assertNotEqual(answer["confidence"], answer["action"]["act_probability"])
        self.assertEqual(answer["calibration"]["raw_temperature"], 1.0)

        score = format_answer(
            {"t": "score", "ins": "rate", "crit": ["low", "high"]},
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0, 1.0],
            {},
        )
        self.assertEqual(score["score"], 0.5)
        self.assertEqual(list(score["legend"]), ["0", "1"])

        noul = format_answer(
            {"t": "noul", "ins": "true?"},
            [0.0, 2.0],
            [0.0, 0.0],
            [1.0, 1.0, 1.0],
            {},
        )
        self.assertEqual(noul["noul"], noul["confidence"])


if __name__ == "__main__":
    unittest.main()
