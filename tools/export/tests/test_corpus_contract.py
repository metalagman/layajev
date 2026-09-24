from __future__ import annotations

import unittest

from laya_export.corpus import case_specifications


class BoundaryTokenizer:
    cls_token_id = 2
    sep_token_id = 3
    mask_token_id = 4
    pad_token_id = 0
    mask_token = "<mask>"

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        return list(range(10, 11 + text.count(" ")))


class CorpusContractTest(unittest.TestCase):
    def test_reviewed_inventory_and_boundaries_are_stable(self) -> None:
        cases, negative = case_specifications(BoundaryTokenizer(), 1024, 256)
        self.assertEqual(len(cases), 14)
        self.assertEqual(len(negative), 2)
        self.assertEqual(cases[0]["id"], "english-choice")
        self.assertEqual(cases[-2]["id"], "exact-fit-choice")
        self.assertEqual(cases[-1]["id"], "right-truncated-choice")
        self.assertEqual(negative[0]["expected_error"], "input_too_long")
        self.assertLess(negative[1]["retained_marker_count"], 300)


if __name__ == "__main__":
    unittest.main()
