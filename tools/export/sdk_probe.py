"""Run deterministic semantic observations against verified upstream source."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys
import types


if len(sys.argv) != 2:
    raise SystemExit("usage: sdk_probe.py SDK_ROOT")
sdk_root = Path(sys.argv[1]).resolve(strict=True)


class Vector(list):
    def __getitem__(self, key):
        value = super().__getitem__(key)
        return Vector(value) if isinstance(key, slice) else value

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Vector(value * other for value in self)
        return Vector(left * right for left, right in zip(self, other, strict=True))

    __rmul__ = __mul__

    def __neg__(self):
        return Vector(-value for value in self)

    def sum(self):
        return sum(self)

    def tolist(self):
        return list(self)


class Matrix:
    def __init__(self, shape, fill):
        self.values = [[fill for _ in range(shape[1])] for _ in range(shape[0])]

    def __setitem__(self, key, value):
        row, columns = key
        indexes = list(range(*columns.indices(len(self.values[row]))))
        if isinstance(value, (int, float, bool)):
            values = [value] * len(indexes)
        else:
            values = list(value)
        for index, item in zip(indexes, values, strict=True):
            self.values[row][index] = item

    def tolist(self):
        return [list(row) for row in self.values]


numpy = types.ModuleType("numpy")
numpy.ndarray = Vector
numpy.asarray = lambda values: Vector(float(value) for value in values)
numpy.log = lambda values: Vector(math.log(value) for value in values)


def numpy_clip(value, low, high):
    if isinstance(value, Vector):
        return Vector(min(high, max(low, item)) for item in value)
    return min(high, max(low, value))


numpy.clip = numpy_clip
sys.modules["numpy"] = numpy

torch = types.ModuleType("torch")
torch.Tensor = object
torch.dtype = object
torch.long = object()
torch.bool = object()
torch.float32 = object()
torch.bfloat16 = object()
torch.float16 = object()
torch.full = lambda shape, value, dtype=None: Matrix(shape, value)
torch.zeros = lambda shape, dtype=None: Matrix(shape, False if dtype is torch.bool else 0)
torch.tensor = lambda values, dtype=None: Vector(values)
nn = types.ModuleType("torch.nn")
nn.Module = object
torch.nn = nn
sys.modules["torch"] = torch
sys.modules["torch.nn"] = nn

spec = importlib.util.spec_from_file_location("pinned_laya_common", sdk_root / "laya" / "common.py")
if spec is None or spec.loader is None:
    raise SystemExit("cannot load pinned laya/common.py")
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


class CodepointTokenizer:
    cls_token_id = 2
    sep_token_id = 3
    mask_token_id = 4
    mask_token = "<mask>"
    pad_token_id = 0

    def __call__(self, text, *, add_special_tokens):
        if add_special_tokens:
            raise ValueError("probe never adds special tokens")
        return {"input_ids": [ord(character) + 10 for character in text]}


QUESTIONS = (
    {"t": "choice", "ins": "Choose <mask> route: привет", "crit": {"alpha": "first", "beta": {"rank": 2}}},
    {"t": "score", "ins": "Rate urgency", "crit": ["low", "medium", "high"]},
    {"t": "noul", "ins": "Does it hold?", "crit": {"false": "нет", "true": "да"}},
)
STATE = '{"emoji":"🧭","text":"mixed Русский + English","n":1}'


def main():
    tokenizer = CodepointTokenizer()
    raw_items = []
    sequences = []
    for question in QUESTIONS:
        ids, markers = common.build_sequence(tokenizer, STATE, question, 1024, 256)
        qtype = common.QTYPES[question["t"]]
        raw_items.append({"ids": ids, "markers": markers, "qtype": qtype})
        sequences.append({"input_ids": ids, "marker_positions": markers, "qtype": qtype})
    batch = common.collate_items([raw_items], tokenizer.pad_token_id)
    observations = {
        "rendered_options": [common.render_options(question) for question in QUESTIONS],
        "sequences": sequences,
        "batch": {
            "input_ids": batch["input_ids"].tolist(),
            "attention_mask": batch["attention_mask"].tolist(),
            "marker_pos": batch["marker_pos"].tolist(),
            "marker_mask": batch["marker_mask"].tolist(),
            "qtype": batch["qtype"].tolist(),
        },
        "temperatures": [common.clamp_temperature(value) for value in (None, "bad", math.nan, math.inf, 0.1, 1.25, 9.0)],
        "temperature_buckets": [common.temp_bucket(qtype, count) for qtype in range(3) for count in (2, 3, 6, 11)],
        "confidences": [
            common.confidence_from_probs(Vector(values), len(values))
            for values in ([1.0], [0.5, 0.5], [0.8, 0.2], [0.7, 0.2, 0.1])
        ],
    }
    json.dump(observations, sys.stdout, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
