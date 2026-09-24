from __future__ import annotations

import json
from pathlib import Path
import struct
import unittest


class ArchitectureContractTest(unittest.TestCase):
    def test_pinned_checkpoint_header_has_complete_decision_components(self) -> None:
        checkpoint = Path("/tmp/laya-official-052592a/model.safetensors")
        if not checkpoint.exists():
            self.skipTest("external official checkpoint is not present")
        with checkpoint.open("rb") as source:
            header_size = struct.unpack("<Q", source.read(8))[0]
            header = json.loads(source.read(header_size))
        tensors = {name: value for name, value in header.items() if name != "__metadata__"}
        self.assertEqual(len(tensors), 170)
        for prefix in ("encoder.", "head.", "type_emb.", "scorer.", "act_head."):
            self.assertTrue(any(name.startswith(prefix) for name in tensors), prefix)
        self.assertEqual(tensors["type_emb.weight"]["shape"], [3, 768])
        self.assertEqual(tensors["act_head.2.weight"]["shape"], [2, 256])
        self.assertEqual(tensors["temperature"]["shape"], [3])


if __name__ == "__main__":
    unittest.main()
