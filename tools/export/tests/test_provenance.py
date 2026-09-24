from __future__ import annotations

import hashlib
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest import mock

from laya_export.fixture import build_fixture, canonical_json
from laya_export.provenance import (
    EXPECTED_SDK_FILES,
    EXPECTED_SDK_REVISION,
    EXPECTED_SOURCE_ID,
    EXPECTED_SOURCE_REVISION,
    load_profile,
    verify_sdk,
    verify_source,
)
from laya_export.sdk_compare import first_mismatch


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PROFILE = REPOSITORY_ROOT / "tools" / "export" / "profiles" / "laya-multilingual-v1.json"


class ProvenanceTest(unittest.TestCase):
    def test_repository_profile_has_exact_official_identities(self) -> None:
        profile = load_profile(PROFILE)
        self.assertEqual(profile["source_model"]["id"], EXPECTED_SOURCE_ID)
        self.assertEqual(profile["source_model"]["revision"], EXPECTED_SOURCE_REVISION)

    def test_source_verification_is_exact_and_rejects_code(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "model.safetensors"
            artifact.write_bytes(b"safe tensor bytes")
            data = artifact.read_bytes()
            profile = {
                "source_files": [
                    {
                        "path": "model.safetensors",
                        "size": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                ]
            }
            self.assertEqual(verify_source(profile, root)[0]["path"], "model.safetensors")
            (root / "model.py").write_text("raise RuntimeError", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "executable"):
                verify_source(profile, root)

    def test_source_verification_rejects_pickle_and_escaping_profile_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "weights.bin").write_bytes(b"pickle-capable")
            with self.assertRaisesRegex(ValueError, "pickle-capable"):
                verify_source({"source_files": []}, root)
            (root / "weights.bin").unlink()
            with self.assertRaisesRegex(ValueError, "invalid source profile path"):
                verify_source(
                    {"source_files": [{"path": "../outside", "size": 0, "sha256": "0" * 64}]},
                    root,
                )

    def test_sdk_verification_reads_version_without_importing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "laya"
            package.mkdir()
            (package / "__init__.py").write_text('__version__ = "0.3.5"\nraise RuntimeError\n', encoding="utf-8")
            (package / "agent.py").touch()
            (package / "common.py").touch()
            (root / "LICENSE").touch()
            (root / ".git").mkdir()
            with (
                mock.patch(
                    "laya_export.provenance.run_git",
                    side_effect=[EXPECTED_SDK_REVISION, ""],
                ),
                mock.patch(
                    "laya_export.provenance.sha256_file",
                    side_effect=lambda path: EXPECTED_SDK_FILES[
                        path.relative_to(root).as_posix()
                    ],
                ),
            ):
                identity = verify_sdk(root)
            self.assertEqual(identity["revision"], EXPECTED_SDK_REVISION)

    def test_sdk_worktree_verification_rejects_wrong_file_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "laya"
            package.mkdir()
            (package / "__init__.py").write_text('__version__ = "0.3.5"\n', encoding="utf-8")
            (package / "agent.py").touch()
            (package / "common.py").touch()
            (root / "LICENSE").touch()
            (root / ".git").mkdir()
            with (
                mock.patch(
                    "laya_export.provenance.run_git",
                    side_effect=[EXPECTED_SDK_REVISION, ""],
                ),
                mock.patch("laya_export.provenance.sha256_file", return_value="0" * 64),
            ):
                with self.assertRaisesRegex(ValueError, "SDK source digest mismatch"):
                    verify_sdk(root)

    def test_sdk_verification_rejects_wrong_revision_before_import(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".git").mkdir()
            with mock.patch("laya_export.provenance.run_git", return_value="0" * 40):
                with self.assertRaisesRegex(ValueError, "revision mismatch"):
                    verify_sdk(root)

    def test_fixture_is_deterministic_and_binds_all_identities(self) -> None:
        with mock.patch.object(socket, "create_connection", side_effect=AssertionError("network attempted")):
            first = canonical_json(build_fixture(PROFILE))
            second = canonical_json(build_fixture(PROFILE))
        self.assertEqual(first, second)
        parsed = json.loads(first)
        self.assertEqual(parsed["sdk_revision"], EXPECTED_SDK_REVISION)
        self.assertEqual(parsed["source_revision"], EXPECTED_SOURCE_REVISION)
        self.assertTrue(parsed["synthetic_only"])

    def test_comparison_reports_first_nested_mismatch(self) -> None:
        self.assertIsNone(first_mismatch({"a": [1.0]}, {"a": [1.0 + 1e-14]}))
        self.assertEqual(first_mismatch({"a": [1]}, {"a": [2]}), "$.a[0]")


if __name__ == "__main__":
    unittest.main()
