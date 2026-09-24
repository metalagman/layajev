from __future__ import annotations

from contextlib import redirect_stderr
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from laya_export import exporter
from laya_export.exporter import atomic_publish, reject_overlap


class ExporterSecurityTest(unittest.TestCase):
    def test_exporter_revision_ignores_later_docs_and_rejects_dirty_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "tools/export/src/laya_export/exporter.py"
            probe = root / "tools/export/sdk_probe.py"
            source.parent.mkdir(parents=True)
            probe.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("initial\n", encoding="utf-8")
            probe.write_text("probe\n", encoding="utf-8")

            def git(*arguments: str) -> str:
                return subprocess.run(
                    [
                        "git", "-C", str(root),
                        "-c", "user.name=Test",
                        "-c", "user.email=test@example.com",
                        "-c", "commit.gpgsign=false",
                        *arguments,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()

            git("init", "-q")
            git("add", "tools/export")
            git("commit", "-qm", "exporter")
            revision = git("rev-parse", "HEAD")
            (root / "README.md").write_text("documentation\n", encoding="utf-8")
            git("add", "README.md")
            git("commit", "-qm", "docs")
            self.assertEqual(exporter.exporter_git_revision(root), revision)

            source.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be committed"):
                exporter.exporter_git_revision(root)
            git("add", "tools/export")
            git("commit", "-qm", "change exporter")
            self.assertEqual(exporter.exporter_git_revision(root), git("rev-parse", "HEAD"))

            (source.parent / "untracked.py").write_text("new\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be committed"):
                exporter.exporter_git_revision(root)

    def test_atomic_publish_makes_complete_directory_visible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage = root / ".bundle.staging"
            destination = root / "bundle"
            stage.mkdir()
            (stage / "ready").write_text("complete", encoding="utf-8")
            atomic_publish(stage, destination)
            self.assertFalse(stage.exists())
            self.assertEqual((destination / "ready").read_text(encoding="utf-8"), "complete")

    def test_atomic_publish_never_replaces_concurrent_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage = root / ".bundle.staging"
            destination = root / "bundle"
            stage.mkdir()
            destination.mkdir()
            (stage / "candidate").write_text("new", encoding="utf-8")
            (destination / "ready").write_text("old", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                atomic_publish(stage, destination)
            self.assertEqual((destination / "ready").read_text(encoding="utf-8"), "old")
            self.assertEqual((stage / "candidate").read_text(encoding="utf-8"), "new")

    def test_reject_overlap_checks_both_containment_directions(self) -> None:
        root = Path("/tmp/source").resolve()
        with self.assertRaisesRegex(ValueError, "overlap"):
            reject_overlap(root, root / "output")
        with self.assertRaisesRegex(ValueError, "overlap"):
            reject_overlap(root / "nested", root)

    def test_failed_export_removes_only_owned_stage(self) -> None:
        cases = (
            ("interruption", KeyboardInterrupt()),
            ("resource", OSError("simulated disk exhaustion")),
            ("parity", ValueError("simulated parity failure")),
        )
        for phase, failure in cases:
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / "source"
                sdk = root / "sdk"
                repository = root / "repository"
                output = root / "bundle"
                for directory in (source, sdk, repository):
                    directory.mkdir()
                source_config = source / "rl_agent_config.json"
                source_config.write_text("{}\n", encoding="utf-8")
                sdk_marker = sdk / "marker"
                sdk_marker.write_text("unchanged\n", encoding="utf-8")
                verifier = root / "bundlecheck"
                verifier.touch()

                def write_graph(_model: object, path: Path) -> None:
                    path.write_bytes(b"incomplete graph")

                build_error = failure if phase == "interruption" else None
                export_error = failure if phase == "resource" else None
                parity_error = failure if phase == "parity" else None
                progress = io.StringIO()
                with (
                    mock.patch.object(exporter, "load_profile", return_value={}),
                    mock.patch.object(exporter, "verify_source"),
                    mock.patch.object(exporter, "verify_sdk"),
                    mock.patch.object(
                        exporter,
                        "build_model",
                        return_value=object(),
                        side_effect=build_error,
                    ),
                    mock.patch.object(exporter, "load_exact_weights", return_value={}),
                    mock.patch.object(
                        exporter,
                        "export_onnx",
                        side_effect=export_error or write_graph,
                    ),
                    mock.patch.object(exporter, "validate_onnx", return_value={}),
                    mock.patch.object(
                        exporter,
                        "compare_runtime",
                        return_value={},
                        side_effect=parity_error,
                    ),
                ):
                    with redirect_stderr(progress):
                        with self.assertRaises(type(failure)):
                            exporter.export_bundle(
                                root / "profile.json",
                                source,
                                sdk,
                                output,
                                0,
                                repository,
                                verifier,
                            )

                self.assertIn("convert: verifying source and SDK", progress.getvalue())
                self.assertNotIn("convert: publishing verified bundle", progress.getvalue())

                self.assertFalse(output.exists())
                self.assertEqual(list(root.glob(".bundle.staging-*")), [])
                self.assertEqual(source_config.read_text(encoding="utf-8"), "{}\n")
                self.assertEqual(sdk_marker.read_text(encoding="utf-8"), "unchanged\n")

    def test_existing_output_is_preserved_without_staging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            sdk = root / "sdk"
            repository = root / "repository"
            output = root / "bundle"
            for directory in (source, sdk, repository, output):
                directory.mkdir()
            ready = output / "ready"
            ready.write_text("previous bundle\n", encoding="utf-8")
            verifier = root / "bundlecheck"
            verifier.touch()
            with (
                mock.patch.object(exporter, "load_profile", return_value={}),
                mock.patch.object(exporter, "verify_source"),
                mock.patch.object(exporter, "verify_sdk"),
            ):
                with self.assertRaisesRegex(ValueError, "already exists"):
                    exporter.export_bundle(
                        root / "profile.json",
                        source,
                        sdk,
                        output,
                        0,
                        repository,
                        verifier,
                    )

            self.assertEqual(ready.read_text(encoding="utf-8"), "previous bundle\n")
            self.assertEqual(list(root.glob(".bundle.staging-*")), [])


if __name__ == "__main__":
    unittest.main()
