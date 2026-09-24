"""Command-line entry point for offline reference operations."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from .corpus import generate_corpus, write_corpus
from .exporter import export_bundle
from .fixture import build_fixture, canonical_json
from .provenance import load_profile, verify_sdk, verify_source
from .sdk_compare import compare_sdk


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="laya-go-reference")
    result.add_argument("--profile", type=Path, required=True)
    commands = result.add_subparsers(dest="command", required=True)
    source = commands.add_parser("verify-source")
    source.add_argument("--source", type=Path, required=True)
    sdk = commands.add_parser("verify-sdk")
    sdk.add_argument("--sdk", type=Path, required=True)
    comparison = commands.add_parser("compare-sdk")
    comparison.add_argument("--sdk", type=Path, required=True)
    comparison.add_argument("--sdk-python", type=Path, required=True)
    comparison.add_argument("--probe", type=Path, required=True)
    export = commands.add_parser("export")
    export.add_argument("--source", type=Path, required=True)
    export.add_argument("--sdk", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--epoch", type=int, required=True)
    export.add_argument("--repository-root", type=Path, required=True)
    export.add_argument("--verifier", type=Path, required=True)
    corpus = commands.add_parser("corpus")
    corpus.add_argument("--source", type=Path, required=True)
    corpus.add_argument("--sdk", type=Path, required=True)
    corpus.add_argument("--bundle", type=Path, required=True)
    corpus.add_argument("--repository-root", type=Path, required=True)
    corpus.add_argument("--output", type=Path, required=True)
    commands.add_parser("fixture")
    return result


def main(arguments: list[str] | None = None) -> int:
    args = parser().parse_args(arguments)
    profile = load_profile(args.profile)
    if args.command == "verify-source":
        print(canonical_json({"source_files": verify_source(profile, args.source)}), end="")
    elif args.command == "verify-sdk":
        print(canonical_json(verify_sdk(args.sdk)), end="")
    elif args.command == "compare-sdk":
        print(canonical_json(compare_sdk(args.sdk, args.sdk_python, args.probe)), end="")
    elif args.command == "fixture":
        print(canonical_json(build_fixture(args.profile)), end="")
    elif args.command == "export":
        print(
            canonical_json(
                export_bundle(
                    args.profile,
                    args.source,
                    args.sdk,
                    args.output,
                    args.epoch,
                    args.repository_root,
                    args.verifier,
                )
            ),
            end="",
        )
    elif args.command == "corpus":
        corpus = generate_corpus(
            args.profile,
            args.source,
            args.sdk,
            args.bundle,
            args.repository_root,
        )
        write_corpus(args.output, corpus)
        print(
            canonical_json(
                {
                    "cases": len(corpus["cases"]),
                    "negative_cases": len(corpus["negative_cases"]),
                    "output": str(args.output),
                    "runs": len(corpus["runs"]),
                }
            ),
            end="",
        )
    else:
        raise AssertionError(f"unhandled command: {args.command}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"laya-go-reference: {error}", file=sys.stderr)
        raise SystemExit(2) from error
