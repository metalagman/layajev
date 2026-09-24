# Contributing to layajev

This repository contains the CLI, Jev-compatible HTTP adapter, pinned model exporter, and release/container operations. The reusable in-process inference library is maintained in [laya-go](https://github.com/metalagman/laya-go).

Use Go 1.26.6 and Go Task v3.53.1 or newer. From a checkout:

```sh
task fmt
task tidy
task verify
task test
task test:race
task lint
task vuln
```

`task fmt` and `task tidy` check without changing files; run `gofmt` and `go mod tidy` before committing if they report differences. Keep tests focused on observable behavior and failure cases. Do not commit model weights, native libraries, credentials, or generated bundles.

The offline exporter has separate Python tests. Install its locked environment with `uv sync --project tools/export --frozen`, then run `task reference:test`. Native integration needs explicitly supplied, checksum-verified ONNX Runtime, tokenizer archive, and bundle paths; see [CLI operations](docs/layajev-runbook.md) and [native dependencies](docs/native-dependencies.md). Do not claim a new platform, precision, or model is supported without native integration evidence.

The [npm release runbook](docs/layajev-npm-release.md#prepare-a-release) documents Omnidist staging and tag-based publication. A pull request should include the relevant test results and update user-facing documentation when behavior changes.
