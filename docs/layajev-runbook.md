# `layajev` operations

`layajev` serves a documented subset of the [TypeSafe Jev API](https://api.typesafe.ai/openapi.json) over one local Laya bundle. It is a protocol adapter, **not** the Jev model or a promise of identical decisions. The `laya-go` library remains acquisition-free. Serving uses local, in-process inference via Go ADK. The only network acquisition is the operator-invoked `fetch` command. This is the source-checkout runbook; see [npm release and npx operations](layajev-npm-release.md) for the published `linux/amd64` CLI and [container operations](layajev-container-runbook.md) for the production image. The released CLI does not include model weights; the production container builds a verified bundle into the image.

## Prerequisites and preparation

Run these commands from the `layajev` checkout. Use Go 1.26.6+, Task v3.53.1+, and the [native artifact prerequisites](native-dependencies.md) for the supported platform. Conversion also needs `uv`, the locked exporter environment, about 5 GiB of free space, and an already-local pinned reference SDK worktree. The source download is about 650 MiB before conversion; the FP32 bundle is roughly 1.3 GiB. No model or native library is included in a source-built executable.

Set explicit paths. `LAYA_SOURCE_DIR` and `LAYA_BUNDLE_DIR` must not exist before their respective creation commands. `LAYA_SDK_DIR` must point to the verified SDK worktree described in [official export operations](official-export-v1.md). Choose a meaningful reproducible `SOURCE_DATE_EPOCH` for bundle provenance; it must be a positive Unix timestamp.

```sh
export LAYA_SOURCE_DIR="$PWD/.cache/laya-source"
export LAYA_SDK_DIR="/path/to/verified/laya-sdk"
export LAYA_BUNDLE_DIR="$PWD/.cache/laya-bundle"
export SOURCE_DATE_EPOCH=1790035200

go run ./cmd/layajev fetch --destination "$LAYA_SOURCE_DIR"
task reference:verify-source
go run ./cmd/layajev convert --source "$LAYA_SOURCE_DIR" --sdk "$LAYA_SDK_DIR" --output "$LAYA_BUNDLE_DIR" --epoch "$SOURCE_DATE_EPOCH"
task bundle:verify
```

`fetch` reads the checked-in export profile, allows only `convaiinnovations/laya-multilingual` at revision `052592a15d198d9ad47da779604259b10b47b7aa`, checks every declared size and SHA-256, then publishes the new source directory. It does not overwrite an existing destination. A failed transfer removes its own staging directory and leaves the destination unpublished. It does not use Hugging Face credentials or external caches. `convert` runs the existing `bundle:export` Taskfile operation; that exporter is offline and verifies the result. For the detailed SDK pin and export prerequisites, see [official export operations](official-export-v1.md).

Both commands write human-readable progress to stderr. `fetch` shows the current file, downloaded bytes (and a percentage from the pinned expected size), verification, and a success line only after publication. `convert` shows the current coarse exporter stage without an ETA or fabricated percentage; its canonical JSON result remains on stdout. An error or Ctrl-C does not print a success line. Redirect stderr separately if a script parses stdout.

## Serve and query

```sh
export LAYA_ONNXRUNTIME_LIBRARY="/absolute/path/libonnxruntime.so.1.29.0"
export LAYA_TOKENIZERS_LIBRARY="/absolute/path/libtokenizers.a"
export CGO_LDFLAGS="-L/absolute/path/containing/libtokenizers.a"
task setup:native
go run -tags=laya_native ./cmd/layajev serve --bundle "$LAYA_BUNDLE_DIR"
```

The default listener is `127.0.0.1:8080`. `serve` verifies the supplied bundle before opening the model. It never fetches or converts. The `laya_native` build tag, `CGO_LDFLAGS` pointing at the static tokenizer library's directory, and the native library paths above are required for real inference; an ordinary Go build intentionally has no native backend. The `layajev` CLI disables ONNX Runtime telemetry itself before native startup. To choose another local address, run `go run -tags=laya_native ./cmd/layajev serve --bundle "$LAYA_BUNDLE_DIR" --listen 127.0.0.1:9090`. A non-loopback address requires `--allow-remote`. The server itself has no TLS or bearer-auth layer; for remote clients, deploy a TLS/authenticating reverse proxy and restrict network access. Do not expose sensitive inputs through an unauthenticated public listener. The API does not log request bodies.

For a deployable container, use the [Docker runbook](layajev-container-runbook.md).
The image build fetches and converts the pinned source into a verified bundle,
then includes it with the native runtime. The running container needs no
model mount, download, or conversion.

First discover the actual local model name:

```sh
curl -sS http://127.0.0.1:8080/v1/models
```

Use the returned `models[0].name` as `model` (normally `laya-multilingual` for the pinned bundle). The `release_date` field describes the converted bundle's provenance date, not the upstream checkpoint's first publication date.

```sh
curl -sS http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"laya-multilingual","state":{"message":"I was charged twice"},"questions":{"billing":{"type":"noul","instructions":"Is this about billing?"},"tone":{"type":"choice","instructions":"What is the tone?","criteria":{"calm":"calm or polite","upset":"upset or hostile"}}}}'
```

A successful response has `model`, `answers` keyed exactly as the supplied questions, and `usage` with `input_tokens` and `output_tokens`. Choice answers include `type`, `choice`, `confidence`, and `probabilities`; score answers include `type`, expected `score`, `confidence`, `legend`, and `probabilities`; noul answers include `type` and the true/yes probability `noul`.

## Compatibility and errors

Supported input: a text, JSON object, or JSON array `state`; a nonempty `questions` object; each question has a nonempty string `instructions`. Choice requires at least two named criteria with nonempty string descriptions. Score requires at least two nonempty string descriptions in array order. Noul allows optional string `true`/`false` descriptions. A batch may mix types. Question keys are processed deterministically in lexical order. Extra request/question fields, non-string structured instructions or descriptions, and other Jev-valid forms not representable by the current Laya constructors receive HTTP `422` with `error.code=unsupported_request` and a field path. This is intentionally a subset of the official Jev schema.

Malformed JSON, duplicate keys, missing required fields, and wrong shapes receive `400 invalid_request`; a body over 1 MiB receives `413 request_too_large`; an unknown model receives `404 model_not_found`. Local inference failure returns `500 inference_failed` without internal details. A canceled in-flight request does not produce a successful answer. Only `GET /v1/models` and `POST /v1/systemone` are served. Do not send `jev-latest` as the model name: it is not this local model.

## Verify, recover, and stop

```sh
task fmt
task tidy
task verify
task test
task test:race
task lint
task vuln
task bundle:verify
```

Run repository-wide lint, vulnerability, and native checks as specified in `Taskfile.yml` before claiming production support. If `fetch` fails, verify network access and available disk space, then retry with a **new** absent destination. If `convert` fails, inspect the exporter error; do not serve a partial output. Restore from a verified immutable bundle or choose a new output path. Stop the foreground server with Ctrl-C; it drains HTTP requests, closes the model, then closes its runtime. Source and bundle directories are operator-owned and are not pruned by `layajev`.
