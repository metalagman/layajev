# Native dependency gate

Story 3 uses one reviewed candidate stack for local, in-process inference. It
does not download native libraries or models at runtime, and the repository
does not contain the large binary artifacts.

| Component | Pinned identity | Reviewed linux/amd64 artifact |
| --- | --- | --- |
| `github.com/yalue/onnxruntime_go` | `v1.36.0`, revision `1f0fb0647fc7091b57ec72dcf0f666d2234e4d04` | unchanged upstream Go module |
| ONNX Runtime | `1.29.0`, revision `2e2543fbe9fae542f921d47a72d21d5a4ef0b710` | `libonnxruntime.so.1.29.0`, SHA-256 `5715f06d8992ca8eeeddcce43df3a7d38f97d537052126f558e912cb312460ca` |
| `github.com/daulet/tokenizers` | `v1.27.0`, revision `f678a7768d5479d9d5a5161c4fc45c8a5ba46146` | local-only API subset in `internal/native/tokenizer` |
| Tokenizers native library | release `v1.27.0`, reported FFI version `1.26.0` | `libtokenizers.a`, SHA-256 `e6862b31745bb7d07980fcee70e49cd3b4318097609180f5d2d3fb394f305d50` |

The exact module sums, release URLs, archive identities, extracted library
identities, licenses, and retained tokenizer API are recorded in
`internal/native/deps/manifest.json`. GitHub release metadata publishes the
same archive SHA-256 values recorded there.

The tokenizers repository tag is `v1.27.0`, while its pinned FFI crate and
exported ABI marker identify themselves as `1.26.0`. The gate checks both the
release artifact digest and the reported FFI version instead of conflating
those two upstream identities.

The upstream ONNX binding already exposes call-owned `RunOptions`,
`Terminate`, and `RunWithOptions`; no local ONNX wrapper patch is needed. The
upstream tokenizer Go package also exposes Hugging Face HTTP, authentication,
download, and cache helpers. Laya therefore carries only the small MIT-licensed
local subset needed to parse an already-local tokenizer, encode text, report
the native version, and clean up. The subset has no acquisition API and does
not import `net/http` or `os/exec`.

The upstream static library includes Rust's standard-library object, which
contains dormant libc `socket`, `connect`, `getaddrinfo`, `send`, and `recv`
references. Those symbols are not an acquisition implementation or reachable
through the retained binding surface. Artifact verification rejects network
references from every non-stdlib archive member and rejects embedded HTTP,
authentication, or Hugging Face endpoint strings. The exact upstream Cargo
lock and FFI source digests are also pinned in the dependency manifest.

## Protected compatibility run

Acquire the two pinned release archives explicitly, verify their archive
digests from the dependency manifest, and extract them outside the repository.
No Taskfile operation performs acquisition. Point the protected gate at the
extracted libraries and at a complete verified bundle:

```sh
export LAYA_ONNXRUNTIME_LIBRARY=/absolute/path/libonnxruntime.so.1.29.0
export LAYA_TOKENIZERS_LIBRARY=/absolute/path/libtokenizers.a
export LAYA_BUNDLE_DIR=/absolute/path/complete-bundle
task native:integration
```

The Taskfile sets `ORT_DISABLE_TELEMETRY=1` before each native process starts.
Applications must set the same upstream opt-out before `NewRuntime`; the library
fails closed when it is absent instead of briefly changing process-global
environment state after other goroutines may exist. The post-initialization API
opt-out remains enabled as defense in depth.

The aggregate operation consists of these independently callable steps:

- `task native:verify` checks exact artifact sizes, digests, ABI symbols,
  runtime versions, bundle identity, graph contract, licenses, and tokenizer
  output.
- `task native:scan` runs with `GOPROXY=off` and rejects production dependencies
  or source capabilities for network clients, child processes, Python, model
  acquisition, authentication, caches, and provider registries.
- `task native:gate` loads the external-data graph; checks the exact five-input,
  two-float-output tensor contract; compares every public result in the
  14-case frozen corpus and all three raw batches; exercises cancellation,
  recovery, bounded FIFO admission, and runtime/model lifecycle; and rejects
  ONNX telemetry files.
- `task native:gate:race` repeats that gate under the Go race detector.
- `task native:stress` opens two real models, predicts concurrently, closes them
  in both orders, and repeats the process twice. It needs about 4 GiB of
  available memory and bounds final retained RSS to 128 MiB over baseline.

The stress-only `laya_native_diagnostics` build uses Linux `/proc` accounting
and glibc `malloc_trim` after complete teardown. This distinguishes reachable
native resources from allocator arena retention; it is diagnostic test code,
not production cleanup logic or evidence about every allocator.

Run the protected operation in an environment where network access is denied
when producing release evidence. The tasks themselves never acquire artifacts,
and missing inputs are actionable failures rather than skipped successes.

This is candidate evidence, not a general platform support declaration.
