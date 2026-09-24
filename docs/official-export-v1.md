# Official checkpoint export v1

The supported trust chain is deliberately direct:

```text
convaiinnovations/laya-multilingual@052592a...
  + NandhaKishorM/laya@573e5b6 semantics
  -> repository-owned locked layajev exporter
  -> complete Laya Go Bundle v1
```

No independently converted model, graph, tensor set, or fixture is a build
input, parity oracle, fallback, or release artifact.

## Inputs

The caller obtains an immutable official checkpoint snapshot and, for protected
reference verification, the pinned SDK source by an external process. The
`layajev fetch` command is an explicit operator-invoked acquisition path;
`laya-go` itself does not authenticate, resolve revisions, download, or manage
caches. The offline `task bundle:export` consumes only already-local verified
inputs. The library has no acquisition path.

The exporter accepts explicit existing local paths and the checked-in
`laya-multilingual-v1` profile. It verifies every expected checkpoint file by
path, size, and SHA-256 before model construction. A raw snapshot is never
modified.

Checkpoint content is data, not executable code. The exporter:

- loads weights only through Safetensors;
- does not deserialize pickle or PyTorch `.bin` checkpoints;
- does not add the checkpoint to Python module search paths;
- does not import Python files from the checkpoint;
- does not enable Transformers remote code;
- performs no network fallback.

The complete model architecture used for export is reviewed and versioned with
the exporter. Its source digest, last committed exporter-source Git revision,
dependency-lock digest, profile digest, and deterministic creation epoch are
written to bundle provenance. Later documentation commits cannot change the
bundle identity; an uncommitted exporter source is rejected.

## Output and failure recovery

Export produces the encoder, decision head, marker scorer, and action head in a
single opset-18 FP32 graph, optionally with deterministic external-data files.
The result is staged beside the requested output and becomes ready only after
ONNX structure, official-reference parity, licenses, manifest, file table, and
all digests pass.

A failed or interrupted export may remove only its own unpublished staging
directory. It must not mutate the checkpoint, SDK input, external cache,
existing ready bundle, or a released artifact.

Publishing is a later, explicit release action. A locally exported bundle, a
GitHub Release asset, a future authorized Hugging Face bundle repository, and
embedded bundle bytes all use the same format and identity.

## Runbook

All artifact operations are explicit, offline, and path-based. The supported
profile is `tools/export/profiles/laya-multilingual-v1.json`; callers do not
pass a repository ID or URL.

Verify an externally acquired raw official snapshot:

```sh
LAYA_SOURCE_DIR=/absolute/convaiinnovations-laya-multilingual-snapshot \
task reference:verify-source
```

Compare preprocessing semantics with the separately supplied pinned SDK. The
Python executable must be the locked local environment used for that SDK:

```sh
LAYA_SDK_DIR=/absolute/NandhaKishorM-laya-573e5b6 \
LAYA_SDK_PYTHON=/absolute/locked-python \
task reference:verify-sdk
```

Export a new bundle. Allow about 5 GiB of free disk space. The destination must
not exist; it is published atomically and never replaces a prior bundle.
`SOURCE_DATE_EPOCH` is required provenance input, not wall-clock discovery.

```sh
LAYA_SOURCE_DIR=/absolute/official-snapshot \
LAYA_SDK_DIR=/absolute/pinned-sdk \
LAYA_BUNDLE_DIR=/absolute/new-complete-bundle \
SOURCE_DATE_EPOCH=1790035200 \
task bundle:export
```

Verify any already-local complete bundle, including one obtained or cached by
external Hugging Face tooling:

```sh
LAYA_BUNDLE_DIR=/absolute/complete-bundle task bundle:verify
```

Regenerate a parity corpus only from a verified complete bundle and the same
explicit official inputs:

```sh
LAYA_SOURCE_DIR=/absolute/official-snapshot \
LAYA_SDK_DIR=/absolute/pinned-sdk \
LAYA_BUNDLE_DIR=/absolute/complete-bundle \
LAYA_FIXTURE_OUTPUT=/absolute/corpus-v1.json \
task reference:generate
```

These operations do not download, log in, publish, register a model, or mutate
an external cache. A raw official snapshot is exporter input; only a complete
bundle is runtime input. Directory and embedded runtime sources are two
representations of those same bundle bytes.
