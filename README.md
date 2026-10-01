# layajev

Run a Jev-compatible decision API on your own machine. `layajev` serves a verified Laya model through a small HTTP API: send a state and questions, and receive typed answers. Inference stays local and in-process—no hosted inference service or Python server is involved.

`layajev` is a compatibility **subset**, not the Jev model or a promise of identical decisions. [See the supported request forms and errors](https://github.com/metalagman/layajev/blob/main/docs/layajev-runbook.md#compatibility-and-errors).

- **Easy to call:** `GET /v1/models` and `POST /v1/systemone` return structured answers for supported choice, score, and noul questions.
- **Local by design:** the server runs inference in its own process; serving does not contact a model host.
- **Explicit model lifecycle:** fetch and conversion are separate, operator-invoked steps, and serving opens a verified bundle.

## Start with npx

On a supported `linux/amd64` host, check the packaged native runtime, then serve an existing verified bundle:

```sh
npx -y @metalagman/layajev@latest doctor
npx -y @metalagman/layajev@latest serve --bundle /absolute/path/to/verified-bundle
```

The API listens on `127.0.0.1:8080` by default. The npm package includes the native runtime, **not model weights**. `doctor` checks the runtime but does not prepare a model. If you do not have a bundle yet, follow the [from-zero model setup](https://github.com/metalagman/layajev/blob/main/docs/layajev-npm-release.md#run-a-published-package); it fetches the pinned official source, verifies it, and converts it explicitly.

Ask the running API a question:

```sh
curl -sS http://127.0.0.1:8080/v1/models

curl -sS http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"laya-multilingual","state":"I was charged twice.","questions":{"billing":{"type":"noul","instructions":"Is this about billing?"}}}'
```

Use the model name returned by `/v1/models` if it differs from `laya-multilingual`. Choice, score, and noul questions are supported within the [documented API subset](https://github.com/metalagman/layajev/blob/main/docs/layajev-runbook.md#compatibility-and-errors).

## Measured resources and performance

Measured on 2026-10-01 with the locally available **`layajev` 0.2.4** native
binary (Go 1.26.6), ONNX Runtime 1.29.0 and the FP32
`convaiinnovations/laya-multilingual` checkpoint at
`052592a15d198d9ad47da779604259b10b47b7aa`. This is a historical CLI measurement,
not qualification of the current standalone release. The host was WSL2
linux/amd64, AMD Ryzen 7 PRO 7840U, **4 logical CPUs / 2 exposed cores**, 31 GiB
RAM and no swap; inference used `CPUExecutionProvider`, with no GPU.

For this workload, budget **4 logical CPUs and 2 GiB of process RAM**, plus
memory for the host and filesystem cache. This is a starting capacity estimate
from the measurements below, not a tested minimum or an enforced container
memory limit. The prepared bundle needs **1.231 GiB of disk**, and the binary
plus ONNX Runtime library another **57.9 MiB**, excluding licenses, npm/Node,
container layers and any source/export environment. Model preparation is a
separate operation; the [from-zero runbook](https://github.com/metalagman/layajev/blob/main/scripts/layajev-from-zero.sh)
budgets 10 GiB disk and 4 GiB RAM.

| Resource or startup metric | Measured value |
| --- | ---: |
| Fresh process to successful `GET /v1/models` (3 runs) | 3.28–3.40 s |
| RSS at API readiness | 629–630 MiB |
| RSS after first short prediction and 1 s idle | 658 MiB |
| First short HTTP prediction after readiness (3 runs) | 55–69 ms |
| RSS after 100 short requests | 665 MiB |
| Peak process RSS across tested workloads | 963 MiB |
| Threads during 4-CPU inference | 12–13 |
| CPU during sustained 4-CPU tests | 3.6–3.7 logical CPU equivalents |

The startup runs used fresh processes with the OS page cache left intact;
these are not cold-disk measurements. RSS and peak RSS come from Linux
`/proc/<pid>/status` (`VmRSS` / `VmHWM`), including native allocations; Go heap
statistics alone would miss them. Filesystem cache and other processes are
outside these figures.

HTTP timings include request parsing, the ADK adapter, tokenization, native
inference and response serialization over loopback. The Python standard-library
client opened a fresh connection per request, used closed-loop workers, and
excluded five warmups before each row. p50/p95 use nearest-rank percentiles;
throughput is completed requests divided by phase wall time. The 20-request
rows provide only a coarse tail-latency estimate. All measured
responses returned HTTP 200 and the expected answer IDs/types.

| Request workload, 4 logical CPUs | Input tokens¹ | Requests | Concurrent clients | p50 | p95 | Requests/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Short state, one `noul` question | 36 | 100 | 1 | 50.5 ms | 66.4 ms | 18.85 |
| Short state, `choice` + `score` + `noul` | 99 | 50 | 1 | 132.4 ms | 159.6 ms | 7.41 |
| State repeated 64 times, one `noul` | 352 | 30 | 1 | 453.5 ms | 636.5 ms | 2.10 |
| State repeated 189 times, one `noul` | 977 | 20 | 1 | 1880.2 ms | 2100.0 ms | 0.53 |
| State repeated 64 times, three mixed questions | 1047 | 20 | 1 | 1400.5 ms | 1506.1 ms | 0.71 |
| Short state, one `noul` question | 36 | 100 | 4 | 215.6 ms | 253.4 ms | 18.16 |

¹ The API's `usage.input_tokens` for the whole request, including question
formatting; mixed-question counts are summed across questions. The base state
was `I was charged twice.` and the `noul` instruction was `Is this about
billing?`. Exact payloads, individual latencies, binary/bundle identities and
resource observations are retained in
[the measurement record](docs/benchmarks/layajev-2026-10-01.json).

Four clients increased queueing latency without increasing throughput: this
version admits one active inference per model, with `--queue-size 16` in these
runs. More HTTP clients do not provide parallel inference on this one model.

| CPUs available to all inference threads | p50, short `noul` | p95 | Requests/s | Average CPU equivalents |
| --- | ---: | ---: | ---: | ---: |
| 1 logical CPU | 415.0 ms | 439.9 ms | 2.40 | 0.94 |
| 2 logical CPUs, one per exposed core | 196.5 ms | 216.0 ms | 5.31 | 1.87 |
| 4 logical CPUs | 50.5 ms | 66.4 ms | 18.85 | 3.63 |

The 1/2-CPU rows each used 100 short requests after five warmups. Native worker
threads changed their own affinity at initialization, escaping the initial
`taskset` mask. For these two rows, every thread was restricted **after API
readiness**, and its affinity was verified again after inference. Consequently,
these rows measure constrained inference, not constrained startup. ONNX Runtime
kept its default worker configuration; these are not results for thread pools
tuned to 1/2 CPUs. `GOMAXPROCS` alone does not limit native ONNX Runtime workers.
The 4-CPU rows used all host CPUs and `GOMAXPROCS=4`.

To run the measured serving configuration against your own prepared bundle:

```sh
ORT_DISABLE_TELEMETRY=1 GOMAXPROCS=4 /path/to/layajev serve \
  --bundle /absolute/path/to/bundle --listen 127.0.0.1:18979 --queue-size 16
curl --fail --silent --show-error http://127.0.0.1:18979/v1/models
curl --fail --silent --show-error http://127.0.0.1:18979/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"laya-multilingual","state":"I was charged twice.","questions":{"billing":{"type":"noul","instructions":"Is this about billing?"}}}'
```

The source snapshot was prepared and measured using the `laya-go` exporter
checkout. In that checkout, it passed `task reference:verify-source`; `task bundle:export`
performed source/SDK verification and ONNX parity checks, and `task
bundle:verify` verified the complete exported bundle. The packaged CLI passed
`doctor`. These measurements do not cover Docker overhead, a RAM-limited
container, queue overflow, maximum question counts, multilingual corpora,
decision accuracy or sustained production load.

## Run it your way

- **npx:** the quickest way to run `doctor`, `fetch`, or `serve` on `linux/amd64`. [CLI and model setup](https://github.com/metalagman/layajev/blob/main/docs/layajev-npm-release.md).
- **Docker:** build a production image that contains its own verified model bundle; no model mount is needed at runtime. [Container and Compose guide](https://github.com/metalagman/layajev/blob/main/docs/layajev-container-runbook.md).
- **Go source:** build or modify the adapter in this repository. [Contributing guide](https://github.com/metalagman/layajev/blob/main/CONTRIBUTING.md).

`serve` never downloads or converts a model. The server has no built-in TLS or authentication: keep the default loopback listener, or place it behind an authenticating reverse proxy before allowing remote traffic. Only the pinned `linux/amd64` native candidate is qualified; other platforms are not advertised as supported.

`layajev` is MIT-licensed. Model and SDK licenses are separate and must be reviewed before redistribution. The reusable inference library lives in [laya-go](https://github.com/metalagman/laya-go).
