# `layajev` container operations

The root [Dockerfile](../Dockerfile) builds a self-contained `linux/amd64`
image. During `docker build`, it installs the exact published
`@metalagman/layajev@0.2.6` native package, fetches the pinned official model
source and SDK, runs the locked exporter, verifies the completed FP32 bundle,
and copies that bundle into `/models` in the final image. Until the first release from this new repository, the image uses the last published CLI (0.2.6); it does not package untagged source changes. **No model directory
or volume is required by `docker run`.** Model acquisition and conversion
remain build-time operations, outside the root `laya` library. Runtime
inference is local and in-process, with no download or Python service.

## Build

On a `linux/amd64` Docker host, from this checkout:

```sh
docker build --platform linux/amd64 -t layajev:local .
docker run --rm --entrypoint /opt/layajev/bin/layajev layajev:local doctor
```

The equivalent repository operations are `task container:build` and
`task container:doctor`. The first build needs access to Docker Hub, Debian
and Ubuntu package archives, npm, GitHub, Go, PyPI, and the pinned Hugging
Face files.
Allow at least 10 GiB of free Docker storage and 4 GiB of available RAM. The
bundle alone is about 1.3 GiB, so the final image is deliberately large.
BuildKit may reuse verified intermediate layers on later builds. The
[from-zero script](../scripts/layajev-from-zero.sh) is reused in
`--bundle-only` mode; it checks pinned sizes and SHA-256 values, a clean
`layajev` exporter checkout, and the final bundle manifest before the final image is
assembled. A failed fetch, conversion, or verification fails the image build.

The runtime layer uses Ubuntu 24.04 because the published native binary
requires glibc 2.39. It contains the native binary, pinned ONNX Runtime and
verified bundle, but no Go, Node, Python, source snapshot, exporter, or model
downloader. The final stage runs `doctor` as UID/GID `65532:65532` and checks
that `/models/manifest.json` exists. Only this `linux/amd64` FP32 combination
is qualified.

## Run behind a reverse proxy

```sh
docker run --rm --name layajev \
  --read-only --tmpfs /tmp:rw,nosuid,nodev,size=64m \
  --publish 127.0.0.1:8080:8080 --stop-timeout 60 \
  layajev:local
```

With Go Task installed, `task container:serve` applies the same settings.
The model is already in the image; do **not** supply `--mount` or
`LAYA_BUNDLE_DIR`. The container listens on `0.0.0.0:8080` inside its
network namespace so Docker can forward the port, while the example exposes
it only on the host's loopback interface. `layajev` has no TLS or
authentication. Put an authenticating TLS reverse proxy in front of it and
restrict network access; never publish the port publicly without that
protection. The CLI disables ONNX Runtime telemetry before native startup.

Two Docker Compose examples use the same self-contained build and no volumes:

```sh
# Local loopback port, suitable behind a host reverse proxy.
docker compose -f compose.yaml up --build -d
docker compose -f compose.yaml ps
docker compose -f compose.yaml down

# No host port; an existing TLS/auth proxy must join layajev-ingress.
docker compose -f docs/examples/compose-proxy.yaml up --build -d
```

The [local Compose file](../compose.yaml) publishes only host loopback. The
[proxy-network example](examples/compose-proxy.yaml) requires an existing
external Docker network named `layajev-ingress` and a separately configured
proxy connected to it. It does not provision TLS or authentication itself;
do not treat the proxy-network example alone as a public deployment.

Readiness is `GET /v1/models`; the Docker healthcheck queries it after a
120-second startup grace period. Verify the running service:

```sh
docker inspect --format '{{.State.Health.Status}}' layajev
curl --fail --silent --show-error http://127.0.0.1:8080/v1/models
curl --fail --silent --show-error http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"laya-multilingual","state":"I was charged twice.","questions":{"billing":{"type":"noul","instructions":"Is this about billing?"}}}'
```

See the [API contract and errors](layajev-runbook.md#compatibility-and-errors)
before integrating a client. The API is a documented Jev-compatible subset,
not an identical Jev model.

## Operate and recover

Treat the image as an immutable pair of binary and model. Rebuild and deploy
a new image to change either one; do not modify `/models` in a running
container. Keep the root filesystem read-only, size `/tmp` and memory for the
workload, and collect stdout/stderr with the deployment platform. Stop with
`docker stop layajev`; the process drains HTTP, then closes the model and
runtime. If startup fails, inspect `docker logs layajev` and the health status.
Roll back to the previous tested image digest, which also restores its bundle.
The image does not own external caches or model files on the host.
