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

## Run it your way

- **npx:** the quickest way to run `doctor`, `fetch`, or `serve` on `linux/amd64`. [CLI and model setup](https://github.com/metalagman/layajev/blob/main/docs/layajev-npm-release.md).
- **Docker:** build a production image that contains its own verified model bundle; no model mount is needed at runtime. [Container and Compose guide](https://github.com/metalagman/layajev/blob/main/docs/layajev-container-runbook.md).
- **Go source:** build or modify the adapter in this repository. [Contributing guide](https://github.com/metalagman/layajev/blob/main/CONTRIBUTING.md).

`serve` never downloads or converts a model. The server has no built-in TLS or authentication: keep the default loopback listener, or place it behind an authenticating reverse proxy before allowing remote traffic. Only the pinned `linux/amd64` native candidate is qualified; other platforms are not advertised as supported.

`layajev` is MIT-licensed. Model and SDK licenses are separate and must be reviewed before redistribution. The reusable inference library lives in [laya-go](https://github.com/metalagman/laya-go).
