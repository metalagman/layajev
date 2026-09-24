# `layajev` npm release and `npx` operations

The omnidist profile publishes two MIT-licensed npm packages, `@metalagman/layajev`
and its optional platform package `@metalagman/layajev-linux-x64`. Only
`linux/amd64` with the pinned native build has been qualified. There is no
Darwin, Windows, ARM, musl, or general Linux compatibility claim. Neither npm
package contains model weights or a Python inference service. Starting with
version 0.2.2, the platform package includes the checksum-pinned ONNX Runtime
shared library, its MIT license, and third-party notices. The 1.3 GiB FP32
bundle remains a deployment-owned input. The release workflow builds from an
exact Git tag and then publishes the verified npm packages. The
[production container](layajev-container-runbook.md) consumes the same exact
published native package and builds a verified bundle into the image.

## Run a published package

For a clean `linux/amd64` machine with no model, SDK, Go, or uv installed, use
the [from-zero smoke script](../scripts/layajev-from-zero.sh). It needs only
Node/npx, Git, curl, tar, sha256sum, and setsid up front. Allow at least 10 GiB
of free disk and 4 GiB RAM. It installs pinned Go and uv under the chosen work
directory, downloads and verifies the pinned SDK and official model source,
hydrates the locked exporter, converts a new local bundle, and checks the
published package's `doctor` and HTTP API. It uses no `sudo`, does not write a
model into the npm package, and leaves the API running until Ctrl-C:

```sh
curl --fail --silent --show-error --location \
  --output layajev-from-zero.sh \
  https://raw.githubusercontent.com/metalagman/layajev/main/scripts/layajev-from-zero.sh
bash layajev-from-zero.sh "$PWD/layajev-test-0.2.6"
```

Inspect the downloaded script before executing it. The first run downloads
several GiB across the model, locked Python environment, and toolchains; a
repeat run reuses local inputs and re-verifies the source and bundle. The API
binds only to `127.0.0.1`.

After a release has actually appeared on npm, use its exact version on a
compatible `linux/amd64` host:

```sh
export LAYAJEV_VERSION=0.2.6
export LAYA_BUNDLE_DIR=/absolute/path/to/verified-bundle
npx -y "@metalagman/layajev@$LAYAJEV_VERSION" serve --bundle "$LAYA_BUNDLE_DIR"
```

Starting with version 0.2.3, check an installed package before
supplying a model with
`npx -y "@metalagman/layajev@$LAYAJEV_VERSION" doctor`. It initializes and
closes the packaged native runtime, but does not validate a model or the HTTP
API.

The tokenizer archive is linked into the packaged binary at build time; no
tokenizer-library path is needed at runtime. The command finds the packaged
ONNX Runtime library beside its executable. An explicitly set
`LAYA_ONNXRUNTIME_LIBRARY` overrides that path and must still pass the pinned
size and SHA-256 checks. The local bundle remains required; starting in v0.2.6,
`layajev` disables ONNX Runtime telemetry itself before native startup.
`CGO_LDFLAGS` is only a build-time setting, not a consumer setting.
Versions 0.2.0 and 0.2.1 do not package ONNX Runtime and require the explicit
library path.
See [native dependencies](native-dependencies.md) for pinned library identities
and [source-checkout operations](layajev-runbook.md) for the Jev-compatible
subset, HTTP examples, and safe binding rules. The default server listens on
`127.0.0.1:8080`; it provides neither TLS nor authentication. Do not expose it
directly on a public interface.

Starting with version 0.2.1, the packaged `fetch` command uses
an embedded pinned source manifest and works outside a source checkout:

```sh
npx -y "@metalagman/layajev@$LAYAJEV_VERSION" fetch --destination source-snapshot
```

Version 0.2.0 has a known bug: `fetch` tries to read the manifest from the
current checkout. For that historical version, run it from a `laya-go` checkout or pass
`--repository-root` explicitly. The `convert` command still invokes the
repository Taskfile and locked exporter, so it requires a source checkout and
its prerequisites. `fetch` downloads the pinned source snapshot, not a
ready-to-serve ONNX bundle; do not pass its destination directly to `serve`.

## Prepare a release

From a reviewed `main` checkout, choose a new stable SemVer version; do not
reuse an npm version already
published. Confirm control of the `@metalagman` scope and both package names
before publication. Review `.omnidist/omnidist.yaml`, the MIT
license, the pinned native dependency hashes, and the candidate diff. Run the
repository's formatting, verification, tests, lint, and native smoke with the
required external artifacts before making support claims.

The tag-driven workflow `.github/workflows/omnidist-release.yml` resolves the
version from the exact Git tag, acquires only checksum-pinned native archives,
and performs `build`, `stage`, and `verify`. Before upload, it packs both staged
packages, installs them together in a clean offline npm project, runs the
installed CLI, and checks that the packaged ONNX Runtime opens and closes.
An operator native qualification should also start the installed server with a verified bundle, query
`GET /v1/models`, and make one `POST /v1/systemone` prediction. After staging
succeeds, the release workflow publishes both npm packages through omnidist.
Staged packages are retained for seven days; the staging job has read-only
repository permission and no npm publish token.

For local reproduction on `linux/amd64`, obtain the same verified
`libtokenizers.a` described in [native dependencies](native-dependencies.md)
without changing the source tree, then run:

```sh
export CGO_LDFLAGS=-L/absolute/path/containing/libtokenizers.a
# Check out the exact release tag before running the following commands.
npx -y @omnidist/omnidist@latest build
npx -y @omnidist/omnidist@latest stage
npx -y @omnidist/omnidist@latest verify
sha256sum .omnidist/default/dist/linux/amd64/layajev \
  .omnidist/default/npm/@metalagman/layajev-linux-x64/bin/layajev
```

Inspect both staged `package.json` files for the same new version, MIT license,
correct names, and a single `linux/x64` optional dependency. Inspect the
staged file list for absence of weights and unintended binaries. Run the
staged binary's `--help`; for a native smoke, use the pinned external runtime
and bundle and query `/v1/models` as described above. The `npx` CLI version
used by the tag workflow is pinned to `@omnidist/omnidist@0.1.37`; local
reproduction with `@latest` must be reviewed if that version has moved.

## Publish through omnidist and recover

Pushing a new release tag is the publication decision. It requires a reviewed
`main`, successful quality/native gates, verified npm account and scope rights,
and a stable version absent from **both** npm packages. The profile uses npm
trusted publishing, not a long-lived `NPM_PUBLISH_TOKEN`. Before the first
release, the package owner must authorize this exact GitHub repository and
workflow as a trusted publisher for **each** package. From an authenticated
maintainer environment, inspect the commands with:

```sh
npx -y @omnidist/omnidist@latest npm trust
```

Then apply the reviewed trust configuration with
`npx -y @omnidist/omnidist@latest npm trust --apply`. This changes npm package
settings and may require npm account verification. The workflow's publish job
requests GitHub OIDC only after the staging job succeeds, verifies the exact
uploaded candidate, and runs omnidist `verify`, `publish --dry-run`, and
`npm publish`. It then compares published npm checksums and runs `doctor` via
a clean `npx` install. From the current `main` commit, choose a new stable
SemVer version absent from both npm packages and push the exact tag:

```sh
: "${LAYAJEV_NEXT_VERSION:?Set an unused MAJOR.MINOR.PATCH version first}"
git tag "v${LAYAJEV_NEXT_VERSION}"
git push origin "v${LAYAJEV_NEXT_VERSION}"
gh run list --workflow omnidist-release.yml --limit 5
```

Set `LAYAJEV_NEXT_VERSION` to the next confirmed unused SemVer version before
running these commands; never assume a
previous upload failed just because npm search or its web page lags. This
workflow publishes only stable `vMAJOR.MINOR.PATCH` tags. It does not create
or push tags or publish a GitHub Release.

Npm publication is not atomic: omnidist sends the platform package before the
meta package. If an upload fails, stop retries, preserve the log, and inspect
both `@metalagman/layajev-linux-x64` and `@metalagman/layajev` at that exact
version. Compare accepted files and checksums against the verified staging;
never rebuild different bytes under the same version or move an existing tag.
Resume only the missing units after checking registry state, or issue a new
version and explain the incomplete one. Finally, verify installation and an
external-bundle smoke through the published `npx` command on `linux/amd64`.
