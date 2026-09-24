#!/usr/bin/env bash
set -Eeuo pipefail

# Build a verified local bundle from the pinned official checkpoint, then
# optionally exercise the published layajev npm package. All generated files
# stay in the chosen work directory; no system packages or global tools are
# installed.

readonly layajev_version=${LAYAJEV_VERSION:-0.2.7}
readonly repository_ref=${LAYAJEV_REPOSITORY_REF:-v$layajev_version}
readonly sdk_commit=573e5b62696ba441230cd6be71d593331b5d23af
readonly sdk_sha256=03931635a92b7609c6c253ac1d4d618ebe4fc54956744db01c027b504fd9c426
readonly go_sha256=708effb774be8237570d0add163225abbdfaf4fca28b2611df167beba4feef89
readonly uv_sha256=6b52a47358deea1c5e173278bf46b2b489747a59ae31f2a4362ed5c6c1c269f7

usage() {
  printf 'Usage: bash %s [--bundle-only] [WORK_DIR]\n' "$0"
  printf 'Default WORK_DIR: ./layajev-test-0.2.7\n'
  printf 'Requires Linux x86_64, git, curl, tar, sha256sum, Node/npx, and setsid.\n'
  printf 'Downloads pinned Go, uv, SDK, model source, and locked exporter dependencies.\n'
  printf 'Allow at least 10 GiB free disk space and 4 GiB RAM.\n'
  printf '%s\n' '--bundle-only verifies the bundle and exits; otherwise Ctrl-C stops the server.'
}

die() {
  printf 'layajev setup: %s\n' "$*" >&2
  exit 1
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  usage
  exit 0
fi
bundle_only=false
if [[ ${1:-} == --bundle-only ]]; then
  bundle_only=true
  shift
fi
if (( $# > 1 )); then
  usage >&2
  exit 2
fi
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || die 'only linux/amd64 is qualified'
for program in git curl tar sha256sum node npx setsid; do
  command -v "$program" >/dev/null 2>&1 || die "missing prerequisite: $program"
done

mkdir -p -- "${1:-$PWD/layajev-test-0.2.7}"
work_dir=$(cd -- "${1:-$PWD/layajev-test-0.2.7}" && pwd -P)
[[ $work_dir != / ]] || die 'WORK_DIR must not be the filesystem root'
tools_dir=$work_dir/tools
repository_dir=${LAYAJEV_REPOSITORY_DIR:-$work_dir/layajev}
source_dir=$work_dir/source-snapshot
sdk_dir=$work_dir/laya-$sdk_commit
bundle_dir=$work_dir/bundle
mkdir -p -- "$tools_dir" "$tools_dir/bin"

download_verified() {
  local url=$1 expected=$2 destination=$3
  if [[ ! -f $destination ]]; then
    [[ ! -e $destination && ! -L $destination ]] ||
      die "download target is not a regular file: $destination"
    printf 'Downloading %s\n' "$url"
    curl --fail --silent --show-error --location --retry 3 \
      --proto '=https' --proto-redir '=https' \
      --output "$destination.part" "$url"
    printf '%s  %s\n' "$expected" "$destination.part" | sha256sum --check --status ||
      die "checksum mismatch: $destination.part"
    mv -- "$destination.part" "$destination"
  fi
  printf '%s  %s\n' "$expected" "$destination" | sha256sum --check --status ||
    die "checksum mismatch: $destination"
}

go_dir=$tools_dir/go
go_archive=$tools_dir/go1.26.6.linux-amd64.tar.gz
if command -v go >/dev/null 2>&1 && [[ $(go version) == 'go version go1.26.6 linux/amd64' ]]; then
  go_bin_dir=$(dirname "$(command -v go)")
else
  if [[ ! -x $go_dir/bin/go ]]; then
    [[ ! -e $go_dir && ! -L $go_dir ]] || die "incomplete Go installation: $go_dir"
    download_verified \
      https://go.dev/dl/go1.26.6.linux-amd64.tar.gz "$go_sha256" "$go_archive"
    tar -xzf "$go_archive" -C "$tools_dir"
  fi
  [[ $("$go_dir/bin/go" version) == 'go version go1.26.6 linux/amd64' ]] ||
    die "unexpected Go version in $go_dir"
  go_bin_dir=$go_dir/bin
fi

uv_dir=$tools_dir/uv-x86_64-unknown-linux-gnu
uv_archive=$tools_dir/uv-0.10.4-linux-amd64.tar.gz
if [[ ! -x $uv_dir/uv ]]; then
  [[ ! -e $uv_dir && ! -L $uv_dir ]] || die "incomplete uv installation: $uv_dir"
  download_verified \
    https://github.com/astral-sh/uv/releases/download/0.10.4/uv-x86_64-unknown-linux-gnu.tar.gz \
    "$uv_sha256" "$uv_archive"
  tar -xzf "$uv_archive" -C "$tools_dir"
fi
[[ $("$uv_dir/uv" --version) == 'uv 0.10.4' ]] || die "unexpected uv version in $uv_dir"

export PATH="$go_bin_dir:$uv_dir:$tools_dir/bin:$PATH"
export GOTOOLCHAIN=local GOPATH="$work_dir/go-path" GOCACHE="$work_dir/go-build-cache"
export GOBIN="$tools_dir/bin" UV_PYTHON_INSTALL_DIR="$work_dir/python"
export npm_config_cache="$work_dir/npm-cache"
if [[ ! -x $tools_dir/bin/task ]]; then
  go install github.com/go-task/task/v3/cmd/task@v3.53.1
fi

if [[ ! -e $repository_dir ]]; then
  [[ ! -L $repository_dir ]] || die "dangling repository path: $repository_dir"
  git clone https://github.com/metalagman/layajev.git "$repository_dir"
  git -C "$repository_dir" checkout --detach "$repository_ref"
fi
[[ -f "$repository_dir/Taskfile.yml" && -f "$repository_dir/tools/export/pyproject.toml" ]] ||
  die 'repository is not a layajev source checkout'
[[ $(git -C "$repository_dir" rev-parse --is-shallow-repository) == false ]] ||
  die 'the exporter requires a complete Git history'
git -C "$repository_dir" diff --quiet || die 'repository has modified tracked files'
git -C "$repository_dir" diff --cached --quiet || die 'repository has staged changes'

printf 'Checking the published native runtime...\n'
npx -y "@metalagman/layajev@$layajev_version" doctor

if [[ ! -e $source_dir ]]; then
  [[ ! -L $source_dir ]] || die "dangling source path: $source_dir"
  printf 'Fetching and verifying the pinned official model source...\n'
  npx -y "@metalagman/layajev@$layajev_version" fetch --destination "$source_dir"
fi

sdk_archive=$work_dir/laya-sdk-$sdk_commit.tar.gz
if [[ ! -e $sdk_dir ]]; then
  [[ ! -L $sdk_dir ]] || die "dangling SDK path: $sdk_dir"
  download_verified \
    "https://codeload.github.com/NandhaKishorM/laya/tar.gz/$sdk_commit" \
    "$sdk_sha256" "$sdk_archive"
  sdk_stage=$(mktemp -d "$work_dir/sdk-stage.XXXXXX")
  tar -xzf "$sdk_archive" -C "$sdk_stage"
  mv -- "$sdk_stage/laya-$sdk_commit" "$sdk_dir"
  rmdir -- "$sdk_stage"
fi

export UV_CACHE_DIR="$repository_dir/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$repository_dir/.cache/export-venv"
printf 'Hydrating the locked exporter and bundle verifier...\n'
uv sync --project "$repository_dir/tools/export" --python 3.12 --frozen
(
  cd "$repository_dir"
  go build -o .cache/bundlecheck ./internal/cmd/bundlecheck
  LAYA_SOURCE_DIR="$source_dir" task reference:verify-source
)

if [[ ! -e $bundle_dir ]]; then
  [[ ! -L $bundle_dir ]] || die "dangling bundle path: $bundle_dir"
  printf 'Converting the verified source into a complete local bundle...\n'
  GOPROXY=off npx -y "@metalagman/layajev@$layajev_version" convert \
    --repository-root "$repository_dir" --source "$source_dir" \
    --sdk "$sdk_dir" --output "$bundle_dir" --epoch 1790035200
fi
(
  cd "$repository_dir"
  LAYA_BUNDLE_DIR="$bundle_dir" GOPROXY=off task bundle:verify
)
[[ ! -L $bundle_dir ]] || die 'serve requires a real bundle directory, not a symlink'
if [[ $bundle_only == true ]]; then
  printf 'Verified bundle ready: %s\n' "$bundle_dir"
  exit 0
fi

listen=127.0.0.1:18977
api=http://$listen
server_pid=
cleanup() {
  if [[ -n $server_pid ]]; then
    kill -TERM -- "-$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf 'Starting the installed API on %s...\n' "$api"
setsid npx -y "@metalagman/layajev@$layajev_version" serve \
  --bundle "$bundle_dir" --listen "$listen" >"$work_dir/server.log" 2>&1 &
server_pid=$!
ready=false
for _ in {1..120}; do
  if curl --fail --silent --max-time 1 "$api/v1/models" >/dev/null; then
    ready=true
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    break
  fi
  sleep 1
done
if [[ $ready != true ]]; then
  sed -n '1,120p' "$work_dir/server.log" >&2
  die 'server did not become ready'
fi

printf 'Models: '
curl --fail --silent --show-error "$api/v1/models"
printf '\nPrediction: '
curl --fail --silent --show-error --max-time 120 \
  -H 'Content-Type: application/json' \
  --data-binary '{"model":"laya-multilingual","state":"I was charged twice.","questions":{"billing":{"type":"noul","instructions":"Is this about billing?"}}}' \
  "$api/v1/systemone" |
  node -e '
    let text = "";
    process.stdin.on("data", chunk => { text += chunk; });
    process.stdin.on("end", () => {
      const result = JSON.parse(text);
      if (result.model !== "laya-multilingual" ||
          typeof result.answers?.billing?.noul !== "number") {
        throw new Error("unexpected prediction response: " + text);
      }
      process.stdout.write(JSON.stringify(result, null, 2) + "\n");
    });
  '

printf '\nAPI is running at %s; bundle: %s\nPress Ctrl-C to stop.\n' "$api" "$bundle_dir"
wait "$server_pid"
