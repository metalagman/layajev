#!/usr/bin/env bash
set -euo pipefail

version=${1:?Usage: verify-published-npm.sh MAJOR.MINOR.PATCH}
if [[ ! $version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  printf 'Expected a stable MAJOR.MINOR.PATCH version\n' >&2
  exit 2
fi

staged_root=${LAYAJEV_STAGED_NPM_ROOT:-}
cache_root=${RUNNER_TEMP:-${TMPDIR:-/tmp}}
packages=(@metalagman/layajev-linux-x64 @metalagman/layajev)

for attempt in {1..90}; do
  ready=true
  for package in "${packages[@]}"; do
    if ! dist=$(npm view "$package@$version" dist --json --prefer-online 2>/dev/null); then
      ready=false
      break
    fi
    shasum=$(jq -r '.shasum // empty' <<<"$dist")
    tarball=$(jq -r '.tarball // empty' <<<"$dist")
    if [[ -z $shasum || -z $tarball ]]; then
      ready=false
      break
    fi
    if [[ -n $staged_root ]]; then
      expected=$(npm pack --dry-run --json "$staged_root/$package" | jq -r '.[0].shasum')
      if [[ $shasum != "$expected" ]]; then
        printf 'Published shasum mismatch for %s: got %s, expected %s\n' "$package" "$shasum" "$expected" >&2
        exit 1
      fi
    fi
    if ! curl --fail --silent --show-error --head --location "$tarball" >/dev/null 2>&1; then
      ready=false
      break
    fi
  done
  if [[ $ready == true ]]; then
    printf 'Both npm packages and tarballs are available for %s\n' "$version"
    break
  fi
  if (( attempt == 90 )); then
    printf 'Timed out waiting for published npm tarballs for %s\n' "$version" >&2
    exit 1
  fi
  sleep 10
done

for attempt in {1..12}; do
  if env -u LAYA_ONNXRUNTIME_LIBRARY -u LAYA_TOKENIZERS_LIBRARY \
    -u ORT_DISABLE_TELEMETRY \
    npm_config_cache="$cache_root/layajev-release-npx-$version-$attempt" \
    npx -y "@metalagman/layajev@$version" doctor; then
    exit 0
  fi
  if (( attempt == 12 )); then
    printf 'Published npx doctor failed after 12 clean installs\n' >&2
    exit 1
  fi
  sleep 20
done
