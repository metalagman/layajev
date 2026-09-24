#!/usr/bin/env bash
set -euo pipefail

require_file() {
  local variable="$1"
  local path="${!variable:-}"
  if [[ -z "${path}" ]]; then
    echo "${variable} is required" >&2
    exit 1
  fi
  if [[ ! -f "${path}" ]]; then
    echo "${variable} must name a regular file: ${path}" >&2
    exit 1
  fi
}

verify_file() {
  local path="$1"
  local expected_size="$2"
  local expected_sha256="$3"
  local observed_size
  local observed_sha256
  observed_size="$(wc -c < "${path}")"
  observed_sha256="$(sha256sum "${path}" | awk '{print $1}')"
  if [[ "${observed_size}" != "${expected_size}" ]]; then
    echo "native artifact size mismatch for ${path}: got ${observed_size}, want ${expected_size}" >&2
    exit 1
  fi
  if [[ "${observed_sha256}" != "${expected_sha256}" ]]; then
    echo "native artifact SHA-256 mismatch for ${path}: got ${observed_sha256}, want ${expected_sha256}" >&2
    exit 1
  fi
}

if [[ "$(go env GOOS)" != "linux" || "$(go env GOARCH)" != "amd64" ]]; then
  echo "the reviewed native candidate requires linux/amd64" >&2
  exit 1
fi
if [[ "$(go env CGO_ENABLED)" != "1" ]]; then
  echo "the reviewed native candidate requires CGO_ENABLED=1" >&2
  exit 1
fi

require_file LAYA_ONNXRUNTIME_LIBRARY
require_file LAYA_TOKENIZERS_LIBRARY
verify_file "${LAYA_ONNXRUNTIME_LIBRARY}" 28497752 5715f06d8992ca8eeeddcce43df3a7d38f97d537052126f558e912cb312460ca
verify_file "${LAYA_TOKENIZERS_LIBRARY}" 49776794 e6862b31745bb7d07980fcee70e49cd3b4318097609180f5d2d3fb394f305d50

if ! nm -D --defined-only "${LAYA_ONNXRUNTIME_LIBRARY}" | rg 'OrtGetApiBase@@VERS_1\.29\.0$' >/dev/null; then
  echo "ONNX Runtime artifact does not export the reviewed 1.29.0 API" >&2
  exit 1
fi
if ! nm -g --defined-only "${LAYA_TOKENIZERS_LIBRARY}" | rg ' tokenizers_version_1_26_0$' >/dev/null; then
  echo "tokenizers artifact does not export the reviewed ABI marker" >&2
  exit 1
fi

# Rust's statically linked standard library contains dormant libc networking
# references even when the application crate has no network dependency. Reject
# any such reference from a non-stdlib archive member, and reject embedded
# network endpoint/authentication strings everywhere in the pinned archive.
unexpected_network_symbols="$(
  nm -A -u "${LAYA_TOKENIZERS_LIBRARY}" |
    rg ' U (socket|connect|getaddrinfo|recv|send)$' |
    rg -v ':std-[^:]+\.std\.[^:]+-cgu\.0\.rcgu\.o:' || true
)"
if [[ -n "${unexpected_network_symbols}" ]]; then
  echo "tokenizers artifact contains networking references outside Rust std:" >&2
  echo "${unexpected_network_symbols}" >&2
  exit 1
fi
if strings "${LAYA_TOKENIZERS_LIBRARY}" | rg -i 'https?://|authorization:|huggingface\.co' >/dev/null; then
  echo "tokenizers artifact contains a network endpoint or authentication marker" >&2
  exit 1
fi
