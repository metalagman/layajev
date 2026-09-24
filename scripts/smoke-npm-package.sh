#!/usr/bin/env bash
set -euo pipefail

# Exercise the staged tarballs exactly as an isolated npm consumer would.
# LAYA_BUNDLE_DIR is optional: when supplied, also exercise the installed API.
test -n "${OMNIDIST_VERSION:-}"
platform_dir=.omnidist/default/npm/@metalagman/layajev-linux-x64
meta_dir=.omnidist/default/npm/@metalagman/layajev
test -f "$platform_dir/package.json"
test -f "$meta_dir/package.json"
test "$(jq -r .version "$platform_dir/package.json")" = "$OMNIDIST_VERSION"
test "$(jq -r .version "$meta_dir/package.json")" = "$OMNIDIST_VERSION"

work_dir="$(mktemp -d)"
server_pid=
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill -- "-$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  rm -rf -- "$work_dir"
}
trap cleanup EXIT
export npm_config_cache="$work_dir/npm-cache"
platform_archive="$(npm pack --silent --pack-destination "$work_dir" "$platform_dir")"
meta_archive="$(npm pack --silent --pack-destination "$work_dir" "$meta_dir")"
consumer="$work_dir/consumer"
mkdir "$consumer"
npm install --offline --ignore-scripts --no-audit --no-fund --prefix "$consumer" \
  "$work_dir/$platform_archive" "$work_dir/$meta_archive"

installed="$consumer/node_modules/.bin/layajev"
test -x "$installed"
test -f "$consumer/node_modules/@metalagman/layajev-linux-x64/bin/libonnxruntime.so.1.29.0"
env -u LAYA_ONNXRUNTIME_LIBRARY -u LAYA_TOKENIZERS_LIBRARY \
  "$installed" --help >/dev/null
doctor_output="$(env -u LAYA_ONNXRUNTIME_LIBRARY -u LAYA_TOKENIZERS_LIBRARY \
  -u ORT_DISABLE_TELEMETRY "$installed" doctor)"
test "$doctor_output" = 'native runtime OK'

if [[ -z "${LAYA_BUNDLE_DIR:-}" ]]; then
  echo 'Installed npm packages and initialized the bundled native runtime.'
  exit 0
fi

test -f "$LAYA_BUNDLE_DIR/manifest.json"
command -v setsid >/dev/null
listen=127.0.0.1:18977
setsid env -u LAYA_ONNXRUNTIME_LIBRARY -u LAYA_TOKENIZERS_LIBRARY \
  -u ORT_DISABLE_TELEMETRY "$installed" serve --bundle "$LAYA_BUNDLE_DIR" \
  --listen "$listen" >"$work_dir/server.log" 2>&1 &
server_pid=$!
for attempt in $(seq 1 120); do
  if curl --fail --silent --show-error --max-time 1 \
    "http://$listen/v1/models" >"$work_dir/models.json" 2>/dev/null; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo 'Installed server exited before readiness:' >&2
    sed -n '1,120p' "$work_dir/server.log" >&2
    exit 1
  fi
  sleep 1
done
model="$(jq -er '.models | if length == 1 then .[0].name else empty end' "$work_dir/models.json")"
jq -n --arg model "$model" \
  '{model: $model, state: "I was charged twice.", questions: {billing: {type: "noul", instructions: "Is this about billing?"}}}' \
  >"$work_dir/request.json"
curl --fail --silent --show-error --max-time 90 -H 'Content-Type: application/json' \
  --data-binary "@$work_dir/request.json" "http://$listen/v1/systemone" \
  >"$work_dir/response.json"
jq -e --arg model "$model" '.model == $model and .answers.billing.type == "noul" and (.answers.billing.noul | type == "number")' \
  "$work_dir/response.json" >/dev/null
echo 'Installed npm package served a real model prediction.'
