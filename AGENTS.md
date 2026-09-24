# Repository instructions

- `layajev` is the executable Jev-compatible adapter; reusable local inference remains in `github.com/metalagman/laya-go`.
- Keep model acquisition in explicit CLI/build operations. `serve` must only use a local verified bundle and in-process inference.
- Follow Google Go style, `gofmt`, narrow interfaces, wrapped errors, and standard-library tests.
- Use `Taskfile.yml` for formatting, tests, exporter, packaging, and container operations.
- Never commit model weights, native libraries, credentials, or generated bundle caches.
- Preserve the documented `linux/amd64` support boundary until integration evidence establishes more.
