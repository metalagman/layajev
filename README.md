# layajev

`layajev` is a local Laya inference server with a documented Jev-compatible API subset. It uses [laya-go](https://github.com/metalagman/laya-go) for in-process inference and Go ADK for request execution. The server never downloads a model; acquisition and conversion are explicit commands.

```sh
go run ./cmd/layajev --help
go run ./cmd/layajev fetch --destination ./source-snapshot
go run ./cmd/layajev convert --source ./source-snapshot --sdk ./pinned-sdk --output ./bundle --epoch 1790035200
go run -tags=laya_native ./cmd/layajev serve --bundle ./bundle
```

`convert` uses the pinned exporter in this checkout and requires Go Task, uv, and verified local model/SDK inputs. Native `doctor` and `serve` require the documented `linux/amd64` libraries. The published npm package remains `@metalagman/layajev`; do not tag a new release until npm trusted publishing is bound to this repository.

For a container that builds the verified model bundle into the image, see [container operations](docs/layajev-container-runbook.md). See [CLI operations](docs/layajev-runbook.md) and [npm release operations](docs/layajev-npm-release.md) for prerequisites, API contract, and release procedure.

Run `task --list-all` for repository operations. The software is MIT licensed; model and SDK licenses are separate and must be reviewed before redistribution.
