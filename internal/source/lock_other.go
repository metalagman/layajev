//go:build !linux

package source

import (
	"context"
	"os"
)

type advisoryLock struct{}

func acquireAdvisoryLock(context.Context, *os.File) (*advisoryLock, error) {
	return nil, sourceError("lock materialization namespace", "", ErrUnsupported, nil)
}

func (l *advisoryLock) Close() error { return nil }

func renameNoReplace(*os.Root, string, string) (bool, error) {
	return false, sourceError("publish materialized bundle", "", ErrUnsupported, nil)
}
