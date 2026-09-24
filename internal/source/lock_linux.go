//go:build linux

package source

import (
	"context"
	"errors"
	"os"
	"time"

	"golang.org/x/sys/unix"
)

type advisoryLock struct {
	file *os.File
}

func acquireAdvisoryLock(ctx context.Context, file *os.File) (*advisoryLock, error) {
	ticker := time.NewTicker(5 * time.Millisecond)
	defer ticker.Stop()
	for {
		if err := ctx.Err(); err != nil {
			return nil, sourceError("wait for materialization lock", "", err, err)
		}
		err := unix.Flock(int(file.Fd()), unix.LOCK_EX|unix.LOCK_NB)
		if err == nil {
			return &advisoryLock{file: file}, nil
		}
		if !errors.Is(err, unix.EWOULDBLOCK) && !errors.Is(err, unix.EAGAIN) {
			return nil, sourceError("lock materialization namespace", "", ErrMaterialization, err)
		}
		select {
		case <-ctx.Done():
			return nil, sourceError("wait for materialization lock", "", ctx.Err(), ctx.Err())
		case <-ticker.C:
		}
	}
}

func (l *advisoryLock) Close() error {
	if l == nil || l.file == nil {
		return nil
	}
	unlockErr := unix.Flock(int(l.file.Fd()), unix.LOCK_UN)
	closeErr := l.file.Close()
	l.file = nil
	if unlockErr != nil {
		return sourceError("unlock materialization namespace", "", ErrMaterialization, unlockErr)
	}
	if closeErr != nil {
		return sourceError("close materialization lock", "", ErrMaterialization, closeErr)
	}
	return nil
}

func renameNoReplace(root *os.Root, oldPath, newPath string) (renamed bool, returnErr error) {
	directory, err := root.Open(".")
	if err != nil {
		return false, sourceError("open publication namespace", "", ErrMaterialization, err)
	}
	defer func() {
		if err := directory.Close(); err != nil {
			returnErr = errors.Join(returnErr, sourceError("close publication namespace", "", ErrMaterialization, err))
		}
	}()
	err = unix.Renameat2(int(directory.Fd()), oldPath, int(directory.Fd()), newPath, unix.RENAME_NOREPLACE)
	if err == nil {
		return true, nil
	}
	if errors.Is(err, unix.EEXIST) {
		return false, nil
	}
	return false, sourceError("publish materialized bundle", "", ErrMaterialization, err)
}
