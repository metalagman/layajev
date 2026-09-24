//go:build linux

package source

import "syscall"

func makeNamedPipe(path string) error {
	return syscall.Mkfifo(path, 0o600)
}
