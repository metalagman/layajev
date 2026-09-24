//go:build !linux

package source

import "errors"

func makeNamedPipe(string) error {
	return errors.New("named pipes are not supported by this test")
}
