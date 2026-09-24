//go:build !laya_native

package layajev

import (
	"context"
	"errors"
	"testing"

	"github.com/metalagman/laya-go"
)

func TestDoctorReportsUnavailableBackend(t *testing.T) {
	t.Parallel()
	command := NewCommand()
	command.SetArgs([]string{"doctor"})
	if err := command.ExecuteContext(context.Background()); !errors.Is(err, laya.ErrNativeUnavailable) {
		t.Errorf("doctor error = %v, want ErrNativeUnavailable", err)
	}
}
