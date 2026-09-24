package bundle_test

import (
	"embed"
	"os"
	"path/filepath"
	"testing"

	"github.com/metalagman/layajev/internal/bundle"
	"github.com/metalagman/layajev/internal/source"
)

//go:embed testdata/embedded
var stageEmbeddedBundle embed.FS

func TestStageEmbeddedBundle(t *testing.T) {
	destination := filepath.Join(t.TempDir(), "stage")
	manifest, err := source.Stage(t.Context(), stageEmbeddedBundle, "testdata/embedded", destination)
	if err != nil {
		t.Fatalf("source.Stage(embed.FS) returned unexpected error: %v", err)
	}
	verified, err := bundle.Verify(t.Context(), os.DirFS(destination), ".")
	if err != nil {
		t.Fatalf("bundle.Verify(staged embed.FS) returned unexpected error: %v", err)
	}
	if verified.ID() != manifest.ID() {
		t.Errorf("staged embed.FS bundle ID = %q, want %q", verified.ID(), manifest.ID())
	}
}
