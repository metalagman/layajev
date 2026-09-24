package source

import (
	"context"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/metalagman/layajev/internal/bundle"
)

func TestOpenDirectoryVerifiesOrdinaryBundle(t *testing.T) {
	dir := copyFixture(t)
	wantFiles := snapshotFiles(t, dir)

	source, err := OpenDirectory(t.Context(), dir)
	if err != nil {
		t.Fatalf("OpenDirectory() returned unexpected error: %v", err)
	}
	manifest := source.Manifest()
	if manifest.ID() == "" {
		t.Error("Manifest().ID() is empty")
	}
	path, err := source.Path(manifest.Model.Path)
	if err != nil {
		t.Fatalf("Path(%q) returned unexpected error: %v", manifest.Model.Path, err)
	}
	wantPath := filepath.Join(dir, filepath.FromSlash(manifest.Model.Path))
	if path != wantPath {
		t.Errorf("Path(%q) = %q, want %q", manifest.Model.Path, path, wantPath)
	}
	if err := source.Close(); err != nil {
		t.Fatalf("Close() returned unexpected error: %v", err)
	}
	if err := source.Close(); err != nil {
		t.Fatalf("second Close() returned unexpected error: %v", err)
	}
	if _, err := source.Path(manifest.Model.Path); !errors.Is(err, ErrClosed) {
		t.Errorf("Path() after Close error = %v, want errors.Is(_, ErrClosed)", err)
	}
	gotFiles := snapshotFiles(t, dir)
	if len(gotFiles) != len(wantFiles) {
		t.Fatalf("caller file count after open/close = %d, want %d", len(gotFiles), len(wantFiles))
	}
	for path, want := range wantFiles {
		if got := gotFiles[path]; got != want {
			t.Errorf("caller file %q changed after open/close", path)
		}
	}
}

func TestOpenDirectoryRejectsInvalidBundles(t *testing.T) {
	tests := []struct {
		name      string
		mutate    func(*testing.T, string)
		wantError error
	}{
		{
			name: "missing manifest",
			mutate: func(t *testing.T, dir string) {
				t.Helper()
				removeFixtureFile(t, filepath.Join(dir, "manifest.json"))
			},
			wantError: bundle.ErrInvalid,
		},
		{
			name: "extra file",
			mutate: func(t *testing.T, dir string) {
				t.Helper()
				writeFixtureFile(t, filepath.Join(dir, "extra.bin"), []byte("extra"))
			},
			wantError: bundle.ErrInvalid,
		},
		{
			name: "tampered artifact",
			mutate: func(t *testing.T, dir string) {
				t.Helper()
				writeFixtureFile(t, filepath.Join(dir, "model.onnx"), []byte("tampered onnx!\n"))
			},
			wantError: bundle.ErrIntegrity,
		},
		{
			name: "raw checkpoint",
			mutate: func(t *testing.T, dir string) {
				t.Helper()
				for path := range snapshotFiles(t, dir) {
					removeFixtureFile(t, filepath.Join(dir, filepath.FromSlash(path)))
				}
				writeFixtureFile(t, filepath.Join(dir, "model.safetensors"), []byte("raw"))
			},
			wantError: bundle.ErrInvalid,
		},
		{
			name: "unsupported manifest",
			mutate: func(t *testing.T, dir string) {
				t.Helper()
				path := filepath.Join(dir, "manifest.json")
				data, err := os.ReadFile(path)
				if err != nil {
					t.Fatalf("ReadFile(manifest): %v", err)
				}
				data = []byte(strings.Replace(string(data), `"schema_version":1`, `"schema_version":2`, 1))
				writeFixtureFile(t, path, data)
			},
			wantError: bundle.ErrUnsupported,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			dir := copyFixture(t)
			test.mutate(t, dir)
			source, err := OpenDirectory(t.Context(), dir)
			if source != nil {
				t.Fatal("OpenDirectory() source is non-nil on failure")
			}
			if !errors.Is(err, test.wantError) {
				t.Errorf("OpenDirectory() error = %v, want errors.Is(_, %v)", err, test.wantError)
			}
		})
	}
}

func TestOpenDirectoryRejectsLinksAndSpecialFiles(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlink fixture requires platform-specific privileges on Windows")
	}
	t.Run("root symlink", func(t *testing.T) {
		dir := copyFixture(t)
		link := filepath.Join(t.TempDir(), "bundle-link")
		if err := os.Symlink(dir, link); err != nil {
			t.Fatalf("Symlink(root): %v", err)
		}
		if _, err := OpenDirectory(t.Context(), link); !errors.Is(err, bundle.ErrInvalid) {
			t.Errorf("OpenDirectory(root symlink) error = %v, want bundle.ErrInvalid", err)
		}
	})
	t.Run("artifact symlink", func(t *testing.T) {
		dir := copyFixture(t)
		model := filepath.Join(dir, "model.onnx")
		backup := filepath.Join(dir, "model.real")
		if err := os.Rename(model, backup); err != nil {
			t.Fatalf("Rename(model): %v", err)
		}
		if err := os.Symlink("model.real", model); err != nil {
			t.Fatalf("Symlink(model): %v", err)
		}
		if _, err := OpenDirectory(t.Context(), dir); !errors.Is(err, ErrUnsupported) {
			t.Errorf("OpenDirectory(artifact symlink) error = %v, want ErrUnsupported", err)
		}
	})
	t.Run("named pipe", func(t *testing.T) {
		if runtime.GOOS != "linux" {
			t.Skip("named pipe fixture is verified on Linux")
		}
		dir := copyFixture(t)
		pipe := filepath.Join(dir, "extra.pipe")
		if err := makeNamedPipe(pipe); err != nil {
			t.Fatalf("makeNamedPipe(): %v", err)
		}
		if _, err := OpenDirectory(t.Context(), dir); !errors.Is(err, bundle.ErrInvalid) {
			t.Errorf("OpenDirectory(named pipe) error = %v, want bundle.ErrInvalid", err)
		}
	})
}

func TestOpenDirectoryCancellationAndCleanup(t *testing.T) {
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := OpenDirectory(canceled, copyFixture(t)); !errors.Is(err, context.Canceled) {
		t.Errorf("OpenDirectory(canceled) error = %v, want context.Canceled", err)
	}

	dir := copyFixture(t)
	var opened *countingRoot
	source, err := openDirectory(t.Context(), dir, func(path string) (directoryRoot, error) {
		root, openErr := os.OpenRoot(path)
		if openErr != nil {
			return nil, openErr
		}
		opened = &countingRoot{Root: root}
		return opened, nil
	})
	if err != nil {
		t.Fatalf("openDirectory(valid) returned unexpected error: %v", err)
	}
	if opened.closes != 0 {
		t.Errorf("successful open close count = %d, want 0", opened.closes)
	}
	if err := source.Close(); err != nil {
		t.Fatalf("Close(valid source) returned unexpected error: %v", err)
	}
	if opened.closes != 1 {
		t.Errorf("closed source close count = %d, want 1", opened.closes)
	}

	badDir := copyFixture(t)
	writeFixtureFile(t, filepath.Join(badDir, "extra.bin"), []byte("extra"))
	var failed *countingRoot
	_, err = openDirectory(t.Context(), badDir, func(path string) (directoryRoot, error) {
		root, openErr := os.OpenRoot(path)
		if openErr != nil {
			return nil, openErr
		}
		failed = &countingRoot{Root: root}
		return failed, nil
	})
	if !errors.Is(err, bundle.ErrInvalid) {
		t.Fatalf("openDirectory(invalid) error = %v, want bundle.ErrInvalid", err)
	}
	if failed.closes != 1 {
		t.Errorf("failed open close count = %d, want 1", failed.closes)
	}

	otherDir := copyFixture(t)
	var mismatched *countingRoot
	_, err = openDirectory(t.Context(), dir, func(string) (directoryRoot, error) {
		root, openErr := os.OpenRoot(otherDir)
		if openErr != nil {
			return nil, openErr
		}
		mismatched = &countingRoot{Root: root}
		return mismatched, nil
	})
	if !errors.Is(err, bundle.ErrInvalid) {
		t.Fatalf("openDirectory(mismatched root) error = %v, want bundle.ErrInvalid", err)
	}
	if mismatched.closes != 1 {
		t.Errorf("mismatched root close count = %d, want 1", mismatched.closes)
	}
}

func TestBundlePathRejectsUndeclaredAndUnsafePaths(t *testing.T) {
	source, err := OpenDirectory(t.Context(), copyFixture(t))
	if err != nil {
		t.Fatalf("OpenDirectory() returned unexpected error: %v", err)
	}
	t.Cleanup(func() { _ = source.Close() })

	for _, path := range []string{".", "../model.onnx", "/model.onnx", `tokenizer\tokenizer.json`, "missing.bin"} {
		t.Run(path, func(t *testing.T) {
			if _, err := source.Path(path); !errors.Is(err, bundle.ErrInvalid) {
				t.Errorf("Path(%q) error = %v, want bundle.ErrInvalid", path, err)
			}
		})
	}
}

func TestBundleReadFileIsBoundedAndRequiresOpenLease(t *testing.T) {
	source, err := OpenDirectory(t.Context(), copyFixture(t))
	if err != nil {
		t.Fatalf("OpenDirectory() returned unexpected error: %v", err)
	}
	data, err := source.ReadFile(t.Context(), "rl_agent_config.json", 3)
	if err != nil {
		t.Fatalf("ReadFile(calibration) returned unexpected error: %v", err)
	}
	if string(data) != "{}\n" {
		t.Errorf("ReadFile(calibration) = %q, want %q", data, "{}\\n")
	}
	if _, err := source.ReadFile(t.Context(), "rl_agent_config.json", 2); !errors.Is(err, bundle.ErrInvalid) {
		t.Errorf("ReadFile(over limit) error = %v, want bundle.ErrInvalid", err)
	}
	if err := source.Close(); err != nil {
		t.Fatalf("Close() returned unexpected error: %v", err)
	}
	if _, err := source.ReadFile(t.Context(), "rl_agent_config.json", 3); !errors.Is(err, ErrClosed) {
		t.Errorf("ReadFile(after close) error = %v, want ErrClosed", err)
	}
}

func TestSourceErrorsDoNotDiscloseExternalPathsOrContent(t *testing.T) {
	secret := "credential-sentinel-do-not-disclose"
	dir := filepath.Join(t.TempDir(), secret)
	if _, err := OpenDirectory(t.Context(), dir); err == nil {
		t.Fatal("OpenDirectory(missing) returned nil error")
	} else if strings.Contains(err.Error(), secret) {
		t.Errorf("OpenDirectory(missing) error disclosed external path: %v", err)
	}

	err := sourceError("operation", strings.Repeat("x", 256), ErrUnsupported, errors.New(secret))
	if strings.Contains(err.Error(), secret) || len(err.Error()) > 256 {
		t.Errorf("source error is not bounded and redacted: %q", err)
	}
}

type countingRoot struct {
	*os.Root
	closes int
}

func (r *countingRoot) Close() error {
	r.closes++
	return r.Root.Close()
}

func copyFixture(t *testing.T) string {
	t.Helper()
	source := filepath.Join("..", "bundle", "testdata", "embedded")
	destination := t.TempDir()
	err := filepath.WalkDir(source, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		relative, err := filepath.Rel(source, path)
		if err != nil {
			return err
		}
		if relative == "." {
			return nil
		}
		target := filepath.Join(destination, relative)
		if entry.IsDir() {
			return os.MkdirAll(target, 0o700)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		return os.WriteFile(target, data, 0o600)
	})
	if err != nil {
		t.Fatalf("copy fixture: %v", err)
	}
	return destination
}

func snapshotFiles(t *testing.T, dir string) map[string]string {
	t.Helper()
	files := make(map[string]string)
	err := filepath.WalkDir(dir, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() {
			return nil
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		relative, err := filepath.Rel(dir, path)
		if err != nil {
			return err
		}
		files[filepath.ToSlash(relative)] = string(data)
		return nil
	})
	if err != nil {
		t.Fatalf("snapshot files: %v", err)
	}
	return files
}

func writeFixtureFile(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatalf("MkdirAll(%q): %v", filepath.Base(path), err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("WriteFile(%q): %v", filepath.Base(path), err)
	}
}

func removeFixtureFile(t *testing.T, path string) {
	t.Helper()
	if err := os.Remove(path); err != nil {
		t.Fatalf("Remove(%q): %v", filepath.Base(path), err)
	}
}
