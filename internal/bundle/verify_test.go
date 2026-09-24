package bundle

import (
	"context"
	"embed"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"strings"
	"testing"
	"testing/fstest"
)

//go:embed testdata/embedded
var embeddedBundle embed.FS

func TestVerifySourceNeutralRepresentations(t *testing.T) {
	huggingFaceRoot := "models--convaiinnovations--laya-multilingual/snapshots/052592a15d198d9ad47da779604259b10b47b7aa"
	huggingFaceFS := prefixedBundle(t, huggingFaceRoot)
	tests := []struct {
		name       string
		fileSystem fs.FS
		root       string
	}{
		{name: "ordinary directory", fileSystem: osDirFS(t), root: "."},
		{name: "nested snapshot", fileSystem: huggingFaceFS, root: huggingFaceRoot},
		{name: "embedded filesystem", fileSystem: embeddedBundle, root: "testdata/embedded"},
	}

	var wantID string
	var wantFiles map[string][]byte
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			manifest, err := Verify(context.Background(), test.fileSystem, test.root)
			if err != nil {
				t.Fatalf("Verify() returned unexpected error: %v", err)
			}
			if wantID == "" {
				wantID = manifest.ID()
				wantFiles = readLogicalFiles(t, test.fileSystem, test.root, manifest)
				return
			}
			if manifest.ID() != wantID {
				t.Errorf("manifest ID = %q, want %q", manifest.ID(), wantID)
			}
			gotFiles := readLogicalFiles(t, test.fileSystem, test.root, manifest)
			for name, want := range wantFiles {
				got := gotFiles[name]
				if string(got) != string(want) {
					t.Errorf("artifact %q differs across representations", name)
				}
			}
		})
	}
}

func TestVerifyRejectsLayoutAndIntegrityFailures(t *testing.T) {
	tests := []struct {
		name      string
		mutate    func(t *testing.T, bundle fstest.MapFS)
		wantError error
		wantPath  string
	}{
		{
			name: "extra file",
			mutate: func(t *testing.T, bundle fstest.MapFS) {
				t.Helper()
				bundle["extra.bin"] = &fstest.MapFile{Data: []byte("extra")}
			},
			wantError: ErrInvalid,
			wantPath:  "extra.bin",
		},
		{
			name: "case-colliding file",
			mutate: func(t *testing.T, bundle fstest.MapFS) {
				t.Helper()
				bundle["MODEL.ONNX"] = &fstest.MapFile{Data: []byte("collision")}
			},
			wantError: ErrInvalid,
			wantPath:  "MODEL.ONNX",
		},
		{
			name: "missing file",
			mutate: func(t *testing.T, bundle fstest.MapFS) {
				t.Helper()
				delete(bundle, "model.onnx")
			},
			wantError: ErrInvalid,
			wantPath:  "model.onnx",
		},
		{
			name: "tampered same size",
			mutate: func(t *testing.T, bundle fstest.MapFS) {
				t.Helper()
				bundle["model.onnx"].Data = []byte("tampered onnx!\n")
			},
			wantError: ErrIntegrity,
			wantPath:  "model.onnx",
		},
		{
			name: "truncated file",
			mutate: func(t *testing.T, bundle fstest.MapFS) {
				t.Helper()
				bundle["model.onnx"].Data = []byte("short")
			},
			wantError: ErrIntegrity,
			wantPath:  "model.onnx",
		},
		{
			name: "symlink entry",
			mutate: func(t *testing.T, bundle fstest.MapFS) {
				t.Helper()
				bundle["extra-link"] = &fstest.MapFile{Mode: fs.ModeSymlink}
			},
			wantError: ErrInvalid,
			wantPath:  "extra-link",
		},
		{
			name: "wrong linked role",
			mutate: func(t *testing.T, bundle fstest.MapFS) {
				t.Helper()
				mutateManifest(t, bundle, func(object map[string]any) {
					files := object["files"].([]any)
					files[4].(map[string]any)["role"] = "notice"
				})
			},
			wantError: ErrInvalid,
			wantPath:  "manifest.json",
		},
		{
			name: "unsupported manifest",
			mutate: func(t *testing.T, bundle fstest.MapFS) {
				t.Helper()
				mutateManifest(t, bundle, func(object map[string]any) {
					object["schema_version"] = float64(2)
				})
			},
			wantError: ErrUnsupported,
			wantPath:  "manifest.json",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			bundle := cloneEmbeddedBundle(t)
			test.mutate(t, bundle)
			_, err := Verify(context.Background(), bundle, ".")
			if !errors.Is(err, test.wantError) {
				t.Fatalf("Verify() error = %v, want errors.Is(_, %v)", err, test.wantError)
			}
			var bundleError *Error
			if !errors.As(err, &bundleError) {
				t.Fatalf("Verify() error type = %T, want *Error", err)
			}
			if bundleError.Path != test.wantPath {
				t.Errorf("Error.Path = %q, want %q", bundleError.Path, test.wantPath)
			}
		})
	}
}

func TestVerifyRejectsRawCheckpointWithGuidance(t *testing.T) {
	raw := fstest.MapFS{
		"model.safetensors": &fstest.MapFile{Data: []byte("raw checkpoint")},
	}
	_, err := Verify(context.Background(), raw, ".")
	if !errors.Is(err, ErrInvalid) {
		t.Fatalf("Verify(raw checkpoint) error = %v, want ErrInvalid", err)
	}
	if !strings.Contains(err.Error(), "exporter input only") || !strings.Contains(err.Error(), "offline bundle exporter") {
		t.Errorf("Verify(raw checkpoint) error = %q, want actionable exporter guidance", err)
	}
}

func TestVerifyPreservesCancellationAndReadFailure(t *testing.T) {
	canceledContext, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := Verify(canceledContext, cloneEmbeddedBundle(t), ".")
	if !errors.Is(err, context.Canceled) || !errors.Is(err, ErrInvalid) {
		t.Fatalf("Verify(canceled) error = %v, want context.Canceled and ErrInvalid", err)
	}

	wantReadError := errors.New("synthetic read failure")
	failing := readErrorFS{
		FS:     cloneEmbeddedBundle(t),
		target: "model.onnx",
		err:    wantReadError,
	}
	_, err = Verify(context.Background(), failing, ".")
	if !errors.Is(err, wantReadError) || !errors.Is(err, ErrIntegrity) {
		t.Fatalf("Verify(read failure) error = %v, want read cause and ErrIntegrity", err)
	}
}

func FuzzVerifyManifest(f *testing.F) {
	valid := cloneEmbeddedBundle(f)["manifest.json"].Data
	f.Add(valid)
	f.Add([]byte(`{}`))
	f.Fuzz(func(t *testing.T, manifest []byte) {
		bundle := cloneEmbeddedBundle(t)
		bundle["manifest.json"].Data = append([]byte(nil), manifest...)
		_, _ = Verify(context.Background(), bundle, ".")
	})
}

func osDirFS(t *testing.T) fs.FS {
	t.Helper()
	return os.DirFS("testdata/embedded")
}

func cloneEmbeddedBundle(t testing.TB) fstest.MapFS {
	t.Helper()
	result := make(fstest.MapFS)
	err := fs.WalkDir(embeddedBundle, "testdata/embedded", func(name string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() {
			return nil
		}
		data, err := fs.ReadFile(embeddedBundle, name)
		if err != nil {
			return err
		}
		logicalName := strings.TrimPrefix(name, "testdata/embedded/")
		result[logicalName] = &fstest.MapFile{Data: append([]byte(nil), data...)}
		return nil
	})
	if err != nil {
		t.Fatalf("clone embedded bundle: %v", err)
	}
	return result
}

func prefixedBundle(t *testing.T, prefix string) fstest.MapFS {
	t.Helper()
	result := make(fstest.MapFS)
	for name, file := range cloneEmbeddedBundle(t) {
		result[prefix+"/"+name] = file
	}
	return result
}

func readLogicalFiles(t *testing.T, fileSystem fs.FS, root string, manifest Manifest) map[string][]byte {
	t.Helper()
	rootFS, err := fs.Sub(fileSystem, root)
	if err != nil {
		t.Fatalf("fs.Sub(%q): %v", root, err)
	}
	result := make(map[string][]byte, len(manifest.Files)+1)
	paths := append([]File{{Path: "manifest.json"}}, manifest.Files...)
	for _, file := range paths {
		data, err := fs.ReadFile(rootFS, file.Path)
		if err != nil {
			t.Fatalf("fs.ReadFile(%q): %v", file.Path, err)
		}
		result[file.Path] = data
	}
	return result
}

func mutateManifest(t *testing.T, bundle fstest.MapFS, mutate func(map[string]any)) {
	t.Helper()
	var object map[string]any
	if err := json.Unmarshal(bundle["manifest.json"].Data, &object); err != nil {
		t.Fatalf("json.Unmarshal(manifest): %v", err)
	}
	mutate(object)
	data, err := json.Marshal(object)
	if err != nil {
		t.Fatalf("json.Marshal(manifest): %v", err)
	}
	bundle["manifest.json"].Data = data
}

type readErrorFS struct {
	fs.FS
	target string
	err    error
}

func (f readErrorFS) Open(name string) (fs.File, error) {
	file, err := f.FS.Open(name)
	if err != nil || name != f.target {
		return file, err
	}
	return &readErrorFile{File: file, err: f.err}, nil
}

type readErrorFile struct {
	fs.File
	err error
}

func (f *readErrorFile) Read([]byte) (int, error) { return 0, f.err }
