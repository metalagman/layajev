package source

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"testing/fstest"

	"github.com/metalagman/layajev/internal/bundle"
)

func TestStageMapFSProducesExactVerifiedBundle(t *testing.T) {
	source := fixtureMapFS(t)
	wantSource := cloneMapFS(source)
	destination := filepath.Join(t.TempDir(), "stage")

	manifest, err := Stage(t.Context(), source, ".", destination)
	if err != nil {
		t.Fatalf("Stage() returned unexpected error: %v", err)
	}
	verified, err := bundle.Verify(t.Context(), os.DirFS(destination), ".")
	if err != nil {
		t.Fatalf("bundle.Verify(staged) returned unexpected error: %v", err)
	}
	if verified.ID() != manifest.ID() {
		t.Errorf("staged bundle ID = %q, want %q", verified.ID(), manifest.ID())
	}
	for name, want := range wantSource {
		got := source[name]
		if got == nil || !bytes.Equal(got.Data, want.Data) || got.Mode != want.Mode {
			t.Errorf("source entry %q changed during staging", name)
		}
	}
	assertRestrictiveTree(t, destination)
}

func TestStageAcceptsNestedRootAndEmptyRoot(t *testing.T) {
	tests := []struct {
		name       string
		fileSystem fs.FS
		root       string
	}{
		{name: "empty root", fileSystem: fixtureMapFS(t), root: ""},
		{name: "nested root", fileSystem: prefixedMapFS("models/laya", fixtureMapFS(t)), root: "models/laya"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			destination := filepath.Join(t.TempDir(), "stage")
			if _, err := Stage(t.Context(), test.fileSystem, test.root, destination); err != nil {
				t.Fatalf("Stage(root=%q) returned unexpected error: %v", test.root, err)
			}
			if _, err := bundle.Verify(t.Context(), os.DirFS(destination), "."); err != nil {
				t.Fatalf("bundle.Verify(staged) returned unexpected error: %v", err)
			}
		})
	}
}

func TestStageRejectsInvalidInputsWithoutCreatingDestination(t *testing.T) {
	tests := []struct {
		name       string
		fileSystem fs.FS
		root       string
		wantError  error
	}{
		{name: "nil filesystem", fileSystem: nil, root: ".", wantError: bundle.ErrInvalid},
		{name: "parent root", fileSystem: fixtureMapFS(t), root: "..", wantError: bundle.ErrInvalid},
		{name: "absolute root", fileSystem: fixtureMapFS(t), root: "/bundle", wantError: bundle.ErrInvalid},
		{name: "backslash root", fileSystem: fixtureMapFS(t), root: `models\bundle`, wantError: bundle.ErrInvalid},
		{
			name: "extra source entry",
			fileSystem: func() fs.FS {
				files := fixtureMapFS(t)
				files["extra.bin"] = &fstest.MapFile{Data: []byte("extra")}
				return files
			}(),
			root:      ".",
			wantError: bundle.ErrInvalid,
		},
		{
			name: "non-regular source entry",
			fileSystem: func() fs.FS {
				files := fixtureMapFS(t)
				files["extra.pipe"] = &fstest.MapFile{Mode: fs.ModeNamedPipe}
				return files
			}(),
			root:      ".",
			wantError: bundle.ErrInvalid,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			destination := filepath.Join(t.TempDir(), "stage")
			_, err := Stage(t.Context(), test.fileSystem, test.root, destination)
			if !errors.Is(err, test.wantError) {
				t.Errorf("Stage() error = %v, want errors.Is(_, %v)", err, test.wantError)
			}
			assertNotExist(t, destination)
		})
	}
}

func TestStageDoesNotReuseOrRemoveExistingDestination(t *testing.T) {
	parent := t.TempDir()
	destination := filepath.Join(parent, "stage")
	if err := os.Mkdir(destination, 0o700); err != nil {
		t.Fatalf("Mkdir(stage): %v", err)
	}
	sentinel := filepath.Join(destination, "caller-owned")
	writeFixtureFile(t, sentinel, []byte("keep"))

	_, err := Stage(t.Context(), fixtureMapFS(t), ".", destination)
	if !errors.Is(err, ErrMaterialization) {
		t.Errorf("Stage(existing) error = %v, want ErrMaterialization", err)
	}
	data, readErr := os.ReadFile(sentinel)
	if readErr != nil || string(data) != "keep" {
		t.Errorf("existing destination changed: data=%q err=%v", data, readErr)
	}
}

func TestStageRejectsSymlinkedParent(t *testing.T) {
	realParent := t.TempDir()
	link := filepath.Join(t.TempDir(), "work-link")
	if err := os.Symlink(realParent, link); err != nil {
		t.Skipf("Symlink(parent) is unavailable: %v", err)
	}
	destination := filepath.Join(link, "stage")
	_, err := Stage(t.Context(), fixtureMapFS(t), ".", destination)
	if !errors.Is(err, ErrMaterialization) {
		t.Errorf("Stage(symlinked parent) error = %v, want ErrMaterialization", err)
	}
	assertNotExist(t, filepath.Join(realParent, "stage"))
}

func TestStageDetectsChangesAndReadFailuresDuringCopy(t *testing.T) {
	readFailure := errors.New("source read failure secret")
	tests := []struct {
		name      string
		replace   func(*testing.T, fs.File, context.CancelFunc) (fs.File, error)
		wantError error
	}{
		{
			name: "short reader",
			replace: func(t *testing.T, file fs.File, _ context.CancelFunc) (fs.File, error) {
				return replacementFile(t, file, func(data []byte) io.Reader { return bytes.NewReader(data[:len(data)-1]) }), nil
			},
			wantError: bundle.ErrIntegrity,
		},
		{
			name: "long reader",
			replace: func(t *testing.T, file fs.File, _ context.CancelFunc) (fs.File, error) {
				return replacementFile(t, file, func(data []byte) io.Reader {
					return bytes.NewReader(append(data, byte(0)))
				}), nil
			},
			wantError: bundle.ErrIntegrity,
		},
		{
			name: "same-size digest change",
			replace: func(t *testing.T, file fs.File, _ context.CancelFunc) (fs.File, error) {
				return replacementFile(t, file, func(data []byte) io.Reader {
					data[0] ^= 0xff
					return bytes.NewReader(data)
				}), nil
			},
			wantError: bundle.ErrIntegrity,
		},
		{
			name: "unreadable after verification",
			replace: func(_ *testing.T, file fs.File, _ context.CancelFunc) (fs.File, error) {
				_ = file.Close()
				return nil, fs.ErrPermission
			},
			wantError: bundle.ErrInvalid,
		},
		{
			name: "read failure",
			replace: func(t *testing.T, file fs.File, _ context.CancelFunc) (fs.File, error) {
				return replacementFile(t, file, func([]byte) io.Reader { return errorReader{err: readFailure} }), nil
			},
			wantError: readFailure,
		},
		{
			name: "canceled read",
			replace: func(t *testing.T, file fs.File, cancel context.CancelFunc) (fs.File, error) {
				return replacementFile(t, file, func(data []byte) io.Reader {
					return &cancelingReader{reader: bytes.NewReader(data), cancel: cancel}
				}), nil
			},
			wantError: context.Canceled,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			changing := &secondOpenFS{
				FS:      fixtureMapFS(t),
				target:  "model.onnx",
				replace: func(t *testing.T, file fs.File) (fs.File, error) { return test.replace(t, file, cancel) },
				t:       t,
			}
			destination := filepath.Join(t.TempDir(), "stage")
			_, err := Stage(ctx, changing, ".", destination)
			if !errors.Is(err, test.wantError) {
				t.Errorf("Stage() error = %v, want errors.Is(_, %v)", err, test.wantError)
			}
			if strings.Contains(err.Error(), "source read failure secret") {
				t.Errorf("Stage() error disclosed source error text: %v", err)
			}
			assertNotExist(t, destination)
		})
	}
}

func TestStageUsesFixedReadBufferForLargeArtifact(t *testing.T) {
	files := fixtureMapFS(t)
	large := bytes.Repeat([]byte("0123456789abcdef"), 1<<17)
	replaceArtifact(t, files, "model.onnx", large)
	observed := &observingFS{FS: files, target: "model.onnx"}
	destination := filepath.Join(t.TempDir(), "stage")

	if _, err := Stage(t.Context(), observed, ".", destination); err != nil {
		t.Fatalf("Stage(large) returned unexpected error: %v", err)
	}
	if observed.maxRead > stageBufferBytes {
		t.Errorf("maximum source Read buffer = %d, want at most %d", observed.maxRead, stageBufferBytes)
	}
	if observed.maxRead == 0 || int64(len(large)) <= int64(observed.maxRead) {
		t.Errorf("large artifact bytes = %d, maximum Read buffer = %d; streaming was not exercised", len(large), observed.maxRead)
	}
}

func fixtureMapFS(t *testing.T) fstest.MapFS {
	t.Helper()
	root := filepath.Join("..", "bundle", "testdata", "embedded")
	result := make(fstest.MapFS)
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
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
		relative, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		result[filepath.ToSlash(relative)] = &fstest.MapFile{Data: data, Mode: 0o600}
		return nil
	})
	if err != nil {
		t.Fatalf("load fixture MapFS: %v", err)
	}
	return result
}

func cloneMapFS(source fstest.MapFS) fstest.MapFS {
	result := make(fstest.MapFS, len(source))
	for name, file := range source {
		copy := *file
		copy.Data = append([]byte(nil), file.Data...)
		result[name] = &copy
	}
	return result
}

func prefixedMapFS(prefix string, source fstest.MapFS) fstest.MapFS {
	result := make(fstest.MapFS, len(source))
	for name, file := range source {
		result[prefix+"/"+name] = file
	}
	return result
}

func replaceArtifact(t *testing.T, files fstest.MapFS, path string, data []byte) {
	t.Helper()
	manifestFile := files["manifest.json"]
	var manifest bundle.Manifest
	if err := json.Unmarshal(manifestFile.Data, &manifest); err != nil {
		t.Fatalf("Unmarshal(manifest): %v", err)
	}
	digest := sha256.Sum256(data)
	found := false
	for index := range manifest.Files {
		if manifest.Files[index].Path != path {
			continue
		}
		manifest.Files[index].Size = int64(len(data))
		manifest.Files[index].SHA256 = hex.EncodeToString(digest[:])
		found = true
		break
	}
	if !found {
		t.Fatalf("manifest does not declare %q", path)
	}
	manifestBytes, err := json.Marshal(manifest)
	if err != nil {
		t.Fatalf("Marshal(manifest): %v", err)
	}
	files["manifest.json"] = &fstest.MapFile{Data: manifestBytes, Mode: 0o600}
	files[path] = &fstest.MapFile{Data: append([]byte(nil), data...), Mode: 0o600}
}

func assertRestrictiveTree(t *testing.T, root string) {
	t.Helper()
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		if info.Mode().Perm()&0o077 != 0 {
			t.Errorf("staged entry %q permissions = %o, want no group/other bits", filepath.Base(path), info.Mode().Perm())
		}
		return nil
	})
	if err != nil {
		t.Fatalf("WalkDir(staged): %v", err)
	}
}

func assertNotExist(t *testing.T, path string) {
	t.Helper()
	if _, err := os.Lstat(path); !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("Lstat(%q) error = %v, want fs.ErrNotExist", filepath.Base(path), err)
	}
}

type secondOpenFS struct {
	fs.FS
	target  string
	opens   int
	replace func(*testing.T, fs.File) (fs.File, error)
	t       *testing.T
}

func (f *secondOpenFS) Open(name string) (fs.File, error) {
	file, err := f.FS.Open(name)
	if err != nil || name != f.target {
		return file, err
	}
	f.opens++
	if f.opens != 2 {
		return file, nil
	}
	return f.replace(f.t, file)
}

func replacementFile(t *testing.T, original fs.File, reader func([]byte) io.Reader) fs.File {
	t.Helper()
	info, err := original.Stat()
	if err != nil {
		_ = original.Close()
		t.Fatalf("Stat(original): %v", err)
	}
	data, err := io.ReadAll(original)
	if err != nil {
		_ = original.Close()
		t.Fatalf("ReadAll(original): %v", err)
	}
	if err := original.Close(); err != nil {
		t.Fatalf("Close(original): %v", err)
	}
	return &scriptedFile{reader: reader(data), info: info}
}

type scriptedFile struct {
	reader io.Reader
	info   fs.FileInfo
}

func (f *scriptedFile) Stat() (fs.FileInfo, error) { return f.info, nil }
func (f *scriptedFile) Read(data []byte) (int, error) {
	return f.reader.Read(data)
}
func (f *scriptedFile) Close() error { return nil }

type errorReader struct{ err error }

func (r errorReader) Read([]byte) (int, error) { return 0, r.err }

type cancelingReader struct {
	reader io.Reader
	cancel context.CancelFunc
}

func (r *cancelingReader) Read(data []byte) (int, error) {
	count, err := r.reader.Read(data)
	r.cancel()
	return count, err
}

type observingFS struct {
	fs.FS
	target  string
	maxRead int
}

func (f *observingFS) Open(name string) (fs.File, error) {
	file, err := f.FS.Open(name)
	if err != nil || name != f.target {
		return file, err
	}
	return &observingFile{File: file, observe: func(size int) {
		if size > f.maxRead {
			f.maxRead = size
		}
	}}, nil
}

type observingFile struct {
	fs.File
	observe func(int)
}

func (f *observingFile) Read(data []byte) (int, error) {
	f.observe(len(data))
	return f.File.Read(data)
}

var _ fs.File = (*scriptedFile)(nil)
