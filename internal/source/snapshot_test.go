//go:build linux

package source

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/metalagman/layajev/internal/bundle"
)

const testRevision = "052592a15d198d9ad47da779604259b10b47b7aa"

func TestOpenDirectoryHuggingFaceSnapshotMatchesOrdinaryBundle(t *testing.T) {
	ordinary := copyFixture(t)
	addExternalData(t, ordinary)
	snapshot := makeSnapshot(t, ordinary)

	want, err := OpenDirectory(t.Context(), ordinary)
	if err != nil {
		t.Fatalf("OpenDirectory(ordinary) returned unexpected error: %v", err)
	}
	t.Cleanup(func() { _ = want.Close() })
	got, err := OpenDirectory(t.Context(), snapshot.dir)
	if err != nil {
		t.Fatalf("OpenDirectory(snapshot) returned unexpected error: %v", err)
	}
	t.Cleanup(func() { _ = got.Close() })
	if got.Manifest().ID() != want.Manifest().ID() {
		t.Errorf("snapshot bundle ID = %q, want %q", got.Manifest().ID(), want.Manifest().ID())
	}
	for _, path := range []string{"model.onnx", "model.onnx.data", "tokenizer/tokenizer.json"} {
		gotPath, err := got.Path(path)
		if err != nil {
			t.Errorf("Path(%q) returned unexpected error: %v", path, err)
			continue
		}
		if wantPath := filepath.Join(snapshot.dir, filepath.FromSlash(path)); gotPath != wantPath {
			t.Errorf("Path(%q) = %q, want %q", path, gotPath, wantPath)
		}
	}
}

func TestOpenDirectoryRejectsUnsafeSnapshotLinks(t *testing.T) {
	tests := []struct {
		name      string
		mutate    func(*testing.T, *snapshotFixture)
		wantError error
	}{
		{
			name: "absolute target",
			mutate: func(t *testing.T, snapshot *snapshotFixture) {
				replaceLink(t, snapshot.path("model.onnx"), snapshot.blobs["model.onnx"])
			},
			wantError: ErrUnsupported,
		},
		{
			name: "directory target",
			mutate: func(t *testing.T, snapshot *snapshotFixture) {
				replaceLink(t, snapshot.path("model.onnx"), "../../blobs")
			},
			wantError: ErrUnsupported,
		},
		{
			name: "dangling target",
			mutate: func(t *testing.T, snapshot *snapshotFixture) {
				replaceLink(t, snapshot.path("model.onnx"), "../../blobs/"+strings.Repeat("0", 64))
			},
			wantError: ErrUnsupported,
		},
		{
			name: "snapshot cycle",
			mutate: func(t *testing.T, snapshot *snapshotFixture) {
				other := snapshot.path("cycle")
				replaceLink(t, snapshot.path("model.onnx"), "cycle")
				replaceLink(t, other, "model.onnx")
			},
			wantError: ErrUnsupported,
		},
		{
			name: "blob multi-hop",
			mutate: func(t *testing.T, snapshot *snapshotFixture) {
				modelBlob := snapshot.blobs["model.onnx"]
				otherBlob := snapshot.blobs["LICENSE.model"]
				removeFixtureFile(t, modelBlob)
				if err := os.Symlink(filepath.Base(otherBlob), modelBlob); err != nil {
					t.Fatalf("Symlink(blob): %v", err)
				}
			},
			wantError: ErrUnsupported,
		},
		{
			name: "outside blob root",
			mutate: func(t *testing.T, snapshot *snapshotFixture) {
				outside := filepath.Join(filepath.Dir(filepath.Dir(filepath.Dir(snapshot.dir))), strings.Repeat("a", 64))
				writeFixtureFile(t, outside, []byte("outside"))
				target, err := filepath.Rel(filepath.Dir(snapshot.path("model.onnx")), outside)
				if err != nil {
					t.Fatalf("Rel(outside): %v", err)
				}
				replaceLink(t, snapshot.path("model.onnx"), target)
			},
			wantError: ErrUnsupported,
		},
		{
			name: "unclean target",
			mutate: func(t *testing.T, snapshot *snapshotFixture) {
				blob := filepath.Base(snapshot.blobs["model.onnx"])
				replaceLink(t, snapshot.path("model.onnx"), "../../blobs/unused/../"+blob)
			},
			wantError: ErrUnsupported,
		},
		{
			name: "digest mismatch",
			mutate: func(t *testing.T, snapshot *snapshotFixture) {
				blob := snapshot.blobs["model.onnx"]
				data, err := os.ReadFile(blob)
				if err != nil {
					t.Fatalf("ReadFile(blob): %v", err)
				}
				data[0] ^= 0xff
				writeFixtureFile(t, blob, data)
			},
			wantError: bundle.ErrIntegrity,
		},
		{
			name: "size mismatch",
			mutate: func(t *testing.T, snapshot *snapshotFixture) {
				blob := snapshot.blobs["model.onnx"]
				data, err := os.ReadFile(blob)
				if err != nil {
					t.Fatalf("ReadFile(blob): %v", err)
				}
				writeFixtureFile(t, blob, append(data, 0))
			},
			wantError: bundle.ErrIntegrity,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			snapshot := makeSnapshot(t, copyFixture(t))
			test.mutate(t, snapshot)
			opened, err := OpenDirectory(t.Context(), snapshot.dir)
			if opened != nil {
				t.Fatal("OpenDirectory() source is non-nil on failure")
			}
			if !errors.Is(err, test.wantError) {
				t.Errorf("OpenDirectory() error = %v, want errors.Is(_, %v)", err, test.wantError)
			}
		})
	}
}

func TestOpenDirectoryRejectsMalformedSnapshotLayout(t *testing.T) {
	snapshot := makeSnapshot(t, copyFixture(t))
	badRoot := filepath.Join(t.TempDir(), "snapshot")
	if err := os.Rename(snapshot.dir, badRoot); err != nil {
		t.Fatalf("Rename(snapshot): %v", err)
	}
	if _, err := OpenDirectory(t.Context(), badRoot); !errors.Is(err, ErrUnsupported) {
		t.Errorf("OpenDirectory(malformed snapshot) error = %v, want ErrUnsupported", err)
	}
}

func TestSnapshotDetectsBlobMutationAfterClassification(t *testing.T) {
	snapshot := makeSnapshot(t, copyFixture(t))
	root, err := os.OpenRoot(snapshot.dir)
	if err != nil {
		t.Fatalf("OpenRoot(snapshot): %v", err)
	}
	links, err := findLinks(t.Context(), root.FS())
	if err != nil {
		t.Fatalf("findLinks() returned unexpected error: %v", err)
	}
	fileSystem, err := openSnapshot(t.Context(), snapshot.dir, root, links)
	if err != nil {
		_ = root.Close()
		t.Fatalf("openSnapshot() returned unexpected error: %v", err)
	}
	t.Cleanup(func() { _ = closeRoots([]directoryRoot{root, fileSystem.blobs}, nil) })

	blob := snapshot.blobs["model.onnx"]
	data, err := os.ReadFile(blob)
	if err != nil {
		t.Fatalf("ReadFile(blob): %v", err)
	}
	removeFixtureFile(t, blob)
	writeFixtureFile(t, blob, data)
	if err := fileSystem.verifyStable(t.Context()); !errors.Is(err, ErrUnsupported) {
		t.Errorf("verifyStable() error = %v, want ErrUnsupported", err)
	}
}

func TestSnapshotDetectsLinkMutationAfterClassification(t *testing.T) {
	snapshot := makeSnapshot(t, copyFixture(t))
	root, err := os.OpenRoot(snapshot.dir)
	if err != nil {
		t.Fatalf("OpenRoot(snapshot): %v", err)
	}
	links, err := findLinks(t.Context(), root.FS())
	if err != nil {
		t.Fatalf("findLinks() returned unexpected error: %v", err)
	}
	fileSystem, err := openSnapshot(t.Context(), snapshot.dir, root, links)
	if err != nil {
		_ = root.Close()
		t.Fatalf("openSnapshot() returned unexpected error: %v", err)
	}
	t.Cleanup(func() { _ = closeRoots([]directoryRoot{root, fileSystem.blobs}, nil) })

	modelLink := snapshot.path("model.onnx")
	otherBlob := snapshot.blobs["LICENSE.model"]
	target, err := filepath.Rel(filepath.Dir(modelLink), otherBlob)
	if err != nil {
		t.Fatalf("Rel(other blob): %v", err)
	}
	replaceLink(t, modelLink, target)
	if err := fileSystem.verifyStable(t.Context()); !errors.Is(err, ErrUnsupported) {
		t.Errorf("verifyStable() error = %v, want ErrUnsupported", err)
	}
}

func TestSnapshotErrorsDoNotDiscloseLinkTargets(t *testing.T) {
	snapshot := makeSnapshot(t, copyFixture(t))
	secret := "credential-target-do-not-disclose"
	replaceLink(t, snapshot.path("model.onnx"), "/tmp/"+secret)
	_, err := OpenDirectory(t.Context(), snapshot.dir)
	if err == nil {
		t.Fatal("OpenDirectory(absolute secret target) returned nil error")
	}
	if strings.Contains(err.Error(), secret) {
		t.Errorf("OpenDirectory() error disclosed link target: %v", err)
	}
}

func TestSnapshotPlatformPolicy(t *testing.T) {
	tests := []struct {
		goos   string
		goarch string
		want   bool
	}{
		{goos: "linux", goarch: "amd64", want: true},
		{goos: "linux", goarch: "arm64", want: false},
		{goos: "windows", goarch: "amd64", want: false},
		{goos: "darwin", goarch: "amd64", want: false},
	}
	for _, test := range tests {
		if got := snapshotPlatformSupported(test.goos, test.goarch); got != test.want {
			t.Errorf("snapshotPlatformSupported(%q, %q) = %t, want %t", test.goos, test.goarch, got, test.want)
		}
	}
}

type snapshotFixture struct {
	dir   string
	blobs map[string]string
}

func (s *snapshotFixture) path(logicalPath string) string {
	return filepath.Join(s.dir, filepath.FromSlash(logicalPath))
}

func makeSnapshot(t *testing.T, source string) *snapshotFixture {
	t.Helper()
	repository := filepath.Join(t.TempDir(), "models--example--laya")
	blobsDir := filepath.Join(repository, blobsDirectory)
	snapshotDir := filepath.Join(repository, snapshotsDirectory, testRevision)
	if err := os.MkdirAll(blobsDir, 0o700); err != nil {
		t.Fatalf("MkdirAll(blobs): %v", err)
	}
	if err := os.MkdirAll(snapshotDir, 0o700); err != nil {
		t.Fatalf("MkdirAll(snapshot): %v", err)
	}
	result := &snapshotFixture{dir: snapshotDir, blobs: make(map[string]string)}
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
		target := filepath.Join(snapshotDir, relative)
		if entry.IsDir() {
			return os.MkdirAll(target, 0o700)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		if filepath.ToSlash(relative) == "manifest.json" {
			return os.WriteFile(target, data, 0o600)
		}
		digest := sha256.Sum256(data)
		blob := filepath.Join(blobsDir, hex.EncodeToString(digest[:]))
		if err := os.WriteFile(blob, data, 0o600); err != nil {
			return err
		}
		linkTarget, err := filepath.Rel(filepath.Dir(target), blob)
		if err != nil {
			return err
		}
		if err := os.Symlink(linkTarget, target); err != nil {
			return err
		}
		result.blobs[filepath.ToSlash(relative)] = blob
		return nil
	})
	if err != nil {
		t.Fatalf("make snapshot: %v", err)
	}
	return result
}

func addExternalData(t *testing.T, dir string) {
	t.Helper()
	const logicalPath = "model.onnx.data"
	data := []byte("synthetic external tensor data")
	writeFixtureFile(t, filepath.Join(dir, logicalPath), data)
	manifestPath := filepath.Join(dir, "manifest.json")
	manifestBytes, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("ReadFile(manifest): %v", err)
	}
	var manifest bundle.Manifest
	if err := json.Unmarshal(manifestBytes, &manifest); err != nil {
		t.Fatalf("Unmarshal(manifest): %v", err)
	}
	digest := sha256.Sum256(data)
	manifest.Model.ExternalData = []string{logicalPath}
	manifest.Files = append(manifest.Files, bundle.File{
		Role:   "model-external-data",
		Path:   logicalPath,
		Size:   int64(len(data)),
		SHA256: hex.EncodeToString(digest[:]),
	})
	sort.Slice(manifest.Files, func(i, j int) bool {
		return manifest.Files[i].Path < manifest.Files[j].Path
	})
	manifestBytes, err = json.Marshal(manifest)
	if err != nil {
		t.Fatalf("Marshal(manifest): %v", err)
	}
	writeFixtureFile(t, manifestPath, manifestBytes)
}

func replaceLink(t *testing.T, path, target string) {
	t.Helper()
	if err := os.Remove(path); err != nil && !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("Remove(link): %v", err)
	}
	if err := os.Symlink(target, path); err != nil {
		t.Fatalf("Symlink(%q): %v", filepath.Base(path), err)
	}
}
