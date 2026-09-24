//go:build linux

package source

import (
	"bytes"
	"context"
	"errors"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/metalagman/layajev/internal/bundle"
)

const (
	materializeHelperEnvironment = "LAYA_TEST_MATERIALIZE_HELPER"
	materializeWorkEnvironment   = "LAYA_TEST_MATERIALIZE_WORKDIR"
)

func TestMaterializePublishesReusesAndRetainsBundles(t *testing.T) {
	workDir := t.TempDir()
	sentinel := filepath.Join(workDir, "caller-owned.txt")
	writeFixtureFile(t, sentinel, []byte("keep"))
	source := fixtureMapFS(t)

	first, err := Materialize(t.Context(), source, ".", workDir)
	if err != nil {
		t.Fatalf("Materialize(first) returned unexpected error: %v", err)
	}
	manifest := first.Manifest()
	firstPath, err := first.Path(manifest.Model.Path)
	if err != nil {
		t.Fatalf("first.Path(model) returned unexpected error: %v", err)
	}
	finalPath := publishedPath(workDir, manifest.ID())
	if want := filepath.Join(finalPath, manifest.Model.Path); firstPath != want {
		t.Errorf("first.Path(model) = %q, want %q", firstPath, want)
	}

	second, err := Materialize(t.Context(), source, ".", workDir)
	if err != nil {
		_ = first.Close()
		t.Fatalf("Materialize(second) returned unexpected error: %v", err)
	}
	if err := first.Close(); err != nil {
		t.Fatalf("first.Close() returned unexpected error: %v", err)
	}
	if _, err := second.Path(manifest.Model.Path); err != nil {
		t.Errorf("second.Path(model) after first close returned unexpected error: %v", err)
	}
	if err := second.Close(); err != nil {
		t.Fatalf("second.Close() returned unexpected error: %v", err)
	}
	if _, err := bundle.Verify(t.Context(), os.DirFS(finalPath), "."); err != nil {
		t.Errorf("retained publication failed verification after all leases closed: %v", err)
	}
	if data, err := os.ReadFile(sentinel); err != nil || string(data) != "keep" {
		t.Errorf("caller-owned WorkDir entry changed: data=%q err=%v", data, err)
	}
	assertDirectoryEmpty(t, filepath.Join(workDir, namespaceName, stagingDirectory))
	entries, err := os.ReadDir(filepath.Join(workDir, namespaceName, bundlesDirectory))
	if err != nil {
		t.Fatalf("ReadDir(bundles): %v", err)
	}
	if len(entries) != 1 {
		t.Errorf("published bundle count = %d, want 1", len(entries))
	}
}

func TestMaterializeConcurrentGoroutinesConverge(t *testing.T) {
	const workers = 12
	workDir := t.TempDir()
	source := fixtureMapFS(t)
	start := make(chan struct{})
	type result struct {
		bundle *Bundle
		err    error
	}
	results := make(chan result, workers)
	var group sync.WaitGroup
	for range workers {
		group.Add(1)
		go func() {
			defer group.Done()
			<-start
			opened, err := Materialize(context.Background(), source, ".", workDir)
			results <- result{bundle: opened, err: err}
		}()
	}
	close(start)
	group.Wait()
	close(results)

	var wantID string
	for result := range results {
		if result.err != nil {
			t.Errorf("Materialize() returned unexpected error: %v", result.err)
			continue
		}
		if result.bundle == nil {
			t.Error("Materialize() returned a nil bundle")
			continue
		}
		if wantID == "" {
			wantID = result.bundle.Manifest().ID()
		} else if got := result.bundle.Manifest().ID(); got != wantID {
			t.Errorf("concurrent bundle ID = %q, want %q", got, wantID)
		}
		if err := result.bundle.Close(); err != nil {
			t.Errorf("Bundle.Close() returned unexpected error: %v", err)
		}
	}
	entries, err := os.ReadDir(filepath.Join(workDir, namespaceName, bundlesDirectory))
	if err != nil {
		t.Fatalf("ReadDir(bundles): %v", err)
	}
	if len(entries) != 1 {
		t.Errorf("published bundle count = %d, want 1", len(entries))
	}
	assertDirectoryEmpty(t, filepath.Join(workDir, namespaceName, stagingDirectory))
}

func TestMaterializeProcessHelper(t *testing.T) {
	if os.Getenv(materializeHelperEnvironment) != "1" {
		return
	}
	workDir := os.Getenv(materializeWorkEnvironment)
	opened, err := Materialize(t.Context(), fixtureMapFS(t), ".", workDir)
	if err != nil {
		t.Fatalf("Materialize(process helper) returned unexpected error: %v", err)
	}
	if err := opened.Close(); err != nil {
		t.Fatalf("Bundle.Close(process helper) returned unexpected error: %v", err)
	}
}

func TestMaterializeConcurrentProcessesConverge(t *testing.T) {
	const workers = 6
	executable, err := os.Executable()
	if err != nil {
		t.Fatalf("Executable(): %v", err)
	}
	workingDirectory, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd(): %v", err)
	}
	workDir := t.TempDir()
	type process struct {
		command *exec.Cmd
		output  strings.Builder
	}
	processes := make([]process, workers)
	for index := range processes {
		command := exec.Command(executable, "-test.run=^TestMaterializeProcessHelper$")
		command.Dir = workingDirectory
		command.Env = append(os.Environ(), materializeHelperEnvironment+"=1", materializeWorkEnvironment+"="+workDir)
		command.Stdout = &processes[index].output
		command.Stderr = &processes[index].output
		processes[index].command = command
		if err := command.Start(); err != nil {
			t.Fatalf("process %d Start(): %v", index, err)
		}
	}
	for index := range processes {
		if err := processes[index].command.Wait(); err != nil {
			t.Errorf("process %d failed: %v\n%s", index, err, processes[index].output.String())
		}
	}
	entries, err := os.ReadDir(filepath.Join(workDir, namespaceName, bundlesDirectory))
	if err != nil {
		t.Fatalf("ReadDir(bundles): %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("published bundle count = %d, want 1", len(entries))
	}
	publication := filepath.Join(workDir, namespaceName, bundlesDirectory, entries[0].Name())
	if _, err := bundle.Verify(t.Context(), os.DirFS(publication), "."); err != nil {
		t.Errorf("process publication verification failed: %v", err)
	}
	assertDirectoryEmpty(t, filepath.Join(workDir, namespaceName, stagingDirectory))
}

func TestMaterializeLockWaitHonorsCancellation(t *testing.T) {
	workDir := t.TempDir()
	opened, err := Materialize(t.Context(), fixtureMapFS(t), ".", workDir)
	if err != nil {
		t.Fatalf("Materialize(initialize) returned unexpected error: %v", err)
	}
	if err := opened.Close(); err != nil {
		t.Fatalf("Bundle.Close(initialize) returned unexpected error: %v", err)
	}
	namespace, err := os.OpenRoot(filepath.Join(workDir, namespaceName))
	if err != nil {
		t.Fatalf("OpenRoot(namespace): %v", err)
	}
	t.Cleanup(func() { _ = namespace.Close() })
	file, err := openLockFile(namespace)
	if err != nil {
		t.Fatalf("openLockFile() returned unexpected error: %v", err)
	}
	held, err := acquireAdvisoryLock(t.Context(), file)
	if err != nil {
		_ = file.Close()
		t.Fatalf("acquireAdvisoryLock() returned unexpected error: %v", err)
	}
	t.Cleanup(func() { _ = held.Close() })

	ctx, cancel := context.WithCancel(context.Background())
	reached := make(chan struct{})
	var once sync.Once
	result := make(chan error, 1)
	source := fixtureMapFS(t)
	go func() {
		_, err := materialize(ctx, source, ".", workDir, materializeHooks{
			checkpoint: func(operation string) error {
				if operation == "wait lock" {
					once.Do(func() { close(reached) })
				}
				return nil
			},
		})
		result <- err
	}()
	<-reached
	cancel()
	if err := <-result; !errors.Is(err, context.Canceled) {
		t.Errorf("materialize(lock canceled) error = %v, want context.Canceled", err)
	}
}

func TestMaterializeRejectsAndPreservesAlteredNamespace(t *testing.T) {
	workDir := t.TempDir()
	opened, err := Materialize(t.Context(), fixtureMapFS(t), ".", workDir)
	if err != nil {
		t.Fatalf("Materialize(initialize) returned unexpected error: %v", err)
	}
	if err := opened.Close(); err != nil {
		t.Fatalf("Bundle.Close(initialize) returned unexpected error: %v", err)
	}
	unrelated := filepath.Join(workDir, namespaceName, "unrelated")
	writeFixtureFile(t, unrelated, []byte("keep"))

	result, err := Materialize(t.Context(), fixtureMapFS(t), ".", workDir)
	if result != nil {
		_ = result.Close()
		t.Fatal("Materialize(altered namespace) returned a non-nil bundle")
	}
	if !errors.Is(err, ErrMaterialization) {
		t.Errorf("Materialize(altered namespace) error = %v, want ErrMaterialization", err)
	}
	if data, readErr := os.ReadFile(unrelated); readErr != nil || string(data) != "keep" {
		t.Errorf("unrelated namespace entry changed: data=%q err=%v", data, readErr)
	}
}

func TestMaterializeRecoversOnlyMarkedStages(t *testing.T) {
	workDir := t.TempDir()
	opened, err := Materialize(t.Context(), fixtureMapFS(t), ".", workDir)
	if err != nil {
		t.Fatalf("Materialize(initialize) returned unexpected error: %v", err)
	}
	if err := opened.Close(); err != nil {
		t.Fatalf("Bundle.Close(initialize) returned unexpected error: %v", err)
	}
	staging := filepath.Join(workDir, namespaceName, stagingDirectory)
	ownedName := strings.Repeat("a", 64) + "-" + strings.Repeat("b", 32)
	unmarkedName := strings.Repeat("c", 64) + "-" + strings.Repeat("d", 32)
	owned := filepath.Join(staging, ownedName)
	unmarked := filepath.Join(staging, unmarkedName)
	malformed := filepath.Join(staging, "caller-owned")
	for _, path := range []string{owned, unmarked, malformed} {
		if err := os.Mkdir(path, 0o700); err != nil {
			t.Fatalf("Mkdir(%q): %v", filepath.Base(path), err)
		}
	}
	writeFixtureFile(t, filepath.Join(owned, ownerMarkerName), []byte(stageMarker))
	writeFixtureFile(t, filepath.Join(owned, "partial"), []byte("owned partial"))
	writeFixtureFile(t, filepath.Join(unmarked, "partial"), []byte("unmarked partial"))
	outside := t.TempDir()
	linked := filepath.Join(staging, strings.Repeat("e", 64)+"-"+strings.Repeat("f", 32))
	if err := os.Symlink(outside, linked); err != nil {
		t.Skipf("Symlink(stage) is unavailable: %v", err)
	}

	reused, err := Materialize(t.Context(), fixtureMapFS(t), ".", workDir)
	if err != nil {
		t.Fatalf("Materialize(recovery) returned unexpected error: %v", err)
	}
	if err := reused.Close(); err != nil {
		t.Fatalf("Bundle.Close(recovery) returned unexpected error: %v", err)
	}
	assertNotExist(t, owned)
	for _, path := range []string{unmarked, malformed, linked, outside} {
		if _, err := os.Lstat(path); err != nil {
			t.Errorf("unowned path %q was removed: %v", filepath.Base(path), err)
		}
	}
}

func TestMaterializeRejectsAndPreservesTamperedPublication(t *testing.T) {
	workDir := t.TempDir()
	opened, err := Materialize(t.Context(), fixtureMapFS(t), ".", workDir)
	if err != nil {
		t.Fatalf("Materialize(first) returned unexpected error: %v", err)
	}
	manifest := opened.Manifest()
	if err := opened.Close(); err != nil {
		t.Fatalf("Bundle.Close(first) returned unexpected error: %v", err)
	}
	modelPath := filepath.Join(publishedPath(workDir, manifest.ID()), filepath.FromSlash(manifest.Model.Path))
	data, err := os.ReadFile(modelPath)
	if err != nil {
		t.Fatalf("ReadFile(model): %v", err)
	}
	data[0] ^= 0xff
	writeFixtureFile(t, modelPath, data)

	result, err := Materialize(t.Context(), fixtureMapFS(t), ".", workDir)
	if result != nil {
		t.Fatal("Materialize(tampered ready) returned a non-nil bundle")
	}
	if !errors.Is(err, bundle.ErrIntegrity) {
		t.Errorf("Materialize(tampered ready) error = %v, want bundle.ErrIntegrity", err)
	}
	got, readErr := os.ReadFile(modelPath)
	if readErr != nil || !bytes.Equal(got, data) {
		t.Errorf("tampered publication was changed: err=%v", readErr)
	}
}

func TestMaterializeFaultBoundariesNeverExposePartialPublication(t *testing.T) {
	operations := []string{
		"verify source",
		"ensure namespace",
		"create namespace",
		"write namespace marker",
		"open lock",
		"wait lock",
		"recover stages",
		"create stage",
		"write stage marker",
		"copy bundle",
		"sync stage",
		"verify stage",
		"publish bundle",
		"sync publication",
		"remove published stage",
		"open publication",
	}
	for _, operation := range operations {
		t.Run(operation, func(t *testing.T) {
			workDir := t.TempDir()
			sentinel := filepath.Join(workDir, "unrelated")
			writeFixtureFile(t, sentinel, []byte("keep"))
			fault := errors.New("injected operation failure")
			opened, err := materialize(t.Context(), fixtureMapFS(t), ".", workDir, materializeHooks{
				checkpoint: func(got string) error {
					if got == operation {
						return fault
					}
					return nil
				},
			})
			if opened != nil {
				_ = opened.Close()
				t.Fatal("materialize(fault) returned a non-nil bundle")
			}
			if !errors.Is(err, fault) {
				t.Errorf("materialize(fault) error = %v, want injected cause", err)
			}
			if data, readErr := os.ReadFile(sentinel); readErr != nil || string(data) != "keep" {
				t.Errorf("unrelated WorkDir entry changed: data=%q err=%v", data, readErr)
			}
			assertPublicationAbsentOrValid(t, workDir)

			retried, retryErr := Materialize(t.Context(), fixtureMapFS(t), ".", workDir)
			if retryErr != nil {
				t.Fatalf("Materialize(retry after %q) returned unexpected error: %v", operation, retryErr)
			}
			if closeErr := retried.Close(); closeErr != nil {
				t.Errorf("retry Bundle.Close() returned unexpected error: %v", closeErr)
			}
			assertPublicationAbsentOrValid(t, workDir)
		})
	}
}

func TestMaterializeCleanupFailureLeavesRecoverableMarkedStage(t *testing.T) {
	workDir := t.TempDir()
	primary := errors.New("copy checkpoint failure")
	cleanup := errors.New("cleanup checkpoint failure")
	_, err := materialize(t.Context(), fixtureMapFS(t), ".", workDir, materializeHooks{
		checkpoint: func(operation string) error {
			switch operation {
			case "copy bundle":
				return primary
			case "cleanup stage":
				return cleanup
			default:
				return nil
			}
		},
	})
	if !errors.Is(err, primary) || !errors.Is(err, cleanup) {
		t.Fatalf("materialize(cleanup failure) error = %v, want both injected causes", err)
	}
	entries, err := os.ReadDir(filepath.Join(workDir, namespaceName, stagingDirectory))
	if err != nil {
		t.Fatalf("ReadDir(staging): %v", err)
	}
	if len(entries) != 1 || !validStageName(entries[0].Name()) {
		t.Fatalf("recoverable stage entries = %v, want one valid owned stage", entryNames(entries))
	}
	recovered, err := Materialize(t.Context(), fixtureMapFS(t), ".", workDir)
	if err != nil {
		t.Fatalf("Materialize(recover) returned unexpected error: %v", err)
	}
	if err := recovered.Close(); err != nil {
		t.Errorf("recovered Bundle.Close() returned unexpected error: %v", err)
	}
	assertDirectoryEmpty(t, filepath.Join(workDir, namespaceName, stagingDirectory))
}

func publishedPath(workDir, bundleID string) string {
	return filepath.Join(workDir, namespaceName, bundlesDirectory, strings.Replace(bundleID, "sha256:", "sha256-", 1))
}

func assertDirectoryEmpty(t *testing.T, path string) {
	t.Helper()
	entries, err := os.ReadDir(path)
	if err != nil {
		t.Fatalf("ReadDir(%q): %v", filepath.Base(path), err)
	}
	if len(entries) != 0 {
		t.Errorf("directory %q entries = %v, want empty", filepath.Base(path), entryNames(entries))
	}
}

func assertPublicationAbsentOrValid(t *testing.T, workDir string) {
	t.Helper()
	bundles := filepath.Join(workDir, namespaceName, bundlesDirectory)
	entries, err := os.ReadDir(bundles)
	if errors.Is(err, fs.ErrNotExist) {
		return
	}
	if err != nil {
		t.Fatalf("ReadDir(bundles): %v", err)
	}
	if len(entries) > 1 {
		t.Fatalf("published entries = %v, want at most one", entryNames(entries))
	}
	if len(entries) == 1 {
		if _, err := bundle.Verify(t.Context(), os.DirFS(filepath.Join(bundles, entries[0].Name())), "."); err != nil {
			t.Errorf("visible publication is partial: %v", err)
		}
	}
}

func entryNames(entries []os.DirEntry) []string {
	names := make([]string, len(entries))
	for index, entry := range entries {
		names[index] = entry.Name()
	}
	return names
}
