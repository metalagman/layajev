// Package source opens verified local bundle sources without acquiring models
// or loading native inference resources.
package source

import (
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/metalagman/layajev/internal/bundle"
)

var (
	// ErrUnsupported indicates that a source uses filesystem behavior outside
	// the deliberately narrow source contract.
	ErrUnsupported = errors.New("laya: unsupported model source")
	// ErrClosed indicates that an operation requires an open source handle.
	ErrClosed = errors.New("laya: model source is closed")
	// ErrMaterialization indicates an I/O failure while constructing a local
	// native-compatible copy of an already-present bundle.
	ErrMaterialization = errors.New("laya: model source materialization failure")
)

// Error describes a source operation failure without including an external
// filesystem path or file content in its message. Path, when present, is a
// slash-relative logical bundle path.
type Error struct {
	Operation string
	Path      string
	Kind      error
	Cause     error
}

// Error returns a bounded description of the source failure.
func (e *Error) Error() string {
	if e == nil {
		return ErrUnsupported.Error()
	}
	kind := e.Kind
	if kind == nil {
		kind = ErrUnsupported
	}
	message := kind.Error()
	if e.Operation != "" {
		message += ": operation " + quote(e.Operation)
	}
	if e.Path != "" {
		message += ": path " + quote(e.Path)
	}
	if e.Cause != nil {
		message += fmt.Sprintf(": cause %T", e.Cause)
	}
	return message
}

// Unwrap exposes both the caller-actionable category and underlying cause.
func (e *Error) Unwrap() []error {
	if e == nil {
		return nil
	}
	var errs []error
	if e.Kind != nil {
		errs = append(errs, e.Kind)
	}
	if e.Cause != nil {
		errs = append(errs, e.Cause)
	}
	return errs
}

type directoryRoot interface {
	FS() fs.FS
	Lstat(string) (os.FileInfo, error)
	Readlink(string) (string, error)
	Close() error
}

type rootOpener func(string) (directoryRoot, error)

// Bundle is a verified concrete bundle source. A Bundle owns its root handle
// and is safe for concurrent use. Callers must keep the underlying directory
// immutable until Close returns.
type Bundle struct {
	mu sync.RWMutex

	roots    []directoryRoot
	sourceFS fs.FS
	rootPath string
	manifest bundle.Manifest
	allowed  map[string]struct{}
	closed   bool
}

// OpenDirectory opens and verifies a complete Bundle v1 in dir. It never
// writes to dir. Symlinks are rejected; the supported Hugging Face snapshot
// link policy is applied by a separate source mode.
func OpenDirectory(ctx context.Context, dir string) (*Bundle, error) {
	return openDirectory(ctx, dir, func(path string) (directoryRoot, error) {
		return os.OpenRoot(path)
	})
}

func openDirectory(ctx context.Context, dir string, open rootOpener) (*Bundle, error) {
	if ctx == nil {
		return nil, sourceError("validate context", "", bundle.ErrInvalid, nil)
	}
	if err := ctx.Err(); err != nil {
		return nil, sourceError("open directory", "", err, err)
	}
	if dir == "" {
		return nil, sourceError("validate directory", "", bundle.ErrInvalid, nil)
	}
	abs, err := filepath.Abs(dir)
	if err != nil {
		return nil, sourceError("resolve directory", "", bundle.ErrInvalid, err)
	}
	abs = filepath.Clean(abs)
	info, err := os.Lstat(abs)
	if err != nil {
		return nil, sourceError("inspect directory", "", bundle.ErrInvalid, err)
	}
	if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return nil, sourceError("inspect directory", "", bundle.ErrInvalid, nil)
	}
	root, err := open(abs)
	if err != nil {
		return nil, sourceError("open directory", "", bundle.ErrInvalid, err)
	}
	roots := []directoryRoot{root}
	fail := func(cause error) (*Bundle, error) { return nil, closeRoots(roots, cause) }
	if err := ctx.Err(); err != nil {
		return fail(sourceError("open directory", "", err, err))
	}
	openedInfo, err := fs.Stat(root.FS(), ".")
	if err != nil {
		return fail(sourceError("inspect opened directory", "", bundle.ErrInvalid, err))
	}
	if !openedInfo.IsDir() || !os.SameFile(info, openedInfo) {
		return fail(sourceError("inspect opened directory", "", bundle.ErrInvalid, nil))
	}
	before, err := snapshotEntries(ctx, root.FS())
	if err != nil {
		return fail(err)
	}
	links, err := findLinks(ctx, root.FS())
	if err != nil {
		return fail(err)
	}
	fileSystem := root.FS()
	var snapshot *snapshotFS
	if len(links) > 0 {
		snapshot, err = openSnapshot(ctx, abs, root, links)
		if err != nil {
			return fail(err)
		}
		roots = append(roots, snapshot.blobs)
		fileSystem = snapshot
	}
	manifest, err := bundle.Verify(ctx, fileSystem, ".")
	if err != nil {
		return fail(err)
	}
	if snapshot != nil {
		if err := snapshot.verifyStable(ctx); err != nil {
			return fail(err)
		}
	}
	after, err := snapshotEntries(ctx, root.FS())
	if err != nil {
		return fail(err)
	}
	if err := compareEntries(before, after); err != nil {
		return fail(err)
	}
	allowed := make(map[string]struct{}, len(manifest.Files)+1)
	allowed["manifest.json"] = struct{}{}
	for _, file := range manifest.Files {
		allowed[file.Path] = struct{}{}
	}
	return &Bundle{
		roots:    roots,
		sourceFS: fileSystem,
		rootPath: abs,
		manifest: manifest,
		allowed:  allowed,
	}, nil
}

func snapshotEntries(ctx context.Context, fileSystem fs.FS) (map[string]os.FileInfo, error) {
	entries := make(map[string]os.FileInfo)
	err := fs.WalkDir(fileSystem, ".", func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return sourceError("inspect entry", path, bundle.ErrInvalid, walkErr)
		}
		if err := ctx.Err(); err != nil {
			return sourceError("inspect entry", path, err, err)
		}
		info, err := entry.Info()
		if err != nil {
			return sourceError("inspect entry", path, bundle.ErrInvalid, err)
		}
		entries[path] = info
		return nil
	})
	if err != nil {
		return nil, err
	}
	return entries, nil
}

func compareEntries(before, after map[string]os.FileInfo) error {
	if len(before) != len(after) {
		return sourceError("verify stable directory", "", bundle.ErrInvalid, nil)
	}
	for path, first := range before {
		second, ok := after[path]
		if !ok || !sameFileIdentity(first, second) {
			return sourceError("verify stable entry", path, bundle.ErrInvalid, nil)
		}
	}
	return nil
}

func findLinks(ctx context.Context, fileSystem fs.FS) ([]string, error) {
	var links []string
	err := fs.WalkDir(fileSystem, ".", func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return sourceError("inspect entry", path, bundle.ErrInvalid, walkErr)
		}
		if err := ctx.Err(); err != nil {
			return sourceError("inspect entry", path, err, err)
		}
		if entry.Type()&fs.ModeSymlink != 0 {
			links = append(links, path)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return links, nil
}

// Manifest returns the verified bundle manifest. Callers must treat the
// returned value and its slice fields as immutable.
func (b *Bundle) Manifest() bundle.Manifest {
	if b == nil {
		return bundle.Manifest{}
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.manifest
}

// Path resolves a declared logical bundle path to a concrete native path. It
// fails after Close and never resolves an undeclared entry.
func (b *Bundle) Path(logicalPath string) (string, error) {
	if b == nil {
		return "", sourceError("resolve path", logicalPath, ErrClosed, nil)
	}
	if !validLogicalPath(logicalPath) {
		return "", sourceError("resolve path", logicalPath, bundle.ErrInvalid, nil)
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	if b.closed || len(b.roots) == 0 || b.sourceFS == nil {
		return "", sourceError("resolve path", logicalPath, ErrClosed, nil)
	}
	if _, ok := b.allowed[logicalPath]; !ok {
		return "", sourceError("resolve path", logicalPath, bundle.ErrInvalid, fs.ErrNotExist)
	}
	info, err := fs.Stat(b.sourceFS, logicalPath)
	if err != nil {
		return "", sourceError("inspect path", logicalPath, bundle.ErrInvalid, err)
	}
	if !info.Mode().IsRegular() {
		return "", sourceError("inspect path", logicalPath, ErrUnsupported, nil)
	}
	return filepath.Join(b.rootPath, filepath.FromSlash(logicalPath)), nil
}

// ReadFile reads one declared regular bundle file up to limit bytes. It checks
// cancellation between reads and fails if the file exceeds the supplied bound.
func (b *Bundle) ReadFile(ctx context.Context, logicalPath string, limit int64) ([]byte, error) {
	if ctx == nil {
		return nil, sourceError("validate context", logicalPath, bundle.ErrInvalid, nil)
	}
	if err := ctx.Err(); err != nil {
		return nil, sourceError("read bundle file", logicalPath, err, err)
	}
	if b == nil {
		return nil, sourceError("read bundle file", logicalPath, ErrClosed, nil)
	}
	if !validLogicalPath(logicalPath) || limit < 0 {
		return nil, sourceError("read bundle file", logicalPath, bundle.ErrInvalid, nil)
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	if b.closed || len(b.roots) == 0 || b.sourceFS == nil {
		return nil, sourceError("read bundle file", logicalPath, ErrClosed, nil)
	}
	if _, ok := b.allowed[logicalPath]; !ok {
		return nil, sourceError("read bundle file", logicalPath, bundle.ErrInvalid, fs.ErrNotExist)
	}
	file, err := b.sourceFS.Open(logicalPath)
	if err != nil {
		return nil, sourceError("open bundle file", logicalPath, bundle.ErrInvalid, err)
	}
	defer file.Close() //nolint:errcheck // The bounded read reports source failures.
	info, err := file.Stat()
	if err != nil {
		return nil, sourceError("inspect bundle file", logicalPath, bundle.ErrInvalid, err)
	}
	if !info.Mode().IsRegular() || info.Size() < 0 || info.Size() > limit {
		return nil, sourceError("inspect bundle file", logicalPath, bundle.ErrInvalid, nil)
	}
	data := make([]byte, 0, info.Size())
	buffer := make([]byte, 32<<10)
	for {
		if err := ctx.Err(); err != nil {
			return nil, sourceError("read bundle file", logicalPath, err, err)
		}
		count, readErr := file.Read(buffer)
		if count < 0 || count > len(buffer) {
			return nil, sourceError("read bundle file", logicalPath, bundle.ErrIntegrity, errors.New("invalid reader count"))
		}
		if contextErr := ctx.Err(); contextErr != nil {
			return nil, sourceError("read bundle file", logicalPath, contextErr, contextErr)
		}
		if count > 0 {
			if int64(len(data)+count) > limit {
				return nil, sourceError("read bundle file", logicalPath, bundle.ErrIntegrity, nil)
			}
			data = append(data, buffer[:count]...)
		}
		if errors.Is(readErr, io.EOF) {
			break
		}
		if readErr != nil {
			return nil, sourceError("read bundle file", logicalPath, bundle.ErrIntegrity, readErr)
		}
		if count == 0 {
			return nil, sourceError("read bundle file", logicalPath, bundle.ErrIntegrity, io.ErrNoProgress)
		}
	}
	return data, nil
}

// Close releases the owned directory root. Close is idempotent. It does not
// modify or remove any caller-owned file.
func (b *Bundle) Close() error {
	if b == nil {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return nil
	}
	if len(b.roots) == 0 {
		b.closed = true
		return nil
	}
	err := closeRoots(b.roots, nil)
	b.roots = nil
	b.sourceFS = nil
	b.closed = true
	return err
}

func closeRoots(roots []directoryRoot, cause error) error {
	errs := []error{cause}
	for index := len(roots) - 1; index >= 0; index-- {
		if roots[index] == nil {
			continue
		}
		if err := roots[index].Close(); err != nil {
			errs = append(errs, sourceError("close directory", "", ErrMaterialization, err))
		}
	}
	return errors.Join(errs...)
}

func validLogicalPath(path string) bool {
	return path != "." && fs.ValidPath(path) && !strings.Contains(path, `\`)
}

func sourceError(operation, path string, kind, cause error) error {
	return &Error{
		Operation: operation,
		Path:      path,
		Kind:      kind,
		Cause:     cause,
	}
}

func quote(value string) string {
	const maxLength = 128
	if len(value) > maxLength {
		value = value[:maxLength] + "…"
	}
	return fmt.Sprintf("%q", value)
}
