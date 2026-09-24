package source

import (
	"context"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/metalagman/layajev/internal/bundle"
)

const (
	huggingFaceModelPrefix = "models--"
	snapshotsDirectory     = "snapshots"
	blobsDirectory         = "blobs"
)

type snapshotFS struct {
	snapshot     directoryRoot
	blobs        directoryRoot
	snapshotPath string
	blobsPath    string
	links        map[string]snapshotLink
}

type snapshotLink struct {
	target   string
	blob     string
	blobInfo os.FileInfo
}

func openSnapshot(ctx context.Context, path string, snapshot directoryRoot, links []string) (*snapshotFS, error) {
	if !snapshotPlatformSupported(runtime.GOOS, runtime.GOARCH) {
		return nil, sourceError("classify snapshot", "", ErrUnsupported, nil)
	}
	revision := filepath.Base(path)
	snapshotsPath := filepath.Dir(path)
	repositoryPath := filepath.Dir(snapshotsPath)
	if filepath.Base(snapshotsPath) != snapshotsDirectory || !validRepositoryName(filepath.Base(repositoryPath)) ||
		!validHexName(revision) {
		return nil, sourceError("classify snapshot", "", ErrUnsupported, nil)
	}
	if err := requireDirectory(repositoryPath); err != nil {
		return nil, err
	}
	if err := requireDirectory(snapshotsPath); err != nil {
		return nil, err
	}
	blobsPath := filepath.Join(repositoryPath, blobsDirectory)
	blobs, err := openDirectoryRoot(blobsPath)
	if err != nil {
		return nil, err
	}
	result := &snapshotFS{
		snapshot:     snapshot,
		blobs:        blobs,
		snapshotPath: path,
		blobsPath:    blobsPath,
		links:        make(map[string]snapshotLink, len(links)),
	}
	for _, logicalPath := range links {
		if err := ctx.Err(); err != nil {
			cause := sourceError("classify snapshot", logicalPath, err, err)
			return nil, closeRoots([]directoryRoot{blobs}, cause)
		}
		link, err := result.classifyLink(logicalPath)
		if err != nil {
			return nil, closeRoots([]directoryRoot{blobs}, err)
		}
		result.links[logicalPath] = link
	}
	return result, nil
}

func snapshotPlatformSupported(goos, goarch string) bool {
	return goos == "linux" && goarch == "amd64"
}

func requireDirectory(path string) error {
	info, err := os.Lstat(path)
	if err != nil {
		return sourceError("classify snapshot", "", ErrUnsupported, err)
	}
	if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return sourceError("classify snapshot", "", ErrUnsupported, nil)
	}
	return nil
}

func openDirectoryRoot(path string) (directoryRoot, error) {
	info, err := os.Lstat(path)
	if err != nil {
		return nil, sourceError("open blob root", "", ErrUnsupported, err)
	}
	if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return nil, sourceError("open blob root", "", ErrUnsupported, nil)
	}
	root, err := os.OpenRoot(path)
	if err != nil {
		return nil, sourceError("open blob root", "", ErrUnsupported, err)
	}
	openedInfo, err := fs.Stat(root.FS(), ".")
	if err != nil {
		_ = root.Close()
		return nil, sourceError("inspect blob root", "", ErrUnsupported, err)
	}
	if !openedInfo.IsDir() || !os.SameFile(info, openedInfo) {
		_ = root.Close()
		return nil, sourceError("inspect blob root", "", ErrUnsupported, nil)
	}
	return root, nil
}

func validRepositoryName(name string) bool {
	if !strings.HasPrefix(name, huggingFaceModelPrefix) {
		return false
	}
	parts := strings.Split(strings.TrimPrefix(name, huggingFaceModelPrefix), "--")
	if len(parts) < 2 {
		return false
	}
	for _, part := range parts {
		if part == "" || part == "." || part == ".." {
			return false
		}
	}
	return true
}

func validHexName(name string) bool {
	if len(name) != 40 && len(name) != 64 {
		return false
	}
	return validLowerHex(name)
}

func validLowerHex(name string) bool {
	for _, character := range name {
		if character < '0' || character > '9' {
			if character < 'a' || character > 'f' {
				return false
			}
		}
	}
	return true
}

func (f *snapshotFS) classifyLink(logicalPath string) (snapshotLink, error) {
	if !validLogicalPath(logicalPath) {
		return snapshotLink{}, sourceError("classify link", logicalPath, ErrUnsupported, nil)
	}
	target, err := f.snapshot.Readlink(logicalPath)
	if err != nil {
		return snapshotLink{}, sourceError("read link", logicalPath, ErrUnsupported, err)
	}
	if filepath.IsAbs(target) || strings.Contains(target, `\`) || filepath.Clean(target) != target {
		return snapshotLink{}, sourceError("classify link", logicalPath, ErrUnsupported, nil)
	}
	linkDirectory := filepath.Join(f.snapshotPath, filepath.FromSlash(filepath.Dir(logicalPath)))
	resolved := filepath.Clean(filepath.Join(linkDirectory, target))
	if filepath.Dir(resolved) != f.blobsPath {
		return snapshotLink{}, sourceError("classify link", logicalPath, ErrUnsupported, nil)
	}
	blob := filepath.Base(resolved)
	if !validHexName(blob) {
		return snapshotLink{}, sourceError("classify link", logicalPath, ErrUnsupported, nil)
	}
	info, err := f.blobs.Lstat(blob)
	if err != nil {
		return snapshotLink{}, sourceError("inspect blob", logicalPath, ErrUnsupported, err)
	}
	if !info.Mode().IsRegular() {
		return snapshotLink{}, sourceError("inspect blob", logicalPath, ErrUnsupported, nil)
	}
	return snapshotLink{target: target, blob: blob, blobInfo: info}, nil
}

func (f *snapshotFS) Open(name string) (fs.File, error) {
	if !fs.ValidPath(name) || strings.Contains(name, `\`) {
		return nil, sourceError("open snapshot entry", name, bundle.ErrInvalid, fs.ErrInvalid)
	}
	info, err := f.snapshot.Lstat(name)
	if err != nil {
		return nil, sourceError("open snapshot entry", name, bundle.ErrInvalid, err)
	}
	if info.Mode()&fs.ModeSymlink == 0 {
		return f.snapshot.FS().Open(name)
	}
	link, err := f.currentLink(name)
	if err != nil {
		return nil, err
	}
	return f.blobs.FS().Open(link.blob)
}

func (f *snapshotFS) ReadDir(name string) ([]fs.DirEntry, error) {
	entries, err := fs.ReadDir(f.snapshot.FS(), name)
	if err != nil {
		return nil, sourceError("read snapshot directory", name, bundle.ErrInvalid, err)
	}
	result := make([]fs.DirEntry, 0, len(entries))
	for _, entry := range entries {
		logicalPath := entry.Name()
		if name != "." {
			logicalPath = name + "/" + entry.Name()
		}
		if entry.Type()&fs.ModeSymlink == 0 {
			result = append(result, entry)
			continue
		}
		link, err := f.currentLink(logicalPath)
		if err != nil {
			return nil, err
		}
		result = append(result, snapshotDirEntry{name: entry.Name(), info: link.blobInfo})
	}
	return result, nil
}

func (f *snapshotFS) currentLink(logicalPath string) (snapshotLink, error) {
	want, ok := f.links[logicalPath]
	if !ok {
		return snapshotLink{}, sourceError("classify link", logicalPath, ErrUnsupported, nil)
	}
	target, err := f.snapshot.Readlink(logicalPath)
	if err != nil {
		return snapshotLink{}, sourceError("read link", logicalPath, ErrUnsupported, err)
	}
	if target != want.target {
		return snapshotLink{}, sourceError("verify stable link", logicalPath, ErrUnsupported, nil)
	}
	info, err := f.blobs.Lstat(want.blob)
	if err != nil {
		return snapshotLink{}, sourceError("inspect blob", logicalPath, ErrUnsupported, err)
	}
	if !sameFileIdentity(want.blobInfo, info) {
		return snapshotLink{}, sourceError("verify stable blob", logicalPath, ErrUnsupported, nil)
	}
	return want, nil
}

func (f *snapshotFS) verifyStable(ctx context.Context) error {
	for path := range f.links {
		if err := ctx.Err(); err != nil {
			return sourceError("verify snapshot", path, err, err)
		}
		if _, err := f.currentLink(path); err != nil {
			return err
		}
	}
	return nil
}

func sameFileIdentity(first, second os.FileInfo) bool {
	return first.Mode() == second.Mode() && first.Size() == second.Size() &&
		first.ModTime().Equal(second.ModTime()) && os.SameFile(first, second)
}

type snapshotDirEntry struct {
	name string
	info os.FileInfo
}

func (e snapshotDirEntry) Name() string               { return e.name }
func (e snapshotDirEntry) IsDir() bool                { return false }
func (e snapshotDirEntry) Type() fs.FileMode          { return e.info.Mode().Type() }
func (e snapshotDirEntry) Info() (fs.FileInfo, error) { return e.info, nil }
