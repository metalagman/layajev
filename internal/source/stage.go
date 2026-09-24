package source

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/metalagman/layajev/internal/bundle"
)

const (
	stageBufferBytes   = 64 << 10
	stageManifestBytes = 1 << 20
)

// Stage verifies a complete bundle below root, copies it with bounded memory
// into a newly created destination directory, and verifies the copy again.
// Stage never publishes or reuses destination; callers provide a unique path
// whose parent already exists.
func Stage(ctx context.Context, fileSystem fs.FS, root, destination string) (manifest bundle.Manifest, err error) {
	if ctx == nil {
		return bundle.Manifest{}, sourceError("validate context", "", bundle.ErrInvalid, nil)
	}
	if fileSystem == nil {
		return bundle.Manifest{}, sourceError("validate filesystem", "", bundle.ErrInvalid, nil)
	}
	if root == "" {
		root = "."
	}
	if root != "." && (!fs.ValidPath(root) || strings.Contains(root, `\`)) {
		return bundle.Manifest{}, sourceError("validate filesystem root", root, bundle.ErrInvalid, nil)
	}
	if destination == "" {
		return bundle.Manifest{}, sourceError("validate staging destination", "", ErrMaterialization, nil)
	}
	if err := ctx.Err(); err != nil {
		return bundle.Manifest{}, sourceError("stage bundle", "", err, err)
	}

	manifest, err = bundle.Verify(ctx, fileSystem, root)
	if err != nil {
		return bundle.Manifest{}, err
	}
	sourceFS, err := fs.Sub(fileSystem, root)
	if err != nil {
		return bundle.Manifest{}, sourceError("open filesystem root", root, bundle.ErrInvalid, err)
	}

	abs, err := filepath.Abs(destination)
	if err != nil {
		return bundle.Manifest{}, sourceError("resolve staging destination", "", ErrMaterialization, err)
	}
	abs = filepath.Clean(abs)
	name := filepath.Base(abs)
	if name == "." || name == string(filepath.Separator) || !fs.ValidPath(name) {
		return bundle.Manifest{}, sourceError("validate staging destination", "", ErrMaterialization, nil)
	}
	parentPath := filepath.Dir(abs)
	parentInfo, err := os.Lstat(parentPath)
	if err != nil {
		return bundle.Manifest{}, sourceError("inspect staging parent", "", ErrMaterialization, err)
	}
	if !parentInfo.IsDir() || parentInfo.Mode()&os.ModeSymlink != 0 {
		return bundle.Manifest{}, sourceError("inspect staging parent", "", ErrMaterialization, nil)
	}
	parent, err := os.OpenRoot(parentPath)
	if err != nil {
		return bundle.Manifest{}, sourceError("open staging parent", "", ErrMaterialization, err)
	}
	defer func() {
		if closeErr := parent.Close(); closeErr != nil {
			err = errors.Join(err, sourceError("close staging parent", "", ErrMaterialization, closeErr))
		}
	}()
	openedInfo, err := fs.Stat(parent.FS(), ".")
	if err != nil || !openedInfo.IsDir() || !os.SameFile(parentInfo, openedInfo) {
		return bundle.Manifest{}, sourceError("inspect opened staging parent", "", ErrMaterialization, err)
	}
	if err := parent.Mkdir(name, 0o700); err != nil {
		return bundle.Manifest{}, sourceError("create staging directory", "", ErrMaterialization, err)
	}
	owned := true
	defer func() {
		if err == nil || !owned {
			return
		}
		if cleanupErr := parent.RemoveAll(name); cleanupErr != nil {
			err = errors.Join(err, sourceError("remove failed staging directory", "", ErrMaterialization, cleanupErr))
		}
	}()
	createdInfo, err := parent.Lstat(name)
	if err != nil || !createdInfo.IsDir() || createdInfo.Mode()&os.ModeSymlink != 0 {
		return bundle.Manifest{}, sourceError("inspect staging directory", "", ErrMaterialization, err)
	}
	stage, err := parent.OpenRoot(name)
	if err != nil {
		return bundle.Manifest{}, sourceError("open staging directory", "", ErrMaterialization, err)
	}
	stageInfo, err := fs.Stat(stage.FS(), ".")
	if err != nil || !stageInfo.IsDir() || !os.SameFile(createdInfo, stageInfo) {
		return bundle.Manifest{}, sourceError("inspect opened staging directory", "", ErrMaterialization, err)
	}
	defer func() {
		if stage == nil {
			return
		}
		if closeErr := stage.Close(); closeErr != nil {
			err = errors.Join(err, sourceError("close staging directory", "", ErrMaterialization, closeErr))
		}
	}()

	buffer := make([]byte, stageBufferBytes)
	manifestDigest := strings.TrimPrefix(manifest.ID(), "sha256:")
	if err := copyStagedFile(ctx, sourceFS, stage, "manifest.json", -1, stageManifestBytes, manifestDigest, buffer); err != nil {
		return bundle.Manifest{}, err
	}
	for _, file := range manifest.Files {
		if err := copyStagedFile(ctx, sourceFS, stage, file.Path, file.Size, file.Size, file.SHA256, buffer); err != nil {
			return bundle.Manifest{}, err
		}
	}
	stagedManifest, err := bundle.Verify(ctx, stage.FS(), ".")
	if err != nil {
		return bundle.Manifest{}, err
	}
	if stagedManifest.ID() != manifest.ID() {
		return bundle.Manifest{}, &bundle.Error{
			Path:     "manifest.json",
			Field:    "sha256",
			Expected: manifest.ID(),
			Observed: stagedManifest.ID(),
			Kind:     bundle.ErrIntegrity,
		}
	}
	if err := stage.Close(); err != nil {
		stage = nil
		return bundle.Manifest{}, sourceError("close staging directory", "", ErrMaterialization, err)
	}
	stage = nil
	owned = false
	return stagedManifest, nil
}

func copyStagedFile(
	ctx context.Context,
	sourceFS fs.FS,
	destination *os.Root,
	path string,
	expectedSize int64,
	maxSize int64,
	expectedDigest string,
	buffer []byte,
) (err error) {
	if err := ctx.Err(); err != nil {
		return sourceError("copy source file", path, err, err)
	}
	input, err := sourceFS.Open(path)
	if err != nil {
		return sourceError("open source file", path, bundle.ErrInvalid, err)
	}
	defer func() {
		if closeErr := input.Close(); closeErr != nil {
			err = errors.Join(err, sourceError("close source file", path, bundle.ErrIntegrity, closeErr))
		}
	}()
	info, err := input.Stat()
	if err != nil {
		return sourceError("inspect source file", path, bundle.ErrInvalid, err)
	}
	if !info.Mode().IsRegular() {
		return sourceError("inspect source file", path, bundle.ErrInvalid, nil)
	}
	if info.Size() < 0 || info.Size() > maxSize || (expectedSize >= 0 && info.Size() != expectedSize) {
		return &bundle.Error{
			Path:     path,
			Field:    "size",
			Expected: expectedSizeDescription(expectedSize, maxSize),
			Observed: fmt.Sprintf("%d", info.Size()),
			Kind:     bundle.ErrIntegrity,
		}
	}
	directory := filepath.ToSlash(filepath.Dir(filepath.FromSlash(path)))
	if directory != "." {
		if err := destination.MkdirAll(directory, 0o700); err != nil {
			return sourceError("create staging subdirectory", path, ErrMaterialization, err)
		}
	}
	output, err := destination.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return sourceError("create staged file", path, ErrMaterialization, err)
	}
	defer func() {
		if output == nil {
			return
		}
		if closeErr := output.Close(); closeErr != nil {
			err = errors.Join(err, sourceError("close staged file", path, ErrMaterialization, closeErr))
		}
	}()

	hash := sha256.New()
	var size int64
	for {
		if err := ctx.Err(); err != nil {
			return sourceError("copy source file", path, err, err)
		}
		count, readErr := input.Read(buffer)
		if count < 0 || count > len(buffer) {
			return sourceError("read source file", path, bundle.ErrIntegrity, errors.New("invalid reader count"))
		}
		if contextErr := ctx.Err(); contextErr != nil {
			return sourceError("copy source file", path, contextErr, contextErr)
		}
		if count > 0 {
			size += int64(count)
			if size > maxSize {
				return &bundle.Error{
					Path:     path,
					Field:    "size",
					Expected: expectedSizeDescription(expectedSize, maxSize),
					Observed: fmt.Sprintf("more than %d", maxSize),
					Kind:     bundle.ErrIntegrity,
				}
			}
			_, _ = hash.Write(buffer[:count])
			if err := writeAll(ctx, output, buffer[:count]); err != nil {
				return sourceError("write staged file", path, ErrMaterialization, err)
			}
		}
		if errors.Is(readErr, io.EOF) {
			break
		}
		if readErr != nil {
			return sourceError("read source file", path, bundle.ErrIntegrity, readErr)
		}
		if count == 0 {
			return sourceError("read source file", path, bundle.ErrIntegrity, io.ErrNoProgress)
		}
	}
	if expectedSize >= 0 && size != expectedSize {
		return &bundle.Error{
			Path:     path,
			Field:    "size",
			Expected: fmt.Sprintf("%d", expectedSize),
			Observed: fmt.Sprintf("%d", size),
			Kind:     bundle.ErrIntegrity,
		}
	}
	observedDigest := hex.EncodeToString(hash.Sum(nil))
	if observedDigest != expectedDigest {
		return &bundle.Error{
			Path:     path,
			Field:    "sha256",
			Expected: expectedDigest,
			Observed: observedDigest,
			Kind:     bundle.ErrIntegrity,
		}
	}
	if err := output.Sync(); err != nil {
		return sourceError("sync staged file", path, ErrMaterialization, err)
	}
	if err := output.Close(); err != nil {
		output = nil
		return sourceError("close staged file", path, ErrMaterialization, err)
	}
	output = nil
	return nil
}

func writeAll(ctx context.Context, writer io.Writer, data []byte) error {
	for len(data) > 0 {
		if err := ctx.Err(); err != nil {
			return err
		}
		count, err := writer.Write(data)
		if count < 0 || count > len(data) {
			return errors.New("invalid writer count")
		}
		data = data[count:]
		if err != nil {
			return err
		}
		if count == 0 {
			return io.ErrShortWrite
		}
	}
	return nil
}

func expectedSizeDescription(expectedSize, maxSize int64) string {
	if expectedSize >= 0 {
		return fmt.Sprintf("%d", expectedSize)
	}
	return fmt.Sprintf("at most %d", maxSize)
}
