package source

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/metalagman/layajev/internal/bundle"
)

const (
	namespaceName       = ".laya-go"
	ownerMarkerName     = "owner-v1"
	lockFileName        = "materialize.lock"
	stagingDirectory    = "staging"
	bundlesDirectory    = "bundles"
	namespaceMarker     = "laya-go materialization namespace v1\n"
	stageMarker         = "laya-go materialization stage v1\n"
	stageRandomBytes    = 16
	namespaceRetryDelay = 5 * time.Millisecond
	namespaceRetries    = 200
)

type materializeHooks struct {
	checkpoint func(string) error
}

func (h materializeHooks) step(operation string) error {
	if h.checkpoint == nil {
		return nil
	}
	if err := h.checkpoint(operation); err != nil {
		return sourceError(operation, "", ErrMaterialization, err)
	}
	return nil
}

// Materialize verifies an fs.FS bundle and returns a retained verified handle
// to its deterministic native-compatible publication below workDir. Published
// bundles are immutable and retained after the returned handle is closed.
func Materialize(ctx context.Context, fileSystem fs.FS, root, workDir string) (*Bundle, error) {
	return materialize(ctx, fileSystem, root, workDir, materializeHooks{})
}

func materialize(
	ctx context.Context,
	fileSystem fs.FS,
	root string,
	workDir string,
	hooks materializeHooks,
) (result *Bundle, returnErr error) {
	if ctx == nil {
		return nil, sourceError("validate context", "", bundle.ErrInvalid, nil)
	}
	if !materializationPlatformSupported(runtime.GOOS, runtime.GOARCH) {
		return nil, sourceError("materialize bundle", "", ErrUnsupported, nil)
	}
	if fileSystem == nil {
		return nil, sourceError("validate filesystem", "", bundle.ErrInvalid, nil)
	}
	if root == "" {
		root = "."
	}
	if root != "." && (!fs.ValidPath(root) || strings.Contains(root, `\`)) {
		return nil, sourceError("validate filesystem root", root, bundle.ErrInvalid, nil)
	}
	if workDir == "" {
		return nil, sourceError("validate work directory", "", ErrMaterialization, nil)
	}
	if err := ctx.Err(); err != nil {
		return nil, sourceError("materialize bundle", "", err, err)
	}
	if err := hooks.step("verify source"); err != nil {
		return nil, err
	}
	manifest, err := bundle.Verify(ctx, fileSystem, root)
	if err != nil {
		return nil, err
	}
	digest := strings.TrimPrefix(manifest.ID(), "sha256:")
	if len(digest) != 64 || !validHexName(digest) {
		return nil, sourceError("validate bundle identity", "manifest.json", bundle.ErrIntegrity, nil)
	}

	workPath, work, err := openWorkDirectory(workDir)
	if err != nil {
		return nil, err
	}
	defer func() {
		if closeErr := work.Close(); closeErr != nil {
			joinMaterializeReturn(&result, &returnErr, sourceError("close work directory", "", ErrMaterialization, closeErr))
		}
	}()
	if err := hooks.step("ensure namespace"); err != nil {
		return nil, err
	}
	namespace, err := ensureNamespace(ctx, work, hooks)
	if err != nil {
		return nil, err
	}
	defer func() {
		if closeErr := namespace.Close(); closeErr != nil {
			joinMaterializeReturn(&result, &returnErr, sourceError("close materialization namespace", "", ErrMaterialization, closeErr))
		}
	}()

	if err := hooks.step("open lock"); err != nil {
		return nil, err
	}
	lockFile, err := openLockFile(namespace)
	if err != nil {
		return nil, err
	}
	if err := hooks.step("wait lock"); err != nil {
		_ = lockFile.Close()
		return nil, err
	}
	lock, err := acquireAdvisoryLock(ctx, lockFile)
	if err != nil {
		_ = lockFile.Close()
		return nil, err
	}
	defer func() {
		if closeErr := lock.Close(); closeErr != nil {
			joinMaterializeReturn(&result, &returnErr, closeErr)
		}
	}()

	if err := hooks.step("recover stages"); err != nil {
		return nil, err
	}
	if err := recoverStages(ctx, namespace, hooks); err != nil {
		return nil, err
	}
	finalName := "sha256-" + digest
	finalRelative := bundlesDirectory + "/" + finalName
	finalPath := filepath.Join(workPath, namespaceName, bundlesDirectory, finalName)
	if exists, err := rootEntryExists(namespace, finalRelative); err != nil {
		return nil, err
	} else if exists {
		if err := hooks.step("verify ready"); err != nil {
			return nil, err
		}
		ready, err := OpenDirectory(ctx, finalPath)
		if err != nil {
			return nil, err
		}
		if ready.Manifest().ID() != manifest.ID() {
			_ = ready.Close()
			return nil, sourceError("verify ready identity", "manifest.json", bundle.ErrIntegrity, nil)
		}
		return ready, nil
	}

	stageName, err := newStageName(digest)
	if err != nil {
		return nil, err
	}
	stageRelative := stagingDirectory + "/" + stageName
	if err := hooks.step("create stage"); err != nil {
		return nil, err
	}
	if err := namespace.Mkdir(stageRelative, 0o700); err != nil {
		return nil, sourceError("create materialization stage", "", ErrMaterialization, err)
	}
	stageOwned := true
	defer func() {
		if !stageOwned {
			return
		}
		if err := hooks.step("cleanup stage"); err != nil {
			joinMaterializeReturn(&result, &returnErr, err)
			return
		}
		if cleanupErr := namespace.RemoveAll(stageRelative); cleanupErr != nil {
			joinMaterializeReturn(
				&result,
				&returnErr,
				sourceError("remove owned stage", "", ErrMaterialization, cleanupErr),
			)
		}
	}()
	if err := hooks.step("write stage marker"); err != nil {
		return nil, err
	}
	if err := writeMarker(namespace, stageRelative+"/"+ownerMarkerName, stageMarker); err != nil {
		return nil, err
	}
	stagePath := filepath.Join(workPath, namespaceName, stagingDirectory, stageName, "bundle")
	if err := hooks.step("copy bundle"); err != nil {
		return nil, err
	}
	stagedManifest, err := Stage(ctx, fileSystem, root, stagePath)
	if err != nil {
		return nil, err
	}
	if stagedManifest.ID() != manifest.ID() {
		return nil, sourceError("verify staged identity", "manifest.json", bundle.ErrIntegrity, nil)
	}
	if err := hooks.step("sync stage"); err != nil {
		return nil, err
	}
	if err := syncTree(ctx, namespace, stageRelative); err != nil {
		return nil, err
	}
	if err := hooks.step("verify stage"); err != nil {
		return nil, err
	}
	staged, err := OpenDirectory(ctx, stagePath)
	if err != nil {
		return nil, err
	}
	if staged.Manifest().ID() != manifest.ID() {
		_ = staged.Close()
		return nil, sourceError("verify staged identity", "manifest.json", bundle.ErrIntegrity, nil)
	}
	if err := staged.Close(); err != nil {
		return nil, err
	}
	if err := hooks.step("publish bundle"); err != nil {
		return nil, err
	}
	renamed, err := renameNoReplace(namespace, stageRelative+"/bundle", finalRelative)
	if err != nil {
		return nil, err
	}
	if !renamed {
		ready, openErr := OpenDirectory(ctx, finalPath)
		if openErr != nil {
			return nil, openErr
		}
		if ready.Manifest().ID() != manifest.ID() {
			_ = ready.Close()
			return nil, sourceError("verify publication winner", "manifest.json", bundle.ErrIntegrity, nil)
		}
		return ready, nil
	}
	if err := hooks.step("sync publication"); err != nil {
		return nil, err
	}
	if err := syncDirectory(namespace, bundlesDirectory); err != nil {
		return nil, err
	}
	if err := hooks.step("remove published stage"); err != nil {
		return nil, err
	}
	if err := namespace.RemoveAll(stageRelative); err != nil {
		return nil, sourceError("remove published stage", "", ErrMaterialization, err)
	}
	stageOwned = false
	if err := syncDirectory(namespace, stagingDirectory); err != nil {
		return nil, err
	}
	if err := hooks.step("open publication"); err != nil {
		return nil, err
	}
	ready, err := OpenDirectory(ctx, finalPath)
	if err != nil {
		return nil, err
	}
	if ready.Manifest().ID() != manifest.ID() {
		_ = ready.Close()
		return nil, sourceError("verify publication identity", "manifest.json", bundle.ErrIntegrity, nil)
	}
	return ready, nil
}

func joinMaterializeReturn(result **Bundle, returnErr *error, err error) {
	if err == nil {
		return
	}
	if *result != nil {
		err = errors.Join(err, (*result).Close())
		*result = nil
	}
	*returnErr = errors.Join(*returnErr, err)
}

func materializationPlatformSupported(goos, goarch string) bool {
	return goos == "linux" && goarch == "amd64"
}

func openWorkDirectory(path string) (string, *os.Root, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", nil, sourceError("resolve work directory", "", ErrMaterialization, err)
	}
	abs = filepath.Clean(abs)
	info, err := os.Lstat(abs)
	if err != nil {
		return "", nil, sourceError("inspect work directory", "", ErrMaterialization, err)
	}
	if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return "", nil, sourceError("inspect work directory", "", ErrMaterialization, nil)
	}
	root, err := os.OpenRoot(abs)
	if err != nil {
		return "", nil, sourceError("open work directory", "", ErrMaterialization, err)
	}
	opened, err := fs.Stat(root.FS(), ".")
	if err != nil || !opened.IsDir() || !os.SameFile(info, opened) {
		_ = root.Close()
		return "", nil, sourceError("inspect opened work directory", "", ErrMaterialization, err)
	}
	return abs, root, nil
}

func ensureNamespace(ctx context.Context, work *os.Root, hooks materializeHooks) (*os.Root, error) {
	for attempt := 0; attempt < namespaceRetries; attempt++ {
		if err := ctx.Err(); err != nil {
			return nil, sourceError("initialize materialization namespace", "", err, err)
		}
		info, err := work.Lstat(namespaceName)
		if errors.Is(err, fs.ErrNotExist) {
			if err := initializeNamespace(work, hooks); err != nil {
				if errors.Is(err, fs.ErrExist) {
					continue
				}
				return nil, err
			}
			continue
		}
		if err != nil {
			return nil, sourceError("inspect materialization namespace", "", ErrMaterialization, err)
		}
		if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
			return nil, sourceError("inspect materialization namespace", "", ErrMaterialization, nil)
		}
		namespace, err := work.OpenRoot(namespaceName)
		if err != nil {
			return nil, sourceError("open materialization namespace", "", ErrMaterialization, err)
		}
		openedInfo, err := fs.Stat(namespace.FS(), ".")
		if err != nil || !openedInfo.IsDir() || !os.SameFile(info, openedInfo) {
			_ = namespace.Close()
			return nil, sourceError("inspect opened materialization namespace", "", ErrMaterialization, err)
		}
		valid, validateErr := validateNamespace(namespace)
		if validateErr != nil {
			_ = namespace.Close()
			return nil, validateErr
		}
		if valid {
			return namespace, nil
		}
		_ = namespace.Close()
		select {
		case <-ctx.Done():
			return nil, sourceError("initialize materialization namespace", "", ctx.Err(), ctx.Err())
		case <-time.After(namespaceRetryDelay):
		}
	}
	return nil, sourceError("initialize materialization namespace", "", ErrMaterialization, nil)
}

func initializeNamespace(work *os.Root, hooks materializeHooks) (returnErr error) {
	if err := hooks.step("create namespace"); err != nil {
		return err
	}
	if err := work.Mkdir(namespaceName, 0o700); err != nil {
		return sourceError("create materialization namespace", "", ErrMaterialization, err)
	}
	owned := true
	defer func() {
		if !owned || returnErr == nil {
			return
		}
		if err := work.RemoveAll(namespaceName); err != nil {
			returnErr = errors.Join(returnErr, sourceError("remove incomplete namespace", "", ErrMaterialization, err))
		}
	}()
	namespace, err := work.OpenRoot(namespaceName)
	if err != nil {
		return sourceError("open new materialization namespace", "", ErrMaterialization, err)
	}
	defer func() {
		if err := namespace.Close(); err != nil {
			returnErr = errors.Join(returnErr, sourceError("close new materialization namespace", "", ErrMaterialization, err))
		}
	}()
	if err := namespace.Mkdir(stagingDirectory, 0o700); err != nil {
		return sourceError("create staging namespace", "", ErrMaterialization, err)
	}
	if err := namespace.Mkdir(bundlesDirectory, 0o700); err != nil {
		return sourceError("create bundles namespace", "", ErrMaterialization, err)
	}
	lock, err := namespace.OpenFile(lockFileName, os.O_RDWR|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return sourceError("create materialization lock", "", ErrMaterialization, err)
	}
	if err := lock.Sync(); err != nil {
		_ = lock.Close()
		return sourceError("sync materialization lock", "", ErrMaterialization, err)
	}
	if err := lock.Close(); err != nil {
		return sourceError("close materialization lock", "", ErrMaterialization, err)
	}
	if err := hooks.step("write namespace marker"); err != nil {
		return err
	}
	if err := writeNamespaceMarker(namespace); err != nil {
		return err
	}
	if err := syncDirectory(namespace, "."); err != nil {
		return err
	}
	owned = false
	return nil
}

func writeNamespaceMarker(root *os.Root) error {
	const initializingMarker = "owner-v1.initializing"
	if err := writeMarker(root, initializingMarker, namespaceMarker); err != nil {
		return err
	}
	if err := root.Rename(initializingMarker, ownerMarkerName); err != nil {
		return sourceError("publish namespace marker", "", ErrMaterialization, err)
	}
	return nil
}

func validateNamespace(root *os.Root) (bool, error) {
	if _, err := root.Lstat(ownerMarkerName); err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return false, nil
		}
		return false, sourceError("inspect namespace marker", "", ErrMaterialization, err)
	}
	entries, err := fs.ReadDir(root.FS(), ".")
	if err != nil {
		return false, sourceError("read materialization namespace", "", ErrMaterialization, err)
	}
	want := map[string]fs.FileMode{
		ownerMarkerName:  0,
		lockFileName:     0,
		stagingDirectory: fs.ModeDir,
		bundlesDirectory: fs.ModeDir,
	}
	if len(entries) != len(want) {
		return false, sourceError("validate materialization namespace", "", ErrMaterialization, nil)
	}
	for _, entry := range entries {
		mode, ok := want[entry.Name()]
		if !ok || entry.Type()&fs.ModeSymlink != 0 || entry.IsDir() != (mode == fs.ModeDir) {
			return false, sourceError("validate materialization namespace", "", ErrMaterialization, nil)
		}
	}
	if err := verifyMarker(root, ownerMarkerName, namespaceMarker); err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return false, nil
		}
		return false, err
	}
	return true, nil
}

func writeMarker(root *os.Root, path, content string) error {
	file, err := root.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return sourceError("create ownership marker", "", ErrMaterialization, err)
	}
	if _, err := io.WriteString(file, content); err != nil {
		_ = file.Close()
		return sourceError("write ownership marker", "", ErrMaterialization, err)
	}
	if err := file.Sync(); err != nil {
		_ = file.Close()
		return sourceError("sync ownership marker", "", ErrMaterialization, err)
	}
	if err := file.Close(); err != nil {
		return sourceError("close ownership marker", "", ErrMaterialization, err)
	}
	return nil
}

func verifyMarker(root *os.Root, path, content string) error {
	info, err := root.Lstat(path)
	if err != nil {
		return sourceError("inspect ownership marker", "", ErrMaterialization, err)
	}
	if !info.Mode().IsRegular() || info.Size() != int64(len(content)) {
		return sourceError("validate ownership marker", "", ErrMaterialization, nil)
	}
	file, err := root.Open(path)
	if err != nil {
		return sourceError("open ownership marker", "", ErrMaterialization, err)
	}
	defer file.Close() //nolint:errcheck // The bounded read reports marker failures.
	data, err := io.ReadAll(io.LimitReader(file, int64(len(content))+1))
	if err != nil {
		return sourceError("read ownership marker", "", ErrMaterialization, err)
	}
	if string(data) != content {
		return sourceError("validate ownership marker", "", ErrMaterialization, nil)
	}
	return nil
}

func openLockFile(root *os.Root) (*os.File, error) {
	info, err := root.Lstat(lockFileName)
	if err != nil {
		return nil, sourceError("inspect materialization lock", "", ErrMaterialization, err)
	}
	if !info.Mode().IsRegular() {
		return nil, sourceError("inspect materialization lock", "", ErrMaterialization, nil)
	}
	file, err := root.OpenFile(lockFileName, os.O_RDWR, 0)
	if err != nil {
		return nil, sourceError("open materialization lock", "", ErrMaterialization, err)
	}
	opened, err := file.Stat()
	if err != nil || !opened.Mode().IsRegular() || !os.SameFile(info, opened) {
		_ = file.Close()
		return nil, sourceError("inspect opened materialization lock", "", ErrMaterialization, err)
	}
	return file, nil
}

func rootEntryExists(root *os.Root, path string) (bool, error) {
	_, err := root.Lstat(path)
	if err == nil {
		return true, nil
	}
	if errors.Is(err, fs.ErrNotExist) {
		return false, nil
	}
	return false, sourceError("inspect materialization entry", "", ErrMaterialization, err)
}

func newStageName(digest string) (string, error) {
	random := make([]byte, stageRandomBytes)
	if _, err := io.ReadFull(rand.Reader, random); err != nil {
		return "", sourceError("generate stage identity", "", ErrMaterialization, err)
	}
	return digest + "-" + hex.EncodeToString(random), nil
}

func validStageName(name string) bool {
	if len(name) != 64+1+stageRandomBytes*2 || name[64] != '-' {
		return false
	}
	return validLowerHex(name[:64]) && validLowerHex(name[65:])
}

func recoverStages(ctx context.Context, namespace *os.Root, hooks materializeHooks) error {
	entries, err := fs.ReadDir(namespace.FS(), stagingDirectory)
	if err != nil {
		return sourceError("read staging namespace", "", ErrMaterialization, err)
	}
	for _, entry := range entries {
		if err := ctx.Err(); err != nil {
			return sourceError("recover stages", "", err, err)
		}
		if !validStageName(entry.Name()) || !entry.IsDir() || entry.Type()&fs.ModeSymlink != 0 {
			continue
		}
		path := stagingDirectory + "/" + entry.Name()
		stage, err := namespace.OpenRoot(path)
		if err != nil {
			continue
		}
		markerErr := verifyMarker(stage, ownerMarkerName, stageMarker)
		closeErr := stage.Close()
		if markerErr != nil || closeErr != nil {
			continue
		}
		if err := hooks.step("recover owned stage"); err != nil {
			return err
		}
		if err := namespace.RemoveAll(path); err != nil {
			return sourceError("remove abandoned stage", "", ErrMaterialization, err)
		}
	}
	return nil
}

func syncTree(ctx context.Context, root *os.Root, path string) error {
	fileSystem, err := fs.Sub(root.FS(), path)
	if err != nil {
		return sourceError("open staged tree", "", ErrMaterialization, err)
	}
	var directories []string
	err = fs.WalkDir(fileSystem, ".", func(name string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if entry.IsDir() {
			directories = append(directories, name)
		}
		return nil
	})
	if err != nil {
		return sourceError("inspect staged tree", "", ErrMaterialization, err)
	}
	for index := len(directories) - 1; index >= 0; index-- {
		name := path
		if directories[index] != "." {
			name += "/" + directories[index]
		}
		if err := syncDirectory(root, name); err != nil {
			return err
		}
	}
	return nil
}

func syncDirectory(root *os.Root, path string) error {
	directory, err := root.Open(path)
	if err != nil {
		return sourceError("open materialization directory", "", ErrMaterialization, err)
	}
	if err := directory.Sync(); err != nil {
		_ = directory.Close()
		return sourceError("sync materialization directory", "", ErrMaterialization, err)
	}
	if err := directory.Close(); err != nil {
		return sourceError("close materialization directory", "", ErrMaterialization, err)
	}
	return nil
}
