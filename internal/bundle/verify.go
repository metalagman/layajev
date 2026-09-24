package bundle

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"strings"
)

const hashBufferBytes = 64 << 10

// Verify checks a complete bundle rooted at root without materializing or
// mutating it. The returned manifest is available only after every declared
// artifact and the exact regular-file allow-list have been verified.
func Verify(ctx context.Context, fileSystem fs.FS, root string) (Manifest, error) {
	if ctx == nil {
		return Manifest{}, invalid("", "context", "non-nil context", "nil", nil)
	}
	if fileSystem == nil {
		return Manifest{}, invalid("", "filesystem", "non-nil fs.FS", "nil", nil)
	}
	if root == "" || (root != "." && (!fs.ValidPath(root) || strings.Contains(root, `\`))) {
		return Manifest{}, invalid(root, "root", "normalized slash-relative fs.FS path", scalar(root), nil)
	}
	if err := ctx.Err(); err != nil {
		return Manifest{}, canceled("manifest.json", err)
	}

	rootFS, err := fs.Sub(fileSystem, root)
	if err != nil {
		return Manifest{}, invalid(root, "root", "readable bundle directory", "invalid", err)
	}
	manifestBytes, err := readManifest(rootFS)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) && isRegular(rootFS, "model.safetensors") {
			return Manifest{}, invalid(
				"manifest.json",
				"bundle",
				"complete runtime bundle; run the offline bundle exporter",
				"raw Safetensors checkpoint is exporter input only",
				err,
			)
		}
		return Manifest{}, err
	}
	manifest, err := Parse(manifestBytes)
	if err != nil {
		return Manifest{}, err
	}
	if err := validateLayout(manifest); err != nil {
		return Manifest{}, err
	}
	if err := verifyFiles(ctx, rootFS, manifest.Files); err != nil {
		return Manifest{}, err
	}
	return manifest, nil
}

func readManifest(fileSystem fs.FS) ([]byte, error) {
	file, err := fileSystem.Open("manifest.json")
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil, &Error{
				Path:     "manifest.json",
				Expected: "regular manifest file",
				Observed: "missing",
				Kind:     ErrInvalid,
				Cause:    err,
			}
		}
		return nil, invalid("manifest.json", "", "readable regular manifest file", "open failed", err)
	}
	defer file.Close() //nolint:errcheck // The complete read below reports content errors.

	info, err := file.Stat()
	if err != nil {
		return nil, invalid("manifest.json", "", "statable regular manifest file", "stat failed", err)
	}
	if !info.Mode().IsRegular() {
		return nil, invalid("manifest.json", "", "regular manifest file", info.Mode().Type().String(), nil)
	}
	if info.Size() < 0 || info.Size() > maxManifestBytes {
		return nil, invalid("manifest.json", "", fmt.Sprintf("at most %d bytes", maxManifestBytes), fmt.Sprintf("%d bytes", info.Size()), nil)
	}
	data, err := io.ReadAll(io.LimitReader(file, maxManifestBytes+1))
	if err != nil {
		return nil, invalid("manifest.json", "", "readable manifest file", "read failed", err)
	}
	if len(data) > maxManifestBytes {
		return nil, invalid("manifest.json", "", fmt.Sprintf("at most %d bytes", maxManifestBytes), fmt.Sprintf("more than %d bytes", maxManifestBytes), nil)
	}
	return data, nil
}

func validateLayout(manifest Manifest) error {
	declared := make(map[string]File, len(manifest.Files))
	roleCounts := make(map[string]int)
	for _, file := range manifest.Files {
		if strings.EqualFold(file.Path, "manifest.json") {
			return invalid("manifest.json", "files", "manifest is never self-declared", file.Path, nil)
		}
		declared[file.Path] = file
		roleCounts[file.Role]++
	}

	wantRoleCounts := []struct {
		role  string
		count int
	}{
		{"model", 1},
		{"model-external-data", len(manifest.Model.ExternalData)},
		{"tokenizer", 1},
		{"tokenizer-config", 1},
		{"agent-config", 1},
		{"license", len(manifest.Provenance.Licenses)},
		{"notice", len(manifest.Provenance.Licenses)},
	}
	for _, want := range wantRoleCounts {
		if got := roleCounts[want.role]; got != want.count {
			return invalid("manifest.json", "files", fmt.Sprintf("%d %s artifact(s)", want.count, want.role), fmt.Sprintf("%d", got), nil)
		}
	}

	required := []struct {
		path string
		role string
	}{
		{manifest.Model.Path, "model"},
		{manifest.Tokenizer.JSONPath, "tokenizer"},
		{manifest.Tokenizer.ConfigPath, "tokenizer-config"},
		{manifest.Calibration.ConfigPath, "agent-config"},
	}
	for _, externalPath := range manifest.Model.ExternalData {
		required = append(required, struct {
			path string
			role string
		}{externalPath, "model-external-data"})
	}
	for _, license := range manifest.Provenance.Licenses {
		required = append(required, struct {
			path string
			role string
		}{license.Path, "license"})
	}
	for _, artifact := range required {
		file, ok := declared[artifact.path]
		if !ok {
			return invalid("manifest.json", "files", "declared "+artifact.role+" path", artifact.path+" missing", nil)
		}
		if file.Role != artifact.role {
			return invalid("manifest.json", "files", artifact.role+" role for "+artifact.path, file.Role, nil)
		}
	}
	return nil
}

func verifyFiles(ctx context.Context, fileSystem fs.FS, files []File) error {
	declared := make(map[string]File, len(files))
	for _, file := range files {
		declared[file.Path] = file
	}
	seen := make(map[string]bool, len(files))
	err := fs.WalkDir(fileSystem, ".", func(filePath string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return invalid(filePath, "", "readable bundle entry", "walk failed", walkErr)
		}
		if err := ctx.Err(); err != nil {
			return canceled(filePath, err)
		}
		if filePath == "." || entry.IsDir() {
			return nil
		}
		if filePath == "manifest.json" {
			if entry.Type().IsRegular() {
				return nil
			}
			return invalid(filePath, "", "regular manifest file", entry.Type().String(), nil)
		}
		declaration, ok := declared[filePath]
		if !ok {
			return invalid(filePath, "files", "declared regular artifact", "undeclared entry", nil)
		}
		if !entry.Type().IsRegular() {
			return invalid(filePath, "files", "regular artifact", entry.Type().String(), nil)
		}
		if err := verifyFile(ctx, fileSystem, declaration); err != nil {
			return err
		}
		seen[filePath] = true
		return nil
	})
	if err != nil {
		return err
	}
	for _, file := range files {
		if !seen[file.Path] {
			return invalid(file.Path, "files", "declared regular artifact", "missing", fs.ErrNotExist)
		}
	}
	return nil
}

func verifyFile(ctx context.Context, fileSystem fs.FS, declaration File) error {
	file, err := fileSystem.Open(declaration.Path)
	if err != nil {
		return invalid(declaration.Path, "files", "readable regular artifact", "open failed", err)
	}
	defer file.Close() //nolint:errcheck // Read and digest failures are reported below.

	info, err := file.Stat()
	if err != nil {
		return invalid(declaration.Path, "files", "statable regular artifact", "stat failed", err)
	}
	if !info.Mode().IsRegular() {
		return invalid(declaration.Path, "files", "regular artifact", info.Mode().Type().String(), nil)
	}
	if info.Size() != declaration.Size {
		return integrity(declaration.Path, "size", fmt.Sprint(declaration.Size), fmt.Sprint(info.Size()), nil)
	}

	hash := sha256.New()
	buffer := make([]byte, hashBufferBytes)
	var size int64
	for {
		if err := ctx.Err(); err != nil {
			return canceled(declaration.Path, err)
		}
		count, readErr := file.Read(buffer)
		if count > 0 {
			size += int64(count)
			_, _ = hash.Write(buffer[:count])
		}
		if errors.Is(readErr, io.EOF) {
			break
		}
		if readErr != nil {
			return integrity(declaration.Path, "sha256", declaration.SHA256, "read failed", readErr)
		}
		if count == 0 {
			return integrity(declaration.Path, "size", fmt.Sprint(declaration.Size), fmt.Sprint(size), io.ErrNoProgress)
		}
	}
	if size != declaration.Size {
		return integrity(declaration.Path, "size", fmt.Sprint(declaration.Size), fmt.Sprint(size), nil)
	}
	observed := hex.EncodeToString(hash.Sum(nil))
	if observed != declaration.SHA256 {
		return integrity(declaration.Path, "sha256", declaration.SHA256, observed, nil)
	}
	return nil
}

func isRegular(fileSystem fs.FS, name string) bool {
	info, err := fs.Stat(fileSystem, name)
	return err == nil && info.Mode().IsRegular()
}

func integrity(filePath, field, expected, observed string, cause error) error {
	return &Error{
		Path:     filePath,
		Field:    field,
		Expected: expected,
		Observed: observed,
		Kind:     ErrIntegrity,
		Cause:    cause,
	}
}

func canceled(filePath string, cause error) error {
	return invalid(filePath, "", "verification to complete", "canceled", cause)
}
