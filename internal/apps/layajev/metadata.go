package layajev

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"
)

// bundleMetadata contains only the identity displayed by the HTTP adapter.
// OpenModelDir performs authoritative verification of the complete bundle.
type bundleMetadata struct {
	Bundle struct {
		ID string `json:"id"`
	} `json:"bundle"`
	Provenance struct {
		CreatedAt string `json:"created_at"`
	} `json:"provenance"`
}

func readBundleMetadata(dir string) (bundleMetadata, error) {
	file, err := os.Open(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return bundleMetadata{}, fmt.Errorf("open manifest: %w", err)
	}
	defer func() { _ = file.Close() }()
	var metadata bundleMetadata
	decoder := json.NewDecoder(io.LimitReader(file, 1<<20))
	if err := decoder.Decode(&metadata); err != nil {
		return bundleMetadata{}, fmt.Errorf("decode manifest metadata: %w", err)
	}
	if metadata.Bundle.ID == "" {
		return bundleMetadata{}, fmt.Errorf("manifest bundle id is empty")
	}
	if _, err := time.Parse(time.RFC3339, metadata.Provenance.CreatedAt); err != nil {
		return bundleMetadata{}, fmt.Errorf("manifest creation time: %w", err)
	}
	return metadata, nil
}
