package layajev

import (
	"os"
	"path/filepath"
	"testing"
)

func TestReadBundleMetadata(t *testing.T) {
	tests := []struct {
		name    string
		content string
		wantErr bool
	}{
		{name: "valid", content: `{"bundle":{"id":"laya-v1"},"provenance":{"created_at":"2026-01-01T00:00:00Z"}}`},
		{name: "missing ID", content: `{"bundle":{},"provenance":{"created_at":"2026-01-01T00:00:00Z"}}`, wantErr: true},
		{name: "bad date", content: `{"bundle":{"id":"laya-v1"},"provenance":{"created_at":"yesterday"}}`, wantErr: true},
		{name: "invalid JSON", content: `{`, wantErr: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "manifest.json"), []byte(test.content), 0o600); err != nil {
				t.Fatalf("write manifest: %v", err)
			}
			metadata, err := readBundleMetadata(dir)
			if (err != nil) != test.wantErr {
				t.Fatalf("readBundleMetadata() error = %v, want error %v", err, test.wantErr)
			}
			if !test.wantErr && metadata.Bundle.ID != "laya-v1" {
				t.Errorf("bundle ID = %q, want laya-v1", metadata.Bundle.ID)
			}
		})
	}
}
