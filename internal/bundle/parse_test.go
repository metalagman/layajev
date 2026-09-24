package bundle

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"
)

func TestParseValidManifest(t *testing.T) {
	data := validManifestBytes(t)
	wantDigest := sha256.Sum256(data)

	manifest, err := Parse(data)
	if err != nil {
		t.Fatalf("Parse(valid) returned unexpected error: %v", err)
	}
	if want := "sha256:" + hex.EncodeToString(wantDigest[:]); manifest.ID() != want {
		t.Errorf("Manifest.ID() = %q, want %q", manifest.ID(), want)
	}
	if manifest.Provenance.Profile.ID != SupportedProfileID {
		t.Errorf("profile ID = %q, want %q", manifest.Provenance.Profile.ID, SupportedProfileID)
	}
	if len(manifest.Model.Inputs) != 5 || len(manifest.Model.Outputs) != 2 {
		t.Errorf("tensor counts = %d inputs, %d outputs; want 5, 2", len(manifest.Model.Inputs), len(manifest.Model.Outputs))
	}

	for index := range data {
		data[index] = 'x'
	}
	if manifest.Bundle.ID != "laya-multilingual" {
		t.Errorf("manifest aliased input bytes: bundle ID = %q", manifest.Bundle.ID)
	}
}

func TestParseRejectsInvalidAndUnsupportedManifests(t *testing.T) {
	tests := []struct {
		name      string
		mutate    func(t *testing.T, data []byte) []byte
		wantError error
		wantField string
	}{
		{
			name: "duplicate field",
			mutate: func(t *testing.T, data []byte) []byte {
				t.Helper()
				return bytes.Replace(data, []byte(`"schema_version": 1`), []byte(`"schema_version": 1, "schema_version": 1`), 1)
			},
			wantError: ErrInvalid,
			wantField: "$.schema_version",
		},
		{
			name:      "unknown field",
			mutate:    mutateObject(func(object map[string]any) { object["unexpected"] = true }),
			wantError: ErrInvalid,
		},
		{
			name: "wrong field case",
			mutate: func(t *testing.T, data []byte) []byte {
				t.Helper()
				return bytes.Replace(data, []byte(`"schema_version"`), []byte(`"SCHEMA_VERSION"`), 1)
			},
			wantError: ErrInvalid,
			wantField: "$.SCHEMA_VERSION",
		},
		{
			name:      "missing bundle",
			mutate:    mutateObject(func(object map[string]any) { delete(object, "bundle") }),
			wantError: ErrInvalid,
			wantField: "$.bundle",
		},
		{
			name:      "missing schema version",
			mutate:    mutateObject(func(object map[string]any) { delete(object, "schema_version") }),
			wantError: ErrInvalid,
			wantField: "$.schema_version",
		},
		{
			name:      "unsupported schema",
			mutate:    mutateObject(func(object map[string]any) { object["schema_version"] = float64(2) }),
			wantError: ErrUnsupported,
			wantField: "schema_version",
		},
		{
			name: "unsupported profile",
			mutate: mutateObject(func(object map[string]any) {
				objectAt(object, "provenance", "profile")["id"] = "unknown-profile"
			}),
			wantError: ErrUnsupported,
			wantField: "provenance.profile.id",
		},
		{
			name: "profile digest substitution",
			mutate: mutateObject(func(object map[string]any) {
				objectAt(object, "provenance", "profile")["sha256"] = strings.Repeat("f", 64)
			}),
			wantError: ErrUnsupported,
			wantField: "provenance.profile.sha256",
		},
		{
			name: "unsupported precision",
			mutate: mutateObject(func(object map[string]any) {
				objectAt(object, "model")["precision"] = "fp16"
			}),
			wantError: ErrUnsupported,
			wantField: "model.precision",
		},
		{
			name: "unsupported input contract",
			mutate: mutateObject(func(object map[string]any) {
				inputs := objectAt(object, "model")["inputs"].([]any)
				inputs[0].(map[string]any)["name"] = "tokens"
			}),
			wantError: ErrUnsupported,
			wantField: "model.inputs",
		},
		{
			name: "unapproved redistribution",
			mutate: mutateObject(func(object map[string]any) {
				objectAt(object, "provenance")["redistribution"] = "review-required"
			}),
			wantError: ErrUnsupported,
			wantField: "provenance.redistribution",
		},
		{
			name: "invalid digest",
			mutate: mutateObject(func(object map[string]any) {
				files := object["files"].([]any)
				files[0].(map[string]any)["sha256"] = "ABC"
			}),
			wantError: ErrInvalid,
			wantField: "files[0].sha256",
		},
		{
			name: "escaping path",
			mutate: mutateObject(func(object map[string]any) {
				files := object["files"].([]any)
				files[0].(map[string]any)["path"] = "../LICENSE.model"
			}),
			wantError: ErrInvalid,
			wantField: "files[0].path",
		},
		{
			name: "case collision",
			mutate: mutateObject(func(object map[string]any) {
				files := object["files"].([]any)
				files[1].(map[string]any)["path"] = "license.MODEL"
			}),
			wantError: ErrInvalid,
			wantField: "files[1].path",
		},
		{
			name: "created epoch mismatch",
			mutate: mutateObject(func(object map[string]any) {
				objectAt(object, "provenance")["source_date_epoch"] = float64(0)
			}),
			wantError: ErrInvalid,
			wantField: "provenance.source_date_epoch",
		},
	}

	valid := validManifestBytes(t)
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := Parse(test.mutate(t, bytes.Clone(valid)))
			if !errors.Is(err, test.wantError) {
				t.Fatalf("Parse() error = %v, want errors.Is(_, %v)", err, test.wantError)
			}
			var bundleError *Error
			if !errors.As(err, &bundleError) {
				t.Fatalf("Parse() error type = %T, want *Error", err)
			}
			if test.wantField != "" && bundleError.Field != test.wantField {
				t.Errorf("BundleError.Field = %q, want %q", bundleError.Field, test.wantField)
			}
		})
	}
}

func TestParseRejectsManifestSizeLimit(t *testing.T) {
	data := bytes.Repeat([]byte(" "), maxManifestBytes+1)
	_, err := Parse(data)
	if !errors.Is(err, ErrInvalid) {
		t.Fatalf("Parse(oversized) error = %v, want ErrInvalid", err)
	}
	if strings.Contains(err.Error(), string(data[:128])) {
		t.Fatal("oversized error exposes manifest content")
	}
}

func TestSupportedProfileDigestMatchesRepositoryFile(t *testing.T) {
	data, err := os.ReadFile("../../tools/export/profiles/laya-multilingual-v1.json")
	if err != nil {
		t.Fatalf("os.ReadFile(profile): %v", err)
	}
	digest := sha256.Sum256(data)
	if got := hex.EncodeToString(digest[:]); got != SupportedProfileSHA256 {
		t.Errorf("profile SHA-256 = %q, compatibility constant = %q", got, SupportedProfileSHA256)
	}
}

func FuzzParse(f *testing.F) {
	f.Add(validManifestBytes(f))
	f.Add([]byte(`{}`))
	f.Add([]byte(`{"schema_version":1,"schema_version":1}`))
	f.Fuzz(func(t *testing.T, data []byte) {
		manifest, err := Parse(data)
		if err == nil && !strings.HasPrefix(manifest.ID(), "sha256:") {
			t.Errorf("successful Parse() ID = %q, want sha256 prefix", manifest.ID())
		}
	})
}

func validManifestBytes(t testing.TB) []byte {
	t.Helper()
	data, err := os.ReadFile("../../schema/testdata/manifest-v1.valid.json")
	if err != nil {
		t.Fatalf("os.ReadFile(valid manifest): %v", err)
	}
	return data
}

func mutateObject(mutate func(map[string]any)) func(*testing.T, []byte) []byte {
	return func(t *testing.T, data []byte) []byte {
		t.Helper()
		var object map[string]any
		if err := json.Unmarshal(data, &object); err != nil {
			t.Fatalf("json.Unmarshal(valid manifest): %v", err)
		}
		mutate(object)
		result, err := json.Marshal(object)
		if err != nil {
			t.Fatalf("json.Marshal(mutated manifest): %v", err)
		}
		return result
	}
}

func objectAt(object map[string]any, path ...string) map[string]any {
	for _, field := range path {
		object = object[field].(map[string]any)
	}
	return object
}
