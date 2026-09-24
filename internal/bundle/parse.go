package bundle

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"math"
	"reflect"
	"regexp"
	"slices"
	"strings"
	"time"
	"unicode/utf8"
)

const (
	maxManifestBytes = 1 << 20
	maxStringBytes   = 512
	maxFiles         = 256
	maxLicenses      = 16
	maxActions       = 32
	maxExternalData  = 64
)

var (
	sha256Pattern      = regexp.MustCompile(`^[0-9a-f]{64}$`)
	gitRevisionPattern = regexp.MustCompile(`^[0-9a-f]{40}$`)
	identifierPattern  = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._/-]*$`)
	versionPattern     = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._+\-]*$`)
)

// Parse decodes, validates, and checks compatibility of exact manifest bytes.
// It retains no reference to data.
func Parse(data []byte) (Manifest, error) {
	if len(data) == 0 {
		return Manifest{}, invalid("manifest.json", "", "non-empty JSON object", "empty", nil)
	}
	if len(data) > maxManifestBytes {
		return Manifest{}, invalid("manifest.json", "", fmt.Sprintf("at most %d bytes", maxManifestBytes), fmt.Sprintf("%d bytes", len(data)), nil)
	}
	if !utf8.Valid(data) {
		return Manifest{}, invalid("manifest.json", "", "valid UTF-8", "invalid UTF-8", nil)
	}
	if !json.Valid(data) {
		var value any
		err := json.Unmarshal(data, &value)
		return Manifest{}, invalid("manifest.json", "", "valid JSON", "malformed", err)
	}
	if err := rejectDuplicateFields(data); err != nil {
		return Manifest{}, err
	}
	var fields any
	if err := json.Unmarshal(data, &fields); err != nil {
		return Manifest{}, invalid("manifest.json", "", "JSON object", "invalid", err)
	}
	if err := validateExactFields(fields, reflect.TypeFor[Manifest](), "$"); err != nil {
		return Manifest{}, err
	}

	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var manifest Manifest
	if err := decoder.Decode(&manifest); err != nil {
		return Manifest{}, invalid("manifest.json", "", "bundle manifest v1 fields", "invalid", err)
	}
	if err := requireEOF(decoder); err != nil {
		return Manifest{}, invalid("manifest.json", "", "one JSON object", "trailing value", err)
	}
	if err := validateManifest(manifest); err != nil {
		return Manifest{}, err
	}
	if err := validateCompatibility(manifest); err != nil {
		return Manifest{}, err
	}

	digest := sha256.Sum256(data)
	manifest.id = "sha256:" + hex.EncodeToString(digest[:])
	return manifest, nil
}

func requireEOF(decoder *json.Decoder) error {
	var trailing any
	err := decoder.Decode(&trailing)
	if errors.Is(err, io.EOF) {
		return nil
	}
	if err != nil {
		return err
	}
	return errors.New("additional JSON value")
}

func rejectDuplicateFields(data []byte) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	if err := consumeJSONValue(decoder, "$", 0); err != nil {
		return err
	}
	return requireEOF(decoder)
}

func consumeJSONValue(decoder *json.Decoder, path string, depth int) error {
	if depth > 64 {
		return invalid("manifest.json", path, "nesting depth at most 64", "too deep", nil)
	}
	token, err := decoder.Token()
	if err != nil {
		return invalid("manifest.json", path, "JSON value", "invalid", err)
	}
	delimiter, ok := token.(json.Delim)
	if !ok {
		return nil
	}
	switch delimiter {
	case '{':
		seen := make(map[string]bool)
		for decoder.More() {
			keyToken, err := decoder.Token()
			if err != nil {
				return invalid("manifest.json", path, "object key", "invalid", err)
			}
			key, ok := keyToken.(string)
			if !ok {
				return invalid("manifest.json", path, "string object key", fmt.Sprintf("%T", keyToken), nil)
			}
			if len(key) > maxStringBytes {
				return invalid("manifest.json", path, "object key at most 512 bytes", fmt.Sprintf("%d bytes", len(key)), nil)
			}
			fieldPath := path + "." + key
			if seen[key] {
				return invalid("manifest.json", fieldPath, "unique field", "duplicate", nil)
			}
			seen[key] = true
			if err := consumeJSONValue(decoder, fieldPath, depth+1); err != nil {
				return err
			}
		}
		if _, err := decoder.Token(); err != nil {
			return invalid("manifest.json", path, "object close", "invalid", err)
		}
	case '[':
		for index := 0; decoder.More(); index++ {
			if err := consumeJSONValue(decoder, fmt.Sprintf("%s[%d]", path, index), depth+1); err != nil {
				return err
			}
		}
		if _, err := decoder.Token(); err != nil {
			return invalid("manifest.json", path, "array close", "invalid", err)
		}
	default:
		return invalid("manifest.json", path, "object or array", string(delimiter), nil)
	}
	return nil
}

func validateExactFields(value any, target reflect.Type, path string) error {
	for target.Kind() == reflect.Pointer {
		target = target.Elem()
	}
	if value == nil {
		return nil
	}
	if target.Kind() == reflect.Struct {
		object, ok := value.(map[string]any)
		if !ok {
			return invalid("manifest.json", path, "object", fmt.Sprintf("%T", value), nil)
		}
		fields := make(map[string]reflect.Type, target.NumField())
		for index := range target.NumField() {
			field := target.Field(index)
			if !field.IsExported() {
				continue
			}
			name := strings.Split(field.Tag.Get("json"), ",")[0]
			if name == "" || name == "-" {
				continue
			}
			fields[name] = field.Type
		}
		for name, child := range object {
			childType, ok := fields[name]
			if !ok {
				return invalid("manifest.json", path+"."+name, "known exact-case field", "unknown", nil)
			}
			if err := validateExactFields(child, childType, path+"."+name); err != nil {
				return err
			}
		}
		for name := range fields {
			if _, ok := object[name]; !ok {
				return invalid("manifest.json", path+"."+name, "required field", "missing", nil)
			}
		}
		return nil
	}
	if target.Kind() == reflect.Slice || target.Kind() == reflect.Array {
		array, ok := value.([]any)
		if !ok {
			return invalid("manifest.json", path, "array", fmt.Sprintf("%T", value), nil)
		}
		for index, child := range array {
			if err := validateExactFields(child, target.Elem(), fmt.Sprintf("%s[%d]", path, index)); err != nil {
				return err
			}
		}
	}
	return nil
}

func validateManifest(manifest Manifest) error {
	if manifest.SchemaVersion < 0 {
		return invalid("manifest.json", "schema_version", "non-negative integer", fmt.Sprint(manifest.SchemaVersion), nil)
	}
	if err := validateIdentifier("bundle.id", manifest.Bundle.ID); err != nil {
		return err
	}
	if err := validateVersion("bundle.version", manifest.Bundle.Version); err != nil {
		return err
	}
	if err := validateIdentifier("provenance.source_model.id", manifest.Provenance.SourceModel.ID); err != nil {
		return err
	}
	if !gitRevisionPattern.MatchString(manifest.Provenance.SourceModel.Revision) {
		return invalid("manifest.json", "provenance.source_model.revision", "40 lowercase hexadecimal characters", scalar(manifest.Provenance.SourceModel.Revision), nil)
	}
	if err := validateIdentifier("provenance.sdk.repository", manifest.Provenance.SDK.Repository); err != nil {
		return err
	}
	if !gitRevisionPattern.MatchString(manifest.Provenance.SDK.Revision) {
		return invalid("manifest.json", "provenance.sdk.revision", "40 lowercase hexadecimal characters", scalar(manifest.Provenance.SDK.Revision), nil)
	}
	if err := validateVersion("provenance.sdk.version", manifest.Provenance.SDK.Version); err != nil {
		return err
	}
	if err := validateIdentifier("provenance.profile.id", manifest.Provenance.Profile.ID); err != nil {
		return err
	}
	if err := validateSHA256("provenance.profile.sha256", manifest.Provenance.Profile.SHA256); err != nil {
		return err
	}
	if err := validateIdentifier("provenance.exporter.name", manifest.Provenance.Exporter.Name); err != nil {
		return err
	}
	if err := validateVersion("provenance.exporter.version", manifest.Provenance.Exporter.Version); err != nil {
		return err
	}
	if err := validateSHA256("provenance.exporter.source_sha256", manifest.Provenance.Exporter.SourceSHA256); err != nil {
		return err
	}
	if !gitRevisionPattern.MatchString(manifest.Provenance.Exporter.GitRevision) {
		return invalid("manifest.json", "provenance.exporter.git_revision", "40 lowercase hexadecimal characters", scalar(manifest.Provenance.Exporter.GitRevision), nil)
	}
	if err := validatePath("provenance.lock.path", manifest.Provenance.Lock.Path); err != nil {
		return err
	}
	if err := validateSHA256("provenance.lock.sha256", manifest.Provenance.Lock.SHA256); err != nil {
		return err
	}
	createdAt, err := time.Parse(time.RFC3339, manifest.Provenance.CreatedAt)
	if err != nil {
		return invalid("manifest.json", "provenance.created_at", "RFC3339 timestamp", "invalid", err)
	}
	if manifest.Provenance.SourceDateEpoch < 0 {
		return invalid("manifest.json", "provenance.source_date_epoch", "non-negative integer", fmt.Sprint(manifest.Provenance.SourceDateEpoch), nil)
	}
	if createdAt.Unix() != manifest.Provenance.SourceDateEpoch {
		return invalid("manifest.json", "provenance.source_date_epoch", fmt.Sprint(createdAt.Unix()), fmt.Sprint(manifest.Provenance.SourceDateEpoch), nil)
	}
	if manifest.Provenance.Redistribution != "review-required" && manifest.Provenance.Redistribution != "approved" {
		return invalid("manifest.json", "provenance.redistribution", "review-required or approved", scalar(manifest.Provenance.Redistribution), nil)
	}
	if len(manifest.Provenance.Licenses) < 2 || len(manifest.Provenance.Licenses) > maxLicenses {
		return invalid("manifest.json", "provenance.licenses", "2..16 entries", fmt.Sprintf("%d entries", len(manifest.Provenance.Licenses)), nil)
	}
	licenseComponents := make(map[string]bool, len(manifest.Provenance.Licenses))
	for index, license := range manifest.Provenance.Licenses {
		prefix := fmt.Sprintf("provenance.licenses[%d]", index)
		if err := validateIdentifier(prefix+".component", license.Component); err != nil {
			return err
		}
		if licenseComponents[strings.ToLower(license.Component)] {
			return invalid("manifest.json", prefix+".component", "unique component", scalar(license.Component), nil)
		}
		licenseComponents[strings.ToLower(license.Component)] = true
		if err := validateString(prefix+".spdx", license.SPDX, 64); err != nil {
			return err
		}
		if err := validatePath(prefix+".path", license.Path); err != nil {
			return err
		}
	}
	if err := validatePositive("preprocessing.max_len", manifest.Preprocessing.MaxLen); err != nil {
		return err
	}
	if err := validatePositive("preprocessing.head_max_len", manifest.Preprocessing.HeadMaxLen); err != nil {
		return err
	}
	if err := validatePositive("preprocessing.option_max_len", manifest.Preprocessing.OptionMaxLen); err != nil {
		return err
	}
	if err := validatePositive("preprocessing.head_reserve", manifest.Preprocessing.HeadReserve); err != nil {
		return err
	}
	if manifest.Preprocessing.HeadMaxLen > manifest.Preprocessing.MaxLen {
		return invalid("manifest.json", "preprocessing.head_max_len", "at most max_len", fmt.Sprint(manifest.Preprocessing.HeadMaxLen), nil)
	}
	if err := validatePath("model.path", manifest.Model.Path); err != nil {
		return err
	}
	if len(manifest.Model.ExternalData) > maxExternalData {
		return invalid("manifest.json", "model.external_data", "at most 64 entries", fmt.Sprintf("%d entries", len(manifest.Model.ExternalData)), nil)
	}
	seenExternal := make(map[string]bool, len(manifest.Model.ExternalData))
	for index, path := range manifest.Model.ExternalData {
		field := fmt.Sprintf("model.external_data[%d]", index)
		if err := validatePath(field, path); err != nil {
			return err
		}
		folded := strings.ToLower(path)
		if seenExternal[folded] {
			return invalid("manifest.json", field, "unique case-insensitive path", scalar(path), nil)
		}
		seenExternal[folded] = true
	}
	if !slices.IsSorted(manifest.Model.ExternalData) {
		return invalid("manifest.json", "model.external_data", "sorted paths", "unsorted", nil)
	}
	if err := validatePath("tokenizer.json_path", manifest.Tokenizer.JSONPath); err != nil {
		return err
	}
	if err := validatePath("tokenizer.config_path", manifest.Tokenizer.ConfigPath); err != nil {
		return err
	}
	for field, tokenID := range map[string]int{
		"tokenizer.special_tokens.cls":  manifest.Tokenizer.SpecialTokens.CLS,
		"tokenizer.special_tokens.sep":  manifest.Tokenizer.SpecialTokens.SEP,
		"tokenizer.special_tokens.mask": manifest.Tokenizer.SpecialTokens.Mask,
		"tokenizer.special_tokens.pad":  manifest.Tokenizer.SpecialTokens.Pad,
	} {
		if tokenID < 0 {
			return invalid("manifest.json", field, "non-negative token ID", fmt.Sprint(tokenID), nil)
		}
	}
	if err := validatePath("calibration.config_path", manifest.Calibration.ConfigPath); err != nil {
		return err
	}
	for field, value := range map[string]float64{
		"calibration.temperature_min":     manifest.Calibration.TemperatureMin,
		"calibration.temperature_max":     manifest.Calibration.TemperatureMax,
		"calibration.invalid_temperature": manifest.Calibration.InvalidTemperature,
	} {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return invalid("manifest.json", field, "finite number", "non-finite", nil)
		}
	}
	if manifest.Calibration.TemperatureMin <= 0 || manifest.Calibration.TemperatureMax < manifest.Calibration.TemperatureMin {
		return invalid("manifest.json", "calibration", "positive ordered temperature bounds", "invalid bounds", nil)
	}
	if len(manifest.Calibration.Actions) == 0 || len(manifest.Calibration.Actions) > maxActions {
		return invalid("manifest.json", "calibration.actions", "1..32 entries", fmt.Sprintf("%d entries", len(manifest.Calibration.Actions)), nil)
	}
	seenActions := make(map[string]bool, len(manifest.Calibration.Actions))
	for index, action := range manifest.Calibration.Actions {
		prefix := fmt.Sprintf("calibration.actions[%d]", index)
		if action.Index != index {
			return invalid("manifest.json", prefix+".index", fmt.Sprint(index), fmt.Sprint(action.Index), nil)
		}
		if err := validateIdentifier(prefix+".id", action.ID); err != nil {
			return err
		}
		if seenActions[action.ID] {
			return invalid("manifest.json", prefix+".id", "unique action ID", scalar(action.ID), nil)
		}
		seenActions[action.ID] = true
	}
	if err := validateString("runtime.engine", manifest.Runtime.Engine, 64); err != nil {
		return err
	}
	if err := validateString("runtime.minimum_version", manifest.Runtime.MinimumVersion, 64); err != nil {
		return err
	}
	if err := validateString("runtime.maximum_exclusive", manifest.Runtime.MaximumExclusive, 64); err != nil {
		return err
	}
	if len(manifest.Runtime.ExecutionProviders) == 0 || len(manifest.Runtime.ExecutionProviders) > 16 {
		return invalid("manifest.json", "runtime.execution_providers", "1..16 entries", fmt.Sprintf("%d entries", len(manifest.Runtime.ExecutionProviders)), nil)
	}
	if len(manifest.Files) < 6 || len(manifest.Files) > maxFiles {
		return invalid("manifest.json", "files", "6..256 entries", fmt.Sprintf("%d entries", len(manifest.Files)), nil)
	}
	seenFiles := make(map[string]bool, len(manifest.Files))
	for index, file := range manifest.Files {
		prefix := fmt.Sprintf("files[%d]", index)
		if !validRole(file.Role) {
			return invalid("manifest.json", prefix+".role", "known artifact role", scalar(file.Role), nil)
		}
		if err := validatePath(prefix+".path", file.Path); err != nil {
			return err
		}
		folded := strings.ToLower(file.Path)
		if seenFiles[folded] {
			return invalid("manifest.json", prefix+".path", "unique case-insensitive path", scalar(file.Path), nil)
		}
		seenFiles[folded] = true
		if file.Size < 0 {
			return invalid("manifest.json", prefix+".size", "non-negative integer", fmt.Sprint(file.Size), nil)
		}
		if err := validateSHA256(prefix+".sha256", file.SHA256); err != nil {
			return err
		}
	}
	filePaths := make([]string, len(manifest.Files))
	for index := range manifest.Files {
		filePaths[index] = manifest.Files[index].Path
	}
	if !slices.IsSorted(filePaths) {
		return invalid("manifest.json", "files", "entries sorted by path", "unsorted", nil)
	}
	return nil
}

func validateString(field, value string, limit int) error {
	if value == "" {
		return invalid("manifest.json", field, "non-empty string", "empty", nil)
	}
	if len(value) > limit || len(value) > maxStringBytes {
		return invalid("manifest.json", field, fmt.Sprintf("at most %d bytes", min(limit, maxStringBytes)), fmt.Sprintf("%d bytes", len(value)), nil)
	}
	if !utf8.ValidString(value) {
		return invalid("manifest.json", field, "valid UTF-8", "invalid UTF-8", nil)
	}
	return nil
}

func validateIdentifier(field, value string) error {
	if err := validateString(field, value, 128); err != nil {
		return err
	}
	if !identifierPattern.MatchString(value) {
		return invalid("manifest.json", field, "identifier", scalar(value), nil)
	}
	return nil
}

func validateVersion(field, value string) error {
	if err := validateString(field, value, 64); err != nil {
		return err
	}
	if !versionPattern.MatchString(value) {
		return invalid("manifest.json", field, "version identifier", scalar(value), nil)
	}
	return nil
}

func validatePositive(field string, value int) error {
	if value <= 0 {
		return invalid("manifest.json", field, "positive integer", fmt.Sprint(value), nil)
	}
	return nil
}

func validateSHA256(field, value string) error {
	if !sha256Pattern.MatchString(value) {
		return invalid("manifest.json", field, "64 lowercase hexadecimal characters", scalar(value), nil)
	}
	return nil
}

func validatePath(field, path string) error {
	if len(path) > 255 {
		return invalid("manifest.json", field, "path at most 255 bytes", fmt.Sprintf("%d bytes", len(path)), nil)
	}
	if path == "." || !fs.ValidPath(path) || strings.Contains(path, `\`) {
		return invalid("manifest.json", field, "normalized contained slash-relative path", scalar(path), nil)
	}
	return nil
}

func validRole(role string) bool {
	switch role {
	case "model", "model-external-data", "tokenizer", "tokenizer-config", "agent-config", "license", "notice":
		return true
	default:
		return false
	}
}

func scalar(value string) string {
	if value == "" {
		return "empty"
	}
	if len(value) > 128 {
		return fmt.Sprintf("%d-byte string", len(value))
	}
	return value
}

func invalid(path, field, expected, observed string, cause error) error {
	return &Error{
		Path:     path,
		Field:    field,
		Expected: expected,
		Observed: observed,
		Kind:     ErrInvalid,
		Cause:    cause,
	}
}
