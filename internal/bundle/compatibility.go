package bundle

import (
	"fmt"
	"slices"
)

const (
	SupportedSchemaVersion        = 1
	SupportedProfileID            = "laya-multilingual-v1"
	SupportedProfileSHA256        = "2dc717bbbf2bacb9c00d57f2ba7279c6a4332900bca84d12c000ebd6401a0235"
	SupportedSourceModelID        = "convaiinnovations/laya-multilingual"
	SupportedSourceModelRevision  = "052592a15d198d9ad47da779604259b10b47b7aa"
	SupportedSDKRepository        = "NandhaKishorM/laya"
	SupportedSDKRevision          = "573e5b62696ba441230cd6be71d593331b5d23af"
	SupportedSDKVersion           = "0.3.5"
	SupportedPreprocessingVersion = 1
	SupportedOpset                = 18
	SupportedPrecision            = "fp32"
	SupportedRuntimeMinimum       = "1.20.0"
	SupportedRuntimeMaximum       = "2.0.0"
)

var (
	supportedInputs = []TensorDescriptor{
		{Name: "input_ids", DType: "int64", Dimensions: []string{"batch", "sequence"}},
		{Name: "attention_mask", DType: "int64", Dimensions: []string{"batch", "sequence"}},
		{Name: "marker_pos", DType: "int64", Dimensions: []string{"batch", "markers"}},
		{Name: "marker_mask", DType: "bool", Dimensions: []string{"batch", "markers"}},
		{Name: "qtype", DType: "int64", Dimensions: []string{"batch"}},
	}
	supportedOutputs = []TensorDescriptor{
		{Name: "logits", DType: "float32", Dimensions: []string{"batch", "markers"}},
		{Name: "act_logits", DType: "float32", Dimensions: []string{"batch", "actions"}},
	}
)

func validateCompatibility(manifest Manifest) error {
	checks := []struct {
		field    string
		observed any
		expected any
	}{
		{"schema_version", manifest.SchemaVersion, SupportedSchemaVersion},
		{"provenance.source_model.id", manifest.Provenance.SourceModel.ID, SupportedSourceModelID},
		{"provenance.source_model.revision", manifest.Provenance.SourceModel.Revision, SupportedSourceModelRevision},
		{"provenance.sdk.repository", manifest.Provenance.SDK.Repository, SupportedSDKRepository},
		{"provenance.sdk.revision", manifest.Provenance.SDK.Revision, SupportedSDKRevision},
		{"provenance.sdk.version", manifest.Provenance.SDK.Version, SupportedSDKVersion},
		{"provenance.profile.id", manifest.Provenance.Profile.ID, SupportedProfileID},
		{"provenance.profile.sha256", manifest.Provenance.Profile.SHA256, SupportedProfileSHA256},
		{"provenance.exporter.name", manifest.Provenance.Exporter.Name, "laya-go-export"},
		{"provenance.lock.path", manifest.Provenance.Lock.Path, "tools/export/uv.lock"},
		{"preprocessing.version", manifest.Preprocessing.Version, SupportedPreprocessingVersion},
		{"preprocessing.max_len", manifest.Preprocessing.MaxLen, 1024},
		{"preprocessing.head_max_len", manifest.Preprocessing.HeadMaxLen, 256},
		{"preprocessing.option_max_len", manifest.Preprocessing.OptionMaxLen, 48},
		{"preprocessing.head_reserve", manifest.Preprocessing.HeadReserve, 16},
		{"preprocessing.truncation_side", manifest.Preprocessing.TruncationSide, "right"},
		{"preprocessing.qtypes.choice", manifest.Preprocessing.QTypes.Choice, 0},
		{"preprocessing.qtypes.score", manifest.Preprocessing.QTypes.Score, 1},
		{"preprocessing.qtypes.noul", manifest.Preprocessing.QTypes.Noul, 2},
		{"model.path", manifest.Model.Path, "model.onnx"},
		{"model.format", manifest.Model.Format, "onnx"},
		{"model.opset", manifest.Model.Opset, SupportedOpset},
		{"model.precision", manifest.Model.Precision, SupportedPrecision},
		{"tokenizer.json_path", manifest.Tokenizer.JSONPath, "tokenizer/tokenizer.json"},
		{"tokenizer.config_path", manifest.Tokenizer.ConfigPath, "tokenizer/tokenizer_config.json"},
		{"calibration.config_path", manifest.Calibration.ConfigPath, "rl_agent_config.json"},
		{"calibration.temperature_min", manifest.Calibration.TemperatureMin, 0.5},
		{"calibration.temperature_max", manifest.Calibration.TemperatureMax, 5.0},
		{"calibration.invalid_temperature", manifest.Calibration.InvalidTemperature, 1.0},
		{"runtime.engine", manifest.Runtime.Engine, "onnxruntime"},
		{"runtime.minimum_version", manifest.Runtime.MinimumVersion, SupportedRuntimeMinimum},
		{"runtime.maximum_exclusive", manifest.Runtime.MaximumExclusive, SupportedRuntimeMaximum},
	}
	for _, check := range checks {
		if fmt.Sprint(check.observed) != fmt.Sprint(check.expected) {
			return unsupported(check.field, fmt.Sprint(check.expected), fmt.Sprint(check.observed))
		}
	}
	if !equalTensorDescriptors(manifest.Model.Inputs, supportedInputs) {
		return unsupported("model.inputs", tensorSummary(supportedInputs), tensorSummary(manifest.Model.Inputs))
	}
	if !equalTensorDescriptors(manifest.Model.Outputs, supportedOutputs) {
		return unsupported("model.outputs", tensorSummary(supportedOutputs), tensorSummary(manifest.Model.Outputs))
	}
	if !slices.Equal(manifest.Calibration.SelectionOrder, []string{"option_bucket", "qtype", "default"}) {
		return unsupported("calibration.selection_order", "option_bucket,qtype,default", fmt.Sprint(manifest.Calibration.SelectionOrder))
	}
	if len(manifest.Calibration.Actions) != 1 || manifest.Calibration.Actions[0] != (Action{Index: 0, ID: "escalate"}) {
		return unsupported("calibration.actions", "0:escalate", fmt.Sprint(manifest.Calibration.Actions))
	}
	if !slices.Equal(manifest.Runtime.ExecutionProviders, []string{"CPUExecutionProvider"}) {
		return unsupported("runtime.execution_providers", "CPUExecutionProvider", fmt.Sprint(manifest.Runtime.ExecutionProviders))
	}
	wantLicenses := []License{
		{Component: "source-model", SPDX: "Apache-2.0", Path: "LICENSE.model"},
		{Component: "sdk-derived-exporter-code", SPDX: "Apache-2.0", Path: "LICENSE.sdk"},
	}
	if !slices.Equal(manifest.Provenance.Licenses, wantLicenses) {
		return unsupported("provenance.licenses", fmt.Sprint(wantLicenses), fmt.Sprint(manifest.Provenance.Licenses))
	}
	if manifest.Provenance.Redistribution != "approved" {
		return unsupported("provenance.redistribution", "approved", manifest.Provenance.Redistribution)
	}
	return nil
}

func equalTensorDescriptors(got, want []TensorDescriptor) bool {
	if len(got) != len(want) {
		return false
	}
	for index := range want {
		if got[index].Name != want[index].Name || got[index].DType != want[index].DType || !slices.Equal(got[index].Dimensions, want[index].Dimensions) {
			return false
		}
	}
	return true
}

func tensorSummary(descriptors []TensorDescriptor) string {
	result := ""
	for index, descriptor := range descriptors {
		if index > 0 {
			result += ","
		}
		result += descriptor.Name + ":" + descriptor.DType + fmt.Sprint(descriptor.Dimensions)
	}
	return result
}

func unsupported(field, expected, observed string) error {
	return &Error{
		Path:     "manifest.json",
		Field:    field,
		Expected: expected,
		Observed: observed,
		Kind:     ErrUnsupported,
	}
}
