package bundle

// Manifest is the parsed Laya Go Bundle v1 manifest. Values are owned by the
// manifest and do not alias the input byte slice passed to Parse.
type Manifest struct {
	SchemaVersion int            `json:"schema_version"`
	Bundle        BundleIdentity `json:"bundle"`
	Provenance    Provenance     `json:"provenance"`
	Preprocessing Preprocessing  `json:"preprocessing"`
	Model         Model          `json:"model"`
	Tokenizer     Tokenizer      `json:"tokenizer"`
	Calibration   Calibration    `json:"calibration"`
	Runtime       Runtime        `json:"runtime"`
	Files         []File         `json:"files"`

	id string
}

// ID returns the content identity derived from the exact manifest bytes.
func (m Manifest) ID() string { return m.id }

// BundleIdentity identifies the logical model bundle.
type BundleIdentity struct {
	ID      string `json:"id"`
	Version string `json:"version"`
}

// Provenance binds a bundle to its official input and exporter.
type Provenance struct {
	SourceModel     SourceModel     `json:"source_model"`
	SDK             SDK             `json:"sdk"`
	Profile         ProfileIdentity `json:"profile"`
	Exporter        Exporter        `json:"exporter"`
	Lock            Lock            `json:"lock"`
	CreatedAt       string          `json:"created_at"`
	SourceDateEpoch int64           `json:"source_date_epoch"`
	Redistribution  string          `json:"redistribution"`
	Licenses        []License       `json:"licenses"`
}

// SourceModel identifies the immutable official model snapshot.
type SourceModel struct {
	ID       string `json:"id"`
	Revision string `json:"revision"`
}

// SDK identifies the upstream behavior used for reference parity.
type SDK struct {
	Repository string `json:"repository"`
	Revision   string `json:"revision"`
	Version    string `json:"version"`
}

// ProfileIdentity identifies the local export profile and its exact bytes.
type ProfileIdentity struct {
	ID     string `json:"id"`
	SHA256 string `json:"sha256"`
}

// Exporter identifies the exact exporter implementation.
type Exporter struct {
	Name         string `json:"name"`
	Version      string `json:"version"`
	SourceSHA256 string `json:"source_sha256"`
	GitRevision  string `json:"git_revision"`
}

// Lock identifies the exact build-time dependency lock.
type Lock struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
}

// License identifies a component license file in the bundle.
type License struct {
	Component string `json:"component"`
	SPDX      string `json:"spdx"`
	Path      string `json:"path"`
}

// Preprocessing declares the immutable input construction contract.
type Preprocessing struct {
	Version        int    `json:"version"`
	MaxLen         int    `json:"max_len"`
	HeadMaxLen     int    `json:"head_max_len"`
	OptionMaxLen   int    `json:"option_max_len"`
	HeadReserve    int    `json:"head_reserve"`
	TruncationSide string `json:"truncation_side"`
	QTypes         QTypes `json:"qtypes"`
}

// QTypes maps public question kinds to model tensor values.
type QTypes struct {
	Choice int `json:"choice"`
	Score  int `json:"score"`
	Noul   int `json:"noul"`
}

// Model declares the complete ONNX graph contract.
type Model struct {
	Path         string             `json:"path"`
	Format       string             `json:"format"`
	Opset        int                `json:"opset"`
	Precision    string             `json:"precision"`
	ExternalData []string           `json:"external_data"`
	Inputs       []TensorDescriptor `json:"inputs"`
	Outputs      []TensorDescriptor `json:"outputs"`
}

// TensorDescriptor identifies one ordered graph input or output.
type TensorDescriptor struct {
	Name       string   `json:"name"`
	DType      string   `json:"dtype"`
	Dimensions []string `json:"dimensions"`
}

// Tokenizer identifies tokenizer artifacts and required special-token IDs.
type Tokenizer struct {
	JSONPath      string        `json:"json_path"`
	ConfigPath    string        `json:"config_path"`
	SpecialTokens SpecialTokens `json:"special_tokens"`
}

// SpecialTokens contains the tokenizer IDs used by preprocessing.
type SpecialTokens struct {
	CLS  int `json:"cls"`
	SEP  int `json:"sep"`
	Mask int `json:"mask"`
	Pad  int `json:"pad"`
}

// Calibration declares temperature and action-output semantics.
type Calibration struct {
	ConfigPath         string   `json:"config_path"`
	TemperatureMin     float64  `json:"temperature_min"`
	TemperatureMax     float64  `json:"temperature_max"`
	InvalidTemperature float64  `json:"invalid_temperature"`
	SelectionOrder     []string `json:"selection_order"`
	Actions            []Action `json:"actions"`
}

// Action maps an action output index to its stable identifier.
type Action struct {
	Index int    `json:"index"`
	ID    string `json:"id"`
}

// Runtime declares native runtime compatibility.
type Runtime struct {
	Engine             string   `json:"engine"`
	MinimumVersion     string   `json:"minimum_version"`
	MaximumExclusive   string   `json:"maximum_exclusive"`
	ExecutionProviders []string `json:"execution_providers"`
}

// File is one content-addressed non-manifest bundle artifact.
type File struct {
	Role   string `json:"role"`
	Path   string `json:"path"`
	Size   int64  `json:"size"`
	SHA256 string `json:"sha256"`
}
