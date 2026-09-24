// Package bundle implements the versioned, source-neutral Laya bundle
// contract. It performs no model acquisition or native loading.
package bundle

import (
	"errors"
	"fmt"
	"strings"
)

var (
	// ErrInvalid indicates malformed or semantically invalid bundle metadata.
	ErrInvalid = errors.New("laya: invalid model bundle")
	// ErrIntegrity indicates a declared artifact does not match its identity.
	ErrIntegrity = errors.New("laya: model bundle integrity failure")
	// ErrUnsupported indicates a valid declaration is not supported by this build.
	ErrUnsupported = errors.New("laya: unsupported model bundle")
)

// Error describes a bundle failure without retaining artifact or request
// content. Path and Field are logical metadata locations. Expected and
// Observed contain bounded scalar descriptions, never file content.
type Error struct {
	Path     string
	Field    string
	Expected string
	Observed string
	Kind     error
	Cause    error
}

// Error returns a redacted description of the bundle failure.
func (e *Error) Error() string {
	if e == nil {
		return ErrInvalid.Error()
	}
	kind := e.Kind
	if kind == nil {
		kind = ErrInvalid
	}
	parts := []string{kind.Error()}
	if e.Path != "" {
		parts = append(parts, "path "+quoted(e.Path))
	}
	if e.Field != "" {
		parts = append(parts, "field "+quoted(e.Field))
	}
	if e.Expected != "" {
		parts = append(parts, "expected "+quoted(e.Expected))
	}
	if e.Observed != "" {
		parts = append(parts, "observed "+quoted(e.Observed))
	}
	if e.Cause != nil {
		parts = append(parts, fmt.Sprintf("cause %T", e.Cause))
	}
	return strings.Join(parts, ": ")
}

// Unwrap exposes both the caller-actionable category and underlying cause.
func (e *Error) Unwrap() []error {
	if e == nil {
		return nil
	}
	errs := make([]error, 0, 2)
	if e.Kind != nil {
		errs = append(errs, e.Kind)
	}
	if e.Cause != nil {
		errs = append(errs, e.Cause)
	}
	return errs
}

func quoted(value string) string {
	const maxLen = 128
	if len(value) > maxLen {
		value = value[:maxLen] + "…"
	}
	return fmt.Sprintf("%q", value)
}
