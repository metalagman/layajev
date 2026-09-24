// Command bundlecheck verifies one already-local complete Laya bundle.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/metalagman/layajev/internal/bundle"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "usage: bundlecheck BUNDLE_DIRECTORY")
		os.Exit(2)
	}
	manifest, err := bundle.Verify(context.Background(), os.DirFS(os.Args[1]), ".")
	if err != nil {
		fmt.Fprintf(os.Stderr, "bundlecheck: %v\n", err)
		os.Exit(1)
	}
	result := struct {
		BundleID string `json:"bundle_id"`
		Files    int    `json:"files"`
	}{
		BundleID: manifest.ID(),
		Files:    len(manifest.Files),
	}
	if err := json.NewEncoder(os.Stdout).Encode(result); err != nil {
		fmt.Fprintf(os.Stderr, "bundlecheck: encode result: %v\n", err)
		os.Exit(1)
	}
}
