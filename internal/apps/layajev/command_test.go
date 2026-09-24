package layajev

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) { return f(request) }

func TestCommandHelp(t *testing.T) {
	t.Parallel()
	command := NewCommand()
	var output bytes.Buffer
	command.SetOut(&output)
	command.SetErr(&output)
	command.SetArgs([]string{"--help"})
	if err := command.ExecuteContext(context.Background()); err != nil {
		t.Fatalf("Execute help: %v", err)
	}
	for _, subcommand := range []string{"serve", "doctor", "fetch", "convert"} {
		if !strings.Contains(output.String(), subcommand) {
			t.Errorf("help missing %q: %s", subcommand, output.String())
		}
	}
}

func TestPackagedONNXRuntimeLibrary(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	executable := filepath.Join(dir, "layajev")
	library := filepath.Join(dir, "libonnxruntime.so.1.29.0")
	if got := packagedONNXRuntimeLibrary(executable); got != "" {
		t.Errorf("missing packaged library = %q, want empty", got)
	}
	if err := os.Mkdir(library, 0o755); err != nil {
		t.Fatalf("create library directory: %v", err)
	}
	if got := packagedONNXRuntimeLibrary(executable); got != "" {
		t.Errorf("directory packaged library = %q, want empty", got)
	}
	if err := os.Remove(library); err != nil {
		t.Fatalf("remove library directory: %v", err)
	}
	if err := os.WriteFile(library, []byte("test library"), 0o644); err != nil {
		t.Fatalf("create library file: %v", err)
	}
	if got := packagedONNXRuntimeLibrary(executable); got != library {
		t.Errorf("packaged library = %q, want %q", got, library)
	}
}

func TestNativeCommandsDisableTelemetry(t *testing.T) {
	for _, test := range []struct {
		name       string
		command    []string
		initialEnv string
	}{
		{name: "doctor absent", command: []string{"doctor"}},
		{name: "doctor conflicting", command: []string{"doctor"}, initialEnv: "0"},
		{name: "serve absent", command: []string{"serve", "--bundle", "missing"}},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Setenv("ORT_DISABLE_TELEMETRY", test.initialEnv)
			if test.initialEnv == "" {
				if err := os.Unsetenv("ORT_DISABLE_TELEMETRY"); err != nil {
					t.Fatalf("unset telemetry environment: %v", err)
				}
			}
			t.Setenv("LAYA_ONNXRUNTIME_LIBRARY", "/nonexistent/native-library.so")
			command := NewCommand()
			command.SetArgs(test.command)
			command.SetOut(io.Discard)
			command.SetErr(io.Discard)
			if err := command.ExecuteContext(context.Background()); err == nil {
				t.Fatal("native command unexpectedly succeeded with an invalid library")
			}
			if got := os.Getenv("ORT_DISABLE_TELEMETRY"); got != "1" {
				t.Errorf("telemetry opt-out = %q, want 1", got)
			}
		})
	}
}

func TestFetchPinned(t *testing.T) {
	t.Parallel()
	content := []byte("official pinned artifact")
	digest := sha256.Sum256(content)
	profile := pinnedProfile{SourceFiles: []sourceFile{{Path: "tokenizer/config.json", Size: int64(len(content)), SHA256: hex.EncodeToString(digest[:])}}}
	profile.SourceModel.ID = officialModelID
	profile.SourceModel.Revision = officialRevision
	client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if !strings.HasSuffix(r.URL.Path, "/tokenizer/config.json") {
			t.Errorf("request path = %q", r.URL.Path)
		}
		return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(bytes.NewReader(content))}, nil
	})}
	destination := filepath.Join(t.TempDir(), "snapshot")
	if err := fetchPinnedFrom(context.Background(), profile, destination, client, "https://example.test"); err != nil {
		t.Fatalf("fetchPinnedFrom: %v", err)
	}
	got, err := os.ReadFile(filepath.Join(destination, "tokenizer", "config.json"))
	if err != nil || !bytes.Equal(got, content) {
		t.Fatalf("fetched artifact = %q, error %v", got, err)
	}
	if err := fetchPinnedFrom(context.Background(), profile, destination, client, "https://example.test"); err == nil {
		t.Error("fetch into existing destination succeeded")
	}
}

func TestFetchRejectsChecksumMismatch(t *testing.T) {
	t.Parallel()
	profile := pinnedProfile{SourceFiles: []sourceFile{{Path: "file.txt", Size: 3, SHA256: strings.Repeat("0", 64)}}}
	profile.SourceModel.ID = officialModelID
	profile.SourceModel.Revision = officialRevision
	client := &http.Client{Transport: roundTripFunc(func(_ *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader("abc"))}, nil
	})}
	destination := filepath.Join(t.TempDir(), "snapshot")
	if err := fetchPinnedFrom(context.Background(), profile, destination, client, "https://example.test"); err == nil || !strings.Contains(err.Error(), "SHA-256 mismatch") {
		t.Errorf("fetch error = %v, want SHA-256 mismatch", err)
	}
	if _, err := os.Lstat(destination); !os.IsNotExist(err) {
		t.Errorf("destination exists after failed verification: %v", err)
	}
}

func TestFetchProgress(t *testing.T) {
	t.Parallel()
	content := bytes.Repeat([]byte("a"), 192<<10)
	digest := sha256.Sum256(content)
	profile := pinnedProfile{SourceFiles: []sourceFile{{Path: "tokenizer/config.json", Size: int64(len(content)), SHA256: hex.EncodeToString(digest[:])}}}
	profile.SourceModel.ID = officialModelID
	profile.SourceModel.Revision = officialRevision
	client := &http.Client{Transport: roundTripFunc(func(_ *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(bytes.NewReader(content))}, nil
	})}
	var output bytes.Buffer
	destination := filepath.Join(t.TempDir(), "snapshot")
	if err := fetchPinnedFrom(context.Background(), profile, destination, client, "https://example.test", &fetchProgress{writer: &output}); err != nil {
		t.Fatalf("fetchPinnedFrom: %v", err)
	}
	progress := output.String()
	for _, item := range []string{"downloading \"tokenizer/config.json\"", "65536/196608 bytes", "196608/196608 bytes (100%)", "verifying \"tokenizer/config.json\"", "verified snapshot ready"} {
		if !strings.Contains(progress, item) {
			t.Errorf("progress missing %q: %s", item, progress)
		}
	}
	if strings.Index(progress, "verifying") > strings.Index(progress, "verified snapshot ready") {
		t.Errorf("success preceded verification: %s", progress)
	}
}

func TestFetchProgressFailureAndCancellation(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		name   string
		cancel bool
	}{
		{name: "checksum mismatch"},
		{name: "canceled", cancel: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			profile := pinnedProfile{SourceFiles: []sourceFile{{Path: "file.txt", Size: 3, SHA256: strings.Repeat("0", 64)}}}
			profile.SourceModel.ID = officialModelID
			profile.SourceModel.Revision = officialRevision
			client := &http.Client{Transport: roundTripFunc(func(_ *http.Request) (*http.Response, error) {
				if test.cancel {
					cancel()
				}
				return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader("abc"))}, nil
			})}
			var output bytes.Buffer
			destination := filepath.Join(t.TempDir(), "snapshot")
			err := fetchPinnedFrom(ctx, profile, destination, client, "https://example.test", &fetchProgress{writer: &output})
			if err == nil {
				t.Fatal("fetch succeeded unexpectedly")
			}
			if test.cancel && !errors.Is(err, context.Canceled) {
				t.Errorf("cancellation error = %v, want context.Canceled", err)
			}
			if strings.Contains(output.String(), "verified snapshot ready") {
				t.Errorf("false success: %s", output.String())
			}
			if _, err := os.Lstat(destination); !os.IsNotExist(err) {
				t.Errorf("destination exists after failure: %v", err)
			}
		})
	}
}

func TestFetchProgressUnknownTotal(t *testing.T) {
	t.Parallel()
	emptyDigest := sha256.Sum256(nil)
	profile := pinnedProfile{SourceFiles: []sourceFile{{Path: "empty.txt", Size: 0, SHA256: hex.EncodeToString(emptyDigest[:])}}}
	profile.SourceModel.ID = officialModelID
	profile.SourceModel.Revision = officialRevision
	client := &http.Client{Transport: roundTripFunc(func(_ *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(""))}, nil
	})}
	var fetchOutput bytes.Buffer
	if err := fetchPinnedFrom(context.Background(), profile, filepath.Join(t.TempDir(), "snapshot"), client, "https://example.test", &fetchProgress{writer: &fetchOutput}); err != nil {
		t.Fatalf("fetch zero-size file: %v", err)
	}
	if !strings.Contains(fetchOutput.String(), "\"empty.txt\": 0 bytes") || strings.Contains(fetchOutput.String(), "%") {
		t.Errorf("zero-size file progress = %q", fetchOutput.String())
	}
	var output bytes.Buffer
	(&fetchProgress{writer: &output}).bytes("empty", 0, 0)
	if got := output.String(); got != "fetch: \"empty\": 0 bytes\n" {
		t.Errorf("unknown total progress = %q", got)
	}
}

func TestConvertValidation(t *testing.T) {
	t.Parallel()
	if err := convert(context.Background(), t.TempDir(), "", "", "", 0); err == nil {
		t.Error("convert accepted empty inputs")
	}
}

func TestConvertProgressStreams(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "Taskfile.yml"), []byte("version: '3'\n"), 0o644); err != nil {
		t.Fatalf("write Taskfile: %v", err)
	}
	bin := t.TempDir()
	script := "#!/bin/sh\n[ \"$1\" = --silent ] && [ \"$2\" = bundle:export ] || exit 9\nprintf '{\"bundle\":\"test\"}\\n'\nprintf 'convert: exporting ONNX graph\\n' >&2\n[ \"$FAKE_TASK_FAIL\" = 1 ] && exit 7\nexit 0\n"
	if err := os.WriteFile(filepath.Join(bin, "task"), []byte(script), 0o755); err != nil {
		t.Fatalf("write fake task: %v", err)
	}
	t.Setenv("PATH", bin+string(os.PathListSeparator)+os.Getenv("PATH"))
	source := t.TempDir()
	sdk := t.TempDir()
	for _, test := range []struct {
		name   string
		fail   bool
		cancel bool
	}{
		{name: "success"},
		{name: "failure", fail: true},
		{name: "cancellation", cancel: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			if test.fail {
				t.Setenv("FAKE_TASK_FAIL", "1")
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if test.cancel {
				cancel()
			}
			var stdout, stderr bytes.Buffer
			output := filepath.Join(t.TempDir(), "bundle")
			err := convertWithStreams(ctx, root, source, sdk, output, 1, &stdout, &stderr)
			if test.fail || test.cancel {
				if err == nil {
					t.Fatal("convert succeeded unexpectedly")
				}
				if strings.Contains(stderr.String(), "bundle ready") {
					t.Errorf("false success on stderr: %s", stderr.String())
				}
				return
			}
			if err != nil {
				t.Fatalf("convertWithStreams: %v", err)
			}
			if got := stdout.String(); got != "{\"bundle\":\"test\"}\n" {
				t.Errorf("stdout = %q, want only JSON", got)
			}
			for _, item := range []string{"starting offline export", "exporting ONNX graph", "bundle ready at"} {
				if !strings.Contains(stderr.String(), item) {
					t.Errorf("stderr missing %q: %s", item, stderr.String())
				}
			}
			if strings.Contains(stderr.String(), "%") || strings.Contains(stderr.String(), "ETA") {
				t.Errorf("stderr claims unknown precision: %s", stderr.String())
			}
			command := NewCommand()
			var commandStdout, commandStderr bytes.Buffer
			command.SetOut(&commandStdout)
			command.SetErr(&commandStderr)
			command.SetArgs([]string{"convert", "--repository-root", root, "--source", source, "--sdk", sdk, "--output", output, "--epoch", "1"})
			if err := command.ExecuteContext(context.Background()); err != nil {
				t.Fatalf("execute convert command: %v", err)
			}
			if commandStdout.String() != stdout.String() || !strings.Contains(commandStderr.String(), "bundle ready at") {
				t.Errorf("command streams: stdout=%q stderr=%q", commandStdout.String(), commandStderr.String())
			}
		})
	}
}

func TestLoadPinnedProfile(t *testing.T) {
	t.Parallel()
	profile, err := loadPinnedProfile("")
	if err != nil {
		t.Fatalf("loadPinnedProfile: %v", err)
	}
	if profile.SourceModel.ID != officialModelID || profile.SourceModel.Revision != officialRevision || len(profile.SourceFiles) == 0 {
		t.Errorf("unexpected pinned source: %+v", profile.SourceModel)
	}
	data, err := os.ReadFile("../../../tools/export/profiles/laya-multilingual-v1.json")
	if err != nil {
		t.Fatalf("read canonical export profile: %v", err)
	}
	var canonical pinnedProfile
	if err := json.Unmarshal(data, &canonical); err != nil {
		t.Fatalf("parse canonical export profile: %v", err)
	}
	if !reflect.DeepEqual(profile, canonical) {
		t.Error("embedded fetch source differs from canonical export profile")
	}
	fromCheckout, err := loadPinnedProfile(filepath.Join("..", "..", ".."))
	if err != nil {
		t.Fatalf("load checkout profile: %v", err)
	}
	if !reflect.DeepEqual(profile, fromCheckout) {
		t.Error("embedded fetch source differs from checkout profile")
	}
}

func TestFetchStandalone(t *testing.T) {
	if os.Getenv("LAYAJEV_TEST_STANDALONE") == "1" {
		command := NewCommand()
		command.SetArgs([]string{"fetch", "--destination", "."})
		err := command.ExecuteContext(context.Background())
		if err == nil || !strings.Contains(err.Error(), "already exists") {
			t.Fatalf("fetch outside checkout error = %v, want existing destination", err)
		}
		return
	}
	command := exec.Command(os.Args[0], "-test.run=^TestFetchStandalone$")
	command.Dir = t.TempDir()
	command.Env = append(os.Environ(), "LAYAJEV_TEST_STANDALONE=1")
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("fetch outside checkout: %v\n%s", err, output)
	}
}
