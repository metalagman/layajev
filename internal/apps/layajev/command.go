package layajev

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"

	"github.com/metalagman/laya-go"
	"github.com/spf13/cobra"
)

// NewCommand builds the layajev command tree without taking process ownership.
// The caller owns signal handling, command execution, and the process exit code.
func NewCommand() *cobra.Command {
	root := &cobra.Command{Use: "layajev", Short: "Serve a local Laya model with a Jev-compatible API subset"}
	root.AddCommand(newServeCommand(), newDoctorCommand(), newFetchCommand(), newConvertCommand())
	return root
}

func newServeCommand() *cobra.Command {
	var cfg ServeConfig
	command := &cobra.Command{
		Use: "serve", Short: "Serve an already-local verified bundle",
		RunE: func(cmd *cobra.Command, _ []string) error {
			if err := configureNativeRuntime(); err != nil {
				return fmt.Errorf("serve: %w", err)
			}
			return Serve(cmd.Context(), cfg)
		},
	}
	command.Flags().StringVar(&cfg.BundleDir, "bundle", "", "directory containing the converted local bundle")
	command.Flags().StringVar(&cfg.ListenAddr, "listen", "127.0.0.1:8080", "TCP address to listen on")
	command.Flags().BoolVar(&cfg.AllowRemote, "allow-remote", false, "allow non-loopback bind; provide TLS and authentication upstream")
	command.Flags().IntVar(&cfg.QueueSize, "queue-size", 16, "maximum waiting local inference requests")
	_ = command.MarkFlagRequired("bundle")
	return command
}

func newDoctorCommand() *cobra.Command {
	return &cobra.Command{
		Use: "doctor", Short: "Verify the local native runtime without a model bundle",
		RunE: func(cmd *cobra.Command, _ []string) error {
			if err := configureNativeRuntime(); err != nil {
				return fmt.Errorf("doctor: %w", err)
			}
			runtime, err := laya.NewRuntime(cmd.Context(), laya.RuntimeOptions{})
			if err != nil {
				return fmt.Errorf("doctor: open native runtime: %w", err)
			}
			if err := runtime.Close(cmd.Context()); err != nil {
				return fmt.Errorf("doctor: close native runtime: %w", err)
			}
			_, err = fmt.Fprintln(cmd.OutOrStdout(), "native runtime OK")
			return err
		},
	}
}

// configureNativeRuntime establishes CLI-owned process policy before opening
// the native runtime. The root laya library never changes its caller's env.
func configureNativeRuntime() error {
	if err := os.Setenv("ORT_DISABLE_TELEMETRY", "1"); err != nil {
		return fmt.Errorf("disable ONNX Runtime telemetry: %w", err)
	}
	if os.Getenv("LAYA_ONNXRUNTIME_LIBRARY") != "" {
		return nil
	}
	executable, err := os.Executable()
	if err != nil {
		return fmt.Errorf("locate executable: %w", err)
	}
	if library := packagedONNXRuntimeLibrary(executable); library != "" {
		if err := os.Setenv("LAYA_ONNXRUNTIME_LIBRARY", library); err != nil {
			return fmt.Errorf("configure packaged ONNX Runtime: %w", err)
		}
	}
	return nil
}

func packagedONNXRuntimeLibrary(executable string) string {
	path := filepath.Join(filepath.Dir(executable), "libonnxruntime.so.1.29.0")
	info, err := os.Stat(path)
	if err != nil || !info.Mode().IsRegular() {
		return ""
	}
	return path
}

func newFetchCommand() *cobra.Command {
	var repositoryRoot, destination string
	command := &cobra.Command{
		Use: "fetch", Short: "Fetch and verify the pinned official source snapshot",
		RunE: func(cmd *cobra.Command, _ []string) error {
			profile, err := loadPinnedProfile(repositoryRoot)
			if err != nil {
				return err
			}
			return fetchPinned(cmd.Context(), profile, destination, nil, &fetchProgress{writer: cmd.ErrOrStderr()})
		},
	}
	command.Flags().StringVar(&repositoryRoot, "repository-root", "", "optional layajev checkout whose pinned export profile overrides the embedded profile")
	command.Flags().StringVar(&destination, "destination", "", "new directory for the verified source snapshot")
	_ = command.MarkFlagRequired("destination")
	return command
}

func newConvertCommand() *cobra.Command {
	var repositoryRoot, sourceDir, sdkDir, outputDir string
	var epoch int64
	command := &cobra.Command{
		Use: "convert", Short: "Convert a pinned local source with the offline Taskfile exporter",
		RunE: func(cmd *cobra.Command, _ []string) error {
			return convertWithStreams(cmd.Context(), repositoryRoot, sourceDir, sdkDir, outputDir, epoch, cmd.OutOrStdout(), cmd.ErrOrStderr())
		},
	}
	command.Flags().StringVar(&repositoryRoot, "repository-root", ".", "layajev checkout containing Taskfile.yml")
	command.Flags().StringVar(&sourceDir, "source", "", "verified official source snapshot directory")
	command.Flags().StringVar(&sdkDir, "sdk", "", "verified pinned reference SDK directory")
	command.Flags().StringVar(&outputDir, "output", "", "new local bundle directory")
	command.Flags().Int64Var(&epoch, "epoch", 0, "SOURCE_DATE_EPOCH for reproducible bundle provenance")
	for _, flag := range []string{"source", "sdk", "output", "epoch"} {
		_ = command.MarkFlagRequired(flag)
	}
	return command
}

func convert(ctx context.Context, repositoryRoot, sourceDir, sdkDir, outputDir string, epoch int64) error {
	return convertWithStreams(ctx, repositoryRoot, sourceDir, sdkDir, outputDir, epoch, os.Stdout, os.Stderr)
}

func convertWithStreams(ctx context.Context, repositoryRoot, sourceDir, sdkDir, outputDir string, epoch int64, stdout, stderr io.Writer) error {
	if sourceDir == "" || sdkDir == "" || outputDir == "" || epoch <= 0 {
		return fmt.Errorf("convert: source, SDK, output and positive epoch are required")
	}
	root, err := filepath.Abs(repositoryRoot)
	if err != nil {
		return fmt.Errorf("convert: resolve repository root: %w", err)
	}
	if _, err := os.Stat(filepath.Join(root, "Taskfile.yml")); err != nil {
		return fmt.Errorf("convert: repository Taskfile: %w", err)
	}
	for _, path := range []string{sourceDir, sdkDir} {
		info, err := os.Stat(path)
		if err != nil || !info.IsDir() {
			return fmt.Errorf("convert: input %q must be an existing directory", path)
		}
	}
	if _, err := os.Lstat(outputDir); err == nil {
		return fmt.Errorf("convert: output %q already exists", outputDir)
	} else if !os.IsNotExist(err) {
		return fmt.Errorf("convert: inspect output: %w", err)
	}
	var paths [3]string
	for i, path := range []string{sourceDir, sdkDir, outputDir} {
		paths[i], err = filepath.Abs(path)
		if err != nil {
			return fmt.Errorf("convert: resolve path %q: %w", path, err)
		}
	}
	_, _ = fmt.Fprintln(stderr, "convert: starting offline export")
	command := exec.CommandContext(ctx, "task", "--silent", "bundle:export")
	command.Dir = root
	command.Env = append(os.Environ(),
		"LAYA_SOURCE_DIR="+paths[0],
		"LAYA_SDK_DIR="+paths[1],
		"LAYA_BUNDLE_DIR="+paths[2],
		"SOURCE_DATE_EPOCH="+strconv.FormatInt(epoch, 10),
	)
	command.Stdout = stdout
	command.Stderr = stderr
	if err := command.Run(); err != nil {
		return fmt.Errorf("convert with Taskfile exporter: %w", err)
	}
	_, _ = fmt.Fprintf(stderr, "convert: bundle ready at %q\n", outputDir)
	return nil
}
