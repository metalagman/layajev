package layajev

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/metalagman/laya-go/adklaya"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/workflow"
)

func TestServeConfigValidate(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name      string
		cfg       ServeConfig
		wantError bool
	}{
		{"default loopback", ServeConfig{BundleDir: "/bundle"}, false},
		{"explicit loopback", ServeConfig{BundleDir: "/bundle", ListenAddr: "[::1]:8080"}, false},
		{"remote rejected", ServeConfig{BundleDir: "/bundle", ListenAddr: "0.0.0.0:8080"}, true},
		{"remote allowed", ServeConfig{BundleDir: "/bundle", ListenAddr: "0.0.0.0:8080", AllowRemote: true}, false},
		{"hostname rejected", ServeConfig{BundleDir: "/bundle", ListenAddr: "example.com:8080"}, true},
		{"missing bundle", ServeConfig{}, true},
		{"negative queue", ServeConfig{BundleDir: "/bundle", QueueSize: -1}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			err := tt.cfg.Validate()
			if (err != nil) != tt.wantError {
				t.Errorf("Validate() error = %v, want error %v", err, tt.wantError)
			}
		})
	}
}

func TestRunPredictionWorkflow(t *testing.T) {
	t.Parallel()
	node := workflow.NewFunctionNode("fake_prediction", func(_ agent.Context, _ string) (adklaya.PredictionOutput, error) {
		return adklaya.PredictionOutput{Results: []adklaya.ResultOutput{{Kind: adklaya.ResultKindNoul, QuestionID: "q"}}}, nil
	}, workflow.NodeConfig{})
	out, err := runPredictionWorkflow(context.Background(), node)
	if err != nil {
		t.Fatalf("runPredictionWorkflow: %v", err)
	}
	if len(out.Results) != 1 || out.Results[0].QuestionID != "q" {
		t.Errorf("output = %+v, want one result for q", out)
	}
}

func TestRunPredictionWorkflowCancellation(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	node := workflow.NewFunctionNode("fake_prediction", func(_ agent.Context, _ string) (adklaya.PredictionOutput, error) {
		t.Error("node ran after cancellation")
		return adklaya.PredictionOutput{}, nil
	}, workflow.NodeConfig{})
	_, err := runPredictionWorkflow(ctx, node)
	if !errors.Is(err, context.Canceled) {
		t.Errorf("runPredictionWorkflow error = %v, want context.Canceled", err)
	}
}

func TestServeMissingBundleDoesNotAcquire(t *testing.T) {
	t.Parallel()
	err := Serve(context.Background(), ServeConfig{BundleDir: t.TempDir()})
	if err == nil || !strings.Contains(err.Error(), "verify local bundle") {
		t.Errorf("Serve error = %v, want missing local bundle verification", err)
	}
}
