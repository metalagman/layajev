//go:build laya_native

package layajev

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/metalagman/laya-go"
)

func TestNativeJevADKPrediction(t *testing.T) {
	dir := nativeBundleDir(t)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	manifest, err := readBundleMetadata(dir)
	if err != nil {
		t.Fatalf("read manifest metadata: %v", err)
	}
	runtime, err := laya.NewRuntime(ctx, laya.RuntimeOptions{})
	if err != nil {
		t.Fatalf("NewRuntime: %v", err)
	}
	t.Cleanup(func() {
		if err := runtime.Close(context.Background()); err != nil {
			t.Errorf("Close runtime: %v", err)
		}
	})
	model, err := runtime.OpenModelDir(ctx, dir, laya.ModelOptions{QueueCapacity: 1})
	if err != nil {
		t.Fatalf("OpenModelDir: %v", err)
	}
	t.Cleanup(func() {
		if err := model.Close(context.Background()); err != nil {
			t.Errorf("Close model: %v", err)
		}
	})
	handler, err := NewHandler(manifest.Bundle.ID, strings.SplitN(manifest.Provenance.CreatedAt, "T", 2)[0], adkPredictor{model: model})
	if err != nil {
		t.Fatalf("NewHandler: %v", err)
	}
	body := `{"model":"` + manifest.Bundle.ID + `","state":"I was charged twice.","questions":{"billing":{"type":"noul","instructions":"Is this message about billing?"}}}`
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/v1/systemone", strings.NewReader(body)).WithContext(ctx))
	if recorder.Code != http.StatusOK {
		t.Fatalf("POST /v1/systemone status = %d, body = %s", recorder.Code, recorder.Body.String())
	}
	var response struct {
		Model   string `json:"model"`
		Answers map[string]struct {
			Type string  `json:"type"`
			Noul float64 `json:"noul"`
		} `json:"answers"`
	}
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	answer, ok := response.Answers["billing"]
	if response.Model != manifest.Bundle.ID || !ok || answer.Type != "noul" || answer.Noul < 0 || answer.Noul > 1 {
		t.Errorf("unexpected response: %+v", response)
	}
}

func TestNativeFxHTTPServer(t *testing.T) {
	dir := nativeBundleDir(t)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	ready := make(chan net.Addr, 1)
	done := make(chan error, 1)
	go func() {
		done <- serveWithReady(ctx, ServeConfig{BundleDir: dir, ListenAddr: "127.0.0.1:0", QueueSize: 1}, ready)
	}()
	var address net.Addr
	select {
	case address = <-ready:
	case err := <-done:
		t.Fatalf("server exited before readiness: %v", err)
	case <-ctx.Done():
		t.Fatalf("server did not start: %v", ctx.Err())
	}
	client := &http.Client{Timeout: 30 * time.Second}
	response, err := client.Get("http://" + address.String() + "/v1/models")
	if err != nil {
		t.Fatalf("GET models: %v", err)
	}
	var models struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	if err := json.NewDecoder(response.Body).Decode(&models); err != nil {
		t.Fatalf("decode models: %v", err)
	}
	if err := response.Body.Close(); err != nil {
		t.Fatalf("close models response: %v", err)
	}
	if response.StatusCode != http.StatusOK || len(models.Models) != 1 {
		t.Fatalf("GET models status %d, models %+v", response.StatusCode, models)
	}
	body := `{"model":"` + models.Models[0].Name + `","state":"I was charged twice.","questions":{"billing":{"type":"noul","instructions":"Is this about billing?"}}}`
	response, err = client.Post("http://"+address.String()+"/v1/systemone", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST systemone: %v", err)
	}
	var result struct {
		Answers map[string]json.RawMessage `json:"answers"`
	}
	if err := json.NewDecoder(response.Body).Decode(&result); err != nil {
		t.Fatalf("decode prediction: %v", err)
	}
	if err := response.Body.Close(); err != nil {
		t.Fatalf("close prediction response: %v", err)
	}
	if response.StatusCode != http.StatusOK || len(result.Answers["billing"]) == 0 {
		t.Fatalf("POST systemone status %d, answers %+v", response.StatusCode, result.Answers)
	}
	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Errorf("server shutdown: %v", err)
		}
	case <-time.After(30 * time.Second):
		t.Error("server did not shut down")
	}
}

func TestNativeFxBindFailureCleanup(t *testing.T) {
	dir := nativeBundleDir(t)
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("occupy listener: %v", err)
	}
	defer listener.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	err = Serve(ctx, ServeConfig{BundleDir: dir, ListenAddr: listener.Addr().String()})
	if err == nil || !strings.Contains(err.Error(), "listen") {
		t.Errorf("Serve on occupied address error = %v, want listen failure", err)
	}
}

func nativeBundleDir(t *testing.T) string {
	t.Helper()
	dir := os.Getenv("LAYA_BUNDLE_DIR")
	if dir == "" || os.Getenv("LAYA_ONNXRUNTIME_LIBRARY") == "" || os.Getenv("LAYA_TOKENIZERS_LIBRARY") == "" {
		t.Skip("requires explicit verified native artifacts and bundle")
	}
	return dir
}
