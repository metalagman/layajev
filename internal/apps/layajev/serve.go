package layajev

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/metalagman/laya-go"
	"github.com/metalagman/laya-go/adklaya"
	"go.uber.org/fx"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
	"google.golang.org/genai"
)

// ServeConfig configures one local HTTP process. BundleDir must remain
// immutable while serving. Exposing a non-loopback listener requires an
// explicit AllowRemote choice and operator-provided TLS/auth termination.
type ServeConfig struct {
	BundleDir   string
	ListenAddr  string
	AllowRemote bool
	QueueSize   int
}

// Validate checks the serve configuration without opening any resources.
func (c ServeConfig) Validate() error {
	if c.BundleDir == "" {
		return fmt.Errorf("serve: bundle directory is required")
	}
	if c.QueueSize < 0 {
		return fmt.Errorf("serve: queue size cannot be negative")
	}
	addr := c.ListenAddr
	if addr == "" {
		addr = "127.0.0.1:8080"
	}
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		return fmt.Errorf("serve: invalid listen address: %w", err)
	}
	ip := net.ParseIP(host)
	if !c.AllowRemote && (ip == nil || !ip.IsLoopback()) {
		return fmt.Errorf("serve: non-loopback listener requires --allow-remote")
	}
	return nil
}

type adkPredictor struct{ model *laya.Model }

func (p adkPredictor) Predict(ctx context.Context, state laya.State, questions []laya.Question) (adklaya.PredictionOutput, error) {
	project := func(_ agent.Context, _ string) (laya.State, error) { return state, nil }
	node, err := adklaya.NewPredictionNode(p.model, project, questions, adklaya.NodeOptions{Name: "predict"})
	if err != nil {
		return adklaya.PredictionOutput{}, fmt.Errorf("create prediction node: %w", err)
	}
	return runPredictionWorkflow(ctx, node)
}

func runPredictionWorkflow(ctx context.Context, node workflow.Node) (adklaya.PredictionOutput, error) {
	a, err := workflowagent.New(workflowagent.Config{
		Name: "layajev_request", Edges: []workflow.Edge{{From: workflow.Start, To: node}},
	})
	if err != nil {
		return adklaya.PredictionOutput{}, fmt.Errorf("create ADK workflow: %w", err)
	}
	service := session.InMemoryService()
	r, err := runner.New(runner.Config{AppName: "layajev", Agent: a, SessionService: service})
	if err != nil {
		return adklaya.PredictionOutput{}, fmt.Errorf("create ADK runner: %w", err)
	}
	if _, err := service.Create(ctx, &session.CreateRequest{AppName: "layajev", UserID: "request", SessionID: "one"}); err != nil {
		return adklaya.PredictionOutput{}, fmt.Errorf("create ADK session: %w", err)
	}
	var output adklaya.PredictionOutput
	found := false
	for event, runErr := range r.Run(ctx, "request", "one", genai.NewContentFromText("local prediction", genai.RoleUser), agent.RunConfig{}) {
		if runErr != nil {
			return adklaya.PredictionOutput{}, fmt.Errorf("run ADK workflow: %w", runErr)
		}
		if value, ok := event.Output.(adklaya.PredictionOutput); ok {
			output = value
			found = true
		}
	}
	if err := ctx.Err(); err != nil {
		return adklaya.PredictionOutput{}, err
	}
	if !found {
		return adklaya.PredictionOutput{}, fmt.Errorf("run ADK workflow: %w: prediction output missing", laya.ErrInvalidOutput)
	}
	return output, nil
}

type serverState struct {
	cfg       ServeConfig
	ready     chan<- net.Addr
	server    *http.Server
	listener  net.Listener
	runtime   *laya.Runtime
	model     *laya.Model
	serveDone chan error
}

func (s *serverState) start(ctx context.Context) error {
	metadata, err := readBundleMetadata(s.cfg.BundleDir)
	if err != nil {
		return fmt.Errorf("verify local bundle: read metadata: %w", err)
	}
	runtime, err := laya.NewRuntime(ctx, laya.RuntimeOptions{})
	if err != nil {
		return fmt.Errorf("open local runtime: %w", err)
	}
	model, err := runtime.OpenModelDir(ctx, s.cfg.BundleDir, laya.ModelOptions{QueueCapacity: s.cfg.QueueSize})
	if err != nil {
		_ = runtime.Close(context.Background())
		return fmt.Errorf("open local model: %w", err)
	}
	date := strings.SplitN(metadata.Provenance.CreatedAt, "T", 2)[0]
	handler, err := NewHandler(metadata.Bundle.ID, date, adkPredictor{model: model})
	if err != nil {
		_ = model.Close(context.Background())
		_ = runtime.Close(context.Background())
		return err
	}
	addr := s.cfg.ListenAddr
	if addr == "" {
		addr = "127.0.0.1:8080"
	}
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		_ = model.Close(context.Background())
		_ = runtime.Close(context.Background())
		return fmt.Errorf("listen %s: %w", addr, err)
	}
	s.runtime, s.model, s.listener = runtime, model, listener
	s.server = &http.Server{Handler: handler, ReadHeaderTimeout: 5 * time.Second}
	s.serveDone = make(chan error, 1)
	go func() {
		err := s.server.Serve(listener)
		if errors.Is(err, http.ErrServerClosed) {
			err = nil
		}
		s.serveDone <- err
	}()
	if s.ready != nil {
		s.ready <- listener.Addr()
	}
	return nil
}

func (s *serverState) stop(ctx context.Context) error {
	var errs []error
	if s.server != nil {
		if err := s.server.Shutdown(ctx); err != nil {
			errs = append(errs, fmt.Errorf("shutdown HTTP: %w", err))
			_ = s.server.Close()
		}
	}
	if s.model != nil {
		if err := s.model.Close(ctx); err != nil {
			errs = append(errs, fmt.Errorf("close model: %w", err))
		}
	}
	if s.runtime != nil {
		if err := s.runtime.Close(ctx); err != nil {
			errs = append(errs, fmt.Errorf("close runtime: %w", err))
		}
	}
	return errors.Join(errs...)
}

// Serve runs the local server until ctx ends or the listener fails. It owns
// the runtime, model, listener, and ADK request sessions, and closes them in
// that order after HTTP shutdown. It never acquires a model.
func Serve(ctx context.Context, cfg ServeConfig) error {
	return serveWithReady(ctx, cfg, nil)
}

func serveWithReady(ctx context.Context, cfg ServeConfig, ready chan<- net.Addr) error {
	if err := cfg.Validate(); err != nil {
		return err
	}
	var state *serverState
	app := fx.New(
		fx.NopLogger,
		fx.Provide(
			func() ServeConfig { return cfg },
			func(config ServeConfig) *serverState { return &serverState{cfg: config, ready: ready} },
		),
		fx.Invoke(func(lc fx.Lifecycle, server *serverState) {
			state = server
			lc.Append(fx.Hook{OnStart: server.start, OnStop: server.stop})
		}),
	)
	if err := app.Err(); err != nil {
		return fmt.Errorf("construct server: %w", err)
	}
	if err := app.Start(ctx); err != nil {
		return fmt.Errorf("start server: %w", err)
	}
	var serveErr error
	select {
	case <-ctx.Done():
	case serveErr = <-state.serveDone:
	}
	stopCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := app.Stop(stopCtx); err != nil {
		return errors.Join(serveErr, fmt.Errorf("stop server: %w", err))
	}
	return serveErr
}
