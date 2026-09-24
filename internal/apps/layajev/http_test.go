package layajev

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/metalagman/laya-go"
	"github.com/metalagman/laya-go/adklaya"
)

type fakePredictor struct {
	calls int
}

func (p *fakePredictor) Predict(ctx context.Context, _ laya.State, questions []laya.Question) (adklaya.PredictionOutput, error) {
	p.calls++
	if err := ctx.Err(); err != nil {
		return adklaya.PredictionOutput{}, err
	}
	out := adklaya.PredictionOutput{Results: make([]adklaya.ResultOutput, 0, len(questions)), Usage: laya.Usage{InputTokens: 3, OutputTokens: 2}}
	for _, q := range questions {
		switch q := q.(type) {
		case laya.ChoiceQuestion:
			criteria := q.Criteria()
			answer := &laya.ChoiceAnswer{Selected: criteria[0].ID, Confidence: 0.8, Probabilities: []laya.CriterionProbability{{CriterionID: criteria[0].ID, Probability: 0.8}, {CriterionID: criteria[1].ID, Probability: 0.2}}}
			out.Results = append(out.Results, adklaya.ResultOutput{Kind: adklaya.ResultKindChoice, QuestionID: q.ID(), Choice: answer})
		case laya.ScoreQuestion:
			answer := &laya.ScoreAnswer{ExpectedLevel: 0.3, Confidence: 0.7, Distribution: []laya.ScoreProbability{{Level: 0, Label: "0", Probability: 0.7}, {Level: 1, Label: "1", Probability: 0.3}}}
			out.Results = append(out.Results, adklaya.ResultOutput{Kind: adklaya.ResultKindScore, QuestionID: q.ID(), Score: answer})
		case laya.NoulQuestion:
			out.Results = append(out.Results, adklaya.ResultOutput{Kind: adklaya.ResultKindNoul, QuestionID: q.ID(), Noul: &laya.NoulAnswer{TrueProbability: 0.9}})
		}
	}
	return out, nil
}

func TestHandlerSubset(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name, body, wantCode, wantPath string
		status                         int
		calls                          int
	}{
		{"mixed", `{"model":"laya-multilingual-v1","state":{"text":"hello"},"questions":{"a":{"type":"choice","instructions":"tone?","criteria":{"calm":"calm","upset":"upset"}},"b":{"type":"score","instructions":"urgency?","criteria":["low","high"]},"c":{"type":"noul","instructions":"spam?"}}}`, "", "", http.StatusOK, 1},
		{"missing instructions", `{"model":"laya-multilingual-v1","state":"hello","questions":{"x":{"type":"noul"}}}`, "unsupported_request", "body.questions.x.instructions", http.StatusUnprocessableEntity, 0},
		{"single score", `{"model":"laya-multilingual-v1","state":"hello","questions":{"x":{"type":"score","instructions":"urgency?","criteria":["low"]}}}`, "unsupported_request", "body.questions.x.criteria", http.StatusUnprocessableEntity, 0},
		{"object instructions", `{"model":"laya-multilingual-v1","state":"hello","questions":{"x":{"type":"noul","instructions":{"task":"spam"}}}}`, "unsupported_request", "body.questions.x.instructions", http.StatusUnprocessableEntity, 0},
		{"malformed", `{`, "invalid_request", "body", http.StatusBadRequest, 0},
		{"duplicate key", `{"model":"laya-multilingual-v1","model":"jev-latest","state":"hello","questions":{"x":{"type":"noul","instructions":"spam?"}}}`, "invalid_request", "body", http.StatusBadRequest, 0},
		{"wrong model", `{"model":"jev-latest","state":"hello","questions":{"x":{"type":"noul","instructions":"spam?"}}}`, "model_not_found", "body.model", http.StatusNotFound, 0},
		{"unknown field", `{"model":"laya-multilingual-v1","state":"hello","questions":{"x":{"type":"noul","instructions":"spam?","extra":1}}}`, "unsupported_request", "body.questions.x.extra", http.StatusUnprocessableEntity, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			predictor := &fakePredictor{}
			h, err := NewHandler("laya-multilingual-v1", "2026-09-23", predictor)
			if err != nil {
				t.Fatalf("NewHandler: %v", err)
			}
			recorder := httptest.NewRecorder()
			h.ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/v1/systemone", strings.NewReader(tt.body)))
			if recorder.Code != tt.status {
				t.Fatalf("status = %d, want %d; body = %s", recorder.Code, tt.status, recorder.Body.String())
			}
			if predictor.calls != tt.calls {
				t.Errorf("predictor calls = %d, want %d", predictor.calls, tt.calls)
			}
			var response map[string]json.RawMessage
			if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
				t.Fatalf("decode response: %v", err)
			}
			if tt.wantCode != "" {
				var apiError struct{ Code, Path string }
				if err := json.Unmarshal(response["error"], &apiError); err != nil {
					t.Fatalf("decode error: %v", err)
				}
				if apiError.Code != tt.wantCode || apiError.Path != tt.wantPath {
					t.Errorf("error = %+v, want code %q path %q", apiError, tt.wantCode, tt.wantPath)
				}
			} else {
				var answers map[string]map[string]any
				if err := json.Unmarshal(response["answers"], &answers); err != nil {
					t.Fatalf("decode answers: %v", err)
				}
				if len(answers) != 3 || answers["a"]["type"] != "choice" || answers["b"]["type"] != "score" || answers["c"]["type"] != "noul" {
					t.Errorf("unexpected answers: %v", answers)
				}
			}
		})
	}
}

func TestModels(t *testing.T) {
	t.Parallel()
	h, err := NewHandler("laya-multilingual-v1", "2026-09-23", &fakePredictor{})
	if err != nil {
		t.Fatalf("NewHandler: %v", err)
	}
	recorder := httptest.NewRecorder()
	h.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	if recorder.Code != http.StatusOK || !strings.Contains(recorder.Body.String(), `"name":"laya-multilingual-v1"`) {
		t.Fatalf("GET /v1/models: status %d body %s", recorder.Code, recorder.Body.String())
	}
}
