package layajev

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/metalagman/laya-go"
	"github.com/metalagman/laya-go/adklaya"
)

const maxRequestBytes = 1 << 20

// Predictor evaluates a request through the application's ADK workflow.
// Implementations must honor cancellation and be safe for concurrent requests.
type Predictor interface {
	Predict(context.Context, laya.State, []laya.Question) (adklaya.PredictionOutput, error)
}

// Handler serves the documented Jev-compatible subset for one local model.
// The caller owns the predictor and must close its underlying model after the
// HTTP server has stopped accepting requests.
type Handler struct {
	modelName   string
	releaseDate string
	predictor   Predictor
}

// NewHandler validates the fixed local model identity, bundle date, and predictor.
// releaseDate is the bundle provenance date in YYYY-MM-DD form, not a claim
// about the upstream checkpoint's original release date.
func NewHandler(modelName, releaseDate string, predictor Predictor) (*Handler, error) {
	if strings.TrimSpace(modelName) == "" || predictor == nil {
		return nil, fmt.Errorf("new handler: model name and predictor are required")
	}
	if _, err := time.Parse("2006-01-02", releaseDate); err != nil {
		return nil, fmt.Errorf("new handler: invalid bundle date: %w", err)
	}
	return &Handler{modelName: modelName, releaseDate: releaseDate, predictor: predictor}, nil
}

// ServeHTTP handles GET /v1/models and POST /v1/systemone.
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch r.URL.Path {
	case "/v1/models":
		if r.Method != http.MethodGet {
			w.Header().Set("Allow", http.MethodGet)
			writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "GET required", "")
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"models": []any{map[string]any{
			"name": h.modelName, "description": "Local Laya bundle; release_date is the bundle provenance date; Jev-compatible API subset", "release_date": h.releaseDate,
		}}})
	case "/v1/systemone":
		if r.Method != http.MethodPost {
			w.Header().Set("Allow", http.MethodPost)
			writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "POST required", "")
			return
		}
		h.systemOne(w, r)
	default:
		writeError(w, http.StatusNotFound, "not_found", "route not found", "")
	}
}

func (h *Handler) systemOne(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, maxRequestBytes+1))
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", "read body failed", "")
		return
	}
	if len(body) > maxRequestBytes {
		writeError(w, http.StatusRequestEntityTooLarge, "request_too_large", "request exceeds 1 MiB", "")
		return
	}
	state, questions, questionByID, apiErr := parseRequest(body, h.modelName)
	if apiErr != nil {
		writeError(w, apiErr.status, apiErr.code, apiErr.message, apiErr.path)
		return
	}
	out, err := h.predictor.Predict(r.Context(), state, questions)
	if err != nil {
		if r.Context().Err() != nil {
			writeError(w, http.StatusRequestTimeout, "request_canceled", "request canceled", "")
			return
		}
		writeError(w, http.StatusInternalServerError, "inference_failed", "local inference failed", "")
		return
	}
	answers, err := mapAnswers(out.Results, questionByID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "invalid_inference_output", "local inference output invalid", "")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"model":   h.modelName,
		"answers": answers,
		"usage":   map[string]int{"input_tokens": out.Usage.InputTokens, "output_tokens": out.Usage.OutputTokens},
	})
}

type requestError struct {
	status  int
	code    string
	message string
	path    string
}

func bad(path, message string) *requestError {
	return &requestError{http.StatusBadRequest, "invalid_request", message, path}
}

func unsupported(path, message string) *requestError {
	return &requestError{http.StatusUnprocessableEntity, "unsupported_request", message, path}
}

func parseRequest(body []byte, localModel string) (laya.State, []laya.Question, map[laya.QuestionID]laya.Question, *requestError) {
	if err := rejectDuplicateKeys(body); err != nil {
		return laya.State{}, nil, nil, bad("body", "invalid JSON or duplicate object key")
	}
	var root map[string]json.RawMessage
	if err := decodeObject(body, &root); err != nil {
		return laya.State{}, nil, nil, bad("body", "expected a JSON object")
	}
	for field := range root {
		if field != "model" && field != "state" && field != "questions" {
			return laya.State{}, nil, nil, unsupported("body."+field, "field is not supported")
		}
	}
	var model string
	if err := json.Unmarshal(root["model"], &model); err != nil || model == "" {
		return laya.State{}, nil, nil, bad("body.model", "nonempty model name required")
	}
	if model != localModel {
		return laya.State{}, nil, nil, &requestError{http.StatusNotFound, "model_not_found", "model not available", "body.model"}
	}
	stateRaw, ok := root["state"]
	if !ok {
		return laya.State{}, nil, nil, bad("body.state", "state required")
	}
	var state laya.State
	if len(stateRaw) > 0 && stateRaw[0] == '"' {
		var text string
		if err := json.Unmarshal(stateRaw, &text); err != nil {
			return laya.State{}, nil, nil, bad("body.state", "invalid text state")
		}
		state = laya.TextState(text)
	} else if len(stateRaw) > 0 && (stateRaw[0] == '{' || stateRaw[0] == '[') {
		var err error
		state, err = laya.JSONState(stateRaw)
		if err != nil {
			return laya.State{}, nil, nil, bad("body.state", "invalid JSON state")
		}
	} else {
		return laya.State{}, nil, nil, bad("body.state", "state must be text, object, or array")
	}
	var rawQuestions map[string]json.RawMessage
	if err := decodeObject(root["questions"], &rawQuestions); err != nil || len(rawQuestions) == 0 {
		return laya.State{}, nil, nil, bad("body.questions", "nonempty questions object required")
	}
	keys := make([]string, 0, len(rawQuestions))
	for key := range rawQuestions {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	questions := make([]laya.Question, 0, len(keys))
	byID := make(map[laya.QuestionID]laya.Question, len(keys))
	for _, key := range keys {
		q, apiErr := parseQuestion(key, rawQuestions[key])
		if apiErr != nil {
			return laya.State{}, nil, nil, apiErr
		}
		questions = append(questions, q)
		byID[q.ID()] = q
	}
	return state, questions, byID, nil
}

func rejectDuplicateKeys(body []byte) error {
	decoder := json.NewDecoder(bytes.NewReader(body))
	if err := visitJSONValue(decoder); err != nil {
		return err
	}
	if _, err := decoder.Token(); err != io.EOF {
		return fmt.Errorf("trailing JSON content")
	}
	return nil
}

func visitJSONValue(decoder *json.Decoder) error {
	token, err := decoder.Token()
	if err != nil {
		return err
	}
	delim, ok := token.(json.Delim)
	if !ok {
		return nil
	}
	switch delim {
	case '{':
		seen := make(map[string]bool)
		for decoder.More() {
			keyToken, err := decoder.Token()
			if err != nil {
				return err
			}
			key, ok := keyToken.(string)
			if !ok || seen[key] {
				return fmt.Errorf("duplicate or invalid object key")
			}
			seen[key] = true
			if err := visitJSONValue(decoder); err != nil {
				return err
			}
		}
	case '[':
		for decoder.More() {
			if err := visitJSONValue(decoder); err != nil {
				return err
			}
		}
	default:
		return fmt.Errorf("unexpected delimiter")
	}
	_, err = decoder.Token()
	return err
}

func parseQuestion(name string, raw json.RawMessage) (laya.Question, *requestError) {
	path := "body.questions." + name
	if name == "" {
		return nil, bad(path, "question name must be nonempty")
	}
	var fields map[string]json.RawMessage
	if err := decodeObject(raw, &fields); err != nil {
		return nil, bad(path, "question must be an object")
	}
	var kind string
	if err := json.Unmarshal(fields["type"], &kind); err != nil || kind == "" {
		return nil, bad(path+".type", "question type required")
	}
	if kind != "choice" && kind != "score" && kind != "noul" {
		return nil, bad(path+".type", "unknown question type")
	}
	for field := range fields {
		if field != "type" && field != "instructions" && field != "criteria" {
			return nil, unsupported(path+"."+field, "field is not supported")
		}
	}
	var instructions string
	if err := json.Unmarshal(fields["instructions"], &instructions); err != nil || instructions == "" {
		return nil, unsupported(path+".instructions", "nonempty string instructions required")
	}
	switch kind {
	case "choice":
		var criteria map[string]json.RawMessage
		if err := decodeObject(fields["criteria"], &criteria); err != nil {
			return nil, bad(path+".criteria", "criteria object required")
		}
		if len(criteria) < 2 {
			return nil, unsupported(path+".criteria", "at least two described choices required")
		}
		keys := make([]string, 0, len(criteria))
		for key := range criteria {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		items := make([]laya.ChoiceCriterion, 0, len(keys))
		for _, key := range keys {
			var description string
			if err := json.Unmarshal(criteria[key], &description); err != nil || key == "" || description == "" {
				return nil, unsupported(path+".criteria."+key, "nonempty string description required")
			}
			items = append(items, laya.ChoiceCriterion{ID: laya.CriterionID(key), Description: description})
		}
		q, err := laya.NewChoiceQuestion(laya.QuestionID(name), instructions, items)
		if err != nil {
			return nil, unsupported(path, "question not representable by Laya")
		}
		return q, nil
	case "score":
		var criteria []json.RawMessage
		if err := json.Unmarshal(fields["criteria"], &criteria); err != nil || criteria == nil {
			return nil, bad(path+".criteria", "criteria array required")
		}
		if len(criteria) < 2 {
			return nil, unsupported(path+".criteria", "at least two described levels required")
		}
		items := make([]laya.ScoreLevel, 0, len(criteria))
		for i, rawLevel := range criteria {
			var description string
			if err := json.Unmarshal(rawLevel, &description); err != nil || description == "" {
				return nil, unsupported(path+".criteria."+strconv.Itoa(i), "nonempty string description required")
			}
			items = append(items, laya.ScoreLevel{Label: strconv.Itoa(i), Description: description})
		}
		q, err := laya.NewScoreQuestion(laya.QuestionID(name), instructions, items)
		if err != nil {
			return nil, unsupported(path, "question not representable by Laya")
		}
		return q, nil
	default:
		var criteria map[string]json.RawMessage
		if len(fields["criteria"]) > 0 && !bytes.Equal(fields["criteria"], []byte("null")) {
			if err := decodeObject(fields["criteria"], &criteria); err != nil {
				return nil, bad(path+".criteria", "noul criteria object required")
			}
		}
		for key := range criteria {
			if key != "true" && key != "false" {
				return nil, unsupported(path+".criteria."+key, "criterion is not supported")
			}
		}
		var trueDescription, falseDescription string
		for key, value := range criteria {
			var description string
			if err := json.Unmarshal(value, &description); err != nil {
				return nil, unsupported(path+".criteria."+key, "string description required")
			}
			if key == "true" {
				trueDescription = description
			} else {
				falseDescription = description
			}
		}
		q, err := laya.NewNoulQuestion(laya.QuestionID(name), instructions, falseDescription, trueDescription)
		if err != nil {
			return nil, unsupported(path, "question not representable by Laya")
		}
		return q, nil
	}
}

func decodeObject(raw []byte, target *map[string]json.RawMessage) error {
	if len(raw) == 0 || raw[0] != '{' {
		return fmt.Errorf("expected object")
	}
	return json.Unmarshal(raw, target)
}

func mapAnswers(results []adklaya.ResultOutput, questions map[laya.QuestionID]laya.Question) (map[string]any, error) {
	if len(results) != len(questions) {
		return nil, fmt.Errorf("result count mismatch")
	}
	answers := make(map[string]any, len(results))
	for _, result := range results {
		q, ok := questions[result.QuestionID]
		if !ok || answers[string(result.QuestionID)] != nil {
			return nil, fmt.Errorf("unexpected or duplicate result")
		}
		switch q := q.(type) {
		case laya.ChoiceQuestion:
			if result.Kind != adklaya.ResultKindChoice || result.Choice == nil {
				return nil, fmt.Errorf("choice result mismatch")
			}
			probabilities := make(map[string]float64, len(result.Choice.Probabilities))
			for _, p := range result.Choice.Probabilities {
				probabilities[string(p.CriterionID)] = p.Probability
			}
			answers[string(q.ID())] = map[string]any{"type": "choice", "choice": result.Choice.Selected, "confidence": result.Choice.Confidence, "probabilities": probabilities}
		case laya.ScoreQuestion:
			if result.Kind != adklaya.ResultKindScore || result.Score == nil {
				return nil, fmt.Errorf("score result mismatch")
			}
			legend := make(map[string]string, len(q.Rubric()))
			for _, level := range q.Rubric() {
				legend[level.Label] = level.Description
			}
			probabilities := make(map[string]float64, len(result.Score.Distribution))
			for _, p := range result.Score.Distribution {
				probabilities[p.Label] = p.Probability
			}
			answers[string(q.ID())] = map[string]any{"type": "score", "score": result.Score.ExpectedLevel, "confidence": result.Score.Confidence, "legend": legend, "probabilities": probabilities}
		case laya.NoulQuestion:
			if result.Kind != adklaya.ResultKindNoul || result.Noul == nil {
				return nil, fmt.Errorf("noul result mismatch")
			}
			answers[string(q.ID())] = map[string]any{"type": "noul", "noul": result.Noul.TrueProbability}
		default:
			return nil, fmt.Errorf("unknown result kind")
		}
	}
	return answers, nil
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func writeError(w http.ResponseWriter, status int, code, message, path string) {
	writeJSON(w, status, map[string]any{"error": map[string]string{"code": code, "message": message, "path": path}})
}
