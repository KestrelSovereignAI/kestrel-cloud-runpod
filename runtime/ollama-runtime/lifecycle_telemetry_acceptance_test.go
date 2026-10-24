package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func waitForTelemetryRequests(requests *sync.WaitGroup, timeout time.Duration) bool {
	done := make(chan struct{})
	go func() {
		requests.Wait()
		close(done)
	}()
	select {
	case <-done:
		return true
	case <-time.After(timeout):
		return false
	}
}

func TestStartupTelemetryCarriesOriginalBindingAndMonotonicIdentity(t *testing.T) {
	started := time.Date(2026, 10, 20, 12, 0, 0, 0, time.UTC)
	binding := strings.Repeat("b", 64)
	values := map[string]string{
		"KESTREL_OLLAMA_BEARER_TOKEN":             testToken,
		"KESTREL_OLLAMA_BEARER_TOKEN_EXPIRES_AT":  started.Add(time.Hour).Format(time.RFC3339),
		"KESTREL_OLLAMA_ALLOWED_MODELS":           "registry.example/team/model:v1@sha256:" + strings.Repeat("a", 64),
		"KESTREL_OLLAMA_REQUIRED_MODEL":           "registry.example/team/model:v1",
		"KESTREL_OLLAMA_MODE":                     "dedicated_pod",
		"KESTREL_OLLAMA_MODEL_STORAGE_PATH":       "/models",
		"KESTREL_OLLAMA_PROVISION_REQUESTED_AT":   started.Add(-time.Second).Format(time.RFC3339Nano),
		"KESTREL_OLLAMA_TELEMETRY_BINDING_SHA256": binding,
	}
	config, err := loadConfig(func(name string) string { return values[name] }, started)
	if err != nil {
		t.Fatal(err)
	}
	_, upstream := testUpstream(t, func(writer http.ResponseWriter, _ *http.Request) {
		writer.WriteHeader(http.StatusOK)
	})
	config.Upstream = upstream
	state := newStartupState("cache_hit", time.Now())
	state.ollamaProcessRunning = true
	state.markReady()
	handler := newRuntimeProxy(config, state, func() time.Time { return started })
	response := httptest.NewRecorder()
	handler.ServeHTTP(
		response,
		authorizedRequest(t, http.MethodGet, "/kestrel/telemetry", ""),
	)
	if response.Code != http.StatusOK {
		t.Fatalf("telemetry returned %d", response.Code)
	}
	var payload map[string]any
	if err := json.Unmarshal(response.Body.Bytes(), &payload); err != nil {
		t.Fatal(err)
	}
	if payload["telemetry_binding_sha256"] != binding {
		t.Fatalf("telemetry binding = %#v, want original digest", payload["telemetry_binding_sha256"])
	}
	if instance, ok := payload["runtime_instance_id"].(string); !ok || instance == "" {
		t.Fatalf("runtime instance identity is unavailable: %#v", payload["runtime_instance_id"])
	}
	if revision, ok := payload["telemetry_revision"].(float64); !ok || revision < 1 {
		t.Fatalf("telemetry revision is unavailable: %#v", payload["telemetry_revision"])
	}
}

func TestProxyTelemetryCountsConcurrentInferenceAsOneBusyInterval(t *testing.T) {
	entered := make(chan struct{}, 2)
	release := make(chan struct{})
	var releaseOnce sync.Once
	releaseRequests := func() {
		releaseOnce.Do(func() { close(release) })
	}
	_, upstream := testUpstream(t, func(writer http.ResponseWriter, _ *http.Request) {
		entered <- struct{}{}
		<-release
		writer.WriteHeader(http.StatusOK)
	})
	config := testConfig(upstream)
	state := newStartupState("cache_hit", time.Now())
	state.ollamaProcessRunning = true
	state.markReady()
	handler := newRuntimeProxy(config, state, time.Now)
	privateBodies := []string{
		`{"model":"registry.example/team/model:v1","prompt":"private-one"}`,
		`{"model":"registry.example/team/model:v1","prompt":"private-two"}`,
	}
	var requests sync.WaitGroup
	defer func() {
		releaseRequests()
		if !waitForTelemetryRequests(&requests, 5*time.Second) {
			t.Errorf("inference handlers did not join after releasing their barrier")
		}
	}()
	requests.Add(len(privateBodies))
	for _, body := range privateBodies {
		go func(payload string) {
			defer requests.Done()
			response := httptest.NewRecorder()
			handler.ServeHTTP(
				response,
				authorizedRequest(t, http.MethodPost, "/api/generate", payload),
			)
			if response.Code != http.StatusOK {
				t.Errorf("inference returned %d", response.Code)
			}
		}(body)
	}
	for range privateBodies {
		select {
		case <-entered:
		case <-time.After(5 * time.Second):
			t.Fatal("inference handler did not reach the telemetry barrier")
		}
	}
	time.Sleep(time.Millisecond)
	inFlight := state.snapshot(config)
	if inFlight.InferenceStarted != 2 || inFlight.InferenceInFlight != 2 {
		t.Fatalf("in-flight counters = started %d, active %d", inFlight.InferenceStarted, inFlight.InferenceInFlight)
	}
	if inFlight.InferenceBusyMilliseconds < 1 {
		t.Fatalf("concurrent busy duration = %d", inFlight.InferenceBusyMilliseconds)
	}
	releaseRequests()
	if !waitForTelemetryRequests(&requests, 5*time.Second) {
		t.Fatal("inference handlers did not join after releasing their barrier")
	}
	completed := state.snapshot(config)
	if completed.InferenceCompleted != 2 || completed.InferenceInFlight != 0 {
		t.Fatalf("completed counters = completed %d, active %d", completed.InferenceCompleted, completed.InferenceInFlight)
	}
	if completed.InferenceBusyMilliseconds < inFlight.InferenceBusyMilliseconds {
		t.Fatal("cumulative busy duration regressed")
	}
	encoded, err := json.Marshal(completed)
	if err != nil {
		t.Fatal(err)
	}
	for _, private := range []string{"private-one", "private-two", config.RequiredModel.Name, testToken} {
		if strings.Contains(string(encoded), private) {
			t.Fatalf("telemetry leaked private inference content %q", private)
		}
	}
}

func TestProxyTelemetryClassifiesEveryUnsuccessfulInferenceResponse(t *testing.T) {
	tests := []struct {
		name          string
		status        int
		wantCompleted uint64
		wantFailed    uint64
	}{
		{"implicit OK", 0, 1, 0},
		{"OK", http.StatusOK, 1, 0},
		{"No Content", http.StatusNoContent, 1, 0},
		{"Found", http.StatusFound, 0, 1},
		{"Bad Request", http.StatusBadRequest, 0, 1},
		{"Not Found", http.StatusNotFound, 0, 1},
		{"Too Many Requests", http.StatusTooManyRequests, 0, 1},
		{"Internal Server Error", http.StatusInternalServerError, 0, 1},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, upstream := testUpstream(t, func(writer http.ResponseWriter, _ *http.Request) {
				if test.status != 0 {
					writer.WriteHeader(test.status)
				}
			})
			config := testConfig(upstream)
			state := newStartupState("cache_hit", time.Now())
			handler := newRuntimeProxy(config, state, time.Now)
			response := httptest.NewRecorder()
			handler.ServeHTTP(
				response,
				authorizedRequest(
					t,
					http.MethodPost,
					"/api/chat",
					`{"model":"registry.example/team/model:v1","messages":[]}`,
				),
			)
			wantStatus := test.status
			if wantStatus == 0 {
				wantStatus = http.StatusOK
			}
			if response.Code != wantStatus {
				t.Fatalf("proxy returned %d, want %d", response.Code, wantStatus)
			}
			snapshot := state.snapshot(config)
			if snapshot.InferenceStarted != 1 || snapshot.InferenceInFlight != 0 ||
				snapshot.InferenceCompleted != test.wantCompleted ||
				snapshot.InferenceFailed != test.wantFailed {
				t.Fatalf("unexpected inference counters for status %d: %#v", test.status, snapshot)
			}
		})
	}
}

func TestInferenceTimestampsSampleInsideMutationLock(t *testing.T) {
	// T2 exercises the repaired callable signature. The retained T1 source and
	// native RED logs separately prove both old-signature behavioral failures;
	// a signature/compile mismatch is not that counterproof.
	for _, phase := range []string{"start", "completion"} {
		t.Run(phase, func(t *testing.T) {
			state := newStartupState("test", time.Now())
			config := testConfig(nil)
			base := time.Date(2026, 10, 20, 12, 0, 0, 0, time.UTC)
			owners := [2]time.Time{}
			if phase == "completion" {
				for index := range owners {
					owners[index] = state.beginInference(func() time.Time { return base })
				}
			}
			clockEntered := make(chan struct{})
			releaseClock := make(chan struct{})
			secondAttempted := make(chan struct{})
			var releaseOnce sync.Once
			releaseFirst := func() { releaseOnce.Do(func() { close(releaseClock) }) }
			var requests sync.WaitGroup
			defer func() {
				releaseFirst()
				if !waitForTelemetryRequests(&requests, 5*time.Second) {
					t.Error("original timestamp mutation goroutines did not join")
				}
			}()
			var clockCalls atomic.Int32
			var outsideLock atomic.Int32
			clock := func() time.Time {
				ordinal := clockCalls.Add(1)
				if state.mu.TryLock() {
					outsideLock.Add(1)
					state.mu.Unlock()
				}
				if ordinal == 1 {
					close(clockEntered)
					<-releaseClock
				}
				return base.Add(time.Duration(ordinal) * time.Second)
			}
			mutate := func(index int) {
				if phase == "start" {
					owners[index] = state.beginInference(clock)
				} else {
					state.finishInference(owners[index], clock, false, false)
				}
			}
			requests.Add(1)
			go func() {
				defer requests.Done()
				mutate(0)
			}()
			select {
			case <-clockEntered:
			case <-time.After(5 * time.Second):
				t.Fatal("first mutation did not enter its actual clock callback")
			}
			requests.Add(1)
			go func() {
				defer requests.Done()
				close(secondAttempted)
				mutate(1)
			}()
			select {
			case <-secondAttempted:
			case <-time.After(5 * time.Second):
				t.Fatal("second mutation did not reach its original helper call")
			}
			releaseFirst()
			if !waitForTelemetryRequests(&requests, 5*time.Second) {
				t.Fatal("timestamp mutations did not join after clock release")
			}
			if clockCalls.Load() != 2 || outsideLock.Load() != 0 {
				t.Fatalf("clock must sample twice inside real state mutex: calls=%d outside=%d", clockCalls.Load(), outsideLock.Load())
			}
			observed := state.snapshot(config)
			first := base.Add(time.Second)
			last := base.Add(2 * time.Second)
			if phase == "start" {
				if !observed.FirstInferenceStartedAt.Equal(first) || !observed.LastInferenceStartedAt.Equal(last) ||
					observed.InferenceStarted != 2 || observed.InferenceInFlight != 2 {
					t.Fatalf("start identity/order changed: %#v", observed)
				}
				for _, owner := range owners {
					state.finishInference(owner, func() time.Time { return last.Add(time.Second) }, false, false)
				}
			} else if !observed.FirstInferenceStartedAt.Equal(base) ||
				!observed.LastInferenceStartedAt.Equal(base) || !observed.LastInferenceCompletedAt.Equal(last) {
				t.Fatalf("completion failed to retain original starts and latest sample: %#v", observed)
			}
			finished := state.snapshot(config)
			if finished.InferenceStarted != 2 || finished.InferenceCompleted != 2 ||
				finished.InferenceFailed != 0 || finished.InferenceCancelled != 0 || finished.InferenceInFlight != 0 {
				t.Fatalf("joined original mutations have unbalanced counters: %#v", finished)
			}
		})
	}
}
