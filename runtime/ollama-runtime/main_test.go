package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

const testToken = "0123456789abcdef0123456789abcdef"

func testConfig(upstream *url.URL) runtimeConfig {
	now := time.Now().UTC()
	digest := strings.Repeat("a", 64)
	pin := modelPin{Name: "registry.example/team/model:v1", Digest: digest}
	return runtimeConfig{
		ListenAddress:      defaultListenAddress,
		Upstream:           upstream,
		BearerToken:        testToken,
		BearerTokenExpires: now.Add(time.Hour),
		RequiredModel:      pin,
		AllowedModels:      map[string]modelPin{pin.Name: pin},
		Mode:               "dedicated_pod",
		ModelStoragePath:   "/models",
		StartupTimeout:     time.Second,
		PollInterval:       time.Millisecond,
		PreloadModel:       true,
		ContainerStartedAt: now,
	}
}

func testUpstream(t *testing.T, handler http.HandlerFunc) (*httptest.Server, *url.URL) {
	t.Helper()
	server := httptest.NewServer(handler)
	parsed, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(server.Close)
	return server, parsed
}

func authorizedRequest(t *testing.T, method, target, body string) *http.Request {
	t.Helper()
	request := httptest.NewRequest(method, target, strings.NewReader(body))
	request.Header.Set("Authorization", "Bearer "+testToken)
	request.Header.Set("Content-Type", "application/json")
	return request
}

func TestLoadConfigRequiresSecretExpiryAndDigestPinnedAllowlist(t *testing.T) {
	started := time.Now().UTC()
	valid := map[string]string{
		"KESTREL_OLLAMA_BEARER_TOKEN":            testToken,
		"KESTREL_OLLAMA_BEARER_TOKEN_EXPIRES_AT": started.Add(time.Hour).Format(time.RFC3339),
		"KESTREL_OLLAMA_ALLOWED_MODELS":          "registry.example/team/model:v1@sha256:" + strings.Repeat("a", 64),
		"KESTREL_OLLAMA_REQUIRED_MODEL":          "registry.example/team/model:v1",
		"KESTREL_OLLAMA_MODE":                    "dedicated_pod",
		"KESTREL_OLLAMA_MODEL_STORAGE_PATH":      "/models",
	}
	getenv := func(name string) string { return valid[name] }
	config, err := loadConfig(getenv, started)
	if err != nil {
		t.Fatal(err)
	}
	if config.RequiredModel.Digest != "sha256:"+strings.Repeat("a", 64) {
		t.Fatalf("unexpected required model pin: %#v", config.RequiredModel)
	}

	tests := []struct {
		name  string
		key   string
		value string
	}{
		{name: "short token", key: "KESTREL_OLLAMA_BEARER_TOKEN", value: "short"},
		{name: "expired token", key: "KESTREL_OLLAMA_BEARER_TOKEN_EXPIRES_AT", value: started.Add(-time.Second).Format(time.RFC3339)},
		{name: "mutable latest", key: "KESTREL_OLLAMA_ALLOWED_MODELS", value: "model:latest"},
		{name: "required absent", key: "KESTREL_OLLAMA_REQUIRED_MODEL", value: "different:v1"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			changed := make(map[string]string, len(valid))
			for key, value := range valid {
				changed[key] = value
			}
			changed[tt.key] = tt.value
			_, err := loadConfig(func(name string) string { return changed[name] }, started)
			if err == nil {
				t.Fatal("expected fail-closed configuration error")
			}
		})
	}
}

func TestModelStoragePathIsBoundToRunpodMode(t *testing.T) {
	tests := []struct {
		mode, path string
		valid      bool
	}{
		{"dedicated_pod", "/models", true},
		{"dedicated_pod", "/workspace/ollama", true},
		{"serverless_load_balancer", "/models", true},
		{"serverless_load_balancer", "/runpod-volume/ollama", true},
		{"dedicated_pod", "/runpod-volume/ollama", false},
		{"serverless_load_balancer", "/workspace/ollama", false},
		{"dedicated_pod", "/models/../private", false},
	}
	for _, tt := range tests {
		_, err := validateModelStoragePath(tt.path, tt.mode)
		if (err == nil) != tt.valid {
			t.Fatalf("mode %s path %s validity=%t error=%v", tt.mode, tt.path, tt.valid, err)
		}
	}
}

func TestModelStorageTreeRejectsSymlinksBeforePrivilegedOwnershipChange(t *testing.T) {
	storage := t.TempDir()
	if err := os.Symlink("/etc", filepath.Join(storage, "escape")); err != nil {
		t.Fatal(err)
	}
	if err := validateModelStorageTree(storage); err == nil || !strings.Contains(err.Error(), "unsupported symlink") {
		t.Fatalf("expected symlink rejection, got %v", err)
	}
}

func TestAnonymousAndWrongBearerAreDeniedOnEveryNonHealthSurface(t *testing.T) {
	_, upstream := testUpstream(t, func(writer http.ResponseWriter, _ *http.Request) {
		writer.WriteHeader(http.StatusNoContent)
	})
	handler := newRuntimeProxy(testConfig(upstream), &startupState{}, time.Now)
	tests := []struct {
		method string
		path   string
		body   string
	}{
		{http.MethodGet, "/api/tags", ""},
		{http.MethodPost, "/api/chat", `{"model":"registry.example/team/model:v1"}`},
		{http.MethodPost, "/api/generate", `{"model":"registry.example/team/model:v1"}`},
		{http.MethodPost, "/api/pull", `{"name":"registry.example/team/model:v1"}`},
		{http.MethodPost, "/api/delete", `{}`},
		{http.MethodGet, "/kestrel/telemetry", ""},
		{http.MethodGet, "/unknown", ""},
	}
	for _, tt := range tests {
		for _, authorization := range []string{"", "Bearer wrong"} {
			request := httptest.NewRequest(tt.method, tt.path, strings.NewReader(tt.body))
			if authorization != "" {
				request.Header.Set("Authorization", authorization)
			}
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, request)
			if response.Code != http.StatusUnauthorized {
				t.Fatalf("%s %s with %q returned %d", tt.method, tt.path, authorization, response.Code)
			}
		}
	}
}

func TestExpiredBearerIsDenied(t *testing.T) {
	_, upstream := testUpstream(t, func(writer http.ResponseWriter, _ *http.Request) {
		writer.WriteHeader(http.StatusOK)
	})
	config := testConfig(upstream)
	handler := newRuntimeProxy(config, &startupState{}, func() time.Time {
		return config.BearerTokenExpires.Add(time.Nanosecond)
	})
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, authorizedRequest(t, http.MethodGet, "/api/tags", ""))
	if response.Code != http.StatusUnauthorized {
		t.Fatalf("expired token returned %d", response.Code)
	}
}

func TestHealthIsPublicButFailClosedUntilExactModelReady(t *testing.T) {
	digest := "sha256:" + strings.Repeat("a", 64)
	_, upstream := testUpstream(t, func(writer http.ResponseWriter, request *http.Request) {
		if request.URL.Path == "/api/tags" {
			writeJSON(writer, http.StatusOK, map[string]any{"models": []map[string]string{{
				"name": "registry.example/team/model:v1", "digest": digest,
			}}})
			return
		}
		writer.WriteHeader(http.StatusOK)
	})
	config := testConfig(upstream)
	state := &startupState{ollamaProcessRunning: true, strategy: "initializing"}
	handler := newRuntimeProxy(config, state, time.Now)

	for _, path := range []string{"/ping", "/health/ready"} {
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, httptest.NewRequest(http.MethodGet, path, nil))
		if response.Code != http.StatusNoContent {
			t.Fatalf("%s before readiness returned %d", path, response.Code)
		}
	}
	live := httptest.NewRecorder()
	handler.ServeHTTP(live, httptest.NewRequest(http.MethodGet, "/health/live", nil))
	if live.Code != http.StatusOK {
		t.Fatalf("liveness returned %d", live.Code)
	}

	state.markReady()
	ready := httptest.NewRecorder()
	handler.ServeHTTP(ready, httptest.NewRequest(http.MethodGet, "/ping", nil))
	if ready.Code != http.StatusOK {
		t.Fatalf("ready ping returned %d", ready.Code)
	}
}

func TestContainerHealthcheckUsesLivenessDuringColdModelPull(t *testing.T) {
	_, endpoint := testUpstream(t, func(writer http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/health/live" {
			t.Fatalf("healthcheck requested %s", request.URL.Path)
		}
		writer.WriteHeader(http.StatusOK)
	})
	t.Setenv("PORT_HEALTH", endpoint.Port())
	if status := runHealthcheck(); status != 0 {
		t.Fatalf("live-but-initializing healthcheck returned %d", status)
	}
}

func TestReadinessRevalidatesExactModelAndCapabilityExpiry(t *testing.T) {
	digest := strings.Repeat("b", 64)
	_, upstream := testUpstream(t, func(writer http.ResponseWriter, request *http.Request) {
		if request.URL.Path == "/api/tags" {
			writeJSON(writer, http.StatusOK, map[string]any{"models": []map[string]string{{
				"name": "registry.example/team/model:v1", "digest": digest,
			}}})
			return
		}
		writer.WriteHeader(http.StatusOK)
	})
	config := testConfig(upstream)
	state := &startupState{ollamaProcessRunning: true, strategy: "cache_hit"}
	state.markReady()

	missing := httptest.NewRecorder()
	newRuntimeProxy(config, state, time.Now).ServeHTTP(
		missing, httptest.NewRequest(http.MethodGet, "/ping", nil),
	)
	if missing.Code != http.StatusServiceUnavailable {
		t.Fatalf("digest mismatch readiness returned %d", missing.Code)
	}

	expired := httptest.NewRecorder()
	newRuntimeProxy(config, state, func() time.Time {
		return config.BearerTokenExpires
	}).ServeHTTP(expired, httptest.NewRequest(http.MethodGet, "/ping", nil))
	if expired.Code != http.StatusServiceUnavailable {
		t.Fatalf("expired capability readiness returned %d", expired.Code)
	}
}

func TestStopChildReturnsAfterExitNotificationWasConsumed(t *testing.T) {
	child := exec.Command("sh", "-c", "exit 0")
	if err := child.Start(); err != nil {
		t.Fatal(err)
	}
	exit := make(chan error, 1)
	go func() { exit <- child.Wait() }()
	if err := <-exit; err != nil {
		t.Fatal(err)
	}

	done := make(chan struct{})
	go func() {
		stopChild(child, exit, true)
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("shutdown waited for a consumed child-exit notification")
	}
}

func TestModelManagementAndArbitraryModelsFailClosed(t *testing.T) {
	var upstreamCalls atomic.Int32
	_, upstream := testUpstream(t, func(writer http.ResponseWriter, _ *http.Request) {
		upstreamCalls.Add(1)
		writer.WriteHeader(http.StatusNoContent)
	})
	handler := newRuntimeProxy(testConfig(upstream), &startupState{}, time.Now)
	tests := []struct {
		path string
		body string
		want int
	}{
		{"/api/create", `{"name":"registry.example/team/model:v1"}`, http.StatusForbidden},
		{"/api/delete", `{"name":"registry.example/team/model:v1"}`, http.StatusForbidden},
		{"/api/push", `{"name":"registry.example/team/model:v1"}`, http.StatusForbidden},
		{"/api/pull", `{"name":"registry.example/team/model:v1"}`, http.StatusForbidden},
		{"/api/pull", `{"name":"unapproved/model:v1"}`, http.StatusForbidden},
		{"/api/pull", `{"model":"registry.example/team/model:v1","name":"unapproved/model:v1"}`, http.StatusForbidden},
		{"/api/chat", `{"model":"unapproved/model:v1","messages":[]}`, http.StatusForbidden},
		{"/future/management", `{}`, http.StatusNotFound},
	}
	for _, tt := range tests {
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, authorizedRequest(t, http.MethodPost, tt.path, tt.body))
		if response.Code != tt.want {
			t.Fatalf("%s returned %d, want %d", tt.path, response.Code, tt.want)
		}
	}
	if upstreamCalls.Load() != 0 {
		t.Fatalf("blocked calls reached Ollama %d times", upstreamCalls.Load())
	}
}

func TestApprovedInferenceStripsWorkloadBearer(t *testing.T) {
	requests := make(chan *http.Request, 1)
	_, upstream := testUpstream(t, func(writer http.ResponseWriter, request *http.Request) {
		requests <- request.Clone(context.Background())
		writer.WriteHeader(http.StatusOK)
	})
	handler := newRuntimeProxy(testConfig(upstream), &startupState{}, time.Now)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, authorizedRequest(t, http.MethodPost, "/api/chat", `{"model":"registry.example/team/model:v1","messages":[]}`))
	if response.Code != http.StatusOK {
		t.Fatalf("/api/chat returned %d", response.Code)
	}
	upstreamRequest := <-requests
	if upstreamRequest.Header.Get("Authorization") != "" {
		t.Fatal("workload bearer was forwarded to Ollama")
	}
}

func TestModelPolicyMatchesOllamaCaseInsensitiveLastKeyWinsParsing(t *testing.T) {
	var upstreamCalls atomic.Int32
	_, upstream := testUpstream(t, func(writer http.ResponseWriter, _ *http.Request) {
		upstreamCalls.Add(1)
		writer.WriteHeader(http.StatusOK)
	})
	handler := newRuntimeProxy(testConfig(upstream), &startupState{}, time.Now)

	response := httptest.NewRecorder()
	handler.ServeHTTP(response, authorizedRequest(
		t,
		http.MethodPost,
		"/api/chat",
		`{"model":"registry.example/team/model:v1","messages":[],"MODEL":"unapproved/model:v1"}`,
	))
	if response.Code != http.StatusForbidden {
		t.Fatalf("case-variant duplicate model returned %d", response.Code)
	}
	if upstreamCalls.Load() != 0 {
		t.Fatal("case-variant duplicate model reached Ollama")
	}

	allowed := httptest.NewRecorder()
	handler.ServeHTTP(allowed, authorizedRequest(
		t,
		http.MethodPost,
		"/api/chat",
		`{"MODEL":"registry.example/team/model:v1","messages":[]}`,
	))
	if allowed.Code != http.StatusOK {
		t.Fatalf("case-insensitive allowed model returned %d", allowed.Code)
	}
}

func TestStreamingChunksAndClientCancellationPropagate(t *testing.T) {
	cancelled := make(chan struct{})
	_, upstream := testUpstream(t, func(writer http.ResponseWriter, request *http.Request) {
		flusher, ok := writer.(http.Flusher)
		if !ok {
			t.Error("test server has no flusher")
			return
		}
		writer.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = io.WriteString(writer, "{\"chunk\":1}\n")
		flusher.Flush()
		<-request.Context().Done()
		close(cancelled)
	})
	server := httptest.NewServer(newRuntimeProxy(testConfig(upstream), &startupState{}, time.Now))
	t.Cleanup(server.Close)
	request, _ := http.NewRequest(http.MethodPost, server.URL+"/api/generate", strings.NewReader(`{"model":"registry.example/team/model:v1","prompt":"private","stream":true}`))
	request.Header.Set("Authorization", "Bearer "+testToken)
	response, err := server.Client().Do(request)
	if err != nil {
		t.Fatal(err)
	}
	line, err := bufio.NewReader(response.Body).ReadString('\n')
	if err != nil || line != "{\"chunk\":1}\n" {
		t.Fatalf("streamed line %q, error %v", line, err)
	}
	_ = response.Body.Close()
	select {
	case <-cancelled:
	case <-time.After(time.Second):
		t.Fatal("upstream request context was not cancelled")
	}
}

func TestBootstrapUsesPinnedCacheAndPreloadsBeforeReadiness(t *testing.T) {
	digest := strings.Repeat("a", 64)
	var generateCalls atomic.Int32
	_, upstream := testUpstream(t, func(writer http.ResponseWriter, request *http.Request) {
		switch request.URL.Path {
		case "/api/version":
			writeJSON(writer, http.StatusOK, map[string]string{"version": "test"})
		case "/api/tags":
			writeJSON(writer, http.StatusOK, map[string]any{"models": []map[string]string{{
				"name": "registry.example/team/model:v1", "digest": digest,
			}}})
		case "/api/generate":
			generateCalls.Add(1)
			writeJSON(writer, http.StatusOK, map[string]bool{"done": true})
		default:
			writeJSON(writer, http.StatusNotFound, nil)
		}
	})
	config := testConfig(upstream)
	state := &startupState{ollamaProcessRunning: true, strategy: "initializing"}
	if err := bootstrap(context.Background(), config, state, http.DefaultClient); err != nil {
		t.Fatal(err)
	}
	snapshot := state.snapshot(config)
	if !snapshot.Ready || !snapshot.CacheHit || snapshot.StartupStrategy != "cache_hit" {
		t.Fatalf("unexpected snapshot: %#v", snapshot)
	}
	if generateCalls.Load() != 1 {
		t.Fatalf("preload called %d times", generateCalls.Load())
	}
}

func TestBootstrapPullsOnlyRequiredPinAndRejectsDigestMismatch(t *testing.T) {
	digest := strings.Repeat("b", 64)
	var pulled atomic.Bool
	var pulledName string
	_, upstream := testUpstream(t, func(writer http.ResponseWriter, request *http.Request) {
		switch request.URL.Path {
		case "/api/version":
			writeJSON(writer, http.StatusOK, map[string]string{"version": "test"})
		case "/api/tags":
			models := []map[string]string{}
			if pulled.Load() {
				models = append(models, map[string]string{"name": "registry.example/team/model:v1", "digest": digest})
			}
			writeJSON(writer, http.StatusOK, map[string]any{"models": models})
		case "/api/pull":
			var payload map[string]any
			_ = json.NewDecoder(request.Body).Decode(&payload)
			pulledName, _ = payload["name"].(string)
			pulled.Store(true)
			writeJSON(writer, http.StatusOK, map[string]bool{"done": true})
		default:
			writeJSON(writer, http.StatusOK, map[string]bool{"done": true})
		}
	})
	config := testConfig(upstream)
	config.PreloadModel = false
	state := &startupState{ollamaProcessRunning: true, strategy: "initializing"}
	err := bootstrap(context.Background(), config, state, http.DefaultClient)
	if err == nil || !strings.Contains(err.Error(), "digest") {
		t.Fatalf("expected digest mismatch, got %v", err)
	}
	if pulledName != config.RequiredModel.Name {
		t.Fatalf("pulled %q, want %q", pulledName, config.RequiredModel.Name)
	}
	if state.snapshot(config).Ready {
		t.Fatal("digest mismatch became ready")
	}
}

func TestTelemetryIsAuthenticatedAndContainsNoCredentialOrModelName(t *testing.T) {
	_, upstream := testUpstream(t, func(writer http.ResponseWriter, _ *http.Request) {
		writer.WriteHeader(http.StatusOK)
	})
	config := testConfig(upstream)
	provision := config.ContainerStartedAt.Add(-2 * time.Second)
	config.ProvisionRequested = &provision
	state := &startupState{ollamaProcessRunning: true, strategy: "cache_hit", cacheHit: true}
	state.markReady()
	handler := newRuntimeProxy(config, state, time.Now)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, authorizedRequest(t, http.MethodGet, "/kestrel/telemetry", ""))
	if response.Code != http.StatusOK {
		t.Fatalf("telemetry returned %d", response.Code)
	}
	body := response.Body.String()
	for _, forbidden := range []string{testToken, config.RequiredModel.Name, config.RequiredModel.Digest} {
		if strings.Contains(body, forbidden) {
			t.Fatalf("telemetry leaked %q: %s", forbidden, body)
		}
	}
	if !strings.Contains(body, `"placement_and_image_pull_ms":2000`) {
		t.Fatalf("telemetry omitted phase duration: %s", body)
	}
}

func TestRequestSizeLimitIsEnforcedBeforeUpstream(t *testing.T) {
	var calls atomic.Int32
	_, upstream := testUpstream(t, func(writer http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		writer.WriteHeader(http.StatusOK)
	})
	handler := newRuntimeProxy(testConfig(upstream), &startupState{}, time.Now)
	body := fmt.Sprintf(`{"model":"registry.example/team/model:v1","prompt":"%s"}`, strings.Repeat("x", maxRequestBytes))
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, authorizedRequest(t, http.MethodPost, "/api/generate", body))
	if response.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("oversized request returned %d", response.Code)
	}
	if calls.Load() != 0 {
		t.Fatal("oversized request reached upstream")
	}
}

func TestReplaceEnvironmentDoesNotDuplicateSecurityCriticalSettings(t *testing.T) {
	result := replaceEnvironment([]string{"HOME=/root", "PATH=/bin", "OLLAMA_HOST=0.0.0.0:1"}, map[string]string{
		"HOME": "/safe", "OLLAMA_HOST": "127.0.0.1:11435",
	})
	joined := strings.Join(result, "\n")
	if strings.Count(joined, "HOME=") != 1 || strings.Count(joined, "OLLAMA_HOST=") != 1 {
		t.Fatalf("security settings were duplicated: %s", joined)
	}
	if !bytes.Contains([]byte(joined), []byte("HOME=/safe")) || !bytes.Contains([]byte(joined), []byte("PATH=/bin")) {
		t.Fatalf("environment replacement lost values: %s", joined)
	}
}

func TestSensitiveEnvironmentIsNotPassedToOllama(t *testing.T) {
	filtered := filterSensitiveEnvironment([]string{
		"PATH=/bin",
		"CUDA_VISIBLE_DEVICES=0",
		"KESTREL_OLLAMA_BEARER_TOKEN=" + testToken,
		"RUNPOD_API_KEY=control-secret",
		"HF_TOKEN=model-secret",
		"HTTPS_PROXY=https://proxy-user:proxy-pass@example.invalid",
	})
	joined := strings.Join(filtered, "\n")
	for _, secret := range []string{testToken, "control-secret", "model-secret", "proxy-pass"} {
		if strings.Contains(joined, secret) {
			t.Fatalf("child environment leaked %q: %s", secret, joined)
		}
	}
	for _, required := range []string{"PATH=/bin", "CUDA_VISIBLE_DEVICES=0"} {
		if !strings.Contains(joined, required) {
			t.Fatalf("child environment lost %q: %s", required, joined)
		}
	}
}
