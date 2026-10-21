// Command kestrel-ollama-runtime supervises Ollama behind a fail-closed proxy.
package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

const (
	defaultListenAddress  = "0.0.0.0:11434"
	defaultUpstream       = "http://127.0.0.1:11435"
	defaultStartupTimeout = 30 * time.Minute
	defaultPollInterval   = 500 * time.Millisecond
	defaultHealthTimeout  = 2 * time.Second
	maxRequestBytes       = 30 << 20
	runtimeUID            = 10001
	runtimeGID            = 10001
)

var (
	runtimeVersion            = "development"
	ollamaVersion             = "unknown"
	pinnedModelRE             = regexp.MustCompile(`^([a-z0-9][a-z0-9._/-]*:[a-z0-9][a-z0-9._-]*)@sha256:([a-f0-9]{64})$`)
	modelDigestRE             = regexp.MustCompile(`^[a-f0-9]{64}$`)
	requiredModelCapabilities = []string{"completion", "tools"}
)

type modelPin struct {
	Name   string `json:"name"`
	Digest string `json:"digest"`
}

type runtimeConfig struct {
	ListenAddress      string
	Upstream           *url.URL
	BearerToken        string
	BearerTokenExpires time.Time
	RequiredModel      modelPin
	AllowedModels      map[string]modelPin
	Mode               string
	ModelStoragePath   string
	StartupTimeout     time.Duration
	PollInterval       time.Duration
	PreloadModel       bool
	ProvisionRequested *time.Time
	ContainerStartedAt time.Time
	TelemetryBinding   string
}

type startupState struct {
	mu                     sync.RWMutex
	ollamaProcessStartedAt *time.Time
	ollamaHealthyAt        *time.Time
	modelPullStartedAt     *time.Time
	modelPullCompletedAt   *time.Time
	modelLoadStartedAt     *time.Time
	modelReadyAt           *time.Time
	readyAt                *time.Time
	cacheHit               bool
	strategy               string
	terminalError          string
	ollamaProcessRunning   bool
	runtimeInstanceID      string
	telemetryRevision      uint64
	containerStartedMono   time.Time
	readyMono              time.Time
	inferenceStarted       uint64
	inferenceCompleted     uint64
	inferenceFailed        uint64
	inferenceCancelled     uint64
	inferenceInFlight      uint64
	firstInferenceStarted  *time.Time
	lastInferenceStarted   *time.Time
	lastInferenceCompleted *time.Time
	busyStartedMono        time.Time
	inferenceBusy          time.Duration
}

type telemetry struct {
	SchemaVersion                     int        `json:"schema_version"`
	RuntimeVersion                    string     `json:"runtime_version"`
	OllamaVersion                     string     `json:"ollama_version"`
	TelemetryBindingSHA256            string     `json:"telemetry_binding_sha256"`
	RuntimeInstanceID                 string     `json:"runtime_instance_id"`
	TelemetryRevision                 uint64     `json:"telemetry_revision"`
	ProvisionRequestedAt              *time.Time `json:"provision_requested_at"`
	ContainerStartedAt                time.Time  `json:"container_started_at"`
	OllamaProcessStartedAt            *time.Time `json:"ollama_process_started_at"`
	OllamaHealthyAt                   *time.Time `json:"ollama_healthy_at"`
	ModelPullStartedAt                *time.Time `json:"model_pull_started_at"`
	ModelPullCompletedAt              *time.Time `json:"model_pull_completed_at"`
	ModelLoadStartedAt                *time.Time `json:"model_load_started_at"`
	ModelReadyAt                      *time.Time `json:"model_ready_at"`
	ReadyAt                           *time.Time `json:"ready_at"`
	PlacementAndImagePullMilliseconds *int64     `json:"placement_and_image_pull_wall_ms"`
	OllamaBootMilliseconds            *int64     `json:"ollama_boot_ms"`
	ModelPullMilliseconds             *int64     `json:"model_pull_ms"`
	ModelLoadMilliseconds             *int64     `json:"model_load_ms"`
	ContainerToReadyMilliseconds      *int64     `json:"container_to_ready_ms"`
	CacheHit                          bool       `json:"cache_hit"`
	StartupStrategy                   string     `json:"startup_strategy"`
	Ready                             bool       `json:"ready"`
	OllamaProcessRunning              bool       `json:"ollama_process_running"`
	TerminalError                     *string    `json:"terminal_error"`
	InferenceStarted                  uint64     `json:"inference_started"`
	InferenceCompleted                uint64     `json:"inference_completed"`
	InferenceFailed                   uint64     `json:"inference_failed"`
	InferenceCancelled                uint64     `json:"inference_cancelled"`
	InferenceInFlight                 uint64     `json:"inference_in_flight"`
	FirstInferenceStartedAt           *time.Time `json:"first_inference_started_at"`
	LastInferenceStartedAt            *time.Time `json:"last_inference_started_at"`
	LastInferenceCompletedAt          *time.Time `json:"last_inference_completed_at"`
	InferenceBusyMilliseconds         int64      `json:"inference_busy_ms"`
	InferenceIdleMilliseconds         int64      `json:"inference_idle_ms"`
}

type runtimeProxy struct {
	config runtimeConfig
	state  *startupState
	proxy  *httputil.ReverseProxy
	now    func() time.Time
}

type exposedRoutePolicy struct {
	readOnly            bool
	requiresModelPolicy bool
	countsInference     bool
}

func main() {
	if len(os.Args) == 2 && os.Args[1] == "healthcheck" {
		os.Exit(runHealthcheck())
	}
	if err := run(); err != nil {
		log.Printf(`{"event":"runtime_exit","error":%q}`, safeError(err))
		os.Exit(1)
	}
}

func run() error {
	startedAt := time.Now()
	config, err := loadConfig(os.Getenv, startedAt.UTC())
	if err != nil {
		return err
	}
	if err = prepareRuntimeFilesystem(config.ModelStoragePath); err != nil {
		return err
	}
	scrubSensitiveProcessEnvironment()
	state := newStartupState("initializing", startedAt)
	handler := newRuntimeProxy(config, state, time.Now)
	server := &http.Server{
		Addr:              config.ListenAddress,
		Handler:           handler,
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       5 * time.Minute,
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	child, childExit, err := startOllama(config, state)
	if err != nil {
		return err
	}
	serverExit := make(chan error, 1)
	go func() {
		log.Printf(`{"event":"proxy_listening","port":%q,"runtime_version":%q}`, config.ListenAddress, runtimeVersion)
		serverExit <- server.ListenAndServe()
	}()
	startupExit := make(chan error, 1)
	go func() {
		startupExit <- bootstrap(ctx, config, state, http.DefaultClient)
	}()

	var terminal error
	childExited := false
	select {
	case <-ctx.Done():
	case err = <-serverExit:
		if !errors.Is(err, http.ErrServerClosed) {
			terminal = fmt.Errorf("proxy server failed: %w", err)
		}
	case err = <-childExit:
		childExited = true
		state.setProcessRunning(false)
		terminal = ollamaExitError(err)
	case err = <-startupExit:
		if err != nil {
			state.setTerminalFailure()
			terminal = fmt.Errorf("runtime initialization failed: %w", err)
		}
		if err == nil {
			select {
			case <-ctx.Done():
			case err = <-serverExit:
				if !errors.Is(err, http.ErrServerClosed) {
					terminal = fmt.Errorf("proxy server failed: %w", err)
				}
			case err = <-childExit:
				childExited = true
				state.setProcessRunning(false)
				terminal = ollamaExitError(err)
			}
		}
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_ = server.Shutdown(shutdownCtx)
	stopChild(child, childExit, childExited)
	return terminal
}

func ollamaExitError(err error) error {
	if err == nil {
		return errors.New("Ollama exited unexpectedly")
	}
	return fmt.Errorf("Ollama exited: %w", err)
}

func loadConfig(getenv func(string) string, startedAt time.Time) (runtimeConfig, error) {
	token := getenv("KESTREL_OLLAMA_BEARER_TOKEN")
	if len(token) < 32 || strings.TrimSpace(token) != token || strings.ContainsAny(token, "\r\n") {
		return runtimeConfig{}, errors.New("KESTREL_OLLAMA_BEARER_TOKEN must be at least 32 non-whitespace characters")
	}
	expiresAt, err := time.Parse(time.RFC3339, getenv("KESTREL_OLLAMA_BEARER_TOKEN_EXPIRES_AT"))
	if err != nil || !expiresAt.After(startedAt) {
		return runtimeConfig{}, errors.New("KESTREL_OLLAMA_BEARER_TOKEN_EXPIRES_AT must be a future RFC3339 timestamp")
	}
	allowed, err := parseAllowedModels(getenv("KESTREL_OLLAMA_ALLOWED_MODELS"))
	if err != nil {
		return runtimeConfig{}, err
	}
	requiredName := strings.TrimSpace(getenv("KESTREL_OLLAMA_REQUIRED_MODEL"))
	required, ok := allowed[requiredName]
	if !ok {
		return runtimeConfig{}, errors.New("KESTREL_OLLAMA_REQUIRED_MODEL must exactly match an operator-pinned allowlist entry")
	}
	telemetryBinding := getenv("KESTREL_OLLAMA_TELEMETRY_BINDING_SHA256")
	if !modelDigestRE.MatchString(telemetryBinding) {
		return runtimeConfig{}, errors.New("KESTREL_OLLAMA_TELEMETRY_BINDING_SHA256 must be 64 lowercase hex characters")
	}
	mode := getenv("KESTREL_OLLAMA_MODE")
	if mode != "dedicated_pod" && mode != "serverless_load_balancer" {
		return runtimeConfig{}, errors.New("KESTREL_OLLAMA_MODE must select dedicated_pod or serverless_load_balancer")
	}
	storagePath, err := validateModelStoragePath(getenv("KESTREL_OLLAMA_MODEL_STORAGE_PATH"), mode)
	if err != nil {
		return runtimeConfig{}, err
	}
	listenAddress := getenv("PORT")
	if listenAddress == "" {
		listenAddress = defaultListenAddress
	} else if _, err = strconv.Atoi(listenAddress); err == nil {
		listenAddress = "0.0.0.0:" + listenAddress
	}
	if !strings.HasPrefix(listenAddress, "0.0.0.0:") {
		return runtimeConfig{}, errors.New("PORT must be a numeric port or an explicit 0.0.0.0 listener")
	}
	upstream, _ := url.Parse(defaultUpstream)
	startupTimeout, err := durationFromSeconds(getenv("KESTREL_OLLAMA_STARTUP_TIMEOUT_SECONDS"), defaultStartupTimeout)
	if err != nil {
		return runtimeConfig{}, fmt.Errorf("invalid KESTREL_OLLAMA_STARTUP_TIMEOUT_SECONDS: %w", err)
	}
	pollInterval, err := durationFromMilliseconds(getenv("KESTREL_OLLAMA_POLL_INTERVAL_MS"), defaultPollInterval)
	if err != nil {
		return runtimeConfig{}, fmt.Errorf("invalid KESTREL_OLLAMA_POLL_INTERVAL_MS: %w", err)
	}
	preload := true
	if raw := getenv("KESTREL_OLLAMA_PRELOAD_MODEL"); raw != "" {
		preload, err = strconv.ParseBool(raw)
		if err != nil {
			return runtimeConfig{}, errors.New("KESTREL_OLLAMA_PRELOAD_MODEL must be true or false")
		}
	}
	var provisionRequested *time.Time
	if raw := getenv("KESTREL_OLLAMA_PROVISION_REQUESTED_AT"); raw != "" {
		parsed, parseErr := time.Parse(time.RFC3339Nano, raw)
		if parseErr != nil || parsed.After(startedAt) {
			return runtimeConfig{}, errors.New("KESTREL_OLLAMA_PROVISION_REQUESTED_AT must be a non-future RFC3339 timestamp")
		}
		parsed = parsed.UTC()
		provisionRequested = &parsed
	}
	return runtimeConfig{
		ListenAddress:      listenAddress,
		Upstream:           upstream,
		BearerToken:        token,
		BearerTokenExpires: expiresAt.UTC(),
		RequiredModel:      required,
		AllowedModels:      allowed,
		Mode:               mode,
		ModelStoragePath:   storagePath,
		StartupTimeout:     startupTimeout,
		PollInterval:       pollInterval,
		PreloadModel:       preload,
		ProvisionRequested: provisionRequested,
		ContainerStartedAt: startedAt,
		TelemetryBinding:   telemetryBinding,
	}, nil
}

func validateModelStoragePath(storagePath, mode string) (string, error) {
	if !strings.HasPrefix(storagePath, "/") || path.Clean(storagePath) != storagePath {
		return "", errors.New("KESTREL_OLLAMA_MODEL_STORAGE_PATH must be normalized and absolute")
	}
	allowedRoots := []string{"/models", "/workspace"}
	if mode == "serverless_load_balancer" {
		allowedRoots = []string{"/models", "/runpod-volume"}
	}
	for _, root := range allowedRoots {
		if storagePath == root || strings.HasPrefix(storagePath, root+"/") {
			return storagePath, nil
		}
	}
	return "", fmt.Errorf("KESTREL_OLLAMA_MODEL_STORAGE_PATH is invalid for %s", mode)
}

func prepareRuntimeFilesystem(storagePath string) error {
	if os.Geteuid() != 0 {
		return fmt.Errorf("runtime init requires root before dropping to UID %d", runtimeUID)
	}
	if err := os.MkdirAll(storagePath, 0o750); err != nil {
		return fmt.Errorf("prepare model storage: %w", err)
	}
	if err := validateModelStorageTree(storagePath); err != nil {
		return err
	}
	if err := filepath.WalkDir(storagePath, func(itemPath string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.Type()&os.ModeSymlink != 0 {
			return fmt.Errorf("model storage contains unsupported symlink %q", itemPath)
		}
		if err := os.Lchown(itemPath, runtimeUID, runtimeGID); err != nil {
			return fmt.Errorf("set model storage ownership for %q: %w", itemPath, err)
		}
		return nil
	}); err != nil {
		return fmt.Errorf("secure model storage: %w", err)
	}
	if err := syscall.Setgroups([]int{runtimeGID}); err != nil {
		return fmt.Errorf("set runtime supplementary groups: %w", err)
	}
	if err := syscall.Setgid(runtimeGID); err != nil {
		return fmt.Errorf("drop runtime group privilege: %w", err)
	}
	if err := syscall.Setuid(runtimeUID); err != nil {
		return fmt.Errorf("drop runtime user privilege: %w", err)
	}
	if os.Geteuid() != runtimeUID || os.Getegid() != runtimeGID {
		return errors.New("runtime privilege drop did not reach the configured UID/GID")
	}
	syscall.Umask(0o027)
	probe, err := os.CreateTemp(storagePath, ".kestrel-write-probe-")
	if err != nil {
		return fmt.Errorf("model storage is not writable by the runtime user: %w", err)
	}
	probePath := probe.Name()
	if err = probe.Close(); err != nil {
		return fmt.Errorf("close model storage write probe: %w", err)
	}
	if err = os.Remove(probePath); err != nil {
		return fmt.Errorf("remove model storage write probe: %w", err)
	}
	return nil
}

func validateModelStorageTree(storagePath string) error {
	return filepath.WalkDir(storagePath, func(itemPath string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.Type()&os.ModeSymlink != 0 {
			return fmt.Errorf("model storage contains unsupported symlink %q", itemPath)
		}
		return nil
	})
}

func parseAllowedModels(raw string) (map[string]modelPin, error) {
	result := make(map[string]modelPin)
	for _, item := range strings.Split(raw, ",") {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		match := pinnedModelRE.FindStringSubmatch(item)
		if match == nil {
			return nil, fmt.Errorf("invalid pinned model %q; expected name:tag@sha256:<64 lowercase hex>", item)
		}
		if _, duplicate := result[match[1]]; duplicate {
			return nil, fmt.Errorf("duplicate model %q in KESTREL_OLLAMA_ALLOWED_MODELS", match[1])
		}
		result[match[1]] = modelPin{Name: match[1], Digest: "sha256:" + match[2]}
	}
	if len(result) == 0 {
		return nil, errors.New("KESTREL_OLLAMA_ALLOWED_MODELS must contain at least one digest-pinned model")
	}
	return result, nil
}

func durationFromSeconds(raw string, fallback time.Duration) (time.Duration, error) {
	return positiveDuration(raw, fallback, time.Second)
}

func durationFromMilliseconds(raw string, fallback time.Duration) (time.Duration, error) {
	return positiveDuration(raw, fallback, time.Millisecond)
}

func positiveDuration(raw string, fallback, unit time.Duration) (time.Duration, error) {
	if raw == "" {
		return fallback, nil
	}
	value, err := strconv.ParseInt(raw, 10, 64)
	if err != nil || value < 1 {
		return 0, errors.New("value must be a positive integer")
	}
	return time.Duration(value) * unit, nil
}

func startOllama(config runtimeConfig, state *startupState) (*exec.Cmd, <-chan error, error) {
	// The parent owns graceful termination. exec.CommandContext would send an
	// immediate SIGKILL as soon as the runtime context is cancelled, bypassing
	// the bounded SIGTERM shutdown below.
	cmd := exec.Command("/bin/ollama", "serve")
	cmd.Env = replaceEnvironment(filterSensitiveEnvironment(os.Environ()), map[string]string{
		"HOME":              "/var/lib/kestrel-ollama",
		"OLLAMA_DEBUG":      "false",
		"OLLAMA_HOST":       strings.TrimPrefix(config.Upstream.String(), "http://"),
		"OLLAMA_KEEP_ALIVE": "-1",
		"OLLAMA_MODELS":     config.ModelStoragePath,
	})
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return nil, nil, fmt.Errorf("start Ollama: %w", err)
	}
	now := time.Now().UTC()
	state.mu.Lock()
	state.ollamaProcessStartedAt = &now
	state.ollamaProcessRunning = true
	state.mu.Unlock()
	exit := make(chan error, 1)
	go func() { exit <- cmd.Wait() }()
	return cmd, exit, nil
}

func scrubSensitiveProcessEnvironment() {
	for _, item := range os.Environ() {
		name, _, found := strings.Cut(item, "=")
		if found && isSensitiveEnvironmentName(name) {
			_ = os.Unsetenv(name)
		}
	}
}

func filterSensitiveEnvironment(environment []string) []string {
	result := make([]string, 0, len(environment))
	for _, item := range environment {
		name, _, found := strings.Cut(item, "=")
		if !found || isSensitiveEnvironmentName(name) {
			continue
		}
		result = append(result, item)
	}
	return result
}

func isSensitiveEnvironmentName(name string) bool {
	upper := strings.ToUpper(name)
	for _, marker := range []string{"AUTHORIZATION", "CREDENTIAL", "PASSWORD", "SECRET", "TOKEN"} {
		if strings.Contains(upper, marker) {
			return true
		}
	}
	return upper == "RUNPOD_API_KEY" || strings.HasSuffix(upper, "_API_KEY") || strings.HasSuffix(upper, "_PROXY")
}

func replaceEnvironment(environment []string, replacements map[string]string) []string {
	result := make([]string, 0, len(environment)+len(replacements))
	for _, item := range environment {
		name, _, found := strings.Cut(item, "=")
		if found {
			if _, replace := replacements[name]; replace {
				continue
			}
		}
		result = append(result, item)
	}
	keys := make([]string, 0, len(replacements))
	for key := range replacements {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	for _, key := range keys {
		result = append(result, key+"="+replacements[key])
	}
	return result
}

func stopChild(child *exec.Cmd, exit <-chan error, alreadyExited bool) {
	// The main select consumes the sole exit notification when Ollama exits on
	// its own. Waiting on that channel a second time would deadlock shutdown.
	if alreadyExited || child == nil || child.Process == nil {
		return
	}
	_ = child.Process.Signal(syscall.SIGTERM)
	select {
	case <-exit:
	case <-time.After(10 * time.Second):
		_ = child.Process.Kill()
		<-exit
	}
}

func bootstrap(parent context.Context, config runtimeConfig, state *startupState, client *http.Client) error {
	ctx, cancel := context.WithTimeout(parent, config.StartupTimeout)
	defer cancel()
	if err := waitForOllama(ctx, config, state, client); err != nil {
		return err
	}
	present, err := exactModelReady(ctx, client, config.Upstream, config.RequiredModel)
	if err != nil {
		return err
	}
	if present {
		state.setStrategy("cache_hit", true)
	} else {
		state.setStrategy("pulled", false)
		state.markPullStarted()
		if err = ollamaJSON(ctx, client, config.Upstream, "/api/pull", map[string]any{
			"name": config.RequiredModel.Name, "stream": false,
		}); err != nil {
			return fmt.Errorf("pull pinned model: %w", err)
		}
		state.markPullCompleted()
		present, err = exactModelReady(ctx, client, config.Upstream, config.RequiredModel)
		if err != nil {
			return err
		}
		if !present {
			return errors.New("pulled model digest does not match its configured pin")
		}
	}
	if config.PreloadModel {
		state.markLoadStarted()
		if err = ollamaJSON(ctx, client, config.Upstream, "/api/generate", map[string]any{
			"model": config.RequiredModel.Name, "prompt": "", "stream": false, "keep_alive": -1,
		}); err != nil {
			return fmt.Errorf("preload pinned model: %w", err)
		}
	}
	state.markReady()
	log.Printf(`{"event":"runtime_ready","cache_hit":%t}`, present && state.snapshot(config).CacheHit)
	return nil
}

func waitForOllama(ctx context.Context, config runtimeConfig, state *startupState, client *http.Client) error {
	ticker := time.NewTicker(config.PollInterval)
	defer ticker.Stop()
	for {
		req, _ := http.NewRequestWithContext(ctx, http.MethodGet, config.Upstream.String()+"/api/version", nil)
		response, err := client.Do(req)
		if err == nil {
			_, _ = io.Copy(io.Discard, io.LimitReader(response.Body, 1<<20))
			_ = response.Body.Close()
			if response.StatusCode >= 200 && response.StatusCode < 300 {
				state.markOllamaHealthy()
				return nil
			}
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf("wait for Ollama: %w", ctx.Err())
		case <-ticker.C:
		}
	}
}

func exactModelReady(ctx context.Context, client *http.Client, upstream *url.URL, pin modelPin) (bool, error) {
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, upstream.String()+"/api/tags", nil)
	response, err := client.Do(req)
	if err != nil {
		return false, fmt.Errorf("list Ollama models: %w", err)
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return false, fmt.Errorf("list Ollama models returned HTTP %d", response.StatusCode)
	}
	var payload struct {
		Models []struct {
			Name         string   `json:"name"`
			Model        string   `json:"model"`
			Digest       string   `json:"digest"`
			Capabilities []string `json:"capabilities"`
		} `json:"models"`
	}
	decoder := json.NewDecoder(io.LimitReader(response.Body, 4<<20))
	if err = decoder.Decode(&payload); err != nil {
		return false, errors.New("Ollama returned invalid model inventory")
	}
	for _, item := range payload.Models {
		name := item.Name
		if name == "" {
			name = item.Model
		}
		if name == pin.Name {
			actual, validActual := normalizedModelDigest(item.Digest)
			expected, validExpected := normalizedModelDigest(pin.Digest)
			if !validActual || !validExpected || subtle.ConstantTimeCompare([]byte(actual), []byte(expected)) != 1 {
				return false, nil
			}
			available := make(map[string]bool, len(item.Capabilities))
			for _, capability := range item.Capabilities {
				available[capability] = true
			}
			for _, capability := range requiredModelCapabilities {
				if !available[capability] {
					return false, fmt.Errorf("configured model lacks required Ollama capability %q", capability)
				}
			}
			return true, nil
		}
	}
	return false, nil
}

func normalizedModelDigest(value string) (string, bool) {
	normalized := strings.TrimPrefix(value, "sha256:")
	return normalized, modelDigestRE.MatchString(normalized)
}

func ollamaJSON(ctx context.Context, client *http.Client, upstream *url.URL, path string, payload any) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, upstream.String()+path, bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	response, err := client.Do(req)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	_, _ = io.Copy(io.Discard, io.LimitReader(response.Body, 4<<20))
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return fmt.Errorf("Ollama returned HTTP %d", response.StatusCode)
	}
	return nil
}

func newRuntimeProxy(config runtimeConfig, state *startupState, now func() time.Time) http.Handler {
	state.requireRuntimeInitialized()
	reverse := &httputil.ReverseProxy{
		Rewrite: func(request *httputil.ProxyRequest) {
			request.SetURL(config.Upstream)
			request.SetXForwarded()
			request.Out.Header.Del("Authorization")
			request.Out.Header.Del("Proxy-Authorization")
			request.Out.Header.Del("X-Kestrel-Ollama-Authorization")
		},
		FlushInterval: -1,
		ErrorHandler: func(writer http.ResponseWriter, _ *http.Request, _ error) {
			writeJSON(writer, http.StatusBadGateway, map[string]string{"error": "Ollama upstream unavailable"})
		},
	}
	return &runtimeProxy{config: config, state: state, proxy: reverse, now: now}
}

func (runtime *runtimeProxy) ServeHTTP(writer http.ResponseWriter, request *http.Request) {
	path := request.URL.Path
	if isHealthPath(path) {
		runtime.serveHealth(writer, request)
		return
	}
	if !runtime.authorized(request) {
		writer.Header().Set("WWW-Authenticate", `Bearer realm="kestrel-ollama"`)
		writeJSON(writer, http.StatusUnauthorized, map[string]string{"error": "unauthorized"})
		return
	}
	if path == "/kestrel/telemetry" {
		if request.Method != http.MethodGet && request.Method != http.MethodHead {
			methodNotAllowed(writer, http.MethodGet)
			return
		}
		writeJSON(writer, http.StatusOK, runtime.state.snapshot(runtime.config))
		return
	}
	if isForbiddenManagementPath(path) {
		writeJSON(writer, http.StatusForbidden, map[string]string{"error": "model mutation is disabled"})
		return
	}
	policy, allowed := policyForRoute(path, request.Method)
	if !allowed {
		writeJSON(writer, http.StatusNotFound, map[string]string{"error": "route is not exposed"})
		return
	}
	if policy.requiresModelPolicy {
		model, status, err := inspectModelRequest(writer, request)
		if err != nil {
			writeJSON(writer, status, map[string]string{"error": err.Error()})
			return
		}
		if _, allowed := runtime.config.AllowedModels[model]; !allowed {
			writeJSON(writer, http.StatusForbidden, map[string]string{"error": "model is not allowlisted"})
			return
		}
		if !policy.countsInference {
			runtime.proxy.ServeHTTP(writer, request)
			return
		}
		started := runtime.state.beginInference(runtime.now())
		tracked := &statusResponseWriter{ResponseWriter: writer}
		var finishOnce sync.Once
		finish := func(cancelled, failed bool) {
			finishOnce.Do(func() {
				runtime.state.finishInference(
					started,
					runtime.now(),
					cancelled,
					failed,
				)
			})
		}
		stopCancellation := context.AfterFunc(request.Context(), func() {
			finish(true, false)
		})
		runtime.proxy.ServeHTTP(tracked, request)
		stopCancellation()
		finish(
			request.Context().Err() != nil,
			tracked.status >= http.StatusInternalServerError,
		)
		return
	}
	runtime.proxy.ServeHTTP(writer, request)
}

type statusResponseWriter struct {
	http.ResponseWriter
	status int
}

func (writer *statusResponseWriter) WriteHeader(status int) {
	writer.status = status
	writer.ResponseWriter.WriteHeader(status)
}

func (writer *statusResponseWriter) Write(value []byte) (int, error) {
	if writer.status == 0 {
		writer.status = http.StatusOK
	}
	return writer.ResponseWriter.Write(value)
}

func (writer *statusResponseWriter) Unwrap() http.ResponseWriter {
	return writer.ResponseWriter
}

func (runtime *runtimeProxy) authorized(request *http.Request) bool {
	if !runtime.now().Before(runtime.config.BearerTokenExpires) {
		return false
	}
	provided := request.Header.Get("Authorization")
	prefix := "Bearer "
	if !strings.HasPrefix(provided, prefix) {
		return false
	}
	token := strings.TrimPrefix(provided, prefix)
	return len(token) == len(runtime.config.BearerToken) && subtle.ConstantTimeCompare([]byte(token), []byte(runtime.config.BearerToken)) == 1
}

func (runtime *runtimeProxy) serveHealth(writer http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodGet && request.Method != http.MethodHead {
		methodNotAllowed(writer, http.MethodGet)
		return
	}
	snapshot := runtime.state.snapshot(runtime.config)
	if request.URL.Path == "/health/live" {
		if snapshot.OllamaProcessRunning {
			writeJSON(writer, http.StatusOK, map[string]string{"status": "live"})
			return
		}
		writeJSON(writer, http.StatusServiceUnavailable, map[string]string{"status": "unavailable"})
		return
	}
	if snapshot.TerminalError != nil || !snapshot.OllamaProcessRunning {
		writeJSON(writer, http.StatusServiceUnavailable, map[string]string{"status": "failed"})
		return
	}
	if snapshot.Ready {
		if !runtime.now().Before(runtime.config.BearerTokenExpires) {
			writeJSON(writer, http.StatusServiceUnavailable, map[string]string{"status": "expired"})
			return
		}
		ctx, cancel := context.WithTimeout(request.Context(), defaultHealthTimeout)
		defer cancel()
		present, err := exactModelReady(
			ctx,
			&http.Client{Timeout: defaultHealthTimeout},
			runtime.config.Upstream,
			runtime.config.RequiredModel,
		)
		if err == nil && present {
			writeJSON(writer, http.StatusOK, map[string]string{"status": "ready"})
			return
		}
		writeJSON(writer, http.StatusServiceUnavailable, map[string]string{"status": "unavailable"})
		return
	}
	writeJSON(writer, http.StatusNoContent, nil)
}

func isHealthPath(path string) bool {
	return path == "/ping" || path == "/health/live" || path == "/health/ready"
}

func isForbiddenManagementPath(path string) bool {
	return path == "/api/create" || path == "/api/copy" || path == "/api/delete" || path == "/api/pull" || path == "/api/push" || strings.HasPrefix(path, "/api/blobs")
}

func policyForRoute(path, method string) (exposedRoutePolicy, bool) {
	var policy exposedRoutePolicy
	switch path {
	case "/", "/api/tags", "/api/ps", "/api/version", "/v1/models":
		policy = exposedRoutePolicy{readOnly: true}
	case "/api/generate", "/api/chat", "/api/embed", "/api/embeddings",
		"/v1/chat/completions", "/v1/completions", "/v1/embeddings":
		policy = exposedRoutePolicy{requiresModelPolicy: true, countsInference: true}
	case "/api/show":
		policy = exposedRoutePolicy{requiresModelPolicy: true}
	default:
		if strings.HasPrefix(path, "/v1/models/") {
			policy = exposedRoutePolicy{readOnly: true}
		} else {
			return exposedRoutePolicy{}, false
		}
	}
	if policy == (exposedRoutePolicy{}) {
		return exposedRoutePolicy{}, false
	}
	if policy.readOnly {
		return policy, method == http.MethodGet || method == http.MethodHead
	}
	return policy, method == http.MethodPost
}

func inspectModelRequest(writer http.ResponseWriter, request *http.Request) (string, int, error) {
	reader := http.MaxBytesReader(writer, request.Body, maxRequestBytes)
	body, err := io.ReadAll(reader)
	_ = reader.Close()
	if err != nil {
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			return "", http.StatusRequestEntityTooLarge, errors.New("request exceeds the 30 MiB limit")
		}
		return "", http.StatusBadRequest, errors.New("request body cannot be read")
	}
	request.Body = io.NopCloser(bytes.NewReader(body))
	var payload struct {
		Model string `json:"model"`
		Name  string `json:"name"`
	}
	// Match encoding/json's struct-key semantics used by Ollama: unknown
	// inference options are ignored, keys match case-insensitively, and the last
	// duplicate wins. Authorizing a differently parsed projection would let the
	// forwarded body select a model other than the one checked here.
	if err = json.Unmarshal(body, &payload); err != nil {
		return "", http.StatusBadRequest, errors.New("request body must be a JSON object with string model identifiers")
	}
	model := payload.Model
	if payload.Model != "" && payload.Name != "" && payload.Model != payload.Name {
		return "", http.StatusBadRequest, errors.New("model and name identifiers must match")
	}
	if model == "" {
		model = payload.Name
	}
	if model == "" {
		return "", http.StatusBadRequest, errors.New("model identifier is required")
	}
	return model, 0, nil
}

func methodNotAllowed(writer http.ResponseWriter, allowed string) {
	writer.Header().Set("Allow", allowed)
	writeJSON(writer, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
}

func writeJSON(writer http.ResponseWriter, status int, payload any) {
	writer.Header().Set("Content-Type", "application/json")
	writer.Header().Set("Cache-Control", "no-store")
	writer.WriteHeader(status)
	if payload != nil && status != http.StatusNoContent {
		_ = json.NewEncoder(writer).Encode(payload)
	}
}

func (state *startupState) setProcessRunning(running bool) {
	state.mu.Lock()
	defer state.mu.Unlock()
	state.ollamaProcessRunning = running
}

func (state *startupState) setTerminalFailure() {
	state.mu.Lock()
	defer state.mu.Unlock()
	state.terminalError = "runtime_initialization_failed"
}

func (state *startupState) setStrategy(strategy string, cacheHit bool) {
	state.mu.Lock()
	defer state.mu.Unlock()
	state.strategy = strategy
	state.cacheHit = cacheHit
}

func (state *startupState) markOllamaHealthy() {
	state.mu.Lock()
	defer state.mu.Unlock()
	now := time.Now().UTC()
	state.ollamaHealthyAt = &now
}

func (state *startupState) markPullStarted() {
	state.mu.Lock()
	defer state.mu.Unlock()
	now := time.Now().UTC()
	state.modelPullStartedAt = &now
}

func (state *startupState) markPullCompleted() {
	state.mu.Lock()
	defer state.mu.Unlock()
	now := time.Now().UTC()
	state.modelPullCompletedAt = &now
}

func (state *startupState) markLoadStarted() {
	state.mu.Lock()
	defer state.mu.Unlock()
	now := time.Now().UTC()
	state.modelLoadStartedAt = &now
}

func (state *startupState) markReady() {
	state.mu.Lock()
	defer state.mu.Unlock()
	now := time.Now().UTC()
	state.modelReadyAt = &now
	state.readyAt = &now
	state.readyMono = time.Now()
}

func (state *startupState) snapshot(config runtimeConfig) telemetry {
	state.mu.Lock()
	defer state.mu.Unlock()
	state.requireRuntimeInitialized()
	state.telemetryRevision++
	terminalError := (*string)(nil)
	if state.terminalError != "" {
		value := state.terminalError
		terminalError = &value
	}
	busy := state.inferenceBusy
	nowMono := time.Now()
	if state.inferenceInFlight > 0 && !state.busyStartedMono.IsZero() {
		busy += nowMono.Sub(state.busyStartedMono)
	}
	idle := time.Duration(0)
	if !state.readyMono.IsZero() {
		elapsed := nowMono.Sub(state.readyMono)
		if elapsed > busy {
			idle = elapsed - busy
		}
	}
	result := telemetry{
		SchemaVersion: 2, RuntimeVersion: runtimeVersion, OllamaVersion: ollamaVersion,
		TelemetryBindingSHA256: config.TelemetryBinding, RuntimeInstanceID: state.runtimeInstanceID,
		TelemetryRevision:    state.telemetryRevision,
		ProvisionRequestedAt: config.ProvisionRequested, ContainerStartedAt: config.ContainerStartedAt,
		OllamaProcessStartedAt: state.ollamaProcessStartedAt, OllamaHealthyAt: state.ollamaHealthyAt,
		ModelPullStartedAt: state.modelPullStartedAt, ModelPullCompletedAt: state.modelPullCompletedAt,
		ModelLoadStartedAt: state.modelLoadStartedAt, ModelReadyAt: state.modelReadyAt, ReadyAt: state.readyAt,
		CacheHit: state.cacheHit, StartupStrategy: state.strategy, Ready: state.readyAt != nil,
		OllamaProcessRunning: state.ollamaProcessRunning, TerminalError: terminalError,
		InferenceStarted: state.inferenceStarted, InferenceCompleted: state.inferenceCompleted,
		InferenceFailed: state.inferenceFailed, InferenceCancelled: state.inferenceCancelled,
		InferenceInFlight: state.inferenceInFlight, FirstInferenceStartedAt: state.firstInferenceStarted,
		LastInferenceStartedAt: state.lastInferenceStarted, LastInferenceCompletedAt: state.lastInferenceCompleted,
		InferenceBusyMilliseconds: busy.Milliseconds(), InferenceIdleMilliseconds: idle.Milliseconds(),
	}
	result.PlacementAndImagePullMilliseconds = elapsedMilliseconds(config.ProvisionRequested, &config.ContainerStartedAt)
	result.OllamaBootMilliseconds = elapsedMilliseconds(state.ollamaProcessStartedAt, state.ollamaHealthyAt)
	result.ModelPullMilliseconds = elapsedMilliseconds(state.modelPullStartedAt, state.modelPullCompletedAt)
	result.ModelLoadMilliseconds = elapsedMilliseconds(state.modelLoadStartedAt, state.modelReadyAt)
	if !state.containerStartedMono.IsZero() && !state.readyMono.IsZero() {
		value := state.readyMono.Sub(state.containerStartedMono).Milliseconds()
		if value >= 0 {
			result.ContainerToReadyMilliseconds = &value
		}
	}
	return result
}

func newStartupState(strategy string, containerStartedAt time.Time) *startupState {
	if containerStartedAt.IsZero() {
		panic("runtime startup origin is unavailable")
	}
	value := make([]byte, 16)
	if _, err := rand.Read(value); err != nil {
		panic("runtime entropy unavailable")
	}
	return &startupState{
		strategy:             strategy,
		runtimeInstanceID:    hex.EncodeToString(value),
		containerStartedMono: containerStartedAt,
	}
}

func (state *startupState) requireRuntimeInitialized() {
	if state.runtimeInstanceID == "" || state.containerStartedMono.IsZero() {
		panic("runtime state was not initialized")
	}
}

func (state *startupState) beginInference(observedAt time.Time) time.Time {
	state.mu.Lock()
	defer state.mu.Unlock()
	startedMono := time.Now()
	observed := observedAt.UTC()
	state.inferenceStarted++
	state.inferenceInFlight++
	if state.firstInferenceStarted == nil {
		state.firstInferenceStarted = &observed
	}
	state.lastInferenceStarted = &observed
	if state.inferenceInFlight == 1 {
		state.busyStartedMono = startedMono
	}
	return startedMono
}

func (state *startupState) finishInference(startedMono time.Time, observedAt time.Time, cancelled, failed bool) {
	state.mu.Lock()
	defer state.mu.Unlock()
	if state.inferenceInFlight == 0 {
		panic("inference telemetry underflow")
	}
	state.inferenceInFlight--
	observed := observedAt.UTC()
	state.lastInferenceCompleted = &observed
	if cancelled {
		state.inferenceCancelled++
	} else if failed {
		state.inferenceFailed++
	} else {
		state.inferenceCompleted++
	}
	if state.inferenceInFlight == 0 {
		if !state.busyStartedMono.IsZero() {
			state.inferenceBusy += time.Since(state.busyStartedMono)
		}
		state.busyStartedMono = time.Time{}
	}
	_ = startedMono
}

func elapsedMilliseconds(start, end *time.Time) *int64 {
	if start == nil || end == nil || end.Before(*start) {
		return nil
	}
	value := end.Sub(*start).Milliseconds()
	return &value
}

func safeError(err error) string {
	if err == nil {
		return ""
	}
	message := err.Error()
	if len(message) > 240 {
		message = message[:240]
	}
	return strings.Map(func(r rune) rune {
		if r < 0x20 || r == 0x7f {
			return ' '
		}
		return r
	}, message)
}

func runHealthcheck() int {
	port := os.Getenv("PORT_HEALTH")
	if port == "" {
		port = os.Getenv("PORT")
	}
	if port == "" {
		port = "11434"
	}
	if strings.Contains(port, ":") {
		_, port, _ = strings.Cut(port, ":")
	}
	client := &http.Client{Timeout: 2 * time.Second}
	// Docker health represents process liveness. Runpod separately probes /ping
	// for 204-initializing versus 200-ready, so a long cold model pull must not
	// make an otherwise live container unhealthy.
	response, err := client.Get("http://127.0.0.1:" + port + "/health/live")
	if err != nil {
		return 1
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return 1
	}
	return 0
}
