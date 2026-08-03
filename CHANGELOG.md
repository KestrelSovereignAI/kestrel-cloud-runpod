# Changelog

All notable changes to `kestrel-cloud-runpod` are documented here.

## [0.7.0] - 2026-08-02

### Added

- Provider-neutral schema-3 Pod capacity quotes bound to exact training
  parameters, live v2 GPU identity/name/hourly price, startup/execution/runtime
  estimates, and an upward-rounded maximum cost ceiling.
- One-attempt catalog Pod acquisition with immutable image digests, scoped
  restart-recoverable bearers, opaque submit/status/result/cancel transport,
  deterministic ambiguous-create recovery, permanent termination, and final
  `/billing/pods` receipts.
- A cheap externally driven reconciliation surface that retains ambiguous and
  billing-pending work without treating an estimate as actual cost.
- An installed, single-pass `kestrel-runpod-reconcile-capacity` command with
  explicit host dependency injection, per-database process locking, bounded
  execution, content-free JSON summaries, and scheduler-facing exit statuses.
- Replayable non-destructive result retrieval followed by an explicit,
  owner/workload-bound acknowledgement, so the private catalog can commit a
  strict-decoded result durably before Pod termination.
- Owner/workload-bound capacity read accessors for non-mutating
  pre-acquisition absence checks and authoritative post-acknowledgement billing
  polling without exposing or coupling to SQLite.

### Changed

- Extracted training Pod lifecycle ownership into the canonical
  `PodCapacityLeaseService`, provider, and repository. Legacy training imports
  are compatibility aliases and cannot independently create or stop Pods.
- Added an atomic versioned migration from `training_pod_leases` to the single
  `pod_capacity_leases` table while preserving active and cleanup-family rows.
- Catalog attempt Pods no longer use persistent/network volumes by default;
  model-cache placement is an explicit benchmark decision.

## [0.6.1] - 2026-08-02

### Fixed

- Persist the caller cleanup token as the root identity for every LoRA training
  fallback attempt, so root-token cleanup releases active children after a
  process restart while returned child tokens retain exact-attempt semantics.
- Make family release durable and terminal-aware across concurrent caller and
  reconciler passes, and recognize deterministic fallback hashes migrated from
  the 0.5 schema without storing credentials or private workload routes.

## [0.6.0] - 2026-08-02

### Added

- The public SDK 0.34 inference-lease provider contract, registered as the
  dedicated `runpod` provider entry point.
- Deterministic Ollama capability/region matching, read-only live quotes, and
  owner-scoped acquire/status/release over the existing durable Runpod lease.
- Pending, exact-model-ready, failed, releasing, released, and expired state
  mapping with crash-safe duplicate acquisition and host-only route secrets.

### Changed

- Ollama route URLs are re-observed into process memory and scrubbed from the
  durable SQLite store; public lease serialization remains non-addressable.
- `manage_gpu` is now strictly an infrastructure/operator surface. The
  provider-neutral inference coordinator is the sole owner of LLM routing.
- Raised the minimum `kestrel-sovereign-sdk` version to 0.34.0.
- Poolless beta-v2 Serverless availability now fails with a specific no-create
  error instead of being silently discarded; no GPU-pool identifier is guessed.
- The typed v2 control client now lists network volumes so cleanup gates can
  prove that no storage resource remains after a disposable live run.
- Full-agent Ollama leases now advertise OpenAI-compatible tool calling, and
  runtime 1.1.0 refuses readiness unless the exact pinned model reports both
  `completion` and `tools`. Tests preserve streaming tool schemas and tool-call
  responses through the proxy.

## [0.5.0] - 2026-08-01

### Added

- Durable SQLite/CAS ownership records for persistent, reused, and newly created LoRA training Pods, persisted before the first billable mutation.
- Explicit cleanup tokens, ownership-aware stop semantics, ambiguous create/start recovery, workload job/result state, and the externally scheduled `kestrel-runpod-reconcile-training` command.
- Cancellation, crash, readiness/route, submission, status, result, teardown, and concurrent-acquisition coverage that retains every owned provider Pod ID until v2 confirms it stopped.

### Changed

- Runpod v2 Pod actions now surface transport/5xx outcomes as ambiguous instead of implying the mutation failed.

## [0.4.0] - 2026-08-01

### Added

- Durable, ownership-scoped private Ollama inference leases backed by SQLite WAL state and external reconciliation.
- Live-cost selection between load-balanced Serverless and dedicated Pods, with readiness, idle, deadline, and authorized-spend gates.
- Crash-safe ambiguous-create recovery, idempotent teardown, cold-start/cost telemetry, and the `kestrel-runpod-reconcile-ollama` command.
- Separate restricted Serverless and Pod workload credentials, anonymous-access rejection for Pod routes, and TLS-only bearer-token routes.

### Changed

- Replaced the legacy in-memory singleton Ollama session helpers with the durable lease contract.

## [0.3.0] - 2026-08-01

### Changed

- Migrated the Runpod integration to the beta v2 REST control plane with typed clients, live catalog placement, and pinned OpenAPI validation.

[0.7.0]: https://github.com/KestrelSovereignAI/kestrel-cloud-runpod/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/KestrelSovereignAI/kestrel-cloud-runpod/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/KestrelSovereignAI/kestrel-cloud-runpod/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/KestrelSovereignAI/kestrel-cloud-runpod/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/KestrelSovereignAI/kestrel-cloud-runpod/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/KestrelSovereignAI/kestrel-cloud-runpod/releases/tag/v0.3.0
