# Changelog

All notable changes to `kestrel-cloud-runpod` are documented here.

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

[0.6.0]: https://github.com/KestrelSovereignAI/kestrel-cloud-runpod/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/KestrelSovereignAI/kestrel-cloud-runpod/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/KestrelSovereignAI/kestrel-cloud-runpod/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/KestrelSovereignAI/kestrel-cloud-runpod/releases/tag/v0.3.0
