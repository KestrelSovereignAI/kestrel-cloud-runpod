# Changelog

All notable changes to `kestrel-cloud-runpod` are documented here.

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

[0.4.0]: https://github.com/KestrelSovereignAI/kestrel-cloud-runpod/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/KestrelSovereignAI/kestrel-cloud-runpod/releases/tag/v0.3.0
