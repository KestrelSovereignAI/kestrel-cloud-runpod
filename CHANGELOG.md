# Changelog

All notable changes to `kestrel-cloud-runpod` are documented here.

## [0.9.0] - 2026-08-04

### Added

- `kestrel_cloud_runpod.signed_invocations`: canonical Ed25519 receipts for
  authenticated agent invocations — signer, verifier, receipt trust pinning and
  the attested request/response pair. A serving boundary can import the signer
  without importing any provider lifecycle code, and an external verifier can
  pin the exact route, owner, companion, agent and public key that identify one
  execution target.
- `kestrel_cloud_runpod.dogfood_contracts`: the product-neutral typed contracts
  (`DogfoodLane`, `DogfoodPhase`, `ResourceType`, `ResourceIdentity`,
  `ExpectedResource`, `ResourcePlan`, `ProviderAttemptIdentity`, `SpendQuote`,
  `PhaseObservation`) extracted from the live dogfood harness so production
  consumers can depend on the shapes without the orchestrator. Nothing in this
  module executes a run, provisions a resource, or spends money.
- `cryptography` as an explicit runtime dependency. It is imported directly by
  `signed_invocations`; it previously resolved only because `kestrel-sovereign`
  happened to pull it in transitively.

### Fixed

- `serialization.load_der_{public,private}_key` raises
  `cryptography.exceptions.UnsupportedAlgorithm`, which is **not** a
  `ValueError`, for well-formed DER carrying an unrecognized algorithm OID.
  Both key-loading sites caught only `ValueError`, so that input class escaped
  `SignedInvocationError` — the module's entire error contract — and skipped
  the `isinstance(..., Ed25519…)` check entirely. A consumer wrapping signer
  construction in `except ValueError` died with an unhandled traceback instead
  of its intended configuration error.
- `PhaseObservation.to_evidence` now validates both of its caller-supplied
  arguments. `run_id` reached the persisted evidence record without passing
  through `_safe_identifier` — the only run_id in either module that did not —
  and `observed_at` raised `AttributeError` rather than `DogfoodSafetyError`
  when handed a deserialized ISO string, because `dogfood_contracts._iso`
  lacked the `isinstance` check its `signed_invocations` twin has. Both are
  the same escape class as the `UnsupportedAlgorithm` leak above.
- `AttestedInvokeResponse.to_payload()` returned the live `phase_evidence`
  mapping on a frozen dataclass, so a caller could mutate signed evidence in
  place through the returned payload and break `verify_phase_evidence`.
- Digest validation applied `_SHA256.fullmatch` directly to caller-supplied
  attributes at four sites. `re.Pattern.fullmatch` raises `TypeError` on
  non-`str` input, so a `bytes` digest — `hashlib.sha256(x).digest()` where
  `.hexdigest()` was meant — escaped `DogfoodSafetyError` entirely. All four
  now route through a guarded helper, matching `signed_invocations._sha256`.
- `PhaseObservation` and `ResourcePlan` kept the caller's live containers after
  validating them, on frozen dataclasses whose projections are
  signature-bound. Appending to the caller's list after construction put
  unvalidated content into `binding_payload()` — the projection Frinz feeds to
  `phase_evidence_sha256` — and changed `ResourcePlan.digest`, which
  `ProviderAttemptIdentity.plan_digest` pins against a billable attempt. Both
  now materialize their sequence fields BEFORE validating rather than after,
  which additionally fixes a one-shot iterable silently emptying itself: a
  generator passed as `state_transitions` validated on the first pass and was
  copied from an exhausted iterator on the second, so the signed projection
  carried `[]` while the emptiness guard was bypassed (a generator is always
  truthy).
- `ResourcePlan.phase` and `ProviderAttemptIdentity.phase` were checked for
  membership in the mutating-phase tuple but never for type. `DogfoodPhase` is
  a `StrEnum`, so a raw `"lora_submit"` satisfies the membership test, then
  reaches `.value` in `to_payload`/`digest` and raises `AttributeError` — the
  same escape from `DogfoodSafetyError` as above, on the signature-bound path,
  reachable by direct construction (the harness's own route).

## [0.8.0] - 2026-08-03

### Added

- The SDK 0.35 owner-scoped inference lease `touch` contract, backed by the
  durable Ollama idle deadline and an exact live route re-observation that
  never provisions replacement capacity.
- Versioned, provider-neutral finite Serverless capacity quotes with exact
  endpoint/profile binding, read-only pre-submit drift validation, and complete
  content-free v2 billing receipts for exclusive endpoint windows.
- A provider-neutral ambiguous-submission settlement method that accepts an
  exact exclusive endpoint/hour allocation and ceiling, then returns strict v2
  ordered per-hour observations plus derived actual, capped, and operator-loss
  cost evidence without requiring a job ID.

### Changed

- Raised the minimum `kestrel-sovereign-sdk` version to 0.35.0.

### Fixed

- Accept the host's canonical quote-lifetime endpoint-hour allocation as a
  conservative superset for both terminal and ambiguous submissions, while
  still requiring it to cover the exact accepted attempt interval and settling
  every reserved hour.
- Quote interactive Serverless for every possible scale-to-zero cold start in
  the accepted session, bind that maximum into SDK metadata, and reject a zero
  idle tail whose invocation-independent cost cannot be bounded.
- Extend authoritative Serverless billing coverage through the accepted idle
  tail, bind every exclusively allocated endpoint-hour bucket into the receipt,
  and wait for the last touched bucket to close before settlement.
- Project Runpod `delayTime` as aggregate pre-execution delay and validate it
  against the accepted queue plus worker-start bounds.
- Export the public placement enums and Serverless cost/allocation helpers, and
  reject per-job execution-timeout or TTL policies outside Runpod's bounds.

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

[0.8.0]: https://github.com/KestrelSovereignAI/kestrel-cloud-runpod/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/KestrelSovereignAI/kestrel-cloud-runpod/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/KestrelSovereignAI/kestrel-cloud-runpod/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/KestrelSovereignAI/kestrel-cloud-runpod/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/KestrelSovereignAI/kestrel-cloud-runpod/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/KestrelSovereignAI/kestrel-cloud-runpod/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/KestrelSovereignAI/kestrel-cloud-runpod/releases/tag/v0.3.0
