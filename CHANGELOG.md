# Changelog

All notable changes to `kestrel-cloud-runpod` are documented here.

## [Unreleased]

### Added

- Read-only CPU environment acquisition planning uses the existing v2 catalog,
  exact SDK request/profile/placement bindings and explicit noncompute cost
  provenance. Include lifetime/cleanup estimates under both cost ceilings and
  emit a short-lived exact CPU Pod intent without registering a visual provider
  or adding capacity custody. No paid create or browser transport is implied.
- Bind intent names to the exact original SDK request digest, and bind
  noncompute policies to the complete requested profile and cost components.
  Canonical policy/profile digests distinguish domain/version and normalize
  numerical policy amounts. Reserve readiness time and check a single catalog
  clock window. Document placement preferences and required realized Pod
  region/shape/billed-rate verification plus teardown before admission.
- Revalidate original typed CPU planning inputs before catalog access and plan
  construction; unchecked copies cannot bypass cost, grace or request bounds.
  Reject timezone-naive catalog snapshots, zero compute rates and extreme
  profile durations before emitting a paid acquisition intent.

### Changed

- Require published SDK 0.43 or newer for canonical visual resource contracts
  and Core 0.54.4 or newer, the first published Core resolving that SDK and its
  TLS trust bundle.

## [0.16.0] - 2026-10-20

### Added

- Add an explicit CPU Pod request variant through the canonical v2 client and
  shared container envelope. Require a CPU flavor and valid vCPU count; refuse
  GPU/CUDA settings, persistent container mounts and global networking.
- Check CPU request shape against the pinned v2 schema, including the existing
  create/inspect/delete and lost-response paths, without live cloud calls.
- Add typed CPU catalog reads and explicit Pod matching against fresh,
  product/count-specific stock, flavor bounds and requested datacenters.
  Preserve unknown rates; compute rate and RAM calculations do not depend on
  ambient decimal precision. Catalog matches do not authorize spending.

### Changed

- Pod request constructors use keyword arguments. GPU and CPU provisioning
  share one transport while retaining distinct compute constraints.
- Preserve missing Pod rates as `None`; reject negative, non-finite and
  unrepresentable rates. Invalid successful create receipts, including deeply
  nested JSON or unexpected HTTP success statuses, remain ambiguous and require
  reconciliation before retrying. Hide container environment and arguments in
  request representations.

## [0.15.1] - 2026-10-20

### Fixed

- Normalize exact-Pod billing once through closed, contiguous UTC-hour records
  with strict identity/count/component/total validation. Missing history stays
  pending; gaps can no longer settle from minimum/maximum timestamps alone.
- Bind observation identity to every normalized hour/component revision while
  retaining the existing Pod receipt wire shape. Repeated poll timestamps do
  not manufacture new identities or claim provider invoice finality.
- Normalize equivalent API numeric spellings and signed zero for new Pod and
  Serverless observations, without rewriting historical receipt wire/digests.
- Share UTC bounds and exact signed Decimal/component arithmetic with Serverless
  billing, preserving small corrections across large amount ranges. The Ollama
  adapter still requires original durable lifecycle/rate provenance.
- Keep catalog billing unresolved after exact cleanup when original realized
  placement/rate evidence is unavailable. Never replace it with the accepted
  quote or emit a settlement-ready receipt without that provenance.

## [0.15.0] - 2026-10-19

### Fixed

- Reuse Core's canonical joined-operation owner instead of a second shield
  loop. Repeated caller cancellation still cannot detach admitted effects or
  receipt tails, and late effect failure keeps precedence while preserving
  original cancellation and native denial/UNKNOWN evidence. Finite observations
  do not mutate shared native source receipts or emit orphan shield diagnostics.
- Retain the public arbitrary-Awaitable contract, including already-admitted
  Tasks, slots-only Future-like producers and read-only native Future subclasses,
  even when new Task allocation is refused. Require Core 0.53.35 for its canonical
  protocol-only joining and accepted-Task publication custody. This does not
  grant admission or change cleanup evidence/billing semantics.
- Require pytest-asyncio 1.4 for test installations: its upstream fix for #724
  closes the default-loop lifecycle gap when synchronous CLI tests call
  `asyncio.run()` after async cases. Strict warning checks remain enabled;
  this is a test-only dependency floor, not a runtime cancellation workaround.

## [0.14.0] - 2026-10-19

### Added

- Original-authority `ServerlessEndpointCleanupIntent` with strict canonical
  codec and an awaited exact persistence-ACK hook before any endpoint DELETE.
  Hosts store it in their original aggregate and supply it after restart; the
  SDK does not add a parallel resource manager, private dependency or ledger.
- Cleanup evidence v2 explicitly distinguishes acknowledged deletion,
  ambiguous deletion, and interrupted-intent current-absence recovery. An
  intent is not a sent DELETE. Historical v1 receipt codecs/identities remain
  unchanged; consumers of new receipts must understand the evidence basis.

### Fixed

- Recover a successfully deleted endpoint after lost host receipt persistence,
  without inventing a job or historical provider error. Recovery and replay
  require exact-ID typed 404 plus complete name/ID inventory. A present endpoint
  requires fresh exact readback and zero-worker inventory before deletion.
- Retain the entire cleanup read/intent-ACK/mutation/absence tail through
  repeated caller cancellation. Failed, mismatched or self-cancelled intent ACK
  sends no DELETE; hosts retain their outer owner through receipt COMMIT.
- Reject non-datetime observations with a clear shared validation error.
  Recovery-time absence may extend a billing interval; missing hours remain
  unknown, never zero, and no cleanup evidence implies financial finality.

## [0.13.0] - 2026-10-19

### Added

- `retired_endpoint_billing` reads the full originally activated Serverless
  endpoint lifecycle after verified teardown, without a job/status dependency
  or synthetic submission. Bounded configurable queries reuse the canonical
  strict hourly parser and component-cost aggregation. Historical costs are
  not truncated to job authorization limits or an expired quote.
- Typed content-free retired-endpoint receipts preserve original activation
  and cleanup lineage, canonical hourly observations, exact derived totals,
  and revision-sensitive identities. They explicitly do not confirm provider
  invoice finality or authorize erasure of unresolved settlement custody.
  Empty/incomplete history stays pending; hosts reconcile overlapping hours
  and revised costs rather than charging receipt totals twice.

## [0.12.1] - 2026-10-17

### Fixed

- Both catalog Pod and private Ollama lease repositories now share one SQLite
  connection owner that closes after successful operations, failed setup,
  rollback, and commit failure. WAL, explicit transaction, CAS, and historical
  migration behavior is unchanged; connection lifetime no longer depends on
  garbage collection. CI rejects resource and unraisable-exception warnings.

## [0.12.0] - 2026-10-17

### Added

- `recover_catalog_result(..., recovery=opaque_private_recovery)` restores an
  already accepted signed handoff after runner process replacement, using only
  the authenticated artifact-only recovery route. It shares exact durable ACK
  intent, actual-response proof, cancellation ownership, and cached-proof replay
  with the ordinary ACK lifecycle. No execution/provision fallback, private
  dependency, result persistence, or fresh credentials are introduced.
- Derive and reserve `CATALOG_POD_REQUEST_SHA256` from the sealed request so the
  private recovery-capable worker can verify its immutable request binding.

### Fixed

- Check the hard deadline for fresh ACK and recovery before intent admission
  and again after capability loading. Cached actual proof still replays after
  expiry/Stop/teardown for historical bookkeeping; intent alone never does.

## [0.11.0] - 2026-10-17

### Changed

- New catalog Pod requests bind artifact handoff protocol 1 into both durable
  identities. The transport returns the opaque signed receipt/result envelope
  and understands `awaiting_ack` and `acknowledged` worker phases. Private
  artifact schemas and HMAC validation remain exclusively in the private host
  and worker packages; no private dependency is published here.
- `acknowledge_catalog_result(..., acknowledgement=opaque_host_ack)` now sends
  the authenticated runner ACK and persists exact, content-free ACK intent and
  observed proof **without teardown** for protocol-1 leases. After confirming
  that proof in its own durable database, the host calls
  `release_catalog_result(..., host_confirmed_acknowledgement_sha256=...)`.
  That declaration is recorded atomically with the cleanup-driving transition.
- Additive SQLite schema 4 preserves historical catalog identities and their
  old result/cleanup behavior. Historical leases cannot authorize new capacity
  or workload submission, including through generic acquisition.

### Fixed

- Bind both runner image-digest environment variables to the immutable pinned
  image, preventing a private runner configuration failure at startup.
- Preserve ACK intent through ambiguous HTTP outcomes and join ACK delivery
  through durable local evidence under repeated cancellation. Authenticated
  status can recover a lost ACK response only against an exact prior intent.
  Cached proof replays after expiry/revocation without loading credentials.
- Prevent stale polls and CAS retries from regressing an acknowledged phase,
  or a late ACK from reviving durable Stop. Stop/deadline cleanup never invents
  acknowledgment, durable host confirmation, or successful completion evidence.

## [0.10.4] - 2026-10-17

### Added

- Separate catalog Pod provisioning from one-shot readiness polling, using the
  existing durable lifecycle. Hosts can release generation admission during a
  cold start and revalidate it before workload submission. Existing acquisition
  callers retain the wait-until-ready API.
- Export `await_owned_effect` so admitted mutation and durable receipt tails
  remain joined through repeated cancellation, including blocking provider
  threads. Late effect failures remain visible; no automatic retry is added.

### Fixed

- Clean up acquisition cancelled during worker health, not only observation or
  sleep. Readiness CAS retries cannot overwrite durable Stop, and cold-worker
  health failures remain STARTING rather than implying cleanup failure.
- Recheck creation authorization after capability I/O and submission state
  after capability loading. Readiness cannot promote an attempt after its
  deadline. Serverless create/delete and Pod mutation threads finish before
  cancellation releases their caller's custody.
- Prevent late workload observations (including CAS retries) from reviving
  durable cancellation. Cleanup capability grace never extends the deadline
  for a new workload submission.

## [0.10.3] - 2026-10-17

### Fixed

- Resolve the default Pod cloud once: `RUNPOD_CLOUD_TYPE`, then
  `manager.cloud_type`, then SECURE. Profiles without an explicit cloud now
  honor the same default as the direct provider. Invalid values fail before
  provider initialization.
- Expose `require_uniform_profile_cloud()` so single-cloud pricing consumers
  can bind quotes to actual configured profiles and reject mixed or mismatched
  cloud policies. Explicit profile overrides and historical cleanup remain
  independent of the current default.

## [0.10.2] - 2026-10-17

### Fixed

- Apply the exact-GPU Serverless pool restriction to private Ollama as well
  as finite jobs. Shared pools cannot satisfy a specific GPU authorization,
  even when their other members are unavailable or filtered by policy.
- Send accepted CUDA minimums in Pod and Serverless create requests,
  normalizing major-only floors to the required major.minor wire format.
- Validate Serverless CUDA and storage readback against the accepted policy.
  Unsupported global mounts and extra network volumes fail before dispatch.
- Check private Ollama endpoint storage before sending workload requests.
  These guards preserve exact created-resource IDs for durable cleanup.
- Version persisted Serverless resource authority explicitly. Historical
  specs keep their exact digests for adoption and cleanup; new quotes and
  creates require the GA version and cannot reuse historical spend authority.
- Bound inventory walks with configurable `max_inventory_pages`; exceeding
  the bound never produces partial cleanup evidence.

## [0.10.1] - 2026-10-17

### Fixed

- Use the generally available `https://api.runpod.io/v2` control plane and
  refresh the reviewed OpenAPI snapshot from that API.
- Follow cursor pagination for Pod, endpoint, and network-volume inventories,
  including cluster Pods. Missing, inconsistent, repeated cursors or duplicate
  resource identities fail clearly instead of returning partial cleanup evidence.
- Quote and revalidate Serverless using `price.serverless`, independently of
  Pod cloud rates. Missing pool rates cannot authorize paid placement.
- Rebuild private Ollama runtime `1.1.1` with Go `1.26.9`, patched Go module
  dependencies, and Ubuntu's fixed OpenSSL packages. Pin the linked
  `klauspost/compress` module to `1.20.2` to fix CVE-2026-63209
  (GHSA-259r-337f-4rfw) and GHSA-fpmh-frq8-7rj8, and verify that exact linked
  version at build time. Audit upstream advisories as well as scanner results;
  fixed high/critical findings block publication of the exact image.

## [0.10.0] - 2026-08-11

### Added

- `RunpodServerlessCapacityProvider.adopt_planned_endpoint(request, quote)`:
  recover the activation receipt for an endpoint a CRASHED activation already
  created.

  `activate_planned_endpoint` already reconciles an ambiguous create, but only
  inside the same call. If the process dies between `create_endpoint` returning
  and the caller persisting the receipt — an instance recycle, an OOM, a deploy
  — the endpoint exists and bills while no receipt does. Every later activation
  then finds the planned name taken and raises, correctly and permanently, so
  the endpoint is never adopted and never deleted. It just bills, with nothing
  left pointing at it.

  Deliberately a separate method rather than a relaxation of activation.
  Activation must keep refusing a pre-existing name — adopting during a
  purchase would treat an unauthorized endpoint as bought — and that refusal
  stays pinned by `test_activation_never_adopts_a_preexisting_named_endpoint`.

  Adoption does NOT check quote freshness, which is the point rather than an
  oversight. The quote has certainly expired: a 60-second TTL cannot outlive a
  crash and a recovery sweep. Refusing on an expired quote does not save the
  money, it guarantees the loss, because nothing else can identify the endpoint
  afterwards. Freshness governs whether we may START spending, never whether we
  may STOP. Safety comes from identity instead, which is strictly stronger than
  a clock: exactly one endpoint may carry the plan-scoped name, and its REST v2
  readback must match this plan's immutable configuration — the endpoint spec
  plus the `gpu_pool` and `data_center_id` the quote fixes — through the same
  `_validate_planned_endpoint_binding` activation uses. Anything failing either
  check is somebody else's and is left alone.

  A readback that fails to TRANSPORT is not an ownership verdict. `RunPodAPIError`
  derives from `RunPodManagerError`, so catching the base class around the read
  would report a 5xx or a rotated key as "not ours" — and a caller told that is
  entitled to stop trying, which is what strands a billing endpoint. Transport
  errors keep their own type and propagate for retry; only a successful read
  whose config does not match the plan produces the ownership verdict.

  The receipt's three instants are all this process's adoption time. The
  endpoint's own `createdAt` IS available (the vendored v2 schema makes it a
  required date-time on the Endpoint resource) and is deliberately unused:
  `create_started_at` means "when this process began the create call", Runpod's
  `createdAt` answers a different question, and mixing a provider clock into
  fields otherwise sourced from `self._clock` would break the receipt's
  `create_started_at <= activated_at <= readback_at` invariant under skew.
  Carrying the provider instant properly needs its own field and a
  contract-version bump rather than being smuggled into one that means
  something else.

  The call mutates nothing — it lists, reads back, and returns a receipt
  carrying `reconciled_after_ambiguous_create=True`, which can be fed to
  `cleanup_planned_endpoint` to delete the endpoint. Adoption imposes no
  freshness constraint, so a receipt adopted while the quote is still fresh
  settles like any activation receipt; in a real recovery the quote has expired
  and `authorize_planned_submission` refuses it. That is a consequence of the
  timing, not a guarantee this method makes.

  Consumers on `>=0.9.0,<0.10` must widen their cap to reach this.

## [0.9.0] - 2026-08-08

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
  now type-check, then materialize, then validate content. That ordering is
  what both failure directions require: validating and copying afterwards
  re-iterates the input, so a one-shot iterable (`(t.name for t in log)`)
  validated on the first pass and was copied from an exhausted iterator on the
  second — the field silently emptied into the signed projection while the
  emptiness guard was bypassed, a generator being always truthy — whereas
  copying first would let `tuple(None)`/`dict(None)` raise a bare `TypeError`,
  which escapes `DogfoodError` entirely since it derives from `RuntimeError`.
  `InvokeReceiptVerifier.__init__` carried the same validate-then-recopy shape:
  a generator of trusts was drained by the validity check, leaving a verifier
  that trusted nothing while blaming later receipts for the empty set.
- Both modules now guarantee their error contract STRUCTURALLY: every public
  entry point is wrapped so any escaping exception becomes
  `SignedInvocationError` / `DogfoodSafetyError`. The escape class was never
  "hostile datetimes" or "hostile mappings" but "caller code runs inside a
  guard" — `utcoffset`, `__iter__`, `__eq__`, `__hash__` and `__str__` all
  execute user code inside validators, so no finite list of hostile input
  types can close it. The specific validators are retained for their
  diagnostics; they are simply no longer what the contract rests on.
  `BaseException` is not caught, so `KeyboardInterrupt` and `SystemExit`
  still propagate.
- Timezone awareness is now decided in ONE place per module (`_require_aware`).
  `utcoffset()` executes caller-supplied `tzinfo` code, so the check itself can
  raise — `TypeError` for a non-`timedelta` return, `ValueError` for an
  out-of-range offset, or anything at all for a `tzinfo` that raises. The check
  existed inline at six sites; a previous fix patched two and left four live on
  `ProviderAttemptIdentity` and `SpendQuote`, both public constructors.
- `canonical_json` and `canonical_sha256` now contain the encoder's own
  recursion limit. `json.dumps` is C-implemented but not depth-unlimited —
  CPython's `_json.c` calls `_Py_EnterRecursiveCall` per container, raising
  `RecursionError` at roughly 1000 on Python 3.11 and 30k on 3.14 — and both
  functions are public and outside the bounded pre-pass.
- `_require_exact_json_values` recursed one Python frame per nesting level with
  no depth bound and no cycle detection, so attacker-controlled `phase_evidence`
  nested ~1000 deep — or any cyclic mapping — raised `RecursionError` on the
  verification path. That is a `RuntimeError`: neither the module's error type
  nor a `ValueError`, so a consumer wrapping the boundary returned 500 rather
  than 4xx. `canonical_json` alone already contained both shapes, because
  `json.dumps` is C-implemented and turns a cycle into `ValueError`, so running
  this pure-Python pre-pass first *removed* containment rather than adding it.
  Both it and its `dogfood_contracts` twin are now depth-bounded, which doubles
  as cycle detection.
- `astimezone(UTC)` raised `OverflowError` near `datetime.min`/`datetime.max`.
  An aware datetime is not necessarily representable after the offset shift,
  and `OverflowError` is an `ArithmeticError` — neither module's error type nor
  even a `ValueError`, so it bypassed both contracts and the `except ValueError`
  wrappers consumers use. Reached from untrusted wire data by an ordinary
  one-minute offset (`"0001-01-01T00:00:00+00:01"`) through every `from_payload`
  carrying a timestamp, through the signing boundary, and — because
  `ProviderAttemptIdentity` validated awareness but never serialized — deferred
  onto `PhaseObservation.binding_payload()`, the projection Frinz signs.
  `SpendQuote`'s five-minute window had the same problem in its own arithmetic.
- `InvokeReceiptVerifier.verify_phase_evidence` had no typedness guard, unlike
  its sibling `verify`. Being a staticmethod it is the natural call site for a
  consumer holding a row it deserialized itself, so a plain dict raised
  `AttributeError` on the verification path.
- `InvokeReceiptSigner.sign` did not validate its payload argument at the
  signing boundary: `sign(None)` raised `TypeError` and `sign("string")` raised
  `ValueError` out of `dict(payload)`.
- `utf8_sha256` raised `UnicodeEncodeError` for a `str` holding a lone
  surrogate. `isinstance(value, str)` does not imply UTF-8-encodable, and
  `json.loads` produces such a string without complaint — a token-truncated
  emoji cuts a surrogate pair and leaves one. `UnicodeEncodeError` is a
  `ValueError` but not a `SignedInvocationError`, so it escaped the module's
  contract on all three paths that digest text, including
  `InvokeReceiptVerifier.verify(input_text=...)` — the verification path, not
  merely key loading.
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
