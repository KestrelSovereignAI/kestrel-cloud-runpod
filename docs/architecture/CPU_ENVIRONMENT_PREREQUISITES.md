# CPU environment acquisition prerequisites

Source slice for [Runpod #43](https://github.com/KestrelSovereignAI/kestrel-cloud-runpod/issues/43)
and [Core #3703](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3703).

`cpu_environment.CpuEnvironmentPlanner` completes read-only placement and
bounded acquisition-intent planning using published SDK 0.43 contracts and the
existing production v2 CPU catalog client. It is not an `EnvironmentProvider`
or a working remote browser. No environment/driver entry point is registered.
The GPU/Ollama offering and its capacity custody are independent and unchanged.

The planner admits an exact trusted CPU profile: digest-pinned image, CPU
flavor/power-of-two count, one explicit data center, disk and maximum lifetime.
The original `EnvironmentRequest` must explicitly target that provider/profile,
image, region and remote browser destination. Files/exec/clipboard/DOM and
native desktop are unsupported prerequisites rather than inferred capabilities.
Allowing a requested visual operation here does not establish driver conformance.

It reads `/catalog/cpus?include=AVAILABILITY&product=POD&vcpuCount=N` through
`RunpodControlPlaneClient`, preserving request-start time. Provider latency
counts toward a short freshness limit. Unknown price, stock, exact data center
or shape fails. This uses the API's
[CPU catalog](https://docs.runpod.io/api-reference-v2/catalog/list-cpu-types)
and existing typed
[CPU Pod create shape](https://docs.runpod.io/api-reference-v2/pods/create-a-pod),
validated against the repository's pinned OpenAPI schema in tests.

The catalog is compute-only. An explicit `CpuNonComputeCostPolicy` supplies
finite disk/transfer/cleanup hourly and fixed estimates with original provenance
and a validity window covering the whole request plus cleanup grace. Neither
value defaults to zero. These are operator estimates, not provider billing caps
or verified paid observations. Unknown components must not be represented as
zero. The plan prices the entire original lifetime plus cleanup grace, includes
fixed reserves and refuses either original ceiling being exceeded. Duration
cost rounds conservatively upward to 12 decimal USD places without ambient
Decimal precision affecting the result.

The immutable plan binds the original request, exact profile, policy, CPU facts,
catalog timestamp, price and expiry. Before a separate admitted create,
`plan.pod_request(original_request, original_profile, original_policy, at=...)`
checks those originals and current bounds again. It emits the existing
`CpuPodCreateRequest`: exact CPU/image/placement/disk, deterministic request-scoped
name, no GPU, no global networking, no ports/args/mounts/private environment.
The returned request's environment mapping is immutable. No method creates,
connects or releases a resource; no CPU authority/custody database is added.

## Remaining production gates

Core owns the canonical visual request/quote/lease/connection/action store.
A complete adapter must persist the exact accepted all-in quote and original
create intent there before calling the existing paid v2 create boundary. A
timeout/malformed accepted response retains original unknown custody. Reconcile
the same original request and all resource fingerprints; ambiguous candidates
must not trigger another create. A deterministic name alone is not an API
idempotency key or sufficient proof of ownership. Release needs actual provider
absence evidence; stopped/expired does not establish that billing ended.

A real worker must prove private authenticated connectivity, actual browser
sandbox admission, network egress policy, current owner/grants/privacy,
priority control and original receipt reconciliation. No public CDP port,
host-shell fallback or unverified capability claim substitutes for that proof.
Files require the negotiated SDK workspace extension with real containment;
manual input requires the separately reviewed/published human-control extension.
Remote selection remains subject to the original host destination/privacy gate.
No local-only session may be sent here because a model or attribution requested it.

The create API does not itself implement the host's fixed expiry or cleanup
guarantee. External lifecycle ownership, termination, measured billing and
unknown cleanup recovery must be composed and verified before registration.
The source tests use synthetic HTTP and pinned provider contracts only; no live
cloud request, paid acquisition, credential, release or immutable tag was used.
