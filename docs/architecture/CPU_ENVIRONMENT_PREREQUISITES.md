# CPU environment acquisition prerequisites

Source slice for [Runpod #43](https://github.com/KestrelSovereignAI/kestrel-cloud-runpod/issues/43)
and [Core #3703](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3703).

`cpu_environment.CpuEnvironmentPlanner` completes read-only placement and
bounded acquisition-intent planning using published SDK 0.43 contracts and the
existing production v2 CPU catalog client. It is not an `EnvironmentProvider`
or a working remote browser. No environment/driver entry point is registered.
The GPU/Ollama offering and its capacity custody are independent and unchanged.

The planner requires an exact trusted requested CPU profile: digest-pinned image,
CPU flavor/power-of-two count, one explicit preferred data center, disk and lifetime.
Profile lifetime is at most seven days and cleanup grace at most one day;
extreme durations fail typed validation before arithmetic or catalog access.
The original `EnvironmentRequest` must explicitly target that provider/profile,
image, region and remote browser destination. Files/exec/clipboard/DOM,
accessibility and native desktop are unsupported prerequisites rather than inferred capabilities.
Allowing a requested visual operation here does not establish driver conformance.
The published base SDK's `takeover` names a logical controller fence; permitting
that request does not advertise manual pixels/input or the optional human channel.

It reads `/catalog/cpus?include=AVAILABILITY&product=POD&vcpuCount=N` through
`RunpodControlPlaneClient`, preserving request-start time. Provider latency
counts toward a short freshness limit. The observed timestamp must lie between
the same call's initial and final trusted clock samples. An injected planner
clock must use the same authoritative wall-clock domain as the production
catalog client's system UTC clock; an arbitrary synthetic epoch is not a
supported production clock. Naive snapshot timestamps are refused explicitly.
Exact freshness expiry fails with a domain error. Unknown or zero compute price,
stock, requested data center or shape fails. A zero catalog compute rate is not
admitted as free capacity. This uses the API's
[CPU catalog](https://docs.runpod.io/api-reference-v2/catalog/list-cpu-types)
and existing typed
[CPU Pod create shape](https://docs.runpod.io/api-reference-v2/pods/create-a-pod),
validated against the repository's pinned OpenAPI schema in tests.

The catalog is compute-only. An explicit `CpuNonComputeCostPolicy` supplies
finite disk/transfer/cleanup hourly and fixed estimates with original provenance,
exact provider/profile/disk/datacenter/full-profile-digest binding, explicit
covered components and a window covering request lifetime plus cleanup grace. Neither
value defaults to zero. Both amounts permit at most 12 fractional places and
24 whole digits; policy timestamps must convert within the UTC calendar range.
Invalid policy values fail typed validation before catalog access. These are
operator estimates, not provider billing caps
or verified paid observations. Unknown components must not be represented as
zero. The plan prices the entire original lifetime plus cleanup grace, includes
fixed reserves and refuses either original ceiling being exceeded. Duration
cost rounds conservatively upward to 12 decimal USD places without ambient
Decimal precision affecting the result.

The immutable plan binds the exact original SDK request digest, profile/policy
digests, CPU facts, catalog timestamp, price and expiry. SDK request Decimal
spellings remain distinct inputs. Profile and policy digests have explicit
kind/version tags; equivalent numerical policy amounts normalize identically.
Before a separate admitted create,
`plan.pod_request(original_request, original_profile, original_policy, at=...)`
checks those originals and current bounds again, then emits from the STORED
original request/profile, never a newly spelled caller request. It emits the
existing `CpuPodCreateRequest`: exact requested CPU/image/disk and placement
preference, with a deterministic request-scoped
name, no GPU, no global networking, no ports/args/mounts/private environment.
The returned request's environment mapping is immutable. No method creates,
connects or releases a resource; no CPU authority/custody database is added.
The unchecked profile emitter is private. Plan expiry also reserves the full
original readiness timeout; a late plan cannot extend the original request TTL.

## Remaining production gates

Core #3703 candidate source owns the intended canonical visual request/quote/
lease/connection/action store. Published Core 0.55.0 does not yet contain that
new SDK visual runtime/provider registration. This planning package requires
published Core >=0.54.4, the first version resolving SDK 0.43 and its TLS bundle.
That dependency compatibility does not satisfy visual runtime readiness.
Core's legacy computer-use feature is a separately authorized host-access
surface. A compatible reviewed Core release and registered runtime remain
prerequisites, not published readiness.
A complete adapter must persist the exact accepted all-in quote and original
create intent there before calling the existing paid v2 create boundary. A
timeout/malformed accepted response retains original unknown custody. Reconcile
the same original request and all resource fingerprints; ambiguous candidates
must not trigger another create. A deterministic name alone is not an API
idempotency key or sufficient proof of ownership. Release needs actual provider
absence evidence; stopped/expired does not establish that billing ended.

The pinned API explicitly describes `dataCenterIds` as PREFERRED placement,
not enforcement. After create, retain original Pod ID/custody and inspect the
realized `dataCenterId`, CPU type/count, image/cloud/disk and Pod `cost` before
connecting, sending private content/credentials or claiming readiness. Require
the actual data center to equal the original authorized region, actual shape
to match the admitted profile. Read the authoritative billed `cost` from a
`RUNNING` Pod. Require the billed rate to be positive, finite, known and no greater than
`plan.catalog.compute_cost_per_hr`, the planned compute-only rate; do not compare
it with `plan.all_in_hourly_usd`. Missing or zero rates never establish readiness,
including on a `RUNNING` Pod. A `0.0` placeholder while
PROVISIONING/STARTING/EXITED/TERMINATED/ERROR, are unknown rather than free or
ready resources. Non-RUNNING observations do not establish an admitted billed
rate even if they contain a numeric value. Pending observations may be polled
only within the original readiness deadline without private content or credentials. A verified
mismatch, or still-unknown facts at that deadline, refuses admission and triggers
teardown of the ORIGINAL owned Pod through canonical cleanup custody.
Teardown errors remain unknown and retain that ID until actual release is proven.
Never accept a different region/rate, launch a replacement or treat a create
preference/catalog match as realized destination/privacy/billing evidence.

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
