# Remote CPU environment implementation boundaries

The current source adds independently useful placement observations and a
verified TLS byte channel. These are not an EnvironmentProvider, ComputerDriver,
workspace transport, worker bootstrap or remote browser readiness claim. No
new entry point is registered, no paid create is enabled and no second custody
store, wallet, action queue or inference integration is added.

## Implemented independent boundaries

`cpu_realized.inspect_cpu_readiness` consumes the production v2 Pod parser's
observation for the EXACT original Pod ID retained by the host. It checks the
accepted plan's request-scoped name, digest-pinned image string, CPU flavor/count,
integral RAM, disk, secure cloud and realized data center. The readback uses
`Pod.globalNetworking`'s pinned response object (`enabled: false`, no `ip` or
`internalDns`), not the request boolean. It requires the exact empty ports,
environment, mounts and args, no registry/template, unlocked state and no GPU or
cluster membership. Optional deconstructed command fields must also be empty.
A missing required fact stays unknown even while provisioning; a known changed
fact is a contradiction. Data-center create preferences
are not enforcement. Only RUNNING has an authoritative compute rate: it must be
known, finite and no greater than the accepted COMPUTE rate, not the larger
all-in rate. Pending zero placeholders do not become free capacity. Initial
readiness has one fixed deadline; retries cannot renew it. Known contradiction
or elapsed readiness preserves the original Pod ID and requires exact teardown.
Unknown observations remain unknown, not absent/released. A repeated name is a
candidate only: duplicates are ambiguous, and no candidate never proves an
in-flight create cannot finish. `startedAt` must fall between original planning and observation; it is an
incarnation hint, not an attestation. A caller clock before planning is unknown,
not evidence of a contradictory Pod. These pure checks are factual correlation, never caller authority.

`remote_worker_tls.PinnedWorkerTLS` sends one bounded opaque newline frame per
native TLS connection. It requires TLS1.3, certificate chain/hostname verification,
an exact pretrusted peer certificate pin, an explicit numeric endpoint and fixed
expiry. The host owns client certificate loading and a dedicated, provisioned CA;
no TOFU or ambient DNS lookup is used. Mutual TLS presents the client certificate
during the handshake, before the application can verify the peer certificate
pin. Therefore the original host must provision a dedicated trust root for the
exact worker destination and load only the intended request-scoped client
identity. A broadly trusted public or cross-tenant CA is not sufficient for
client-identity confidentiality; the later certificate pin does not retroactively
protect that handshake. No CA or identity is discovered from the endpoint. The ORIGINAL async policy admission remains retained through
connect, handshake, final write and response. Its synchronous checker runs after
handshake immediately before the final write with no intervening await. After
write, transport loss/denied delivery requires original-ID reconciliation; there
are no automatic retries. A close-notify timeout or other close-only error after write is also unknown.
Socket retirement cannot replace an already classified admission/framing error.
Native route/TLS exception messages and their attached contexts/causes are not
returned by the transport; errors expose only content-free classification.
Cancellation retains the original host's unknown custody and physically joins
the socket retirement even under repeated cancellation. Control and data calls share no client queue.
Server priority dispatch and physical input retirement still require the real
worker. The byte channel does not authenticate SDK scope/session/epoch by itself:
the reviewed worker/driver must enforce those original bindings, current epoch,
credentials and original receipts through every actual effect and transfer.

Tests use the production v2 GET parser over MockTransport plus actual native TLS
sockets with a locally owned CA and exact client/server certificates. They test
post-handshake revocation before bytes, post-write revocation, receipt loss,
certificate/hostname/owner denial, fixed expiry, bounded frames, independent
concurrent transport and an owned native peer that withholds close_notify.
Nominal running and pending Pod fixtures validate against the pinned response
schema before intentional missing/contradictory-fact tests modify them. They do not run Chromium, cloud isolation, an AMD64 image,
a remote worker or real Runpod provisioning, and are not worker conformance.

## Canonical acquisition custody gap

Published SDK0.43 `EnvironmentProvider.acquire(request, quote)` has no original
first-dispatch checkpoint or persistence callback. Core persists `acquiring`
before BOTH initial acquire and restart reconciliation. Runpod POST `/pods` has
no pinned idempotency-key contract; names are not unique create keys. Therefore
name absence cannot authorize repeating an unknown POST. A provider-local store
or borrowing GPU training leases would create competing visual custody and is
not a solution.

The proposed shared optional SDK follow-on is separate from the human-control
release. It must let Core allocate/persist the attempt ID and exact private
create payload plus original TLS identity material ONCE on the SAME encrypted
request row before any POST. Restart returns those originals. An atomic
at-most-once dispatch checkpoint, including lost checkpoint acknowledgments,
allows no resend after dispatching/unknown. Late original factual results may
preserve only original resource custody; they authorize no new I/O. Unknown
without a known Pod cannot become released because an inventory read was empty.
The separately frozen SDK source candidate `fbdd802939c70b490b16fc629be8c9ca736ae581`
defines optional acquisition-custody API version 1, private preparation storage
and pure checkpoint transitions. It remains unpublished and is not imported or
advertised by this package. Core request-row integration and production transport
composition still need implementation and review. No private store fallback or
provider-specific authority resolver is implemented here.
Actual production HTTP native final-write checking must compose with Core's
original current checker, not merely sample it before `to_thread`/network await.
Core's current native HTTP admission helper accepts asynchronous clients; the
existing canonical Runpod transport uses synchronous HTTPX. The helper cannot
simply be wrapped around that client's `to_thread` call. A shared synchronous
transport extension or a deliberate canonical asynchronous migration must keep
the existing parsing/custody pipeline and GPU behavior, with an actual delayed
native-write test. No second create client or compatibility shim is added here.

## Transport, image and cloud gates

The pinned Runpod v2 schema's SSH proxy is interactive-shell-only: no forwarding,
SCP or SFTP. Direct SSH requires a published TCP mapping and provides no
independent host-key provenance in that schema. Suggested unchecked host-key
options are explicitly unsuitable. A provider proxy's HTTPS certificate alone
also does not prove the original request-bound worker identity. See the official
[connection guide](https://docs.runpod.io/pods/connect-to-a-pod) and repository
`vendor/runpod-v2-openapi.yaml` PodSsh/PodSshEndpoint/Pod definitions.

A real remote profile must compose authenticated API provenance for the exact
Pod/incarnation/address, pretrusted request/lease-bound TLS identity, an issued
consumer certificate bound to original scope/session/epoch and the existing
worker's single input/receipt/workspace state engine. No public CDP/VNC, broad
cloud API key forwarding, unchecked SSH, host fallback or parallel GUI queue.
A published CPU-appropriate registry MANIFEST digest is required: a local Docker
configuration digest is a different identity. Local linux/aarch64 evidence does
not establish Runpod linux/amd64 browser namespaces, seccomp, capabilities,
egress denial, authenticated priority control or filesystem containment.

Core's accepted all-in quote, original remote destination/privacy/cost approval,
durable create/release custody, exact physical absence and measured billing
reconciliation remain mandatory. Inference stays independent. Every workspace
transfer retains BOTH original current source/destination policies through the
last write and canonical artifact publication. Human input requires the actual
separately published compatible SDK extension and worker support; logical
`takeover` is not that channel. Real Runpod conformance and any paid trial remain
separate explicit operator approval gates. No cloud or image publication has
been performed by this source slice.
