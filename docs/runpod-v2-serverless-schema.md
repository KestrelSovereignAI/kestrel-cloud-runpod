# Runpod Serverless v2 schema provenance

The production management contract for this package is the OpenAPI document
served by the Runpod v2 API itself:

- Source: `https://api.runpod.io/v2/openapi.yaml`
- Observed SHA-256: `202db6255f134dda8d1881913a0a52c63a53cf3e8909ca167628075017c4c415`
- Observed: 2026-10-17

That schema requires `type: QUEUE`, puts `idleTimeout` under `workers`, and
uses `scaling: {type: QUEUE_DELAY, queueDelay: ...}`. In August 2026, the
checked-in schema in `runpod/docs` main was stale (SHA-256
`64635d12b9e03d2ad7136e0375baaed7d2a8ac1b13ef4698e652f94e6e5db1f2`):
it described `QUEUE_BASED` and the older `scaling.value/idleTimeout` layout.

The October refresh also requires cursor pagination for resource inventories
and supplies a distinct `price.serverless` rate per GPU per hour. Pod cloud
rates do not price Serverless workers. The client consumes all inventory
pages and uses the product-specific rate for authorization.

Before changing request models, compare the
API-served schema first, update the pinned digest and contract tests together,
and review every create/readback field. Do not infer the production wire shape
from the rendered example or GitHub copy alone.

## Durable endpoint cleanup (0.14)

The host permanently closes original endpoint admission and drains original
effects before calling `cleanup_planned_endpoint`. It must keep that physical
owner through the final receipt COMMIT. The SDK retains its complete admitted
read/intent/mutation/absence tail through caller cancellation, but does not own
the host's database, cross-replica lock or funds ledger.

Supply `persist_intent`, an async callback that commits the exact immutable
`ServerlessEndpointCleanupIntent` and returns the acknowledged typed value.
Store it in the existing endpoint aggregate, not a second cleanup registry.
The callback must be bounded by the host's persistence policy. A failed,
ambiguous, mismatched or self-cancelled ACK never authorizes DELETE. A caller's
cancellation cannot detach a valid already-admitted cleanup tail.

On restart, pass the committed `prior_intent`. Exact-ID GET takes precedence
over inventory: if the endpoint exists, fresh original readback and zero-worker
verification precede deletion. If GET reports typed 404, complete paginated
inventory must also contain neither original ID nor original name. Any other
read, parsing or pagination error remains unresolved. These are current
observations under the provider's read semantics, not a consistency guarantee
or proof the DELETE was ever sent. A missing endpoint without prior committed
cleanup evidence cannot fabricate a receipt.

New receipts carry `cleanup_evidence_version: 2` and `cleanup_basis`:

- `delete_acknowledged`: this invocation observed DELETE acknowledgment.
- `ambiguous_delete_verified_absent`: this invocation observed an ambiguous
  delete error and then verified current absence.
- `interrupted_intent_verified_absent`: restart recovered current absence
  against a prior intent; DELETE delivery and acknowledgment remain unknown.

Only the ambiguous-delete basis sets `reconciled_after_delete_error` true.
The legacy `delete_started_at` field denotes cleanup-operation start, before
intent ACK/wire send. Never infer delivery from that timestamp. Original v1
receipts retain their exact old field set/digests; new hosts must support the
explicit v2 basis rather than treating that boolean as a delivery proof.

Recovery uses the current `verified_absent_at`, conservatively extending the
possible billable interval. Missing later cost records can remain pending;
cleanup does not infer zero, release holds, or authorize private erasure.

The vendor documents [endpoint deletion](https://docs.runpod.io/api-reference-v2/serverless/delete-a-serverless-endpoint),
[worker inventory](https://docs.runpod.io/api-reference-v2/serverless/list-serverless-endpoint-workers),
and [global Serverless billing](https://docs.runpod.io/api-reference-v2/billing/get-serverless-billing-history).
Application transaction acknowledgment is explicitly the host/SDK boundary.
