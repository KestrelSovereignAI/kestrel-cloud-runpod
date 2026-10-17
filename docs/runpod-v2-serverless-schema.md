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
