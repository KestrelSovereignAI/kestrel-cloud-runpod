# Runpod Serverless v2 schema provenance

The production management contract for this package is the OpenAPI document
served by the Runpod v2 API itself:

- Source: `https://api.runpod.io/v2/openapi.json`
- Observed SHA-256: `0bbdd828569233765e310e773e34586b33a6e38f55afde989ebd670152ed5c13`
- Observed: 2026-08-03

That schema requires `type: QUEUE`, puts `idleTimeout` under `workers`, and
uses `scaling: {type: QUEUE_DELAY, queueDelay: ...}`. The checked-in schema in
`runpod/docs` main was stale at the observation time (SHA-256
`64635d12b9e03d2ad7136e0375baaed7d2a8ac1b13ef4698e652f94e6e5db1f2`):
it described `QUEUE_BASED` and the older `scaling.value/idleTimeout` layout.

Runpod labels REST v2 beta. Before changing request models, compare the
API-served schema first, update the pinned digest and contract tests together,
and review every create/readback field. Do not infer the production wire shape
from the rendered example or GitHub copy alone.
