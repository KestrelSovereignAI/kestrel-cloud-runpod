# kestrel-cloud-runpod

Runpod GPU cloud provider for Kestrel Sovereign agents. Provision Pods, run LoRA training, manage Pod lifecycle, and submit queue-based Serverless jobs without using Runpod's v1 or GraphQL infrastructure APIs.

See [CHANGELOG.md](CHANGELOG.md) for release notes.

## Installation

```bash
uv pip install kestrel-cloud-runpod
```

The feature is auto-discovered by Kestrel Sovereign via the `kestrel_sovereign.features` entry point — install it alongside `kestrel-sovereign` and `RunPodFeature` registers itself at startup.

## Configuration

| Variable | Description |
|----------|-------------|
| `RUNPOD_API_KEY` | RunPod API key (required) |
| `RUNPOD_SERVERLESS_API_KEY` | Restricted Serverless invocation key for Ollama endpoint probes and model pulls |
| `RUNPOD_OLLAMA_BEARER_TOKEN` | Workload-scoped token enforced by the reviewed Ollama Pod image |
| `RUNPOD_CONTROL_PLANE_BASE_URL` | Optional beta/dev override; must end in `/v2` |
| `RUNPOD_USER_AGENT` | Optional non-empty application User-Agent override |
| `RUNPOD_OLLAMA_IMAGE` | Reviewed private Ollama Pod/Serverless image |
| `RUNPOD_OLLAMA_LEASE_DB` | Optional absolute override for the durable SQLite lease store |

Optional `[runpod]` section in `kestrel.toml` for default profile preferences.

## What's provided

- `RunPodFeature` — agent-facing tools for pod search, provisioning, training, lifecycle
- Standalone API: `RunPodManager` for direct programmatic use
- `RunpodControlPlaneClient` — typed v2 catalog, Pod, Serverless endpoint, worker/log, and billing client
- `RunpodServerlessClient` — typed queue job run/status/cancel/retry/health client
- Durable RunPod-backed private Ollama leases with readiness and cost gates

## Architecture

- [Runpod v2 execution platform](docs/architecture/RUNPOD_V2_EXECUTION_PLATFORM.md) — the accepted control-plane, Serverless, catalog inference, and private Ollama design.

Runpod has two distinct v2 services:

| Service | Default base | Authentication | Use |
| --- | --- | --- | --- |
| Control plane | `https://v2-rest.runpod.io/v2` | Bearer API key | Catalog, Pods, endpoint definitions, workers/logs, billing |
| Serverless data plane | `https://api.runpod.ai/v2` | Bearer API key | Queue job run, status, cancel, retry, and health |

Both clients set an explicit application User-Agent because the beta control plane rejects generic/default clients at its edge. Base URLs are injectable for testing but must end in `/v2`; there is no v1 or GraphQL production fallback.

### Direct client example

```python
from kestrel_cloud_runpod import RunpodControlPlaneClient, RunpodServerlessClient
from kestrel_cloud_runpod.models import ComputeProduct

control = RunpodControlPlaneClient(api_key="...")
offers = control.list_gpus(products=(ComputeProduct.SERVERLESS,))

jobs = RunpodServerlessClient(api_key="...")
job = jobs.run("endpoint-id", {"prompt": "hello"})
status = jobs.status("endpoint-id", job.id)
```

Create calls are never retried automatically. If a connection failure or 5xx makes a Pod, endpoint, or queue-job creation result ambiguous, the client raises `RunPodAmbiguousResultError` with `reconcile_required = True`. Ollama leases persist the request fingerprint and deterministic resource name before creation, then recover by listing before any replacement could be authorized.

Private Ollama callers submit a stable `OllamaLeaseRequest` with owner/workload IDs, the exact model, placement constraints, expected warm utilization, an idle timeout, hard deadline, and maximum spend. The service compares current Pod and Serverless catalog offers. Bursty traffic can use native load-balanced Serverless; sustained sessions can use a dedicated Pod. Queue Serverless is not selected for interactive streaming. `lease.public_route_url` remains `None` until both Runpod health and Ollama `/api/tags` prove the requested model is ready. An external scheduler must run `RunPodManager.reconcile_ollama_leases()` periodically so expiry and teardown retries survive requester crashes. The `kestrel-runpod-reconcile-ollama` command performs one fail-fast pass and is suitable for a timer or job runner.

Control-plane, Serverless data-plane, and Pod workload credentials are separate. This package does not ship the runtime image: `RUNPOD_OLLAMA_IMAGE` must identify a separately built and reviewed image that enforces `Authorization: Bearer $KESTREL_OLLAMA_BEARER_TOKEN`. The provider injects that value from `RUNPOD_OLLAMA_BEARER_TOKEN` and refuses to publish a Pod route unless an anonymous `/api/tags` probe receives `401` or `403`. The token is never returned in lease state. AUTO mode considers only products whose scoped credential is configured.

`accrued_estimated_cost` is intentionally a billing-safe upper bound. Dedicated Pods accrue their continuous live catalog rate. Until the Serverless readiness API exposes active worker-seconds or billing reconciliation supplies them, Serverless uses the same wall-clock bound; it may release early, but it cannot authorize spend beyond the caller's cap. The selection estimate still compares the expected Serverless initialization, active, and idle-tail window against the expected Pod session.

### Configuration migration from 0.2

Profiles no longer select a hardcoded `gpu_type_id` or record a `cost_per_hr`. Replace legacy fields with workload constraints:

```toml
[profiles.image]
name = "Large-memory image generation"
task_type = "image"
image_name = "runpod/kestrel-flux:latest"
min_vram_gb = 40
min_cuda_version = "12.8"
max_cost_per_hr = 3.00
gpu_count = 1
```

At Pod creation time, the direct provider queries v2 with product-specific availability, rejects offers outside the profile constraints, and records the selected GPU and offered live rate on the session. Legacy `gpu_type_id`, `vram_gb`, `cost_per_hr`, and `template_id` fields fail with migration guidance. Use `registry_id` for a v2 registry credential.

The old private CLI SSH helper is also gone. `RunPodManager.get_logs()` uses the v2 SSE Pod log endpoint. Arbitrary commands must be exposed as scoped workload HTTP operations.

### OpenAPI beta pin

The reviewed v2 schema is pinned in `vendor/runpod-v2-openapi.yaml` with its checksum in `vendor/runpod-v2-openapi.lock.json`. Unit/contract CI validates the operations and shapes Kestrel consumes. A weekly/manual workflow compares the live schema and reports semantic drift without overwriting the pin.

## Dependencies

- `kestrel-sovereign-sdk>=0.2,<1` — base `Feature`, `tool`, `ToolCategory`, `BackendType`
- `kestrel-sovereign>=0.7,<1` — `kestrel.toml` unified-config loader (runtime)
- `httpx>=0.27,<1`
- `requests>=2.32,<3`

## Development

```bash
uv pip install -e '.[test]'
uv run pytest
python scripts/check_runpod_openapi.py --check-pin
```

An authenticated smoke test is opt-in, read-only, and lists the GPU catalog only:

```bash
RUNPOD_API_KEY=... uv run pytest --run-cloud tests/test_runpod_smoke.py
```

## License

Apache-2.0
