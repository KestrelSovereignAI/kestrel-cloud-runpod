# kestrel-cloud-runpod

Runpod GPU cloud provider for Kestrel Sovereign agents. Provision Pods, run LoRA training, manage Pod lifecycle, and submit queue-based Serverless jobs without using Runpod's v1 or GraphQL infrastructure APIs.

## Installation

```bash
uv pip install kestrel-cloud-runpod
```

The feature is auto-discovered by Kestrel Sovereign via the `kestrel_sovereign.features` entry point — install it alongside `kestrel-sovereign` and `RunPodFeature` registers itself at startup.

## Configuration

| Variable | Description |
|----------|-------------|
| `RUNPOD_API_KEY` | RunPod API key (required) |
| `RUNPOD_CONTROL_PLANE_BASE_URL` | Optional beta/dev override; must end in `/v2` |
| `RUNPOD_USER_AGENT` | Optional non-empty application User-Agent override |

Optional `[runpod]` section in `kestrel.toml` for default profile preferences.

## What's provided

- `RunPodFeature` — agent-facing tools for pod search, provisioning, training, lifecycle
- Standalone API: `RunPodManager` for direct programmatic use
- `RunpodControlPlaneClient` — typed v2 catalog, Pod, Serverless endpoint, worker/log, and billing client
- `RunpodServerlessClient` — typed queue job run/status/cancel/retry/health client
- RunPod-backed Ollama integration (when running large models on rented GPUs)

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

Create calls are never retried automatically. If a connection failure or 5xx makes a Pod, endpoint, or queue-job creation result ambiguous, the client raises `RunPodAmbiguousResultError` with `reconcile_required = True`. The compatibility manager preserves that type, and its LoRA/Ollama helpers halt instead of trying a replacement profile. A production caller must persist its attempt/fingerprint and reconcile by listing or status lookup before authorizing another create.

This package supplies the v2 vendor boundary; it does not turn the legacy in-memory manager TTL into a durable lease. Durable Serverless dispatch belongs to [frinz#688](https://github.com/KestrelSovereignAI/frinz/issues/688), and durable private-inference ownership/reaping belongs to [kestrel-cloud-runpod#9](https://github.com/KestrelSovereignAI/kestrel-cloud-runpod/issues/9). Until those consumers land, process-local expiry is not a billing-safety guarantee.

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
