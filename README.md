# kestrel-cloud-runpod

Runpod GPU cloud provider for Kestrel Sovereign agents. Provision Pods, run LoRA training, manage Pod lifecycle, and submit queue-based Serverless jobs without using Runpod's v1 or GraphQL infrastructure APIs.

See [CHANGELOG.md](CHANGELOG.md) for release notes.

## Installation

```bash
uv pip install kestrel-cloud-runpod
```

The feature is auto-discovered by Kestrel Sovereign via the
`kestrel_sovereign.features` entry point. Private inference is independently
registered as `runpod` in
`kestrel_sovereign.inference_lease_providers`; Kestrel core interacts only with
the public SDK lease contract and never imports this package.

## Configuration

| Variable | Description |
|----------|-------------|
| `RUNPOD_API_KEY` | RunPod API key (required) |
| `RUNPOD_SERVERLESS_API_KEY` | Restricted Serverless invocation key for Ollama endpoint probes and model pulls |
| `RUNPOD_OLLAMA_BEARER_TOKEN` | Workload-scoped token enforced by the reviewed Ollama Pod image |
| `RUNPOD_CONTROL_PLANE_BASE_URL` | Optional beta/dev override; must end in `/v2` |
| `RUNPOD_USER_AGENT` | Optional non-empty application User-Agent override |
| `RUNPOD_OLLAMA_IMAGE` | Immutable `ghcr.io/kestrelsovereignai/kestrel-cloud-runpod-ollama-runtime@sha256:...` reference |
| `RUNPOD_OLLAMA_ALLOWED_MODELS` | Comma-separated operator allowlist of `name:tag@sha256:<digest>` model pins |
| `RUNPOD_OLLAMA_LEASE_DB` | Optional absolute override for the durable SQLite lease store |
| `RUNPOD_TRAINING_LEASE_DB` | Optional absolute override for durable training Pod ownership state |

Runtime settings and profiles live in the standalone
`$KESTREL_HOME/runpod_config.toml`; [runpod_config.toml.example](runpod_config.toml.example)
is the canonical shape. `RunPodManager(config=...)` may receive that same
mapping explicitly. This package does not read a `[runpod]` section from
`kestrel.toml`.

## What's provided

- `RunPodFeature` — agent-facing tools for pod search, provisioning, training, lifecycle
- Standalone API: `RunPodManager` for direct programmatic use
- `RunpodControlPlaneClient` — typed v2 catalog, Pod, Serverless endpoint, worker/log, and billing client
- `RunpodServerlessClient` — typed queue job run/status/cancel/retry/health client
- Durable RunPod-backed private Ollama leases with readiness and cost gates
- Provider-neutral SDK 0.34 inference leases with OpenAI-compatible host-only routes
- Durable LoRA training Pod ownership, cleanup tokens, and restart reconciliation

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

Kestrel normally reaches that lifecycle through
`RunpodInferenceLeaseProvider`. `quote()` reads the v2 catalog but creates
nothing; `acquire()` returns `PENDING` after the one durable billable mutation;
and `status()` reconciles readiness until the exact allowlisted model is loaded
behind an authenticated `/v1` route. Quote selection includes the configured
Serverless or Pod cold-start estimate, hourly ceiling, expected session, and
Serverless idle tail. Requests that cannot meet their region, readiness,
privacy, concurrency, or total-cost limit fail before provisioning.

Set `quote_ttl_seconds`, `serverless_estimated_ready_seconds`, and
`pod_estimated_ready_seconds` in `[ollama_leases]` from measured p95 startup
data. Configure at least one exact `profiles.ollama.allowed_data_center_ids`;
the provider advertises those normalized IDs as its regions and constrains the
v2 create request to the quoted region. A shared model network volume reduces
download time, but can narrow placement availability and Serverless requires a
single writer.

Runpod's beta v2 catalog currently reports the PRO 6000 MIG 1g.24gb and
2g.48gb products as available for Serverless while returning `pool = null`.
The same v2 create contract requires a canonical GPU pool ID. Kestrel therefore
rejects those offers before creation and never guesses a pool from a marketing
name or GPU ID; see [#21](https://github.com/KestrelSovereignAI/kestrel-cloud-runpod/issues/21).
A separately valid Pod offer may still be quoted explicitly as Pod capacity.

The route endpoint and bearer are reconstructed from a fresh authenticated
provider observation and live only in the SDK `InferenceRoute`. They are never
written to provider lease rows or public metadata. On process restart,
`status()` re-observes the same deterministic Runpod resource and returns a
fresh host-only route without creating duplicate capacity.

`manage_gpu` remains available for explicit operator start/stop/status/log
controls, but it no longer attaches, detaches, or reports an LLM route. The
provider-neutral inference coordinator is the only LLM routing owner.

Control-plane, Serverless data-plane, and Pod workload credentials are separate. The reviewed [private Ollama runtime](runtime/ollama-runtime/README.md) is published independently to GHCR. `RUNPOD_OLLAMA_IMAGE` must select it by immutable digest; mutable tags and other repositories fail before a billable create call. The runtime enforces bearer authentication on every non-health route, expires the workload capability at the lease deadline, permits only digest-pinned operator models, and never receives Kestrel's control-plane credential. Every operator-allowlisted model must support Ollama `completion` and `tools`; readiness revalidates both so Kestrel's default full-agent route never falls back to a tool-free lane. The provider refuses to publish a Pod route unless an anonymous `/api/tags` probe receives `401` or `403`. Tokens are never returned in lease state. AUTO mode considers only products whose scoped credential is configured.

For load-balanced Serverless, use a Runpod key restricted to the one endpoint as the scoped inference capability. Runpod authenticates it at the edge and the workload proxy verifies the same bearer defensively. For a dedicated Pod, `RUNPOD_OLLAMA_BEARER_TOKEN` is the scoped capability. The provider rejects either workload credential when it matches `RUNPOD_API_KEY`, and rejects one credential reused across both products. Rotate the Pod value per bounded lease/deployment; never reuse the full control-plane key. Both modes expose `/ping` on port 11434, returning `204` during model preparation and `200` only while Ollama is live, the capability is unexpired, and the exact pinned model remains present.

`accrued_estimated_cost` is intentionally a billing-safe upper bound. Dedicated Pods accrue their continuous live catalog rate. Until the Serverless readiness API exposes active worker-seconds or billing reconciliation supplies them, Serverless uses the same wall-clock bound; it may release early, but it cannot authorize spend beyond the caller's cap. The selection estimate still compares the expected Serverless initialization, active, and idle-tail window against the expected Pod session.

Training Pods use the same durability rule. `start_training_pod()` reserves a SQLite/CAS cleanup token before a configured or reusable Pod is resumed, or before a deterministic Pod name is created. A Pod newly started by that call is either returned with a route, confirmed stopped, or retained as retryable state with its provider ID. A Pod already running before the call is marked `preexisting_running` and is never stopped by this lease. Submission, status, result, and cancellation operations heartbeat the same record and preserve the provider job/Pod IDs on failure. Run `kestrel-runpod-reconcile-training` from an external timer; process-local TTLs are not cleanup.

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

- `kestrel-sovereign-sdk>=0.34,<1` — features, tools, and inference-lease contracts
- `kestrel-sovereign>=0.13.1,<1` — standalone Kestrel config-file loader (runtime)
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
