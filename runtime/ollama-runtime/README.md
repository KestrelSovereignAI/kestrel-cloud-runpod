# Kestrel private Ollama runtime

This directory is the single deployable workload contract used by durable
Ollama leases. It contains no Runpod lifecycle code or control-plane
credential. The Python lease service creates and tears down capacity; this
image only starts Ollama, prepares one approved model, and serves it through a
streaming reverse proxy.

## Security contract

- `/ping`, `/health/live`, and `/health/ready` are the only anonymous routes.
- Every other route requires the exact `Authorization: Bearer ...` capability
  and rejects it at the configured expiry.
- The proxy removes the bearer before forwarding to loopback Ollama.
- Model pulls and inference are restricted to `KESTREL_OLLAMA_ALLOWED_MODELS`.
  Each allowlist entry has the form `name:tag@sha256:<64 lowercase hex>`. The
  worker refuses readiness when the local digest differs from the configured
  pin. Every allowlisted model is a full-agent route and must report Ollama's
  `completion` and `tools` capabilities; the worker refuses readiness when
  either is absent.
- Ollama create, copy, delete, pull, push, and blob-management APIs are disabled
  externally. Bootstrap is the sole digest-verifying model-pull path.
  Unrecognized future routes fail closed instead of being forwarded.
- The entrypoint uses root only to reject cache symlinks and assign an attached
  cache tree to numeric UID/GID `10001`; it drops all user and group privilege
  before Ollama starts or the public listener opens. Neither request bodies nor
  authorization values are logged.

The bearer is a workload-scoped inference capability, not a Runpod control
key. `KESTREL_OLLAMA_BEARER_TOKEN_EXPIRES_AT` must be a future RFC3339 value.
The Pod/LB definition must rotate both values together for every bounded
lease.

## Startup and cache contract

The runtime starts its public proxy immediately. `/ping` returns `204` while
Ollama boots, the pinned model is verified or pulled into `/models`, and the
model is preloaded. It returns `200` only after those phases complete, matching
[Runpod's load-balancer health contract](https://docs.runpod.io/serverless/load-balancing/overview).
After startup every readiness probe revalidates Ollama, the exact digest, and
the model's completion/tool support. A mismatch, missing capability, expired
workload capability, or upstream failure returns `503`.

The provider sets `KESTREL_OLLAMA_MODEL_STORAGE_PATH` to `/models` for ephemeral
container storage, `/workspace/ollama` for a Pod volume, or
`/runpod-volume/ollama` for a Serverless network volume, matching Runpod's
product-specific mount conventions. These paths are exclusively for public
model weights; user prompts, responses, tokens, and private state must not be
written there. Fresh root-owned mounts are initialized safely before privilege
drop; symlinks in a reused cache fail closed. Serverless network-volume mode
requires one worker because Runpod does not serialize concurrent writers to a
shared volume. Do not assume a reusable volume is available in the selected
zone. The lease policy must choose cached, baked, or pull-on-start from the
measurements gathered in the live release gate.

Authenticated `GET /kestrel/telemetry` reports content-free timestamps and
durations for provider request to container start, Ollama boot, model pull,
model preload, and total readiness. The first interval includes placement,
host image download, and container scheduling because code inside a container
cannot observe those provider phases separately. The live gate correlates it
with Runpod worker/Pod timestamps to isolate image-pull time. Runpod LB also
measures the `/ping` `204` to `200` cold-start interval.

## Build and local validation

The runtime is independently versioned as `1.1.0`; it does not change the
Python package version. All build and final bases are pinned by immutable OCI
digest, and the resulting image is linux/amd64-only for the reviewed Runpod
GPU target. The image retains the GPU backends from Ollama `0.32.5` but rebuilds
that exact source commit with Go `1.26.5` and explicit fixed module versions;
the official binary currently contains fixed high-severity dependency findings
and is overwritten in the final filesystem rather than waived.

```bash
docker buildx build --platform linux/amd64 \
  --tag kestrel-ollama-runtime:test \
  --load runtime/ollama-runtime
```

The GitHub runtime workflow runs Go tests with the race detector, validates the
template, builds from this secret-free context, generates an SPDX SBOM, and
fails on fixed high/critical vulnerabilities. On `main`, it publishes the
version and commit tags to GHCR with build provenance; production configuration
must use the resulting `repository@sha256:...` reference, never a mutable tag.
