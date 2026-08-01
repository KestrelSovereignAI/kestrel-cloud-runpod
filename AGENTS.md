# kestrel-cloud-runpod — Agent Instructions

See [README.md](README.md) for package overview.

## Code Index

- [REPO_MAP.md](REPO_MAP.md) — generated per-file index (one-line purpose + public Python symbols; regenerated nightly by `.github/workflows/repo-map.yml`).

## Package Structure

```
kestrel-cloud-runpod/
├── runpod_config.toml(.example)   # GPU/pod profiles (${ENV_VAR} placeholders, no secrets)
├── kestrel_cloud_runpod/
│   ├── feature.py                 # RunPodFeature — feature entry point + tools
│   ├── core.py                    # RunPodManagerCore — SDK ops, profiles, sessions
│   ├── manager.py                 # pod lifecycle management
│   ├── providers.py               # provider wiring
│   ├── models.py                  # typed models/contracts
│   ├── ollama.py                  # manager integration for durable Ollama leases
│   ├── ollama_contracts.py        # lease types and live-cost selection policy
│   ├── ollama_repository.py       # SQLite WAL/revision-CAS lease state
│   ├── ollama_provider.py         # Runpod v2 LB/Pod capacity adapter
│   ├── ollama_service.py          # acquire/readiness/release/reconciliation state machine
│   ├── ollama_reconciler.py       # one-shot external scheduler entry point
│   └── training.py                # training-job orchestration
└── tests/
```

## Entry Points

- `kestrel_sovereign.features`: `RunPodFeature = "kestrel_cloud_runpod.feature:RunPodFeature"`

## Key Files to Read First

1. `kestrel_cloud_runpod/feature.py` — tools exposed to the agent
2. `kestrel_cloud_runpod/core.py` — RunPod SDK operations and profile loading
3. `tests/test_runpod_model_contracts.py` — the model contracts to keep green

## Running Tests

```bash
uv pip install -e '.[test]' && python -m pytest
```

## Agent-Specific Instructions

- Pods cost real money — anything that provisions must have a matching
  teardown path, and tests must never hit the live RunPod API.
- The tracked `runpod_config.toml` mirrors the `.example` with `${ENV_VAR}`
  placeholders — secrets come from the environment; never commit literal keys.
