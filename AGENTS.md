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
│   ├── ollama.py                  # Ollama-on-RunPod serving
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
