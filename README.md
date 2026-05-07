# kestrel-cloud-runpod

RunPod GPU cloud provider for Kestrel Sovereign agents. Provision pods, run LoRA training, manage pod lifecycle, plus a RunPod-backed Ollama provider.

## Installation

```bash
uv pip install kestrel-cloud-runpod
```

The feature is auto-discovered by Kestrel Sovereign via the `kestrel_sovereign.features` entry point — install it alongside `kestrel-sovereign` and `RunPodFeature` registers itself at startup.

## Configuration

| Variable | Description |
|----------|-------------|
| `RUNPOD_API_KEY` | RunPod API key (required) |

Optional `[runpod]` section in `kestrel.toml` for default profile preferences.

## What's provided

- `RunPodFeature` — agent-facing tools for pod search, provisioning, training, lifecycle
- Standalone API: `RunPodManager` for direct programmatic use
- RunPod-backed Ollama integration (when running large models on rented GPUs)

## Dependencies

- `kestrel-sovereign-sdk>=0.2,<1` — base `Feature`, `tool`, `ToolCategory`, `BackendType`
- `kestrel-sovereign>=0.7,<1` — `kestrel.toml` unified-config loader (runtime)
- `runpod>=1.8.1`

## Development

```bash
uv pip install -e '.[test]'
uv run pytest
```

## License

Apache-2.0
