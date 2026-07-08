# kestrel-cloud-runpod — Repo Map

Auto-generated file-tree + per-file purpose index. Do **not** edit by hand —
regenerate via `python scripts/generate_repo_map.py` (refreshed nightly by
`.github/workflows/repo-map.yml`). No timestamp on purpose: the nightly job
commits only when the tree actually changes; `git log REPO_MAP.md` has the date.

**Scope:** 24 tracked files (13 `.py`, 3 `.md`, 8 other). Excludes caches, lockfiles, and build artifacts.

**Format per file:** `path — one-line purpose` plus the public top-level Python symbols on the next line
(classes and functions; private `_name` skipped).

---
## Top-level files

Repo entry points and standard project files.

- **README.md** — kestrel-cloud-runpod — RunPod GPU cloud provider for Kestrel Sovereign agents.
- **AGENTS.md** — kestrel-cloud-runpod — Agent Instructions — See [README.md](README.md) for package overview.
- **LICENSE** — —
- **.gitignore** — —
- **REPO_MAP.md** — kestrel-cloud-runpod — Repo Map — Auto-generated file-tree + per-file purpose index.
- **pyproject.toml** — (configuration)
- **runpod_config.toml** — (configuration)
- **runpod_config.toml.example** — —

## `.github/`

- **.github/workflows/ci.yml** — (configuration)
- **.github/workflows/publish.yml** — (configuration)
- **.github/workflows/repo-map.yml** — (configuration)

## `kestrel_cloud_runpod/`

- **kestrel_cloud_runpod/__init__.py** — RunPod GPU management for Kestrel.
- **kestrel_cloud_runpod/core.py** — RunPod Core Manager Operations.
  - `class RunPodManagerCore`
- **kestrel_cloud_runpod/feature.py** — —
  - `class RunPodFeature`
- **kestrel_cloud_runpod/manager.py** — RunPod Manager - Combined Class.
  - `class RunPodManager`
- **kestrel_cloud_runpod/models.py** — RunPod Data Models and Exceptions.
  - `class PodStatus`; `class GPUProfile`; `class RunPodSession`; `class RunPodManagerError`
- **kestrel_cloud_runpod/ollama.py** — RunPod Ollama Cloud Server Methods.
  - `class RunPodOllamaMixin`
- **kestrel_cloud_runpod/providers.py** — RunPod GPU Providers.
  - `class GPUProvider`; `class DirectRunPodProvider`; `class ManagedRunPodProvider`
- **kestrel_cloud_runpod/training.py** — RunPod LoRA Training Methods.
  - `class RunPodTrainingMixin`

## `scripts/`

- **scripts/generate_repo_map.py** — Generate REPO_MAP.md — a file-tree + per-file purpose index for this repo.
  - `class FileEntry`; `def repo_name()`; `def tracked_files()`; `def is_excluded(path)`; `def first_sentence(text, max_chars)`; `def summarize_python(path)`; `def summarize_markdown(path)`; `def summarize_other(path)`; `…`

## `tests/`

- **tests/conftest.py** — Test configuration for kestrel-cloud-runpod.
  - `def pytest_addoption(parser)`; `def pytest_collection_modifyitems(config, items)`
- **tests/test_runpod_feature.py** — —
  - `class FakeRunPodManager`; `class DummyLLMService`; `async def runpod_feature(monkeypatch)`; `async def test_manage_gpu_start_and_stop(runpod_feature)`; `async def test_image_generation_tears_down_session(runpod_feature)`; `async def test_manage_gpu_unknown_action_returns_failed(runpod_feature)`; `async def test_start_unknown_profile_returns_failed(runpod_feature)`; `async def test_start_invalid_ttl_returns_failed(runpod_feature)`; `…`
- **tests/test_runpod_logs.py** — —
  - `def mock_runpod()`; `def mock_paramiko()`; `def mock_utils()`; `class TestRunPodLogs`
- **tests/test_runpod_model_contracts.py** — Contracts for RunPod profile-owned model defaults.
  - `async def test_resume_stopped_pod_requires_profile_default_model()`; `async def test_start_ollama_pod_uses_profile_default_model_without_hidden_fallback()`; `async def test_start_ollama_pod_resumes_existing_pod_without_new_model_override()`
