# kestrel-cloud-runpod — Repo Map

Auto-generated file-tree + per-file purpose index. Do **not** edit by hand —
regenerate via `python scripts/generate_repo_map.py` (refreshed nightly by
`.github/workflows/repo-map.yml`). No timestamp on purpose: the nightly job
commits only when the tree actually changes; `git log REPO_MAP.md` has the date.

**Scope:** 35 tracked files (20 `.py`, 4 `.md`, 11 other). Excludes caches, lockfiles, and build artifacts.

**Format per file:** `path — one-line purpose` plus the public top-level Python symbols on the next line
(classes and functions; private `_name` skipped).

---
## Top-level files

Repo entry points and standard project files.

- **README.md** — kestrel-cloud-runpod — Runpod GPU cloud provider for Kestrel Sovereign agents.
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
- **.github/workflows/runpod-openapi-drift.yml** — (configuration)

## `docs/`

- **docs/architecture/RUNPOD_V2_EXECUTION_PLATFORM.md** — Runpod v2 execution platform — **Status:** Accepted for staged delivery

## `kestrel_cloud_runpod/`

- **kestrel_cloud_runpod/__init__.py** — RunPod GPU management for Kestrel.
- **kestrel_cloud_runpod/clients.py** — Runpod v2 HTTP clients with one authenticated transport boundary.
  - `class RunpodTransport`; `class RunpodControlPlaneClient`; `class RunpodServerlessClient`
- **kestrel_cloud_runpod/core.py** — RunPod Core Manager Operations.
  - `class RunPodManagerCore`
- **kestrel_cloud_runpod/feature.py** — —
  - `class RunPodFeature`
- **kestrel_cloud_runpod/manager.py** — RunPod Manager - Combined Class.
  - `class RunPodManager`
- **kestrel_cloud_runpod/models.py** — Typed contracts for the Runpod v2 control and data planes.
  - `class RunPodManagerError`; `class RateLimit`; `class RunPodAPIError`; `class RunPodAmbiguousResultError`; `class CloudType`; `class ComputeProduct`; `class Availability`; `class FlashBoot`; `…`
- **kestrel_cloud_runpod/ollama.py** — RunPod Ollama Cloud Server Methods.
  - `class RunPodOllamaMixin`
- **kestrel_cloud_runpod/placement.py** — Deterministic GPU placement from live Runpod v2 catalog offers.
  - `def select_gpu(offers, requirements)`
- **kestrel_cloud_runpod/providers.py** — Provider adapters backed by the Runpod v2 REST control plane.
  - `class GPUProvider`; `class DirectRunPodProvider`; `class ManagedRunPodProvider`
- **kestrel_cloud_runpod/training.py** — RunPod LoRA Training Methods.
  - `class RunPodTrainingMixin`

## `scripts/`

- **scripts/check_runpod_openapi.py** — Validate the pinned Runpod v2 OpenAPI file and report upstream drift.
  - `def load_openapi(source)`; `def sha256(data)`; `def read_lock()`; `def check_pin()`; `def fetch_live(source)`; `def semantic_diff(pinned, live)`; `def check_live()`; `def main()`
- **scripts/generate_repo_map.py** — Generate REPO_MAP.md — a file-tree + per-file purpose index for this repo.
  - `class FileEntry`; `def repo_name()`; `def tracked_files()`; `def is_excluded(path)`; `def first_sentence(text, max_chars)`; `def summarize_python(path)`; `def summarize_markdown(path)`; `def summarize_other(path)`; `…`

## `tests/`

- **tests/conftest.py** — Test configuration for kestrel-cloud-runpod.
  - `def pytest_addoption(parser)`; `def pytest_collection_modifyitems(config, items)`
- **tests/test_runpod_clients.py** — HTTP and request-shape contracts for both Runpod v2 services.
  - `def test_control_client_auth_user_agent_catalog_query_and_timeouts()`; `def test_default_user_agent_is_explicit_and_non_generic()`; `def test_catalog_availability_requires_one_product_context()`; `def test_rfc9457_problem_is_typed_and_does_not_expose_body()`; `def test_safe_get_retries_using_retry_after_and_records_rate_limit()`; `def test_safe_get_uses_rate_limit_reset_when_retry_after_is_absent()`; `def test_create_is_not_retried_after_ambiguous_server_error()`; `def test_create_timeout_is_ambiguous_and_not_retried()`; `…`
- **tests/test_runpod_feature.py** — —
  - `class FakeRunPodManager`; `class DummyLLMService`; `async def runpod_feature(monkeypatch)`; `async def test_manage_gpu_start_and_stop(runpod_feature)`; `async def test_image_generation_tears_down_session(runpod_feature)`; `async def test_manage_gpu_unknown_action_returns_failed(runpod_feature)`; `async def test_start_unknown_profile_returns_failed(runpod_feature)`; `async def test_start_invalid_ttl_returns_failed(runpod_feature)`; `…`
- **tests/test_runpod_model_contracts.py** — Contracts for RunPod profile-owned model defaults.
  - `async def test_resume_stopped_pod_requires_profile_default_model()`; `async def test_start_ollama_pod_uses_profile_default_model_without_hidden_fallback()`; `async def test_start_ollama_pod_resumes_existing_pod_without_new_model_override()`
- **tests/test_runpod_openapi_contract.py** — Parity tests between Kestrel's typed calls and the pinned beta schema.
  - `def test_pin_checksum_and_validator_script()`; `def test_all_consumed_control_plane_operations_match_pin()`; `def test_consumed_request_and_response_shapes_match_pin()`; `def test_typed_create_and_update_payloads_validate_against_pin()`; `def test_runtime_base_url_is_explicit_despite_beta_schema_server_discrepancy()`; `def test_semantic_diff_reports_breaking_and_additive_changes()`; `def test_production_package_contains_no_v1_graphql_or_legacy_sdk_calls()`
- **tests/test_runpod_placement.py** — Live-catalog placement policy tests.
  - `def test_placement_prefers_availability_then_live_price_and_records_snapshot()`; `def test_serverless_mig_uses_product_availability_not_pod_max_count()`; `def test_placement_fails_closed_for_incompatible_offers(offer, requirements)`; `def test_placement_enforces_allowed_data_centers()`; `def test_placement_requires_cuda_filtered_catalog_provenance()`
- **tests/test_runpod_provider.py** — Compatibility-provider migration tests for the v2 client boundary.
  - `def test_direct_provider_selects_from_live_catalog_and_records_placement()`; `def test_direct_provider_fails_before_catalog_when_required_env_is_unset(monkeypatch)`; `def test_profile_rejects_persistent_volume_below_runpod_floor()`; `def test_direct_provider_lifecycle_and_v2_logs()`; `def test_private_cli_ssh_execution_has_clear_migration_error()`; `async def test_manager_get_logs_uses_provider_v2_log_stream()`; `def test_legacy_profile_fields_fail_with_migration_guidance()`; `def test_manager_maps_v2_http_runtime_port_to_pod_proxy_url()`; `…`
- **tests/test_runpod_smoke.py** — Opt-in, read-only authenticated smoke checks for the beta v2 API.
  - `def test_live_v2_gpu_catalog_is_read_only_and_typed()`

## `vendor/`

- **vendor/runpod-v2-openapi.lock.json** — (configuration)
- **vendor/runpod-v2-openapi.yaml** — (configuration)
