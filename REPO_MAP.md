# kestrel-cloud-runpod — Repo Map

Auto-generated file-tree + per-file purpose index. Do **not** edit by hand —
regenerate via `python scripts/generate_repo_map.py` (refreshed nightly by
`.github/workflows/repo-map.yml`). No timestamp on purpose: the nightly job
commits only when the tree actually changes; `git log REPO_MAP.md` has the date.

**Scope:** 95 tracked files (70 `.py`, 7 `.md`, 18 other). Excludes caches, lockfiles, and build artifacts.

**Format per file:** `path — one-line purpose` plus the public top-level Python symbols on the next line
(classes and functions; private `_name` skipped).

---
## Top-level files

Repo entry points and standard project files.

- **README.md** — kestrel-cloud-runpod — Runpod GPU cloud provider for Kestrel Sovereign agents.
- **AGENTS.md** — kestrel-cloud-runpod — Agent Instructions — See [README.md](README.md) for package overview.
- **LICENSE** — —
- **.gitignore** — —
- **CHANGELOG.md** — Changelog — All notable changes to `kestrel-cloud-runpod` are documented here.
- **REPO_MAP.md** — kestrel-cloud-runpod — Repo Map — Auto-generated file-tree + per-file purpose index.
- **pyproject.toml** — (configuration)
- **runpod_config.toml** — (configuration)
- **runpod_config.toml.example** — —

## `.github/`

- **.github/workflows/ci.yml** — (configuration)
- **.github/workflows/ollama-runtime.yml** — (configuration)
- **.github/workflows/publish.yml** — (configuration)
- **.github/workflows/repo-map.yml** — (configuration)
- **.github/workflows/runpod-openapi-drift.yml** — (configuration)

## `docs/`

- **docs/architecture/CATALOG_POD_CAPACITY.md** — Catalog Pod capacity — ## Decision
- **docs/architecture/RUNPOD_V2_EXECUTION_PLATFORM.md** — Runpod v2 execution platform — **Status:** Accepted for staged delivery

## `kestrel_cloud_runpod/`

- **kestrel_cloud_runpod/__init__.py** — RunPod GPU management for Kestrel.
- **kestrel_cloud_runpod/clients.py** — Runpod v2 HTTP clients with one authenticated transport boundary.
  - `class RunpodTransport`; `class RunpodControlPlaneClient`; `class RunpodServerlessClient`
- **kestrel_cloud_runpod/core.py** — RunPod Core Manager Operations.
  - `class RunPodManagerCore`
- **kestrel_cloud_runpod/feature.py** — —
  - `class RunPodFeature`
- **kestrel_cloud_runpod/inference_provider.py** — SDK inference-lease adapter backed by durable Runpod Ollama capacity.
  - `class RunpodInferenceLeaseProvider`
- **kestrel_cloud_runpod/manager.py** — RunPod Manager - Combined Class.
  - `class RunPodManager`
- **kestrel_cloud_runpod/models.py** — Typed contracts for the Runpod v2 control and data planes.
  - `class RunPodManagerError`; `class RateLimit`; `class RunPodAPIError`; `class RunPodAmbiguousResultError`; `class CloudType`; `class ComputeProduct`; `class Availability`; `class FlashBoot`; `…`
- **kestrel_cloud_runpod/ollama.py** — Runpod private-Ollama lease integration for :class:`RunPodManager`.
  - `class RunPodOllamaMixin`
- **kestrel_cloud_runpod/ollama_contracts.py** — Typed contracts and cost policy for durable private-Ollama leases.
  - `class OllamaLeaseMode`; `class OllamaResourceType`; `class OllamaLeaseState`; `class OllamaTeardownState`; `class OllamaLeaseConflictError`; `class OllamaLeaseAuthorizationError`; `class OllamaLeaseReadinessError`; `class OllamaLeaseTeardownError`; `…`
- **kestrel_cloud_runpod/ollama_provider.py** — Runpod v2 capacity, readiness, and teardown adapter for Ollama leases.
  - `class RunpodOllamaDeployment`; `class RunpodOllamaCapacityProvider`
- **kestrel_cloud_runpod/ollama_reconciler.py** — One-shot entry point for externally scheduled Ollama lease reconciliation.
  - `async def reconcile_once(manager_factory)`; `def main()`
- **kestrel_cloud_runpod/ollama_repository.py** — Transactional SQLite persistence for private-Ollama leases.
  - `class SQLiteOllamaLeaseRepository`; `def lease_database_path(config)`; `def request_from_lease(lease)`
- **kestrel_cloud_runpod/ollama_runtime.py** — Canonical configuration contract for the authenticated Ollama workload.
  - `def require_immutable_ollama_image(image)`; `def build_ollama_runtime_environment(profile_environment)`; `def parse_ollama_model_allowlist(raw)`
- **kestrel_cloud_runpod/ollama_service.py** — Restart-safe lifecycle orchestration for private-Ollama leases.
  - `class OllamaLeaseService`
- **kestrel_cloud_runpod/placement.py** — Deterministic GPU placement from live Runpod v2 catalog offers.
  - `def select_gpu(offers, requirements)`
- **kestrel_cloud_runpod/pod_capacity_contracts.py** — Durable contracts for the canonical billable Runpod Pod lifecycle.
  - `class TrainingPodSource`; `class TrainingPodOwnership`; `class TrainingPodState`; `class TrainingPodCleanupState`; `class PodCapacityBillingState`; `class CatalogPodWorkloadState`; `class PodCapacityConstraints`; `class PodCapacityQuoteRequest`; `…`
- **kestrel_cloud_runpod/pod_capacity_provider.py** — Runpod REST v2 adapter for the canonical durable Pod capacity lifecycle.
  - `class TrainingPodObservation`; `class CreatedTrainingPod`; `class PodCapacityCreatedMismatchError`; `class TrainingPodCapacityProvider`; `class RunpodPodCapacityProvider`
- **kestrel_cloud_runpod/pod_capacity_quote.py** — Read-only composition for live Runpod Pod capacity quotes.
  - `class PodCapacityQuoteProvider`; `class PodCapacityQuoteService`
- **kestrel_cloud_runpod/pod_capacity_reconciler.py** — Installed one-shot command for externally scheduled capacity reconciliation.
  - `class PodCapacityReconcilerConfigurationError`; `class PodCapacityReconcilerBusyError`; `class PodCapacityReconcileSummary`; `async def reconcile_once(service_factory)`; `def load_service_factory(reference)`; `def run_cli(argv)`; `def main()`
- **kestrel_cloud_runpod/pod_capacity_repository.py** — SQLite WAL persistence for restart-safe Pod capacity ownership.
  - `class SQLitePodCapacityRepository`; `def pod_capacity_database_path(config)`
- **kestrel_cloud_runpod/pod_capacity_service.py** — Restart-safe acquisition and cleanup state machine for billable Pods.
  - `class PodCapacityLeaseService`
- **kestrel_cloud_runpod/pod_transport.py** — Narrow authenticated transport for one disposable catalog Pod attempt.
  - `class CatalogPodTransportError`; `class CatalogPodTransportConflictError`; `class CatalogPodWorkloadObservation`; `class CatalogPodWorkloadTransport`
- **kestrel_cloud_runpod/providers.py** — Provider adapters backed by the Runpod v2 REST control plane.
  - `class GPUProvider`; `class DirectRunPodProvider`; `class ManagedRunPodProvider`
- **kestrel_cloud_runpod/serverless_capacity_contracts.py** — Provider-neutral contracts for finite Serverless capacity and billing.
  - `class ServerlessCapacityConstraints`; `class ServerlessEndpointProfile`; `class ServerlessCapacityQuoteRequest`; `class ServerlessCapacityQuote`; `def serverless_billing_hour_starts(attempt_started_at, billable_coverage_until)`; `class ServerlessBillingAttempt`; `class ServerlessBillingReceipt`; `def serverless_worker_cost_usd(hourly_rate_usd, gpu_count, seconds)`; `…`
- **kestrel_cloud_runpod/serverless_capacity_provider.py** — Runpod REST v2 adapter for finite Serverless quotes and billing.
  - `class RunpodServerlessCapacityProvider`
- **kestrel_cloud_runpod/training.py** — RunPod LoRA Training Methods.
  - `class RunPodTrainingMixin`
- **kestrel_cloud_runpod/training_contracts.py** — Compatibility names for the canonical Pod capacity contracts.
- **kestrel_cloud_runpod/training_provider.py** — Compatibility names for the canonical Pod capacity provider.
- **kestrel_cloud_runpod/training_reconciler.py** — One-shot entry point for externally scheduled training Pod reconciliation.
  - `class TrainingReconcileManager`; `async def reconcile_once(manager_factory)`; `def main()`
- **kestrel_cloud_runpod/training_repository.py** — Compatibility import for the canonical Pod capacity repository.
- **kestrel_cloud_runpod/training_service.py** — Compatibility import for the canonical Pod capacity lease service.

## `runtime/`

- **runtime/ollama-runtime/.dockerignore** — —
- **runtime/ollama-runtime/Dockerfile** — —
- **runtime/ollama-runtime/README.md** — Kestrel private Ollama runtime — This directory is the single deployable workload contract used by durable Ollama leases.
- **runtime/ollama-runtime/go.mod** — —
- **runtime/ollama-runtime/main.go** — —
- **runtime/ollama-runtime/main_test.go** — —
- **runtime/ollama-runtime/runpod-template.json** — (configuration)

## `scripts/`

- **scripts/check_runpod_openapi.py** — Validate the pinned Runpod v2 OpenAPI file and report upstream drift.
  - `def load_openapi(source)`; `def sha256(data)`; `def read_lock()`; `def check_pin()`; `def fetch_live(source)`; `def semantic_diff(pinned, live)`; `def check_live()`; `def main()`
- **scripts/generate_repo_map.py** — Generate REPO_MAP.md — a file-tree + per-file purpose index for this repo.
  - `class FileEntry`; `def repo_name()`; `def tracked_files()`; `def is_excluded(path)`; `def first_sentence(text, max_chars)`; `def summarize_python(path)`; `def summarize_markdown(path)`; `def summarize_other(path)`; `…`

## `tests/`

- **tests/conftest.py** — Test configuration for kestrel-cloud-runpod.
  - `def pytest_addoption(parser)`; `def pytest_collection_modifyitems(config, items)`
- **tests/ollama_test_support.py** — Shared deterministic fixtures for Ollama lease tests.
  - `class MutableClock`; `def make_request(clock)`; `def make_decision(product)`; `class FakeOllamaProvider`; `def serverless_plan(rate)`
- **tests/pod_capacity_test_support.py** — Deterministic doubles for generic catalog Pod capacity tests.
  - `class MutableClock`; `def constraints()`; `def quote(clock)`; `def request(clock)`; `def profile()`; `class FakeCapabilityStore`; `class FakeCapacityProvider`; `class FakeWorkloadTransport`; `…`
- **tests/serverless_capacity_test_support.py** — Shared fixtures for finite Serverless capacity tests.
  - `class MutableClock`; `def constraints()`; `def profile()`; `def request()`; `def offer()`; `def endpoint(endpoint_profile)`; `def quote(clock)`; `def attempt(item)`
- **tests/test_inference_provider.py** — SDK provider boundary tests for durable private Runpod inference.
  - `def test_dedicated_entry_point_loads_sdk_provider_contract()`; `def test_capabilities_are_deterministic_and_policy_scoped(tmp_path)`; `async def test_quote_is_read_only_and_acquire_returns_pending(tmp_path)`; `async def test_quote_expires_before_estimated_cold_start_window_closes(tmp_path)`; `async def test_catalog_refresh_cannot_consume_remaining_cold_start_window(tmp_path)`; `async def test_status_returns_only_exact_authenticated_ready_route(tmp_path)`; `async def test_restart_reconciles_same_lease_without_duplicate_capacity(tmp_path)`; `async def test_owner_isolation_precedes_status_or_release_mutation(tmp_path)`; `…`
- **tests/test_ollama_contracts.py** — Cost selection and public-route contracts for Ollama leases.
  - `def test_bursty_session_selects_lower_effective_serverless_cost()`; `def test_sustained_session_selects_pod_at_live_rates()`; `def test_forced_mode_and_cost_cap_fail_closed()`; `def test_request_fingerprint_changes_with_billing_policy()`; `def test_model_tags_normalize_implicit_latest()`; `def test_request_rejects_non_finite_or_non_positive_cost(invalid_cost)`; `def test_request_rejects_fractional_duration()`; `def test_provider_error_is_redacted_before_durable_state()`
- **tests/test_ollama_mixin.py** — Manager integration tests for the durable Ollama lease surface.
  - `async def test_mixin_delegates_only_to_injected_durable_service()`; `def test_mixin_has_no_in_memory_or_managed_provider_fallback()`; `def test_mixin_builds_one_configured_durable_service(tmp_path, monkeypatch)`
- **tests/test_ollama_provider.py** — Runpod v2 adapter tests with in-memory HTTP/control-plane doubles.
  - `def test_workload_credentials_must_not_reuse_control_or_cross_product_keys()`; `async def test_network_volume_cache_rejects_concurrent_serverless_writers(monkeypatch)`; `async def test_plan_uses_product_specific_live_catalog_prices(monkeypatch)`; `async def test_poolless_serverless_availability_fails_clearly_without_create(monkeypatch)`; `async def test_pooled_serverless_offer_survives_unrelated_poolless_offer(monkeypatch)`; `async def test_serverless_provision_is_load_balanced_and_configuration_owned(monkeypatch)`; `async def test_observe_requires_provider_health_and_reads_exact_model()`; `async def test_pod_observation_uses_v2_status_and_runtime_route()`; `…`
- **tests/test_ollama_reconciler.py** — External one-shot reconciler entry point tests.
  - `async def test_reconcile_once_constructs_manager_and_runs_single_pass()`
- **tests/test_ollama_repository.py** — Persistence, idempotency, and restart tests for Ollama leases.
  - `def test_insert_is_idempotent_and_survives_repository_restart(tmp_path)`; `def test_reused_lease_id_with_changed_request_is_rejected(tmp_path)`; `def test_compare_and_set_rejects_stale_revision(tmp_path)`; `def test_request_recovery_rejects_corrupt_constraints(tmp_path, constraints_json)`; `def test_request_recovery_rejects_non_requested_state(tmp_path)`; `def test_database_path_requires_absolute_explicit_configuration(monkeypatch)`; `def test_database_environment_expansion_fails_when_missing(monkeypatch)`
- **tests/test_ollama_runtime.py** — Authenticated Ollama runtime and Runpod template contract tests.
  - `def test_runtime_image_must_use_reviewed_repository_and_digest()`; `def test_runtime_environment_is_owned_and_lease_bounded()`; `def test_runtime_environment_rejects_unapproved_model_and_overrides()`; `def test_runtime_environment_rejects_weak_or_expired_capability()`; `def test_runtime_environment_restricts_mode_specific_storage_roots()`; `def test_runpod_template_covers_pod_and_native_load_balancer()`; `def test_container_contract_is_pinned_nonroot_and_secret_free()`; `def test_tracked_configs_share_the_runtime_policy_surface()`; `…`
- **tests/test_ollama_service.py** — Lifecycle, crash recovery, readiness, cost, and teardown tests.
  - `async def test_acquire_pulls_missing_model_and_only_returns_ready_route(tmp_path)`; `async def test_duplicate_acquire_does_not_create_second_resource(tmp_path)`; `async def test_reconciler_recovers_crash_after_requested_insert(tmp_path)`; `async def test_concurrent_duplicate_acquire_cannot_double_provision(tmp_path)`; `async def test_release_is_idempotent(tmp_path)`; `async def test_teardown_failure_keeps_provider_id_and_reconciler_retries(tmp_path)`; `async def test_restart_reconciles_resource_created_before_id_was_persisted(tmp_path)`; `async def test_release_does_not_orphan_late_ambiguous_creation(tmp_path)`; `…`
- **tests/test_pod_capacity_architecture.py** — One-source-of-truth and public/private package-boundary tests.
  - `def test_training_names_are_aliases_not_a_second_writable_lifecycle()`; `def test_public_cloud_package_has_no_private_catalog_dependency()`
- **tests/test_pod_capacity_contracts.py** — Generic quote, identity, cost, environment, and persistence contracts.
  - `def test_request_binds_exact_parameter_digest_quote_cost_and_image()`; `def test_request_refuses_control_database_and_service_owned_environment(key)`; `def test_persisted_spec_contains_only_secret_identity_digest_and_expiry()`; `def test_changed_attempt_environment_changes_durable_identity()`; `def test_billing_receipt_matches_private_terminal_contract()`; `def test_billing_receipt_rejects_non_authoritative_dimensions(changes, message)`; `def test_zero_cost_receipt_requires_no_billed_interval_without_a_pod()`
- **tests/test_pod_capacity_evidence.py** — Durable content-free realized Pod and worker evidence contract tests.
  - `def worker_envelope(clock)`; `async def successful_workload(tmp_path)`; `async def test_realized_lifecycle_worker_and_billing_evidence_survive_restart(tmp_path)`; `async def test_duplicate_observations_and_cas_race_preserve_first_timestamps(tmp_path)`; `async def test_worker_evidence_binding_replay_and_strict_allowlist(tmp_path)`; `async def test_public_projection_is_content_free_and_omits_capabilities(tmp_path)`; `async def test_v2_catalog_row_migrates_with_explicit_missing_evidence(tmp_path)`; `def test_lifecycle_timestamp_order_and_worker_startup_split_are_validated()`
- **tests/test_pod_capacity_provider.py** — Runpod v2 quote, exact recovery, termination, and billing adapter tests.
  - `async def test_quote_binds_parameter_digest_exact_gpu_rate_and_timing()`; `async def test_create_injects_exact_image_gpu_and_attempt_environment()`; `async def test_created_metadata_mismatch_preserves_known_pod_id()`; `async def test_create_rejects_realized_placement_outside_quote(override, expected_message)`; `async def test_exact_recovery_rejects_mismatch_and_multiple_matches()`; `async def test_final_billing_waits_for_records_then_returns_content_free_receipt()`; `async def test_final_billing_waits_until_records_cover_termination()`
- **tests/test_pod_capacity_quote.py** — Tests for read-only Pod capacity quote composition.
  - `def test_quote_service_requires_an_explicit_profile_catalog()`; `async def test_quote_service_delegates_without_a_repository()`
- **tests/test_pod_capacity_reconciler.py** — Installed, locked, content-free Pod-capacity reconciler command tests.
  - `def test_distribution_installs_capacity_reconciler_entry_point()`; `async def test_reconcile_once_constructs_service_and_runs_one_pass(tmp_path)`; `def test_cli_runs_once_without_provisioning_and_emits_content_free_json(tmp_path)`; `def test_cli_missing_factory_fails_before_loading_or_provisioning()`; `def test_cli_auth_configuration_error_never_exposes_message_or_secret()`; `def test_cli_missing_catalog_dependency_fails_before_reconcile(tmp_path)`; `def test_cli_lock_contention_is_retryable_and_never_runs_service(tmp_path)`; `def test_cli_timeout_is_retryable_and_content_free(tmp_path)`; `…`
- **tests/test_pod_capacity_service.py** — End-to-end durable catalog Pod lifecycle tests without live resources.
  - `async def test_acquire_injects_attempt_env_and_persists_no_secret(tmp_path)`; `async def test_identical_replay_adopts_ready_lease_but_changed_env_conflicts(tmp_path)`; `async def test_ambiguous_create_never_retries_and_delayed_visibility_is_adopted(tmp_path)`; `async def test_known_created_mismatch_is_durably_terminated_and_billed(tmp_path)`; `async def test_missing_realized_placement_is_durably_terminated_and_billed(tmp_path)`; `async def test_submit_success_result_replays_until_ack_then_billing_reconciles(tmp_path)`; `async def test_result_replays_after_host_restart_until_durable_ack(tmp_path)`; `async def test_restart_finishes_teardown_after_ack_transition_crash(tmp_path)`; `…`
- **tests/test_pod_capacity_transport.py** — Authenticated single-attempt workload transport tests.
  - `async def test_transport_uses_anonymous_health_and_bearer_for_job_routes()`; `async def test_transport_rejects_local_hash_mismatch_before_network()`; `async def test_transport_maps_remote_conflict_without_echoing_body()`; `async def test_transport_returns_result_unchanged_but_checks_attempt_binding()`; `async def test_transport_requires_tls_and_refuses_redirects()`
- **tests/test_release_metadata.py** — Release metadata alignment tests.
  - `def test_project_version_has_matching_release_notes()`
- **tests/test_runpod_clients.py** — HTTP and request-shape contracts for both Runpod v2 services.
  - `def test_control_client_auth_user_agent_catalog_query_and_timeouts()`; `def test_default_user_agent_is_explicit_and_non_generic()`; `def test_catalog_availability_requires_one_product_context()`; `def test_rfc9457_problem_is_typed_and_does_not_expose_body()`; `def test_safe_get_retries_using_retry_after_and_records_rate_limit()`; `def test_safe_get_uses_rate_limit_reset_when_retry_after_is_absent()`; `def test_create_is_not_retried_after_ambiguous_server_error()`; `def test_create_timeout_is_ambiguous_and_not_retried()`; `…`
- **tests/test_runpod_feature.py** — —
  - `class FakeRunPodManager`; `class DummyLLMService`; `async def runpod_feature(monkeypatch)`; `async def test_manage_gpu_start_and_stop(runpod_feature)`; `async def test_image_generation_tears_down_session(runpod_feature)`; `async def test_manage_gpu_unknown_action_returns_failed(runpod_feature)`; `async def test_start_unknown_profile_returns_failed(runpod_feature)`; `async def test_start_invalid_ttl_returns_failed(runpod_feature)`; `…`
- **tests/test_runpod_model_contracts.py** — Contracts for RunPod profile-owned model defaults.
  - `async def test_resume_stopped_pod_requires_profile_default_model()`
- **tests/test_runpod_openapi_contract.py** — Parity tests between Kestrel's typed calls and the pinned beta schema.
  - `def test_pin_checksum_and_validator_script()`; `def test_all_consumed_control_plane_operations_match_pin()`; `def test_consumed_request_and_response_shapes_match_pin()`; `def test_typed_create_and_update_payloads_validate_against_pin()`; `def test_runtime_base_url_is_explicit_despite_beta_schema_server_discrepancy()`; `def test_semantic_diff_reports_breaking_and_additive_changes()`; `def test_production_package_contains_no_v1_graphql_or_legacy_sdk_calls()`
- **tests/test_runpod_placement.py** — Live-catalog placement policy tests.
  - `def test_placement_prefers_availability_then_live_price_and_records_snapshot()`; `def test_serverless_mig_uses_product_availability_not_pod_max_count()`; `def test_placement_fails_closed_for_incompatible_offers(offer, requirements)`; `def test_placement_enforces_allowed_data_centers()`; `def test_placement_requires_cuda_filtered_catalog_provenance()`
- **tests/test_runpod_provider.py** — Compatibility-provider migration tests for the v2 client boundary.
  - `def test_direct_provider_selects_from_live_catalog_and_records_placement()`; `def test_direct_provider_fails_before_catalog_when_required_env_is_unset(monkeypatch)`; `def test_profile_rejects_persistent_volume_below_runpod_floor()`; `def test_direct_provider_lifecycle_and_v2_logs()`; `def test_private_cli_ssh_execution_has_clear_migration_error()`; `async def test_manager_get_logs_uses_provider_v2_log_stream()`; `def test_legacy_profile_fields_fail_with_migration_guidance()`; `def test_manager_maps_v2_http_runtime_port_to_pod_proxy_url()`; `…`
- **tests/test_runpod_smoke.py** — Opt-in, read-only authenticated smoke checks for the beta v2 API.
  - `def test_live_v2_gpu_catalog_is_read_only_and_typed()`; `def test_live_v2_network_volume_inventory_is_read_only_and_typed()`
- **tests/test_serverless_capacity_contracts.py** — Contract tests for finite Serverless quotes and billing evidence.
  - `def test_public_package_exports_serverless_construction_dependencies()`; `def test_quote_round_trip_binds_normalized_parameters_and_exact_cost_math()`; `def test_quote_rejects_unsafe_or_inconsistent_dimensions(changes, message)`; `def test_quote_deserialization_rejects_raw_or_content_bearing_fields()`; `def test_quote_serialization_is_content_free()`; `def test_endpoint_profile_requires_exact_scale_to_zero_pool_and_data_center()`; `def test_request_requires_bounded_queue_billable_and_non_worker_estimates()`; `def test_serverless_job_policy_accepts_provider_boundaries_and_rejects_overflow()`; `…`
- **tests/test_serverless_capacity_provider.py** — REST v2 adapter tests for Serverless quote and billing reconciliation.
  - `async def test_quote_is_exact_read_only_v2_catalog_and_endpoint_observation()`; `async def test_billing_requires_a_separate_restricted_job_status_client()`; `async def test_quote_uses_existing_selector_for_mig_cuda_region_and_availability()`; `async def test_quote_rejects_endpoint_profile_mismatch(raw_change, message)`; `async def test_quote_rejects_shared_pool_and_unsafe_catalog_numbers()`; `async def test_submission_validation_rejects_rate_pool_workload_and_expiry_drift()`; `async def test_final_billing_binds_terminal_job_and_complete_endpoint_window()`; `async def test_final_billing_reserves_idle_tail_across_hour_boundary()`; `…`
- **tests/test_training_contracts.py** — Public durable training ownership contracts.
  - `def test_request_requires_ordered_aware_deadlines()`; `def test_create_request_does_not_accept_a_provider_id()`; `def test_lifecycle_error_exposes_cleanup_authority_without_provider_detail()`; `def test_request_fingerprint_changes_with_deadline()`; `def test_child_request_fingerprint_is_bound_to_its_cleanup_family()`; `def test_request_defaults_root_for_pre_family_callers()`
- **tests/test_training_mixin.py** — Manager integration for durable training capacity and workload recovery.
  - `class FakeWorkloadClient`; `async def test_manager_session_tracks_job_result_and_confirmed_stop(tmp_path, monkeypatch)`; `async def test_submission_failure_stops_capacity_and_preserves_record(tmp_path, monkeypatch)`; `async def test_status_failure_keeps_job_and_cleanup_authority(tmp_path, monkeypatch)`; `async def test_cancel_stops_pod_and_stop_failure_is_retryable(tmp_path, monkeypatch)`
- **tests/test_training_provider.py** — Runpod REST v2 training capacity adapter contracts.
  - `async def test_observe_resolves_exact_http_proxy_route()`; `async def test_create_uses_provider_v2_boundary_and_validates_id()`; `async def test_successful_create_with_unusable_identity_remains_reconcilable(response)`; `async def test_find_by_name_rejects_duplicate_provider_resources()`; `async def test_stop_requires_v2_confirmation()`; `async def test_cancelled_start_waits_for_inflight_v2_mutation_to_resolve()`
- **tests/test_training_reconciler.py** — External one-shot training reconciliation entry point.
  - `async def test_reconcile_once_constructs_manager_and_runs_one_pass()`
- **tests/test_training_repository.py** — Persistence, CAS, restart, and cross-process exclusion tests.
  - `def test_reservation_survives_repository_restart(tmp_path)`; `def test_revision_cas_rejects_stale_writer(tmp_path)`; `def test_second_active_token_cannot_claim_same_pod(tmp_path)`; `def test_database_path_is_explicit_absolute_and_expands_environment(tmp_path, monkeypatch)`; `def test_pre_family_database_migrates_with_legacy_attempt_self_rooted(tmp_path)`; `def test_family_release_gate_blocks_a_late_fallback_reservation(tmp_path)`; `def test_versioned_table_migration_preserves_every_lifecycle_shape(tmp_path)`
- **tests/test_training_service.py** — Billing-safe training acquisition, cancellation, and reconciliation tests.
  - `async def test_resumed_pod_missing_route_is_stopped(tmp_path)`; `async def test_readiness_exception_stop_failure_retains_retryable_pod_after_restart(tmp_path)`; `async def test_preexisting_running_pod_route_failure_is_never_stopped(tmp_path)`; `async def test_ambiguous_create_reconciles_exact_name_and_stops_pod(tmp_path)`; `async def test_restart_recovers_crash_between_create_and_id_persistence(tmp_path)`; `async def test_reconciler_leaves_live_inflight_create_owned_by_acquirer(tmp_path)`; `async def test_restart_before_persistent_discovery_never_stops_preexisting_pod(tmp_path)`; `async def test_restart_reconciles_submitted_job_at_hard_deadline(tmp_path)`; `…`
- **tests/training_test_support.py** — Deterministic doubles for durable training Pod lifecycle tests.
  - `class MutableClock`; `def training_profile(profile_id)`; `def training_request(clock)`; `class FakeTrainingProvider`; `def training_service(tmp_path, clock, provider)`

## `vendor/`

- **vendor/runpod-v2-openapi.lock.json** — (configuration)
- **vendor/runpod-v2-openapi.yaml** — (configuration)
