# kestrel-cloud-runpod — Repo Map

Auto-generated file-tree + per-file purpose index. Do **not** edit by hand —
regenerate via `python scripts/generate_repo_map.py` (refreshed nightly by
`.github/workflows/repo-map.yml`). No timestamp on purpose: the nightly job
commits only when the tree actually changes; `git log REPO_MAP.md` has the date.

**Scope:** 125 tracked files (96 `.py`, 8 `.md`, 21 other). Excludes caches, lockfiles, and build artifacts.

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
- **docs/runpod-v2-serverless-schema.md** — Runpod Serverless v2 schema provenance — The production management contract for this package is the OpenAPI document served by the Runpod v2 API itself:

## `kestrel_cloud_runpod/`

- **kestrel_cloud_runpod/__init__.py** — RunPod GPU management for Kestrel.
- **kestrel_cloud_runpod/_one_shot_reconciler.py** — Shared, content-free argument boundary for one-shot reconcilers.
  - `def run_one_shot_reconciler(argv)`
- **kestrel_cloud_runpod/billing.py** — Shared UTC coverage and exact monetary arithmetic for provider observations.
  - `def billing_hour_bounds(started_at, coverage_until)`; `def exact_decimal_sum(values)`; `def exact_timedelta_microseconds(value)`; `def ceil_timedelta_seconds(value)`; `def floor_timedelta_seconds(value)`; `def exact_decimal_integer_scale(value, multiplier)`; `def conservative_decimal_ratio_float(value, divisor)`; `def billing_exposure_ratio_float(value)`; `…`
- **kestrel_cloud_runpod/clients.py** — Runpod v2 HTTP clients with one authenticated transport boundary.
  - `def is_retryable_runpod_api_error(error)`; `def is_exact_runpod_not_found(error)`; `class RunpodTransport`; `class RunpodControlPlaneClient`; `class RunpodServerlessClient`
- **kestrel_cloud_runpod/core.py** — RunPod Core Manager Operations.
  - `class RunPodManagerCore`
- **kestrel_cloud_runpod/cpu_catalog.py** — CPU catalog facts and explicit Pod matching, never paid-create authority.
  - `def validate_cpu_query(product, vcpu_count)`; `class CpuDataCenter`; `class CpuOffer`; `class CpuPodCatalogMatch`; `class CpuCatalogSnapshot`
- **kestrel_cloud_runpod/dogfood_contracts.py** — Product-neutral typed contracts shared with the live dogfood harness.
  - `class DogfoodError`; `class DogfoodSafetyError`; `class ResourceType`; `class DogfoodLane`; `class DogfoodPhase`; `class ResourceIdentity`; `class ExpectedResource`; `class ResourcePlan`; `…`
- **kestrel_cloud_runpod/effect_custody.py** — Retain admitted effects until their actual mutation and receipt tail finishes.
  - `async def await_owned_effect(effect)`
- **kestrel_cloud_runpod/feature.py** — —
  - `class RunPodFeature`
- **kestrel_cloud_runpod/inference_provider.py** — SDK inference-lease adapter backed by durable Runpod Ollama capacity.
  - `class RunpodInferenceLeaseProvider`
- **kestrel_cloud_runpod/manager.py** — RunPod Manager - Combined Class.
  - `class RunPodManager`
- **kestrel_cloud_runpod/models.py** — Typed contracts for the Runpod v2 control and data planes.
  - `class RunPodManagerError`; `class RunPodLegacyAdmissionRetiredError`; `class RunPodMountPolicyMismatchError`; `class ResourceFieldState`; `class ResourceFieldEvidence`; `def environment_values_sha256(environment)`; `def normalize_cuda_floor(value)`; `class EndpointMountAssessment`; `…`
- **kestrel_cloud_runpod/ollama.py** — Runpod private-Ollama lease integration for :class:`RunPodManager`.
  - `class RunPodOllamaMixin`
- **kestrel_cloud_runpod/ollama_contracts.py** — Typed contracts and cost policy for durable private-Ollama leases.
  - `class OllamaLeaseMode`; `class OllamaResourceType`; `class OllamaLeaseState`; `class OllamaTeardownState`; `class OllamaProvisioningEvidenceState`; `class OllamaProviderRatePrecision`; `class OllamaCleanupBasis`; `class OllamaRuntimeTelemetryState`; `…`
- **kestrel_cloud_runpod/ollama_provider.py** — Runpod v2 capacity, readiness, and teardown adapter for Ollama leases.
  - `class RunpodOllamaDeployment`; `class RunpodOllamaCapacityProvider`
- **kestrel_cloud_runpod/ollama_reconciler.py** — One-shot entry point for externally scheduled Ollama lease reconciliation.
  - `async def reconcile_once(manager_factory)`; `def run_cli(argv)`; `def main()`
- **kestrel_cloud_runpod/ollama_repository.py** — Transactional SQLite persistence for private-Ollama leases.
  - `class OllamaWriterDrainAttestation`; `class SQLiteOllamaLeaseRepository`; `def lease_database_path(config)`; `def request_from_lease(lease)`
- **kestrel_cloud_runpod/ollama_runtime.py** — Canonical configuration contract for the authenticated Ollama workload.
  - `def require_immutable_ollama_image(image)`; `def build_ollama_runtime_environment(profile_environment)`; `def parse_ollama_model_allowlist(raw)`
- **kestrel_cloud_runpod/ollama_service.py** — Restart-safe lifecycle orchestration for private-Ollama leases.
  - `class OllamaLeaseService`
- **kestrel_cloud_runpod/placement.py** — Deterministic GPU placement from live Runpod v2 catalog offers.
  - `def select_gpu(offers, requirements)`
- **kestrel_cloud_runpod/pod_billing.py** — One exact-Pod billing observation path for training and inference consumers.
  - `async def read_exact_pod_billing(client)`
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
  - `class ServerlessCapacityConstraints`; `class ServerlessEndpointSpec`; `class ServerlessEndpointProfile`; `class PlannedServerlessEndpoint`; `class ServerlessCapacityQuoteRequest`; `class PlannedServerlessCapacityQuoteRequest`; `class PlannedServerlessCapacityQuote`; `class ServerlessEndpointActivationReceipt`; `…`
- **kestrel_cloud_runpod/serverless_capacity_provider.py** — Runpod REST v2 adapter for finite Serverless quotes and billing.
  - `class ServerlessEndpointActivationError`; `class ServerlessEndpointCleanupError`; `class RunpodServerlessCapacityProvider`; `async def read_exact_serverless_billing(client)`
- **kestrel_cloud_runpod/signed_invocations.py** — Canonical Ed25519 receipts for authenticated agent invocations.
  - `class SignedInvocationError`; `def canonical_json(value)`; `def canonical_sha256(value)`; `def utf8_sha256(value)`; `def phase_evidence_sha256(phase, evidence_payload)`; `def base64url_encode(value)`; `def base64url_decode(value, name)`; `class AttestedInvokeRequest`; `…`
- **kestrel_cloud_runpod/sqlite_store.py** — Canonical transaction and connection lifetime for durable lease stores.
  - `def lease_database_connection(path)`
- **kestrel_cloud_runpod/training.py** — RunPod LoRA Training Methods.
  - `class RunPodTrainingMixin`
- **kestrel_cloud_runpod/training_contracts.py** — Compatibility names for the canonical Pod capacity contracts.
- **kestrel_cloud_runpod/training_provider.py** — Compatibility names for the canonical Pod capacity provider.
- **kestrel_cloud_runpod/training_reconciler.py** — One-shot entry point for externally scheduled training Pod reconciliation.
  - `class TrainingReconcileManager`; `async def reconcile_once(manager_factory)`; `def run_cli(argv)`; `def main()`
- **kestrel_cloud_runpod/training_repository.py** — Compatibility import for the canonical Pod capacity repository.
- **kestrel_cloud_runpod/training_service.py** — Compatibility import for the canonical Pod capacity lease service.

## `runtime/`

- **runtime/ollama-runtime/.dockerignore** — —
- **runtime/ollama-runtime/Dockerfile** — —
- **runtime/ollama-runtime/README.md** — Kestrel private Ollama runtime — This directory is the single deployable workload contract used by durable Ollama leases.
- **runtime/ollama-runtime/go.mod** — —
- **runtime/ollama-runtime/lifecycle_telemetry_acceptance_test.go** — —
- **runtime/ollama-runtime/main.go** — —
- **runtime/ollama-runtime/main_test.go** — —
- **runtime/ollama-runtime/runpod-template.json** — (configuration)

## `scripts/`

- **scripts/check_distribution.py** — Verify actual release archives contain only canonical source and metadata.
  - `def verify_distribution(artifact)`; `def main()`
- **scripts/check_runpod_openapi.py** — Validate the pinned Runpod v2 OpenAPI file and report upstream drift.
  - `def load_openapi(source)`; `def sha256(data)`; `def read_lock()`; `def check_pin()`; `def fetch_live(source)`; `def semantic_diff(pinned, live)`; `def check_live()`; `def main()`
- **scripts/generate_repo_map.py** — Generate REPO_MAP.md — a file-tree + per-file purpose index for this repo.
  - `class FileEntry`; `def repo_name()`; `def tracked_files()`; `def is_excluded(path)`; `def first_sentence(text, max_chars)`; `def summarize_python(path)`; `def summarize_markdown(path)`; `def summarize_other(path)`; `…`

## `tests/`

- **tests/conftest.py** — Test configuration for kestrel-cloud-runpod.
  - `def pytest_addoption(parser)`; `def pytest_collection_modifyitems(config, items)`
- **tests/fixtures/serverless-0.10.1-authority.json** — (configuration)
- **tests/fixtures/serverless_billing_v015_numeric_wire.json** — (configuration)
- **tests/ollama_test_support.py** — Shared deterministic fixtures for Ollama lease tests.
  - `class MutableClock`; `def make_request(clock)`; `def runtime_telemetry_payload(binding, timestamp)`; `def make_decision(product)`; `def non_compute_cost_policy()`; `def non_compute_cost_policies()`; `class FakeOllamaProvider`; `def serverless_plan(rate)`; `…`
- **tests/pod_capacity_test_support.py** — Deterministic doubles for generic catalog Pod capacity tests.
  - `def native_httpx_connect_failure()`; `def complete_pod_payload()`; `def native_pod_action_harness()`; `def acknowledgement()`; `class MutableClock`; `def constraints(gpu_count)`; `def quote(clock, gpu_count)`; `def request(clock, gpu_count)`; `…`
- **tests/serverless_capacity_test_support.py** — Shared fixtures for finite Serverless capacity tests.
  - `def legacy_planned_authority()`; `class MutableClock`; `def constraints()`; `def endpoint_spec()`; `def profile()`; `def planned_endpoint()`; `def request()`; `def planned_request()`; `…`
- **tests/test_billing_arithmetic.py** — Exact shared billing primitives, independent of a caller's Decimal context.
  - `def test_signed_deltas_retain_one_to_one_e_minus_29_without_rounding()`; `def test_signed_sum_across_two_hundred_orders_of_magnitude(order)`; `def test_signed_sum_rejects_nonfinite_or_wrong_unit(invalid)`; `def test_empty_sum_is_exact_zero()`; `def test_sum_ignores_ambient_exponent_limits_and_traps()`; `def test_exact_decimal_sum_preserves_published_zero_exponent(values, expected)`; `def test_sum_rejects_pathological_exponent_span_before_large_allocation()`; `def test_exposure_sum_uses_one_context_independent_directed_boundary()`; `…`
- **tests/test_catalog_artifact_handoff.py** — Restart/cancellation/Stop safety of the public opaque artifact handoff.
  - `async def pending(tmp_path, transport, store)`; `def test_evidence_is_content_free_strict_and_canonical()`; `async def test_exact_ack_intent_and_observed_proof_survive_restart_and_revocation(tmp_path)`; `async def test_ack_timeout_preserves_job_and_exact_intent_until_authenticated_recovery(tmp_path)`; `async def test_crash_after_remote_ack_before_local_receipt_is_recoverable(tmp_path)`; `async def test_ack_tail_remains_owned_through_repeated_cancellation_and_stop(tmp_path, stop)`; `async def test_stop_during_capability_load_prevents_ack_dispatch(tmp_path)`; `async def test_status_recovery_preserves_ack_fact_when_stop_wins_in_flight(tmp_path)`; `…`
- **tests/test_catalog_handoff_recovery.py** — Public opaque recovery boundary tests; not private-worker or GPU proof.
  - `def recovery_payload()`; `class RecoveryTransport`; `async def test_recovery_records_actual_proof_without_execution_or_private_persistence(tmp_path)`; `async def test_fresh_ack_or_recovery_checks_hard_deadline_twice(tmp_path, operation, during_load)`; `async def test_recovery_tail_owns_snapshot_and_actual_proof_through_cancellation(tmp_path, stop)`; `async def test_stop_during_capability_load_prevents_recovery_and_keeps_only_intent(tmp_path)`; `async def test_recovery_without_live_capability_never_invents_actual_proof(tmp_path)`; `async def test_recovery_transport_failure_preserves_intent_and_retries_without_resubmission(tmp_path)`; `…`
- **tests/test_contract_containment.py** — Every public entry point raises ONLY its module's declared error type.
  - `def test_signed_invocations_helpers_contain_every_hostile_value(label, fn, value)`; `def test_dogfood_helpers_contain_every_hostile_value(label, fn, value)`; `def test_dogfood_from_payload_contains_a_hostile_value_in_every_field(name, cls, payload)`; `def test_dogfood_from_payload_contains_a_hostile_envelope(value)`; `def test_signed_from_payload_contains_a_hostile_value_in_every_field()`; `def test_signed_from_payload_contains_a_hostile_envelope(value)`; `def test_dogfood_constructors_contain_a_hostile_value_in_every_field()`; `def test_signer_and_verifier_boundaries_contain_every_hostile_value(value)`; `…`
- **tests/test_cpu_catalog.py** — Read-only CPU discovery and matching; all provider traffic is synthetic.
  - `def offer()`; `def request()`; `def snapshot()`; `def match(catalog, pod_request)`; `def test_cpu_catalog_fixture_matches_pinned_schema()`; `def test_query_binds_product_count_and_observation_time(product)`; `def test_invalid_query_never_reaches_transport(product, count)`; `def test_malformed_catalog_is_refused_without_private_echo(raw)`; `…`
- **tests/test_cpu_pod_requests.py** — CPU/GPU wire variants through the existing v2 transport; no cloud calls.
  - `def cpu_request()`; `def pod()`; `def test_cpu_and_gpu_requests_use_the_same_pinned_schema()`; `def test_invalid_cpu_counts_are_refused_before_any_transport(count)`; `def test_cpu_count_is_explicit_without_requesting_ram(count)`; `def test_gpu_and_ram_constraints_cannot_be_added_to_cpu_variant(fields)`; `def test_cpu_flavor_is_required(cpu_id)`; `def test_cpu_rejects_unsupported_networking_and_storage(fields)`; `…`
- **tests/test_distribution_boundary.py** — Build real archives with unrelated local metadata and inspect their bytes.
  - `def native_distribution(tmp_path)`; `def test_ci_and_publication_use_the_same_gate_before_artifact_upload()`; `def test_real_sdist_rebuilds_the_exact_wheel(native_distribution, tmp_path)`; `def test_real_sdist_and_wheel_cannot_collect_nested_metadata(native_distribution)`; `def test_real_archive_envelope_verifier_and_cli(native_distribution)`; `def test_install_metadata_cannot_drift_with_valid_source_and_record(native_distribution, tmp_path, mutation)`; `def test_metadata_frontend_absence_refuses_without_backend_fallback(native_distribution, monkeypatch)`; `def test_native_frontend_failure_refuses_install_metadata(native_distribution, tmp_path, monkeypatch, failure)`; `…`
- **tests/test_dogfood_contracts.py** — Coverage for the product-neutral contracts extracted from ``dogfood.py``.
  - `def test_frinz_facing_surface_is_present(name)`; `def test_module_does_not_import_the_harness_or_the_transport()`; `def test_the_import_guard_catches_every_spelling(source)`; `def test_every_phase_wire_value_is_pinned()`; `def test_every_lane_and_resource_type_wire_value_is_pinned()`; `def test_resource_mutating_phases_are_pinned()`; `def test_content_free_guard_rejects_private_material(payload)`; `def test_content_free_guard_admits_digests_and_timings(payload)`; `…`
- **tests/test_effect_custody.py** — Owned effects keep cancellation from detaching mutation/receipt tails.
  - `def test_owned_effect_metadata_requires_fixed_core_and_test_loop_custody()`; `async def test_success_and_failure_preserve_original_outcomes()`; `async def test_repeated_cancellation_joins_real_thread_and_receipt_tail(failure)`; `async def test_effect_self_cancellation_does_not_loop_forever()`; `async def test_late_effect_failure_retains_original_cancellation_and_native_evidence(kind)`; `async def test_joined_failure_emits_no_orphaned_shield_future_diagnostics()`; `async def test_existing_awaitable_contract_is_not_narrowed_to_coroutines(kind)`; `async def test_real_task_admission_refusal_closes_only_the_original_unstarted_effect(kind)`; `…`
- **tests/test_exact_pod_billing.py** — Exact Pod billing through the native HTTP client; no Runpod account access.
  - `def native_billing()`; `async def test_native_complete_exact_pod_history(native_billing)`; `async def test_native_missing_middle_hour_is_pending_not_min_max_coverage(native_billing)`; `async def test_native_rejects_component_total_disagreement(native_billing)`; `async def test_native_rejects_metadata_total_not_equal_to_records(native_billing)`; `async def test_native_component_revision_changes_receipt_without_extra_charge(native_billing)`; `async def test_native_rejects_foreign_pod_even_with_matching_query(native_billing)`; `async def test_native_rejects_duplicate_overlapping_hour(native_billing)`; `…`
- **tests/test_inference_provider.py** — SDK provider boundary tests for durable private Runpod inference.
  - `def test_dedicated_entry_point_loads_sdk_provider_contract()`; `def test_capabilities_are_deterministic_and_policy_scoped(tmp_path)`; `async def test_quote_is_read_only_and_acquire_returns_pending(tmp_path)`; `async def test_directed_component_total_quote_validates_on_acquire(tmp_path)`; `async def test_sdk_quote_and_acquire_expose_and_persist_all_in_cost_authority(tmp_path)`; `async def test_sdk_quote_enforces_all_in_ceiling_boundary(tmp_path, overhead, available)`; `async def test_sdk_acquire_revalidates_exact_non_compute_policy(tmp_path)`; `async def test_quote_expires_before_estimated_cold_start_window_closes(tmp_path)`; `…`
- **tests/test_ollama_contracts.py** — Cost selection and public-route contracts for Ollama leases.
  - `def test_bursty_session_selects_lower_effective_serverless_cost()`; `def test_serverless_quote_covers_every_possible_scale_to_zero_cycle()`; `def test_zero_idle_tail_has_no_finite_serverless_cold_start_bound()`; `def test_placement_plan_rejects_missing_or_cross_product_cold_start_bound()`; `def test_sustained_session_selects_pod_at_live_rates()`; `def test_forced_mode_and_cost_cap_fail_closed()`; `def test_auto_uses_all_in_ceiling_not_compute_only_affordability()`; `def test_all_in_ceiling_exact_boundary_and_tiny_overage(extra_overhead, affordable)`; `…`
- **tests/test_ollama_effect_custody.py** — Original Ollama effects must physically finish before cancellation escapes.
  - `class ThreadedProvider`; `def service(tmp_path, provider, clock)`; `async def test_cancellation_joins_original_thread_and_durable_receipt(tmp_path, effect)`; `async def test_release_during_post_cannot_discard_late_exact_acknowledgment(tmp_path)`; `async def test_late_exact_ack_does_not_replace_a_different_untrusted_locator(tmp_path)`; `async def test_original_ack_folds_after_same_id_discovery_reaches_ready(tmp_path)`; `async def test_different_valid_candidate_does_not_discard_original_post_custody(tmp_path)`; `async def test_original_post_ack_reopens_only_cleanup_after_candidate_is_terminal(tmp_path)`; `…`
- **tests/test_ollama_lifecycle_followon_acceptance.py** — Native acceptance for Ollama lifecycle billing and telemetry custody.
  - `async def test_runpod_provider_uses_exact_generic_serverless_billing_binding(billing_server)`; `async def test_receiptless_cleanup_only_accepts_receipt_for_exact_resource()`; `async def test_runpod_provider_uses_exact_generic_pod_billing_binding(native_billing)`; `async def test_pod_lifecycle_billing_requires_its_exact_canonical_hour_bounds(tmp_path, coverage)`; `async def test_serverless_lifecycle_billing_requires_complete_ordered_hours(tmp_path, billing_server, coverage, accepted)`; `async def test_terminated_lease_remains_due_for_exact_lifecycle_billing(tmp_path, billing_server)`; `async def test_repeated_exact_receipt_becomes_durable_latest_without_duplicate_fact(tmp_path, billing_server)`; `async def test_error_poll_preserves_successful_latest_freshness_across_restart(tmp_path, billing_server)`; `…`
- **tests/test_ollama_mixin.py** — Manager integration tests for the durable Ollama lease surface.
  - `async def test_mixin_delegates_only_to_injected_durable_service()`; `def test_mixin_has_no_in_memory_or_managed_provider_fallback()`; `def test_mixin_builds_one_configured_durable_service(tmp_path, monkeypatch)`
- **tests/test_ollama_provider.py** — Runpod v2 adapter tests with in-memory HTTP/control-plane doubles.
  - `def test_workload_credentials_must_not_reuse_control_or_cross_product_keys()`; `async def test_network_volume_cache_rejects_concurrent_serverless_writers(monkeypatch)`; `async def test_private_ollama_rejects_shared_serverless_gpu_pool(monkeypatch)`; `async def test_plan_uses_product_specific_live_catalog_prices(monkeypatch)`; `async def test_poolless_serverless_availability_fails_clearly_without_create(monkeypatch)`; `async def test_pooled_serverless_offer_survives_unrelated_poolless_offer(monkeypatch)`; `async def test_serverless_provision_is_load_balanced_and_configuration_owned(monkeypatch)`; `async def test_prepared_request_is_single_use_safe_and_mutation_detected(monkeypatch)`; `…`
- **tests/test_ollama_reconciler.py** — External one-shot reconciler entry point tests.
  - `async def test_reconcile_once_constructs_manager_and_runs_single_pass()`; `def test_cli_empty_arguments_preserve_one_shot_manager_contract()`
- **tests/test_ollama_repository.py** — Persistence, idempotency, and restart tests for Ollama leases.
  - `def test_insert_is_idempotent_and_survives_repository_restart(tmp_path)`; `async def test_cleanup_claim_folds_latest_same_lineage_owned_target(tmp_path)`; `async def test_cleanup_claim_rejects_every_changed_lineage(tmp_path)`; `def test_two_repositories_cannot_claim_one_reconciliation_visit(tmp_path)`; `def test_two_processes_cannot_claim_one_reconciliation_visit(tmp_path)`; `def test_runtime_refuses_incomplete_current_schema_before_any_ddl(tmp_path)`; `def test_attested_migration_repairs_supported_incomplete_current_schema(tmp_path)`; `async def test_attested_v6_trigger_repair_preserves_every_current_fact(tmp_path)`; `…`
- **tests/test_ollama_runtime.py** — Authenticated Ollama runtime and Runpod template contract tests.
  - `def test_runtime_image_must_use_reviewed_repository_and_digest()`; `def test_runtime_environment_is_owned_and_lease_bounded()`; `def test_runtime_environment_rejects_unapproved_model_and_overrides()`; `def test_runtime_environment_rejects_weak_or_expired_capability()`; `def test_runtime_environment_restricts_mode_specific_storage_roots()`; `def test_runpod_template_covers_pod_and_native_load_balancer()`; `def test_container_contract_is_pinned_nonroot_and_secret_free()`; `def test_tracked_configs_share_the_runtime_policy_surface()`; `…`
- **tests/test_ollama_service.py** — Lifecycle, crash recovery, readiness, cost, and teardown tests.
  - `async def test_acquire_pulls_missing_model_and_only_returns_ready_route(tmp_path)`; `async def test_duplicate_acquire_does_not_create_second_resource(tmp_path)`; `async def test_integer_and_float_money_spellings_share_first_acceptance(tmp_path)`; `async def test_unrepresentable_public_integer_cost_fails_before_admission(tmp_path, field_name)`; `async def test_requested_restart_recovers_exact_original_fingerprint_inputs(tmp_path)`; `async def test_touch_renews_only_after_reobserving_the_exact_ready_route(tmp_path)`; `async def test_touch_route_failure_does_not_renew_or_replace_capacity(tmp_path)`; `async def test_reconciler_recovers_crash_after_requested_insert(tmp_path)`; `…`
- **tests/test_one_shot_reconciler.py** — Shared argument boundary for installed one-shot reconcilers.
  - `def test_help_exits_before_coroutine_factory_or_event_loop(capsys)`; `def test_invalid_argument_is_generic_and_never_constructs_operation()`; `def test_empty_arguments_run_exactly_one_operation()`
- **tests/test_pod_capacity_architecture.py** — One-source-of-truth and public/private package-boundary tests.
  - `def test_training_names_are_aliases_not_a_second_writable_lifecycle()`; `def test_public_cloud_package_has_no_private_catalog_dependency()`
- **tests/test_pod_capacity_contracts.py** — Generic quote, identity, cost, environment, and persistence contracts.
  - `def test_request_binds_exact_parameter_digest_quote_cost_and_image()`; `def test_request_refuses_control_database_and_service_owned_environment(key)`; `def test_persisted_spec_contains_only_secret_identity_digest_and_expiry()`; `def test_changed_attempt_environment_changes_durable_identity()`; `def test_new_handoff_protocol_is_persisted_and_changes_capacity_identity()`; `def test_old_stored_spec_keeps_its_exact_pre_handoff_fingerprint_and_name(tmp_path)`; `def test_unknown_handoff_protocol_is_not_treated_as_legacy(invalid)`; `def test_billing_receipt_matches_private_terminal_contract()`; `…`
- **tests/test_pod_capacity_evidence.py** — Durable content-free realized Pod and worker evidence contract tests.
  - `def worker_envelope(clock)`; `async def successful_workload(tmp_path)`; `async def test_realized_lifecycle_worker_and_billing_evidence_survive_restart(tmp_path)`; `async def test_duplicate_observations_and_cas_race_preserve_first_timestamps(tmp_path)`; `async def test_worker_evidence_binding_replay_and_strict_allowlist(tmp_path)`; `async def test_public_projection_is_content_free_and_omits_capabilities(tmp_path)`; `async def test_v2_catalog_row_migrates_with_explicit_missing_evidence(tmp_path)`; `def test_lifecycle_timestamp_order_and_worker_startup_split_are_validated()`
- **tests/test_pod_capacity_provider.py** — Runpod v2 quote, exact recovery, termination, and billing adapter tests.
  - `async def test_pod_cleanup_only_accepts_one_clean_canonical_404(operation, failure_stage, error_kind)`; `async def test_pod_cleanup_rejects_a_foreign_action_or_readback_identity(operation, failure_stage)`; `async def test_native_no_body_pod_action_requires_exact_readback(operation, readback, expected, error_type)`; `async def test_native_terminal_action_dto_remains_direct_evidence(operation)`; `async def test_native_action_404_remains_idempotent_without_readback(operation)`; `async def test_native_no_body_start_does_not_invent_starting_evidence()`; `async def test_catalog_release_keeps_cleanup_and_billing_pending_while_pod_runs(tmp_path)`; `async def test_quote_binds_parameter_digest_exact_gpu_rate_and_timing()`; `…`
- **tests/test_pod_capacity_quote.py** — Tests for read-only Pod capacity quote composition.
  - `def test_quote_service_requires_an_explicit_profile_catalog()`; `async def test_quote_service_delegates_without_a_repository()`
- **tests/test_pod_capacity_reconciler.py** — Installed, locked, content-free Pod-capacity reconciler command tests.
  - `def test_distribution_installs_capacity_reconciler_entry_point()`; `async def test_reconcile_once_constructs_service_and_runs_one_pass(tmp_path)`; `def test_cli_runs_once_without_provisioning_and_emits_content_free_json(tmp_path)`; `def test_cli_missing_factory_fails_before_loading_or_provisioning()`; `def test_cli_auth_configuration_error_never_exposes_message_or_secret()`; `def test_cli_missing_catalog_dependency_fails_before_reconcile(tmp_path)`; `def test_cli_lock_contention_is_retryable_and_never_runs_service(tmp_path)`; `def test_cli_timeout_is_retryable_and_content_free(tmp_path)`; `…`
- **tests/test_pod_capacity_service.py** — End-to-end durable catalog Pod lifecycle tests without live resources.
  - `async def test_provision_and_poll_share_lifecycle_without_waiting_or_recreating(tmp_path)`; `async def test_readiness_poll_validates_scope_and_never_revives_stop(tmp_path)`; `async def test_cancellation_or_stop_during_health_cannot_promote_or_submit(tmp_path, stop)`; `async def test_readiness_timeout_terminates_without_replacement(tmp_path)`; `async def test_stop_while_loading_capability_prevents_submission(tmp_path)`; `async def test_readiness_cas_retry_does_not_overwrite_concurrent_stop(tmp_path, monkeypatch)`; `async def test_unhealthy_poll_is_one_shot_and_health_cannot_cross_deadline(tmp_path)`; `async def test_capability_io_cannot_extend_creation_authorization(tmp_path)`; `…`
- **tests/test_pod_capacity_transport.py** — Authenticated single-attempt workload transport tests.
  - `async def test_transport_uses_anonymous_health_and_bearer_for_job_routes()`; `async def test_transport_rejects_local_hash_mismatch_before_network()`; `async def test_transport_maps_remote_conflict_without_echoing_body()`; `async def test_transport_returns_result_unchanged_but_checks_attempt_binding()`; `async def test_transport_requires_tls_and_refuses_redirects()`; `async def test_transport_forwards_opaque_ack_and_requires_exact_authenticated_proof()`; `async def test_ack_cannot_accept_missing_changed_or_earlier_phase_proof(changed)`; `async def test_ack_http_error_never_echoes_private_receipt(status)`; `…`
- **tests/test_release_metadata.py** — Release metadata alignment tests.
  - `def test_project_version_has_matching_release_notes()`
- **tests/test_retired_endpoint_billing.py** — Historical billing through the real HTTP client; never a paid provider call.
  - `def billing_server()`; `async def test_retired_endpoint_billing_needs_no_job_or_live_endpoint(billing_server)`; `async def test_generic_serverless_billing_needs_no_catalog_authority(billing_server)`; `async def test_generic_serverless_billing_waits_for_closed_hour(billing_server)`; `async def test_generic_serverless_billing_invalid_batch_before_io(billing_server, batch_hours)`; `async def test_generic_serverless_billing_never_exposes_partial_history(billing_server)`; `async def test_generic_serverless_billing_repeated_cancel_joins_native_http(billing_server)`; `async def test_generic_serverless_billing_invalid_identity_before_io(billing_server, endpoint_id)`; `…`
- **tests/test_runpod_clients.py** — HTTP and request-shape contracts for both Runpod v2 services.
  - `def test_control_client_auth_user_agent_catalog_query_and_timeouts()`; `def test_inventory_walks_all_pages_including_an_empty_intermediate_page(method, key, path, factory)`; `def test_inventory_rejects_missing_or_ambiguous_pagination(metadata)`; `def test_inventory_rejects_cursor_cycles_instead_of_hanging()`; `def test_inventory_bounds_a_provider_that_keeps_issuing_unique_cursors()`; `def test_inventory_page_bound_is_strictly_positive_and_integral(limit)`; `def test_inventory_rejects_duplicate_resources_across_pages()`; `def test_inventory_does_not_return_partial_success_after_a_later_page_fails()`; `…`
- **tests/test_runpod_feature.py** — —
  - `async def test_legacy_feature_refuses_direct_admission_before_acknowledged_post()`; `class FakeRunPodManager`; `class DummyLLMService`; `async def runpod_feature(monkeypatch)`; `async def test_manage_gpu_start_and_stop(runpod_feature)`; `async def test_image_generation_tears_down_session(runpod_feature)`; `async def test_manage_gpu_unknown_action_returns_failed(runpod_feature)`; `async def test_start_unknown_profile_returns_failed(runpod_feature)`; `…`
- **tests/test_runpod_ga_authority.py** — GA creation and storage readback must enforce accepted workload constraints.
  - `def test_cuda_floor_is_sent_in_both_create_requests(floor, expected)`; `def test_invalid_cuda_constraints_fail_before_creation(floor)`; `def test_endpoint_create_explicitly_binds_both_storage_representations()`; `def test_partial_allowed_endpoint_mount_evidence_is_unavailable_not_foreign()`; `def test_historical_spec_round_trips_without_rewriting_its_authority()`; `def test_legacy_wire_shape_is_readable_but_the_transport_cannot_create_from_it()`; `def test_unknown_or_noncanonical_resource_version_cannot_be_deserialized(version)`; `def test_removing_a_ga_version_marker_cannot_downgrade_its_saved_authority()`; `…`
- **tests/test_runpod_model_contracts.py** — Contracts for RunPod profile-owned model defaults.
  - `def test_cuda_floor_requires_ascii_wire_numbers(floor)`; `def test_resource_authority_version_is_exported_with_public_contracts()`; `async def test_resume_stopped_pod_refuses_ownerless_admission_before_provider_action()`
- **tests/test_runpod_openapi_contract.py** — Parity tests between Kestrel's typed calls and the pinned v2 schema.
  - `def test_pin_checksum_and_validator_script()`; `def test_all_consumed_control_plane_operations_match_pin()`; `def test_consumed_request_and_response_shapes_match_pin()`; `def test_typed_create_and_update_payloads_validate_against_pin()`; `def test_runtime_base_url_matches_the_ga_schema_server()`; `def test_semantic_diff_reports_breaking_and_additive_changes()`; `def test_production_package_contains_no_v1_graphql_or_legacy_sdk_calls()`
- **tests/test_runpod_placement.py** — Live-catalog placement policy tests.
  - `def test_placement_prefers_availability_then_live_price_and_records_snapshot()`; `def test_serverless_uses_pool_price_for_ranking_and_authorization()`; `def test_serverless_rejects_shared_pool_before_filtering_policy(other_available)`; `def test_pod_can_select_an_exact_gpu_even_when_its_serverless_pool_is_shared()`; `def test_incompatible_gpu_is_not_misdiagnosed_as_a_shared_pool(product)`; `def test_pod_uses_cloud_rate_even_when_serverless_rate_differs()`; `def test_legacy_price_for_defaults_to_pods_and_missing_serverless_rate_is_explicit()`; `def test_serverless_rejects_missing_or_unsafe_pool_rate(pool_price)`; `…`
- **tests/test_runpod_provider.py** — Compatibility-provider migration tests for the v2 client boundary.
  - `def test_direct_provider_selects_from_live_catalog_and_records_placement()`; `def test_direct_provider_never_exposes_invalid_partial_create_evidence()`; `def test_cloud_policy_reaches_catalog_and_actual_create_payload(monkeypatch, environment_cloud, manager_cloud, profile_cloud, …)`; `def test_invalid_cloud_configuration_fails_before_provider(monkeypatch, invalid, source)`; `def test_uniform_cloud_binds_actual_profiles_not_mutable_environment(monkeypatch)`; `def test_single_cloud_pricing_rejects_mixed_explicit_profiles(monkeypatch)`; `def test_invalid_metadata_cloud_never_reads_catalog_or_creates(invalid)`; `def test_changed_default_cannot_redirect_saved_resource_cleanup(monkeypatch)`; `…`
- **tests/test_runpod_smoke.py** — Opt-in, read-only authenticated smoke checks for the v2 API.
  - `def test_live_v2_gpu_catalog_is_read_only_and_typed()`; `def test_live_v2_network_volume_inventory_is_read_only_and_typed()`
- **tests/test_serverless_capacity_contracts.py** — Contract tests for finite Serverless quotes and billing evidence.
  - `def test_endpoint_spec_round_trip_is_canonical_and_digest_protected()`; `def test_shared_aware_datetime_validator_rejects_non_datetimes_clearly(value)`; `def test_cleanup_intent_round_trip_is_original_authority_not_wire_or_zero_proof()`; `def test_cleanup_intent_rejects_invalid_typed_authority(field, value)`; `def test_cleanup_intent_requires_genuine_ordered_aware_observations(boundary)`; `def test_cleanup_intent_codec_rejects_drift_or_unversioned_fields(change)`; `def test_original_cleanup_receipt_codec_and_identity_remain_exact(reconciled)`; `def test_cleanup_v2_names_evidence_without_fabricating_delete_delivery(basis, reconciled)`; `…`
- **tests/test_serverless_capacity_provider.py** — REST v2 adapter tests for Serverless quote and billing reconciliation.
  - `async def test_pre_upgrade_authority_still_cleans_up_its_exact_created_endpoint()`; `async def test_pre_upgrade_crashed_activation_can_still_be_adopted_for_cleanup()`; `async def test_pre_upgrade_plan_cannot_authorize_new_spend()`; `async def test_legacy_activation_cannot_authorize_a_new_job_submission(recover_first)`; `async def test_serverless_cleanup_only_accepts_one_clean_canonical_404(error_kind)`; `async def test_serverless_cleanup_requires_exact_typed_endpoint_readback(readback)`; `async def test_created_endpoint_with_unaccepted_cuda_or_storage_keeps_cleanup_identity(changes)`; `async def test_create_ack_retains_exact_id_but_invalid_optional_evidence_never_activates()`; `…`
- **tests/test_signed_invocations.py** — Production contract tests for server-issued invocation receipts.
  - `def test_attested_request_is_exact_digest_bound_and_hides_input_from_repr()`; `def test_safe_identifiers_are_limited_to_255_characters()`; `def test_attested_request_rejects_every_wrong_string_field_type(field, wrong_value)`; `def test_canonical_digest_and_phase_evidence_reject_ambiguous_values()`; `def test_signed_response_round_trips_and_rejects_unknown_fields()`; `def test_receipt_trust_rejects_every_wrong_string_field_type(field, wrong_value)`; `def test_receipt_rejects_every_wrong_payload_string_field_type(field, wrong_value)`; `def test_receipt_rejects_every_wrong_envelope_string_field_type(section, field, wrong_value)`; `…`
- **tests/test_sqlite_store.py** — Real SQLite connection ownership, transaction, and repository regressions.
  - `def opened_connections(monkeypatch)`; `def test_connection_pragmas_commit_and_deterministic_close(tmp_path, opened_connections)`; `def test_failed_operation_rolls_back_and_closes(tmp_path, opened_connections)`; `def test_failed_commit_rolls_back_and_closes(tmp_path, opened_connections)`; `def test_failed_configuration_closes_real_connection(tmp_path, monkeypatch, opened_connections, failed_pragma)`; `def test_repository_operations_and_failures_close_connections(tmp_path, opened_connections, kind)`; `def test_repository_initialization_failure_closes_connection(tmp_path, opened_connections, repository_class)`
- **tests/test_training_contracts.py** — Public durable training ownership contracts.
  - `def test_request_requires_ordered_aware_deadlines()`; `def test_create_request_does_not_accept_a_provider_id()`; `def test_lifecycle_error_exposes_cleanup_authority_without_provider_detail()`; `def test_request_fingerprint_changes_with_deadline()`; `def test_child_request_fingerprint_is_bound_to_its_cleanup_family()`; `def test_request_defaults_root_for_pre_family_callers()`
- **tests/test_training_mixin.py** — Manager integration for durable training capacity and workload recovery.
  - `class FakeWorkloadClient`; `async def test_legacy_lora_inference_admission_fails_before_provider_post(tmp_path)`; `async def test_legacy_lora_inference_does_not_resume_name_matched_foreign_pod(tmp_path)`; `async def test_manager_session_tracks_job_result_and_confirmed_stop(tmp_path, monkeypatch)`; `async def test_submission_failure_stops_capacity_and_preserves_record(tmp_path, monkeypatch)`; `async def test_status_failure_keeps_job_and_cleanup_authority(tmp_path, monkeypatch)`; `async def test_cancel_stops_pod_and_stop_failure_is_retryable(tmp_path, monkeypatch)`
- **tests/test_training_provider.py** — Runpod REST v2 training capacity adapter contracts.
  - `async def test_observe_resolves_exact_http_proxy_route()`; `async def test_create_uses_provider_v2_boundary_and_validates_id()`; `async def test_successful_create_with_unusable_identity_remains_reconcilable(response)`; `async def test_find_by_name_rejects_duplicate_provider_resources()`; `async def test_stop_requires_v2_confirmation()`; `async def test_cancelled_start_waits_for_inflight_v2_mutation_to_resolve()`
- **tests/test_training_reconciler.py** — External one-shot training reconciliation entry point.
  - `async def test_reconcile_once_constructs_manager_and_runs_one_pass()`; `def test_cli_empty_arguments_preserve_one_shot_manager_contract()`
- **tests/test_training_repository.py** — Persistence, CAS, restart, and cross-process exclusion tests.
  - `def test_reservation_survives_repository_restart(tmp_path)`; `def test_revision_cas_rejects_stale_writer(tmp_path)`; `def test_second_active_token_cannot_claim_same_pod(tmp_path)`; `def test_database_path_is_explicit_absolute_and_expands_environment(tmp_path, monkeypatch)`; `def test_pre_family_database_migrates_with_legacy_attempt_self_rooted(tmp_path)`; `def test_family_release_gate_blocks_a_late_fallback_reservation(tmp_path)`; `def test_versioned_table_migration_preserves_every_lifecycle_shape(tmp_path)`
- **tests/test_training_service.py** — Billing-safe training acquisition, cancellation, and reconciliation tests.
  - `async def test_resumed_pod_missing_route_is_stopped(tmp_path)`; `async def test_readiness_exception_stop_failure_retains_retryable_pod_after_restart(tmp_path)`; `async def test_preexisting_running_pod_route_failure_is_never_stopped(tmp_path)`; `async def test_ambiguous_create_reconciles_exact_name_and_stops_pod(tmp_path)`; `async def test_restart_recovers_crash_between_create_and_id_persistence(tmp_path)`; `async def test_reconciler_leaves_live_inflight_create_owned_by_acquirer(tmp_path)`; `async def test_restart_before_persistent_discovery_never_stops_preexisting_pod(tmp_path)`; `async def test_restart_reconciles_submitted_job_at_hard_deadline(tmp_path)`; `…`
- **tests/training_test_support.py** — Deterministic doubles for durable training Pod lifecycle tests.
  - `class MutableClock`; `def training_profile(profile_id)`; `def training_request(clock)`; `class FakeTrainingProvider`; `def training_service(tmp_path, clock, provider)`

## `vendor/`

- **vendor/runpod-v2-openapi.lock.json** — (configuration)
- **vendor/runpod-v2-openapi.yaml** — (configuration)
