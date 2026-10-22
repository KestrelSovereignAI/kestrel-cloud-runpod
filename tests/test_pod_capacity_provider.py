"""Runpod v2 quote, exact recovery, termination, and billing adapter tests."""

import hashlib
from datetime import timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest
from pod_capacity_test_support import (
    PARAMETERS_SHA,
    FakeCapabilityStore,
    FakeCapacityProvider,
    FakeWorkloadTransport,
    MutableClock,
    complete_pod_payload,
    native_pod_action_harness,
    profile,
    request,
    service,
)

from kestrel_cloud_runpod.billing import billing_hour_bounds
from kestrel_cloud_runpod.models import (
    Availability,
    BillingPage,
    GPUOffer,
    RunPodAPIError,
    RunPodManagerError,
)
from kestrel_cloud_runpod.pod_capacity_contracts import (
    PodCapacityBillingState,
    PodCapacityCleanupState,
    PodCapacityLifecycleError,
    PodCapacityQuoteRequest,
    PodCapacitySpec,
    PodCapacityState,
    attempt_environment_sha256,
)
from kestrel_cloud_runpod.pod_capacity_provider import (
    PodCapacityCreatedMismatchError,
    RunpodPodCapacityProvider,
)


def _offer() -> GPUOffer:
    return GPUOffer(
        id="NVIDIA RTX PRO 4500",
        name="RTX PRO 4500",
        pool=None,
        manufacturer="NVIDIA",
        memory_gb=32,
        secure=True,
        community=False,
        secure_price_per_hr=0.4,
        community_price_per_hr=0.0,
        secure_max_count=8,
        community_max_count=0,
        availability=Availability.HIGH,
        data_centers=({"id": "US-TX-3", "availability": "HIGH"},),
        availability_min_cuda_version="12.8",
    )


def _spec(clock: MutableClock) -> PodCapacitySpec:
    item = request(clock)
    return PodCapacitySpec(
        request=item,
        capability_secret_id="secret:catalog-capability-0001",
        capability_token_sha256="a" * 64,
        capability_expires_at=item.bearer_expires_at,
        attempt_environment_sha256=attempt_environment_sha256(item.attempt_environment),
    )


def _billing_history(created, terminated, amount, *, partial=False) -> BillingPage:
    first, until = billing_hour_bounds(created, terminated)
    amounts = {
        "totalAmount": amount or 0,
        "gpuAmount": amount or 0,
        "cpuAmount": 0,
        "diskAmount": 0,
    }
    records = (
        ()
        if amount is None
        else (
            {
                "podId": "pod-catalog-1",
                "startTime": first.isoformat(),
                "endTime": (
                    until - timedelta(seconds=1) if partial else until
                ).isoformat(),
                **amounts,
            },
        )
    )
    return BillingPage(
        records=records,
        metadata={
            "query": {
                "podId": "pod-catalog-1",
                "bucketSize": "hour",
                "startTime": first.isoformat(),
                "endTime": until.isoformat(),
            },
            "recordCount": len(records),
            "uniquePodCount": 1 if records else 0,
            "totals": amounts,
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["stop", "terminate"])
@pytest.mark.parametrize("failure_stage", ["action", "readback"])
@pytest.mark.parametrize(
    "error_kind",
    ["clean-404", "unknown-404", "structural-404", "authorization"],
)
async def test_pod_cleanup_only_accepts_one_clean_canonical_404(
    operation,
    failure_stage,
    error_kind,
) -> None:
    """Stop/terminate must not turn an untrusted 404 into exact absence."""

    class UnknownNotFoundError(RunPodAPIError):
        pass

    error_type = UnknownNotFoundError if error_kind == "unknown-404" else RunPodAPIError
    status_code = 401 if error_kind == "authorization" else 404
    error = error_type(
        title="Not found" if status_code == 404 else "Unauthorized",
        detail="bounded Pod cleanup response",
        status_code=status_code,
    )
    if error_kind == "structural-404":
        error.__cause__ = ValueError("invalid Pod response")

    class DirectProvider:
        def __init__(self):
            self.action_calls = []
            self.read_calls = []

        def stop_pod(self, pod_id):
            self.action_calls.append(("stop", pod_id))
            if operation == "stop" and failure_stage == "action":
                raise error
            return {"id": pod_id, "status": "RUNNING"}

        def terminate_pod(self, pod_id):
            self.action_calls.append(("terminate", pod_id))
            if operation == "terminate" and failure_stage == "action":
                raise error
            return {"id": pod_id, "status": "RUNNING"}

        def get_status(self, pod_id):
            self.read_calls.append(pod_id)
            if failure_stage == "readback":
                raise error
            return {"id": pod_id, "status": "RUNNING"}

    direct = DirectProvider()
    provider = RunpodPodCapacityProvider(direct)
    method = getattr(provider, operation)
    if error_kind == "clean-404":
        assert await method("pod-exact") is True
    else:
        with pytest.raises(error_type) as caught:
            await method("pod-exact")
        assert caught.value is error
    assert direct.action_calls == [(operation, "pod-exact")]
    assert direct.read_calls == ([] if failure_stage == "action" else ["pod-exact"])


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["stop", "terminate"])
@pytest.mark.parametrize("failure_stage", ["action", "readback"])
async def test_pod_cleanup_rejects_a_foreign_action_or_readback_identity(
    operation,
    failure_stage,
) -> None:
    class DirectProvider:
        def stop_pod(self, pod_id):
            return {
                "id": "foreign-pod" if failure_stage == "action" else pod_id,
                "status": "RUNNING",
            }

        def terminate_pod(self, pod_id):
            return {
                "id": "foreign-pod" if failure_stage == "action" else pod_id,
                "status": "RUNNING",
            }

        def get_status(self, pod_id):
            return {
                "id": "foreign-pod" if failure_stage == "readback" else pod_id,
                "status": "RUNNING",
            }

    provider = RunpodPodCapacityProvider(DirectProvider())
    with pytest.raises(RunPodManagerError, match="requested Pod ID"):
        await getattr(provider, operation)("pod-exact")


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["stop", "terminate"])
@pytest.mark.parametrize(
    ("readback", "expected", "error_type"),
    [
        ("running", False, None),
        ("terminal", True, None),
        ("absent", True, None),
        ("foreign", None, RunPodManagerError),
        ("authorization", None, RunPodAPIError),
        ("structural", None, RunPodAPIError),
        ("transport", None, RunPodAPIError),
    ],
)
async def test_native_no_body_pod_action_requires_exact_readback(
    operation,
    readback,
    expected,
    error_type,
) -> None:
    """A 204 acknowledges the action but supplies no terminal-state evidence."""

    with native_pod_action_harness() as (state, _direct, provider):
        state["readback"] = readback
        if error_type is None:
            assert await getattr(provider, operation)(state["pod_id"]) is expected
        else:
            with pytest.raises(error_type):
                await getattr(provider, operation)(state["pod_id"])

        assert state["calls"][0] == (
            "POST",
            f"/v2/pods/{state['pod_id']}/action",
            {"action": operation},
        )
        assert state["calls"][1:] == [("GET", f"/v2/pods/{state['pod_id']}", None)]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["stop", "terminate"])
async def test_native_terminal_action_dto_remains_direct_evidence(operation) -> None:
    with native_pod_action_harness() as (state, _direct, provider):
        state["action_status"] = 200
        state["action_payload"] = complete_pod_payload(
            pod_id=state["pod_id"], status="TERMINATED"
        )

        assert await getattr(provider, operation)(state["pod_id"]) is True
        assert state["calls"] == [
            (
                "POST",
                f"/v2/pods/{state['pod_id']}/action",
                {"action": operation},
            )
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["stop", "terminate"])
async def test_native_action_404_remains_idempotent_without_readback(operation) -> None:
    with native_pod_action_harness() as (state, _direct, provider):
        state["action_status"] = 404
        state["action_payload"] = {"title": "not found"}

        assert await getattr(provider, operation)(state["pod_id"]) is True
        assert state["calls"] == [
            (
                "POST",
                f"/v2/pods/{state['pod_id']}/action",
                {"action": operation},
            )
        ]


@pytest.mark.asyncio
async def test_native_no_body_start_does_not_invent_starting_evidence() -> None:
    with native_pod_action_harness() as (state, direct, provider):
        result = direct.resume_pod(state["pod_id"])
        assert result == {"id": state["pod_id"]}

        await provider.start(state["pod_id"], gpu_count=1)
        assert state["calls"] == [
            (
                "POST",
                f"/v2/pods/{state['pod_id']}/action",
                {"action": "start"},
            ),
            (
                "POST",
                f"/v2/pods/{state['pod_id']}/action",
                {"action": "start"},
            ),
        ]


@pytest.mark.asyncio
async def test_catalog_release_keeps_cleanup_and_billing_pending_while_pod_runs(
    tmp_path,
) -> None:
    """A durable lease cannot record termination from a no-body action ACK."""

    clock = MutableClock()
    fake = FakeCapacityProvider(clock)
    store = FakeCapabilityStore(clock)
    runtime = service(tmp_path, clock, fake, store, FakeWorkloadTransport())
    catalog_request = request(clock)
    await runtime.acquire_catalog(catalog_request)

    with native_pod_action_harness() as (state, _direct, provider):
        state["readback"] = "running"
        fake.terminate = provider.terminate
        with pytest.raises(PodCapacityLifecycleError):
            await runtime.release(catalog_request.capacity_id, reason="test cleanup")

        durable = runtime.repository.get(catalog_request.capacity_id)
        assert durable is not None
        assert durable.state is PodCapacityState.RELEASING
        assert durable.cleanup_state is PodCapacityCleanupState.PENDING
        assert durable.terminated_at is None
        assert durable.evidence.lifecycle.stop_confirmed_at is None
        assert durable.billing_state is PodCapacityBillingState.PENDING
        assert durable.billing_receipt is None
        assert durable.settlement_ready is False
        assert fake.billing_calls == fake.billing_rates == []
        assert state["calls"] == [
            (
                "POST",
                f"/v2/pods/{state['pod_id']}/action",
                {"action": "terminate"},
            ),
            ("GET", f"/v2/pods/{state['pod_id']}", None),
        ]


@pytest.mark.asyncio
async def test_quote_binds_parameter_digest_exact_gpu_rate_and_timing() -> None:
    clock = MutableClock()
    client = SimpleNamespace(list_gpus=lambda **_: [_offer()])
    direct = SimpleNamespace(client=client)
    provider = RunpodPodCapacityProvider(direct, clock=clock)

    result = await provider.quote(
        PodCapacityQuoteRequest(
            constraints=request(clock).quote.constraints,
            workload_kind="catalog-lora",
            parameters_sha256=PARAMETERS_SHA,
            estimated_startup_seconds=60,
            estimated_execution_seconds=300,
            maximum_runtime_seconds=600,
        )
    )

    assert result.gpu_type_id == "NVIDIA RTX PRO 4500"
    assert result.gpu_display_name == "RTX PRO 4500"
    assert result.hourly_cost_usd == Decimal("0.4")
    assert result.estimated_cost_usd == Decimal("0.040000")
    assert result.cost_ceiling_usd == Decimal("0.066667")
    assert result.parameters_sha256 == PARAMETERS_SHA
    assert result.provider_quote_id.startswith("runpod-pod:")


@pytest.mark.asyncio
async def test_create_injects_exact_image_gpu_and_attempt_environment() -> None:
    clock = MutableClock()
    spec = _spec(clock)
    captured = {}

    def start_pod(effective_profile, metadata):
        captured.update(profile=effective_profile, metadata=metadata)
        return {
            "id": "pod-catalog-1",
            "name": spec.request.resource_name,
            "image": spec.request.image_reference,
            "gpu": {"id": spec.request.quote.gpu_type_id, "count": 1},
            "cloud": "SECURE",
            "dataCenterId": "US-TX-3",
            "cost": 0.4,
            "_kestrel_placement": spec.request.quote.placement,
        }

    direct = SimpleNamespace(start_pod=start_pod)
    provider = RunpodPodCapacityProvider(direct, clock=clock)
    environment = {"CATALOG_WORKER_MODE": "pod"}

    created = await provider.create(
        profile=profile(),
        resource_name=spec.request.resource_name,
        companion_id=spec.request.owner_id,
        environment=environment,
        capacity_spec=spec,
    )

    assert created.provider_pod_id == "pod-catalog-1"
    assert captured["profile"].image_name == spec.request.image_reference
    assert captured["profile"].allowed_gpu_ids == (spec.request.quote.gpu_type_id,)
    assert captured["metadata"]["env_overrides"] == environment


@pytest.mark.asyncio
async def test_created_metadata_mismatch_preserves_known_pod_id() -> None:
    clock = MutableClock()
    spec = _spec(clock)

    def start_pod(effective_profile, metadata):
        return {
            "id": "pod-catalog-mismatch",
            "name": spec.request.resource_name,
            "image": "wrong@sha256:" + "f" * 64,
            "gpu": {"id": spec.request.quote.gpu_type_id, "count": 1},
            "cloud": "SECURE",
            "_kestrel_placement": spec.request.quote.placement,
        }

    provider = RunpodPodCapacityProvider(
        SimpleNamespace(start_pod=start_pod), clock=clock
    )

    with pytest.raises(PodCapacityCreatedMismatchError) as raised:
        await provider.create(
            profile=profile(),
            resource_name=spec.request.resource_name,
            companion_id=spec.request.owner_id,
            environment={"CATALOG_WORKER_MODE": "pod"},
            capacity_spec=spec,
        )

    assert raised.value.provider_pod_id == "pod-catalog-mismatch"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("override", "expected_message"),
    [
        ({"dataCenterId": "EU-RO-1"}, "accepted capacity"),
        ({"cost": 0.41}, "accepted capacity"),
        ({"cloud": "COMMUNITY"}, "accepted capacity"),
    ],
)
async def test_create_rejects_realized_placement_outside_quote(
    override, expected_message
) -> None:
    clock = MutableClock()
    spec = _spec(clock)

    def start_pod(effective_profile, metadata):
        payload = {
            "id": "pod-catalog-mismatch",
            "name": spec.request.resource_name,
            "image": spec.request.image_reference,
            "gpu": {"id": spec.request.quote.gpu_type_id, "count": 1},
            "cloud": "SECURE",
            "dataCenterId": "US-TX-3",
            "cost": 0.4,
            "_kestrel_placement": spec.request.quote.placement,
        }
        payload.update(override)
        return payload

    provider = RunpodPodCapacityProvider(
        SimpleNamespace(start_pod=start_pod), clock=clock
    )
    with pytest.raises(PodCapacityCreatedMismatchError, match=expected_message):
        await provider.create(
            profile=profile(),
            resource_name=spec.request.resource_name,
            companion_id=spec.request.owner_id,
            environment={"CATALOG_WORKER_MODE": "pod"},
            capacity_spec=spec,
        )


@pytest.mark.asyncio
async def test_exact_recovery_rejects_mismatch_and_multiple_matches() -> None:
    clock = MutableClock()
    spec = _spec(clock)
    valid = {
        "id": "pod-catalog-1",
        "name": spec.request.resource_name,
        "image": spec.request.image_reference,
        "gpu": {"id": spec.request.quote.gpu_type_id, "count": 1},
        "cloud": "SECURE",
        "dataCenterId": "US-TX-3",
        "cost": 0.4,
    }
    direct = SimpleNamespace(list_pods=lambda: [valid])
    provider = RunpodPodCapacityProvider(direct, clock=clock)
    recovered = await provider.find_exact(spec.request.resource_name, spec)
    assert recovered is not None
    assert recovered.provider_pod_id == "pod-catalog-1"
    assert recovered.realized_placement is not None
    assert recovered.realized_placement.data_center_id == "US-TX-3"

    direct.list_pods = lambda: [{**valid, "image": "wrong@sha256:" + "f" * 64}]
    with pytest.raises(RunPodManagerError, match="mismatched"):
        await provider.find_exact(spec.request.resource_name, spec)

    direct.list_pods = lambda: [valid, {**valid, "id": "pod-catalog-2"}]
    with pytest.raises(RunPodManagerError, match="Multiple"):
        await provider.find_exact(spec.request.resource_name, spec)


@pytest.mark.asyncio
async def test_final_billing_waits_for_records_then_returns_content_free_receipt() -> (
    None
):
    clock = MutableClock()
    spec = _spec(clock)
    terminated = clock()
    created = terminated - timedelta(minutes=10)
    pages = [
        _billing_history(created, terminated, None),
        _billing_history(created, terminated, 0.061),
    ]
    client = SimpleNamespace(pod_billing=lambda **_: pages.pop(0))
    provider = RunpodPodCapacityProvider(SimpleNamespace(client=client), clock=clock)

    assert (
        await provider.final_billing(
            "pod-catalog-1",
            capacity_spec=spec,
            created_at=created,
            terminated_at=terminated,
            realized_hourly_rate_usd=Decimal("0.4"),
        )
        is None
    )
    receipt = await provider.final_billing(
        "pod-catalog-1",
        capacity_spec=spec,
        created_at=created,
        terminated_at=terminated,
        realized_hourly_rate_usd=Decimal("0.4"),
    )
    assert receipt is not None
    assert receipt.actual_cost_usd == Decimal("0.061")
    # Provider buckets cover a full hour, not 600 seconds of GPU activity.
    assert receipt.billed_seconds == 3600
    assert receipt.provider_billing_id.startswith("runpod-billing:")
    assert hashlib.sha256(b"private").hexdigest() not in str(receipt)


@pytest.mark.asyncio
async def test_final_billing_waits_until_records_cover_termination() -> None:
    clock = MutableClock()
    spec = _spec(clock)
    terminated = clock()
    created = terminated - timedelta(minutes=10)
    page = _billing_history(created, terminated, 0.06, partial=True)
    provider = RunpodPodCapacityProvider(
        SimpleNamespace(client=SimpleNamespace(pod_billing=lambda **_: page)),
        clock=clock,
    )

    assert (
        await provider.final_billing(
            "pod-catalog-1",
            capacity_spec=spec,
            created_at=created,
            terminated_at=terminated,
            realized_hourly_rate_usd=Decimal("0.4"),
        )
        is None
    )


@pytest.mark.parametrize("realized", [Decimal("0.38"), Decimal("1.6"), Decimal("0")])
@pytest.mark.asyncio
async def test_receipt_records_the_realized_rate_not_the_accepted_one(
    realized,
) -> None:
    """Two checks require the receipt rate to equal the REALIZED rate.

    The placement is only required to come in at or under the accepted rate,
    so recording the accepted rate made settlement unreachable on an ordinary
    price drop and on the terminal cost of 0.0 the v2 spec documents. Zero is
    included deliberately: it is the case the recovery path adopts.
    """
    clock = MutableClock()
    spec = _spec(clock)
    terminated = clock()
    created = terminated - timedelta(seconds=600)
    clock.value += timedelta(seconds=1)
    page = _billing_history(created, terminated, 0.061)
    client = SimpleNamespace(pod_billing=lambda **_: page)
    provider = RunpodPodCapacityProvider(SimpleNamespace(client=client), clock=clock)

    receipt = await provider.final_billing(
        "pod-catalog-1",
        capacity_spec=spec,
        created_at=created,
        terminated_at=terminated,
        realized_hourly_rate_usd=realized,
    )

    assert receipt is not None
    assert receipt.hourly_price_usd == realized
    # The accepted rate is the quote's per-GPU price x count; it must NOT be
    # what lands here unless it happens to equal the realized rate.
    accepted = spec.request.quote.hourly_cost_usd * Decimal(
        spec.request.quote.constraints.gpu_count
    )
    if realized != accepted:
        assert receipt.hourly_price_usd != accepted
