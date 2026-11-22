"""Exercise the controller with real SQLite and HTTP on loopback only."""

import json
import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
from kestrel_sdk.llm import (
    InferenceLeaseOwnershipError, InferenceLeaseProviderUnavailableError,
    InferenceLeaseRequest, InferenceLeaseState,
)
from ollama_test_support import OwnedLoopbackHTTPServer, runtime_telemetry_payload
from pydantic import SecretStr

from kestrel_cloud_runpod.clients import RunpodControlPlaneClient
from kestrel_cloud_runpod.models import EndpointCreateRequest, FlashBoot, RunPodAPIError
from kestrel_cloud_runpod.ollama_contracts import OLLAMA_REQUIRE_CACHED_MODEL_ENV, OllamaLeaseConflictError
from kestrel_cloud_runpod.ollama_repository import SQLiteOllamaLeaseRepository
from kestrel_cloud_runpod.registered_ollama import RegisteredOllamaEndpoint, RegisteredOllamaLifecycle
from kestrel_cloud_runpod.registered_ollama_journal import RegisteredActivationState, RegisteredOllamaJournal
from kestrel_cloud_runpod.ollama_runtime import OLLAMA_RUNTIME_IMAGE_REPOSITORY
from kestrel_cloud_runpod.ollama_runtime import cache_only_ollama_runtime_version
from kestrel_cloud_runpod.registered_inference_provider import RegisteredInferenceLeaseProvider
from kestrel_cloud_runpod.registered_ollama_admission import RegisteredQuotePolicy


@pytest.fixture
def home(tmp_path):
    start = datetime(2026, 11, 22, tzinfo=UTC)
    config = EndpointCreateRequest(
        name="kite-example-parked",
        image=OLLAMA_RUNTIME_IMAGE_REPOSITORY + "@sha256:c316c4e8a71cc801844b0651b4b174d79c75a7ca6725eb75b00138862c8d5cd3",
        endpoint_type="LOAD_BALANCER",
        gpu_pools=("ADA_24",),
        scaling={"type": "REQUEST_COUNT", "requestCount": 1},
        workers_max=0, idle_timeout_seconds=5,
        ports=("11434/http",), env={"PORT": "11434"},
        registry_id="registry", data_center_ids=("US-NC-1",),
        network_volume_ids=("volume",), min_cuda_version="12.8",
        flashboot=FlashBoot.OFF,
    )
    endpoint = RegisteredOllamaEndpoint("owned-exact", start, config)
    state = {**config.to_payload(), "id": endpoint.endpoint_id, "createdAt": start.isoformat()}
    state["workers"]["initTimeout"] = None
    applied = []
    drop = set()
    busy = []
    data_reads = []
    model = "gpt-oss:20b"
    manifest = "a" * 64
    base = "/v2/serverless/owned-exact"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            return

        def send_json(self, body, status=200):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == base:
                if self.headers.get("Authorization") == "Bearer fixture-invoke":
                    self.send_json({"title": "Forbidden"}, 403)
                else:
                    self.send_json(state)
            elif self.path == "/v2/account":
                denied = self.headers.get("Authorization") == "Bearer fixture-invoke"
                self.send_json({}, 403 if denied else 200)
            elif self.path == base + "/workers":
                count = len(busy)
                self.send_json({
                    "workers": busy, "endpointVersion": 1, "summary": {
                        "idle": count, "total": count, "running": 0,
                        "initializing": 0, "throttled": 0, "unhealthy": 0,
                    },
                })
            elif self.path.startswith("/v2/catalog/gpus?"):
                self.send_json({"gpus": [{
                    "id": "NVIDIA GeForce RTX 4090", "name": "RTX 4090",
                    "pool": "ADA_24", "manufacturer": "NVIDIA", "memory": 24,
                    "secure": True, "community": False, "availability": "LOW",
                    "price": {"secure": 0.7, "community": 0.5, "serverless": 1.34},
                    "maxCount": {"secure": 1, "community": 1},
                    "dataCenters": [{"id": "US-NC-1", "availability": "LOW"}],
                }]})
            elif self.path.startswith("/v2/network-volumes"):
                self.send_json({
                    "networkVolumes": [{
                        "id": "volume", "name": "example-cache", "size": 25,
                        "dataCenter": "US-NC-1", "type": "STANDARD",
                    }],
                    "pagination": {"hasNextPage": False, "nextCursor": None},
                })
            elif self.path == "/v2/owned-exact/health":
                ok = self.headers.get("Authorization") in ("Bearer fixture-invoke", "Bearer another-broad-key")
                self.send_json({}, 200 if ok else 401)
            elif self.path in ("/kestrel/telemetry", "/api/tags"):
                data_reads.append(self.path)
                if self.headers.get("Authorization") != "Bearer fixture-invoke":
                    self.send_json({"error": "unauthorized"}, 401)
                    return
                if state["workers"]["max"] != 1:
                    self.send_json({"error": "parked"}, 409)
                    return
                if self.path == "/kestrel/telemetry":
                    timestamp = state["env"]["KESTREL_OLLAMA_PROVISION_REQUESTED_AT"]
                    busy[:] = [{
                        "id": "worker-" + str(len(applied)), "status": "IDLE", "version": 1,
                        "isStale": False, "gpuCount": 1, "image": endpoint.baseline.image,
                        "startedAt": timestamp, "gpuTypeId": "NVIDIA GeForce RTX 4090",
                        "dataCenterId": "US-NC-1",
                    }]
                    self.send_json({
                        **runtime_telemetry_payload(state["env"]["KESTREL_OLLAMA_TELEMETRY_BINDING_SHA256"], timestamp),
                        "runtime_version": cache_only_ollama_runtime_version(endpoint.baseline.image),
                    })
                else:
                    if state.get("_replace_during_observation"):
                        busy[0]["id"] = "replacement-worker"
                        ticks[0] += timedelta(seconds=91)
                    self.send_json({"models": [{"name": model, "digest": manifest}]})
            else:
                self.send_json({"title": "Unexpected route"}, 404)

        def do_PATCH(self):
            if self.path != base:
                self.send_json({"title": "Unexpected route"}, 404)
                return
            patch = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            applied.append(patch)
            state["workers"].update(patch["workers"])
            state["env"] = patch["env"]
            if len(applied) in drop:
                self.close_connection = True
                return
            self.send_json(state)

    db = tmp_path / "existing-lease-family.sqlite"
    SQLiteOllamaLeaseRepository(db)
    ticks = [start]

    def controller(client):
        clock = lambda: ticks[0]
        journal = RegisteredOllamaJournal(db, clock=clock)
        return RegisteredOllamaLifecycle(client=client, endpoint=endpoint, journal=journal, clock=clock)

    def activate(client, name="first"):
        now = ticks[0]
        deadline = now + timedelta(seconds=200)
        return controller(client).activate(
            lease_id=name, owner_id="original-agent", admission_sha256="1" * 64,
            active_env={
                "KESTREL_OLLAMA_BEARER_TOKEN": "{{ RUNPOD_SECRET_fixture_invoke }}",
                "KESTREL_OLLAMA_BEARER_TOKEN_EXPIRES_AT": deadline.isoformat(),
                OLLAMA_REQUIRE_CACHED_MODEL_ENV: "true",
            },
            first_patch_before=now + timedelta(seconds=30),
            ready_before=now + timedelta(seconds=60),
            hard_deadline=deadline, idle_seconds=10,
        )

    with OwnedLoopbackHTTPServer(
        server_factory=lambda: ThreadingHTTPServer(("127.0.0.1", 0), Handler),
        thread_factory=lambda server: Thread(target=server.serve_forever, daemon=True),
        client_factory=lambda server: RunpodControlPlaneClient(
            api_key="fixture-management", base_url=f"http://127.0.0.1:{server.server_port}/v2",
            max_safe_retries=0,
        ),
    ) as client:
        def provider():
            lifecycle = controller(client)
            implementation = RegisteredInferenceLeaseProvider()
            implementation._lifecycle = lifecycle
            implementation._policy = RegisteredQuotePolicy(
                endpoint.digest, SQLiteOllamaLeaseRepository(db).journal_identity(),
                "US-NC-1", model, "sha256:" + manifest, 20, 30, 5, 120, 150, Decimal("0.01"),
            )
            implementation._base_url = client.transport.base_url.removesuffix("/v2")
            implementation._gpu_id = "NVIDIA GeForce RTX 4090"
            implementation._secret_ref = "{{ RUNPOD_SECRET_fixture_invoke }}"
            implementation._model_digest = "sha256:" + manifest
            implementation._model_pin = model + "@" + implementation._model_digest
            implementation._model_path = "/runpod-volume/ollama"
            implementation._key = SecretStr("fixture-invoke")
            implementation._health_base_url = client.transport.base_url
            implementation._cache_size_gb = 25
            return implementation

        yield client, controller, activate, ticks, applied, drop, busy, state, provider, data_reads


def test_restart_parks_the_same_endpoint_and_proves_worker_absence_before_second_lease(home):
    client, controller, activate, ticks, applied, _, busy, state, *_ = home
    row, fresh = activate(client)
    assert fresh and row.state is RegisteredActivationState.ACTIVE
    assert activate(client) == (row, False)
    assert len(applied) == 1
    ticks[0] += timedelta(seconds=61)
    restarted = controller(client)
    assert restarted.journal.due() == (row,)
    busy.append({"id": "first-worker"})
    assert restarted.park(row).state is RegisteredActivationState.PARKING
    assert len(applied) == 2
    assert state["workers"]["max"] == 0
    busy.clear()
    assert controller(client).park(row).state is RegisteredActivationState.PARKED
    second, fresh = activate(client, "second")
    assert fresh
    assert second.intent.endpoint_id == row.intent.endpoint_id
    assert second.intent.cache_id == row.intent.cache_id
    assert second.intent.active_environment_sha256 != row.intent.active_environment_sha256
    assert [p["workers"]["max"] for p in applied] == [1, 0, 1]


def test_applied_but_lost_responses_are_observed_after_restart_not_replayed(home):
    client, controller, activate, _ticks, applied, drop, _busy, _state, *_ = home
    drop.update({1, 2})
    with pytest.raises(RunPodAPIError):
        activate(client)
    row = controller(client).journal.get("first")
    assert row.state is RegisteredActivationState.ARMING
    assert activate(client) == (row, False)
    with pytest.raises(RunPodAPIError):
        controller(client).park(row)
    parked = controller(client).park(row)
    assert parked.state is RegisteredActivationState.PARKED
    assert [p["workers"]["max"] for p in applied] == [1, 0]


def test_foreign_env_never_gets_park_patch_or_released_from_journal(home):
    client, controller, activate, ticks, applied, _drop, _busy, state, *_ = home
    row, _fresh = activate(client)
    state["env"] = {"PORT": "foreign"}
    ticks[0] += timedelta(seconds=61)
    assert controller(client).park(row) == row
    assert len(applied) == 1
    assert controller(client).journal.due() == (row,)


def test_restart_before_parking_patch_still_cuts_workers_to_zero_without_reopening(home):
    client, controller, activate, ticks, applied, _, busy, state, *_ = home
    row, _fresh = activate(client)
    controller(client).journal.claim_park(row.intent)
    ticks[0] += timedelta(seconds=201)
    busy.append({"id": "previous-worker"})
    assert controller(client).reap_due()[0].state is RegisteredActivationState.FENCED
    assert state["workers"]["max"] == 0
    busy.clear()
    assert controller(client).reap_due()[0].state is RegisteredActivationState.FENCED_ZERO
    assert controller(client).reap_due() == ()
    assert [p["workers"]["max"] for p in applied] == [1, 0]
    with pytest.raises(OllamaLeaseConflictError, match="already reserved"):
        controller(client).journal.reserve(
            type(row.intent)(
                "next", row.intent.owner_id, row.intent.endpoint_id, row.intent.cache_id,
                row.intent.admission_sha256, row.intent.registration_sha256,
                row.intent.parked_environment_sha256, "a" * 64, "b" * 64,
                ticks[0] + timedelta(seconds=10), ticks[0] + timedelta(seconds=20),
                ticks[0] + timedelta(seconds=40), 10,
            )
        )


def sdk_request(now):
    return InferenceLeaseRequest(
        request_id="first", owner_id="kite-owner", model="gpt-oss:20b", runtime="ollama",
        allowed_regions=("us-nc-1",), capabilities=("chat", "streaming"),
        max_hourly_cost_usd=Decimal("1.50"), max_total_cost_usd=Decimal("0.35"),
        expected_session_seconds=90, idle_ttl_seconds=10, ready_deadline_seconds=90,
        requested_at=now,
    )


READINESS_READS = ["/kestrel/telemetry", "/kestrel/telemetry", "/api/tags"]


@pytest.mark.asyncio
async def test_sdk_lease_two_cycles_use_same_endpoint_and_independent_restart_reaper(home):
    client, controller, _activate, ticks, applied, _drop, busy, _state, provider, data_reads = home
    request = sdk_request(ticks[0])
    requester = provider()
    quote = await requester.quote(request)
    first = await asyncio.wait_for(requester.acquire(request, quote), 5)
    assert first.state is InferenceLeaseState.READY and first.route is not None
    assert first.metadata["cache_hit"] and first.metadata["worker_id"] == "worker-1"
    assert first.route.endpoint.get_secret_value().endswith("/v1")
    restarted = provider()
    assert (await restarted.status(request.owner_id, first.lease_id)).state is InferenceLeaseState.READY
    with pytest.raises(InferenceLeaseOwnershipError):
        await restarted.release("foreign-owner", first.lease_id)
    assert data_reads == READINESS_READS
    ticks[0] += timedelta(seconds=11)
    assert controller(client).reap_due()[0].state is RegisteredActivationState.PARKING
    busy.clear()
    assert controller(client).reap_due()[0].state is RegisteredActivationState.PARKED
    end = await restarted.status(request.owner_id, first.lease_id)
    assert end.state is InferenceLeaseState.RELEASED and end.route is None
    second_request = replace(request, request_id="second", requested_at=ticks[0])
    second = await asyncio.wait_for(
        restarted.acquire(second_request, await restarted.quote(second_request)), 5,
    )
    assert second.state is InferenceLeaseState.READY
    assert second.metadata["worker_id"] == "worker-3"
    assert [patch["workers"]["max"] for patch in applied] == [1, 0, 1]
    assert data_reads == READINESS_READS * 2


@pytest.mark.asyncio
async def test_distinct_broad_key_is_refused_before_any_activation_patch(home):
    *_, ticks, applied, _drop, _busy, _state, provider, data_reads = home
    requester = provider()
    requester._key = SecretStr("another-broad-key")
    request = sdk_request(ticks[0])
    quote = await requester.quote(request)
    with pytest.raises(InferenceLeaseProviderUnavailableError, match="not excluded"):
        await requester.acquire(request, quote)
    assert applied == [] and data_reads == []


@pytest.mark.parametrize("replace_worker", [False, True])
@pytest.mark.asyncio
async def test_missing_or_replaced_worker_cannot_reuse_historical_route(home, replace_worker):
    _client, _controller, _activate, ticks, _applied, _drop, busy, _state, provider, data_reads = home
    requester = provider()
    request = sdk_request(ticks[0])
    first = await asyncio.wait_for(requester.acquire(request, await requester.quote(request)), 5)
    if replace_worker:
        busy[0]["id"] = "other-worker"
    else:
        busy.clear()
    status = await requester.status(request.owner_id, first.lease_id)
    assert status.route is None and status.state is InferenceLeaseState.RELEASING
    assert data_reads == READINESS_READS


@pytest.mark.asyncio
async def test_concurrent_revoke_wins_over_stale_active_control_observation(home):
    client, controller, _activate, ticks, _applied, _drop, _busy, _state, provider, data_reads = home
    requester = provider()
    request = sdk_request(ticks[0])
    first = await asyncio.wait_for(requester.acquire(request, await requester.quote(request)), 5)
    seen, finish = asyncio.Event(), asyncio.Event()
    observe_worker = requester._owns_ready_worker

    async def delayed_observation(*args):
        observed = await observe_worker(*args)
        seen.set()
        await finish.wait()
        return observed

    requester._owns_ready_worker = delayed_observation
    async with asyncio.TaskGroup() as group:
        racing = group.create_task(requester.status(request.owner_id, first.lease_id))
        await asyncio.wait_for(seen.wait(), 5)
        journal = controller(client).journal
        journal.revoke(journal.get(first.lease_id).intent)
        finish.set()
    assert racing.result().route is None
    assert racing.result().state is InferenceLeaseState.RELEASING
    assert data_reads == READINESS_READS


@pytest.mark.asyncio
async def test_worker_replaced_between_certifying_telemetry_and_readback_never_gets_a_route(home):
    _client, _controller, _activate, ticks, applied, _drop, _busy, state, provider, data_reads = home
    state["_replace_during_observation"] = True
    requester = provider()
    request = sdk_request(ticks[0])
    lease = await asyncio.wait_for(requester.acquire(request, await requester.quote(request)), 5)
    assert lease.route is None
    assert lease.state is InferenceLeaseState.RELEASING
    assert data_reads == READINESS_READS
    assert [patch["workers"]["max"] for patch in applied] == [1, 0]


@pytest.mark.asyncio
async def test_release_and_unresolved_status_report_durable_monotonic_timestamps(home):
    _client, _controller, _activate, ticks, _applied, _drop, _busy, _state, provider, _reads = home
    requester = provider()
    original = sdk_request(ticks[0])
    request = replace(original, owner_id="owners/kite", request_id="runs/first")
    first = await asyncio.wait_for(requester.acquire(request, await requester.quote(request)), 5)
    ticks[0] += timedelta(seconds=1)
    released = await requester.release(request.owner_id, first.lease_id)
    ticks[0] += timedelta(seconds=1)
    polled = await requester.status(request.owner_id, first.lease_id)
    assert released.state is polled.state is InferenceLeaseState.RELEASING
    assert polled.updated_at >= released.updated_at >= first.updated_at
