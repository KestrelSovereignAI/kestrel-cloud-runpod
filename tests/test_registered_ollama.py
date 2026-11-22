"""Exercise the controller with real SQLite and HTTP on loopback only."""

import json
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
from ollama_test_support import OwnedLoopbackHTTPServer

from kestrel_cloud_runpod.clients import RunpodControlPlaneClient
from kestrel_cloud_runpod.models import EndpointCreateRequest, FlashBoot, RunPodAPIError
from kestrel_cloud_runpod.ollama_contracts import OLLAMA_REQUIRE_CACHED_MODEL_ENV, OllamaLeaseConflictError
from kestrel_cloud_runpod.ollama_repository import SQLiteOllamaLeaseRepository
from kestrel_cloud_runpod.registered_ollama import RegisteredOllamaEndpoint, RegisteredOllamaLifecycle
from kestrel_cloud_runpod.registered_ollama_journal import RegisteredActivationState, RegisteredOllamaJournal
from kestrel_cloud_runpod.ollama_runtime import OLLAMA_RUNTIME_IMAGE_REPOSITORY


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
                self.send_json(state)
            elif self.path == base + "/workers":
                count = len(busy)
                self.send_json({
                    "workers": busy, "summary": {
                        "idle": count, "total": count, "running": 0,
                        "initializing": 0, "throttled": 0, "unhealthy": 0,
                    },
                })
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
        yield client, controller, activate, ticks, applied, drop, busy, state


def test_restart_parks_the_same_endpoint_and_proves_worker_absence_before_second_lease(home):
    client, controller, activate, ticks, applied, _, busy, state = home
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
    client, controller, activate, _ticks, applied, drop, _busy, _state = home
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
    client, controller, activate, ticks, applied, _drop, _busy, state = home
    row, _fresh = activate(client)
    state["env"] = {"PORT": "foreign"}
    ticks[0] += timedelta(seconds=61)
    assert controller(client).park(row) == row
    assert len(applied) == 1
    assert controller(client).journal.due() == (row,)


def test_restart_before_parking_patch_still_cuts_workers_to_zero_without_reopening(home):
    client, controller, activate, ticks, applied, _, busy, state = home
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
