"""Real native TLS sockets with pretrusted identities; no cloud or browser claim."""

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from dataclasses import replace
from hashlib import sha256
import ssl
import socket
import threading

from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
import pytest
from kestrel_sdk.resources import ResourceOutcomeUnknown, ResourceUnsupported
from kestrel_sdk.operator import OperatorAuthorizationError
from kestrel_cloud_runpod.remote_worker_tls import PinnedWorkerTarget, PinnedWorkerTLS


@asynccontextmanager
async def admission(flags):
    active = True

    def current():
        if not active or not flags["current"]:
            raise OperatorAuthorizationError("original operation retired")

    try:
        yield current
    finally:
        active = False


@pytest.fixture
async def tls_peer(tmp_path):
    now = datetime.now(UTC)
    key = Ed25519PrivateKey.generate()
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "owned-test-ca")])
    ca = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(1)
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .sign(key, None)
    )
    ca_path = tmp_path / "ca.pem"
    issued_files = [ca_path]
    ca_path.write_bytes(ca.public_bytes(serialization.Encoding.PEM))

    def leaf(label, server):
        private = Ed25519PrivateKey.generate()
        cert = (
            x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, label)]))
            .issuer_name(name)
            .public_key(private.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=1))
            .not_valid_after(now + timedelta(minutes=5))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(
                x509.ExtendedKeyUsage(
                    [ExtendedKeyUsageOID.SERVER_AUTH if server else ExtendedKeyUsageOID.CLIENT_AUTH]
                ),
                critical=True,
            )
            .add_extension(x509.SubjectAlternativeName([x509.DNSName(label)]), critical=False)
            .sign(key, None)
        )
        cert_path = tmp_path / (label + ".pem")
        key_path = tmp_path / (label + ".key")
        issued_files.extend((cert_path, key_path))
        cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        key_path.write_bytes(
            private.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        return cert, cert_path, key_path

    cert, server_cert, server_key = leaf("original-worker.invalid", True)
    owner_cert, client_cert, client_key = leaf("original-owner.invalid", False)
    _, other_cert, other_key = leaf("other-owner.invalid", False)

    def context(client=True, with_client_cert=True, replacement=False):
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT if client else ssl.PROTOCOL_TLS_SERVER)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_3
        ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.load_verify_locations(cafile=str(ca_path))
        if not client:
            ctx.load_cert_chain(str(server_cert), str(server_key))
        elif with_client_cert:
            ctx.load_cert_chain(
                str(other_cert if replacement else client_cert),
                str(other_key if replacement else client_key),
            )
        return ctx

    received = []
    hold = asyncio.Event()
    entered = asyncio.Event()
    tasks = set()

    async def serve(reader, writer):
        task = asyncio.current_task()
        tasks.add(task)
        try:
            # The fixture binds the exact issued client certificate as well as
            # CA trust. CA membership alone is not an owner/session grant.
            peer = writer.get_extra_info("ssl_object").getpeercert(binary_form=True)
            if (
                sha256(peer).digest()
                != sha256(owner_cert.public_bytes(serialization.Encoding.DER)).digest()
            ):
                return
            raw = await reader.readline()
            if not raw:
                return
            received.append(raw)
            if raw == b"hold\n":
                entered.set()
                await hold.wait()
            if raw == b"lost\n":
                return
            if raw == b"oversize\n":
                writer.write(b"x" * 1024 + b"\n")
                await writer.drain()
                return
            writer.write(b'{"schema_version":1,"result":"original"}\n')
            await writer.drain()
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, ssl.SSLError):
                pass
            tasks.discard(task)

    server = await asyncio.start_server(serve, "127.0.0.1", 0, ssl=context(False))
    port = server.sockets[0].getsockname()[1]
    target = PinnedWorkerTarget(
        "127.0.0.1",
        port,
        "original-worker.invalid",
        sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest(),
        now + timedelta(minutes=2),
    )
    try:
        yield target, context, received, hold, entered
    finally:
        hold.set()
        server.close()
        await server.wait_closed()
        for task in tuple(tasks):
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for path in issued_files:
            path.unlink()
        assert not any(path.exists() for path in issued_files)


async def test_real_pinned_mutual_tls_carries_one_frame_under_original_admission(tls_peer):
    target, context, received, _, _ = tls_peer
    flags = {"current": True}
    response = await PinnedWorkerTLS(target, context()).exchange(
        b"original\n", admission=admission(flags)
    )
    assert response.endswith(b"\n") and b"original" in response
    assert received == [b"original\n"]


@pytest.mark.parametrize("kind", ["wrong-pin", "wrong-host", "missing-client-cert", "other-owner"])
async def test_wrong_peer_or_owner_sends_no_private_rpc(tls_peer, kind):
    target, context, received, _, _ = tls_peer
    ctx = context()
    if kind == "wrong-pin":
        target = replace(target, certificate_sha256="0" * 64)
    elif kind == "wrong-host":
        target = replace(target, server_name="replacement.invalid")
    elif kind == "missing-client-cert":
        ctx = context(with_client_cert=False)
    else:
        ctx = context(replacement=True)
    with pytest.raises(
        (ResourceUnsupported, ssl.SSLError, ConnectionError, ResourceOutcomeUnknown)
    ):
        await PinnedWorkerTLS(target, ctx).exchange(
            b"private-secret\n", admission=admission({"current": True})
        )
    assert not received


async def test_original_revocation_during_real_handshake_is_checked_before_final_write(
    tls_peer, monkeypatch
):
    target, context, received, _, _ = tls_peer
    flags = {"current": True}
    original = asyncio.open_connection
    handshake = asyncio.Event()
    resume = asyncio.Event()

    async def delay(*args, **kwargs):
        result = await original(*args, **kwargs)
        handshake.set()
        await resume.wait()
        return result

    monkeypatch.setattr(asyncio, "open_connection", delay)
    task = asyncio.create_task(
        PinnedWorkerTLS(target, context()).exchange(b"private-secret\n", admission=admission(flags))
    )
    try:
        await asyncio.wait_for(handshake.wait(), 2)
        flags["current"] = False
        resume.set()
        with pytest.raises(OperatorAuthorizationError):
            await task
        await asyncio.sleep(0.01)
        assert not any(received)
    finally:
        resume.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_post_write_revocation_and_lost_receipt_keep_unknown_original_outcome(tls_peer):
    target, context, received, hold, entered = tls_peer
    flags = {"current": True}
    channel = PinnedWorkerTLS(target, context())
    task = asyncio.create_task(channel.exchange(b"hold\n", admission=admission(flags)))
    await asyncio.wait_for(entered.wait(), 2)
    flags["current"] = False
    hold.set()
    with pytest.raises(ResourceOutcomeUnknown):
        await task
    with pytest.raises(ResourceOutcomeUnknown):
        await channel.exchange(b"lost\n", admission=admission({"current": True}))
    assert received == [b"hold\n", b"lost\n"]


async def test_priority_exchange_does_not_wait_for_blocked_ordinary_call(tls_peer):
    target, context, received, hold, entered = tls_peer
    channel = PinnedWorkerTLS(target, context())
    blocked = asyncio.create_task(
        channel.exchange(b"hold\n", admission=admission({"current": True}))
    )
    try:
        await asyncio.wait_for(entered.wait(), 2)
        result = await asyncio.wait_for(
            channel.exchange(b"stop\n", admission=admission({"current": True})), 1
        )
        assert b"original" in result and not blocked.done()
        hold.set()
        await blocked
    finally:
        hold.set()
        if not blocked.done():
            blocked.cancel()
        await asyncio.gather(blocked, return_exceptions=True)


async def test_expiry_and_unbounded_input_refuse_before_socket(tls_peer):
    target, context, received, _, _ = tls_peer
    channel = PinnedWorkerTLS(target, context(), clock=lambda: target.expires_at)
    with pytest.raises(ResourceUnsupported, match="expired"):
        await channel.exchange(b"private\n", admission=admission({"current": True}))
    for raw in (b"no-terminator", b"one\ntwo\n", b"x" * (channel.MAX_REQUEST + 1) + b"\n"):
        with pytest.raises(ResourceUnsupported):
            await channel.exchange(raw, admission=admission({"current": True}))
    assert not received


@pytest.mark.parametrize("mode", ["unverified", "old-tls"])
def test_unverified_tls_context_is_never_admitted(mode):
    target = PinnedWorkerTarget(
        "127.0.0.1", 443, "original.invalid", "a" * 64, datetime.now(UTC) + timedelta(minutes=1)
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    if mode == "unverified":
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        context.minimum_version = ssl.TLSVersion.TLSv1_3
    else:
        context.minimum_version = ssl.TLSVersion.TLSv1_2
    with pytest.raises(ResourceUnsupported, match="verified hostname and TLS1.3"):
        PinnedWorkerTLS(target, context)


async def test_bounded_response_loss_is_unknown_without_retry(tls_peer):
    target, context, received, _, _ = tls_peer
    channel = PinnedWorkerTLS(target, context())
    channel.MAX_RESPONSE = 64
    with pytest.raises(ResourceOutcomeUnknown):
        await channel.exchange(b"oversize\n", admission=admission({"current": True}))
    assert received == [b"oversize\n"]


async def test_reconfigured_tls_context_cannot_reuse_an_earlier_admission(tls_peer):
    target, context, received, _, _ = tls_peer
    configured = context()
    channel = PinnedWorkerTLS(target, configured)
    configured.check_hostname = False
    with pytest.raises(ResourceUnsupported, match="verified hostname"):
        await channel.exchange(b"private-secret\n", admission=admission({"current": True}))
    assert not received


async def test_revocation_during_native_socket_retirement_refuses_private_delivery(
    tls_peer, monkeypatch
):
    target, context, received, _, _ = tls_peer
    flags = {"current": True}
    closing, resume = asyncio.Event(), asyncio.Event()
    original_open = asyncio.open_connection

    async def delayed_close(*args, **kwargs):
        reader, writer = await original_open(*args, **kwargs)
        original_wait = writer.wait_closed

        async def wait_closed():
            await original_wait()
            closing.set()
            await resume.wait()

        writer.wait_closed = wait_closed
        return reader, writer

    monkeypatch.setattr(asyncio, "open_connection", delayed_close)
    task = asyncio.create_task(
        PinnedWorkerTLS(target, context()).exchange(
            b"original\n",
            admission=admission(flags),
        )
    )
    try:
        await asyncio.wait_for(closing.wait(), 2)
        flags["current"] = False
        resume.set()
        with pytest.raises(ResourceOutcomeUnknown):
            await task
        assert received == [b"original\n"]
    finally:
        resume.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def test_invalid_private_route_is_not_echoed_in_validation_diagnostics():
    with pytest.raises(ValueError, match="numeric endpoint") as error:
        PinnedWorkerTarget(
            "private-route-secret.invalid", 443, "worker", "a" * 64, datetime.now(UTC)
        )
    assert "private-route-secret" not in str(error.value)
    assert error.value.__context__ is None and error.value.__cause__ is None


@pytest.mark.parametrize("address", [1, True, b"\x7f\x00\x00\x01", "fe80::1%private-zone"])
def test_numeric_endpoint_requires_literal_text_without_scope_or_retained_parse_error(address):
    with pytest.raises(ValueError, match="numeric endpoint") as error:
        PinnedWorkerTarget(address, 443, "worker", "a" * 64, datetime.now(UTC))
    assert error.value.__context__ is None and error.value.__cause__ is None


async def test_native_connect_failure_has_no_private_route_in_exception_chain(tls_peer):
    target, context, _, _, _ = tls_peer
    # Bind our own socket without listening: this exact owned endpoint refuses.
    with socket.socket() as owned:
        owned.bind(("127.0.0.1", 0))
        target = replace(target, port=owned.getsockname()[1])
        with pytest.raises(ResourceUnsupported, match="connection failed") as error:
            await PinnedWorkerTLS(target, context()).exchange(
                b"private\n", admission=admission({"current": True})
            )
    assert "127.0.0.1" not in str(error.value) and str(target.port) not in str(error.value)
    assert error.value.__context__ is None and error.value.__cause__ is None


@pytest.mark.parametrize("close_error", [TimeoutError("private-shutdown"), OSError("private-route")])
@pytest.mark.parametrize("outcome", ["response", "wrong-pin", "unframed"])
async def test_close_error_cannot_replace_original_classification(tls_peer, monkeypatch, close_error, outcome):
    target, context, _, _, _ = tls_peer
    original_open = asyncio.open_connection

    async def close_error_open(*args, **kwargs):
        reader, writer = await original_open(*args, **kwargs)
        original_wait = writer.wait_closed

        async def wait_closed():
            await original_wait()  # Actual native socket is retired first.
            raise close_error

        writer.wait_closed = wait_closed
        return reader, writer

    monkeypatch.setattr(asyncio, "open_connection", close_error_open)
    if outcome == "wrong-pin":
        target = replace(target, certificate_sha256="0" * 64)
    expected = ResourceUnsupported if outcome == "wrong-pin" else ResourceOutcomeUnknown
    message = "certificate does not match" if outcome == "wrong-pin" else (
        "framing is uncertain" if outcome == "unframed" else "original-ID reconciliation"
    )
    with pytest.raises(expected, match=message) as error:
        await PinnedWorkerTLS(target, context()).exchange(
            b"lost\n" if outcome == "unframed" else b"original\n",
            admission=admission({"current": True}),
        )
    assert error.value.__context__ is None and error.value.__cause__ is None
    assert "private-shutdown" not in str(error.value) and "private-route" not in str(error.value)


async def test_native_peer_withholding_close_notify_keeps_postwrite_unknown_and_reaps_socket(
    tls_peer, monkeypatch
):
    target, context, _, _, _ = tls_peer
    release = threading.Event()
    received = []
    failures = []
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(3)
    original_port = listener.getsockname()[1]
    retired = []
    close_failures = []
    original_open = asyncio.open_connection

    async def observe_native_close(*args, **kwargs):
        reader, writer = await original_open(*args, **kwargs)
        original_wait = writer.wait_closed

        async def wait_closed():
            try:
                await original_wait()
            except Exception as error:
                # Observe the real CPython close result; inject no error.
                close_failures.append((type(error), str(error)))
                raise

        writer.wait_closed = wait_closed
        return reader, writer

    monkeypatch.setattr(asyncio, "open_connection", observe_native_close)

    def serve():
        try:
            with listener.accept()[0] as connection:
                connection.settimeout(3)
                with context(False).wrap_socket(connection, server_side=True) as tls:
                    retired.append(tls)
                    frame = b""
                    while not frame.endswith(b"\n"):
                        frame += tls.recv(1024)
                    received.append(frame)
                    tls.sendall(b"original-receipt\n")
                    # SSLSocket performs no background read. Unlike an asyncio
                    # server it cannot auto-answer close_notify while withheld.
                    if not release.wait(4):
                        raise AssertionError("owned peer exceeded fixed lifetime")
        except BaseException as error:
            failures.append(error)

    child = threading.Thread(target=serve, name="owned-close-notify-peer")
    child.start()
    try:
        with pytest.raises(ResourceOutcomeUnknown, match="original-ID reconciliation") as error:
            await PinnedWorkerTLS(replace(target, port=original_port), context()).exchange(
                b"original-action-id\n", admission=admission({"current": True}), timeout_seconds=0.3
            )
        assert received == [b"original-action-id\n"] and child.is_alive()
        assert close_failures == [(TimeoutError, "SSL shutdown timed out")]
        assert error.value.__context__ is None and error.value.__cause__ is None
    finally:
        release.set()
        listener.close()
        await asyncio.to_thread(child.join, 4)
        assert not child.is_alive() and listener.fileno() == -1
        assert len(retired) == 1 and retired[0].fileno() == -1
        assert not failures



async def test_repeated_cancellation_keeps_native_close_join_owned(tls_peer, monkeypatch):
    target, context, received, _, _ = tls_peer
    original_open = asyncio.open_connection
    closing, resume, joined = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def delayed_close(*args, **kwargs):
        reader, writer = await original_open(*args, **kwargs)
        original_wait = writer.wait_closed

        async def wait_closed():
            await original_wait()
            closing.set()
            await resume.wait()
            joined.set()

        writer.wait_closed = wait_closed
        return reader, writer

    monkeypatch.setattr(asyncio, "open_connection", delayed_close)
    task = asyncio.create_task(PinnedWorkerTLS(target, context()).exchange(
        b"original\n", admission=admission({"current": True})
    ))
    try:
        await asyncio.wait_for(closing.wait(), 2)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and not joined.is_set()
        resume.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert joined.is_set() and received == [b"original\n"]
    finally:
        resume.set()
        await asyncio.gather(task, return_exceptions=True)
