"""Bounded pinned TLS byte transport, not a computer driver or worker server.

An original host operation must retain authority across exchange. This module
adds no admission policy, worker state engine, retries, bootstrap or queue. A
remote image must separately expose the same reviewed worker protocol and prove
its sandbox/egress; a successful TLS exchange cannot establish those facts.
"""

from __future__ import annotations

import asyncio
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
import hmac
import ipaddress
import re
import ssl
from typing import Callable

from kestrel_sdk.resources import ResourceOutcomeUnknown, ResourceUnsupported
from .effect_custody import await_owned_effect


@dataclass(frozen=True)
class PinnedWorkerTarget:
    """Trusted original private connection facts; never caller-chosen authority."""

    address: str = field(repr=False)
    port: int
    server_name: str = field(repr=False)
    certificate_sha256: str = field(repr=False)
    expires_at: datetime

    def __post_init__(self):
        try:
            ipaddress.ip_address(self.address)  # No ambient DNS or rebinding.
        except (ValueError, TypeError):
            raise ValueError("private worker requires a numeric endpoint") from None
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("invalid private worker port")
        if not re.fullmatch(r"[a-f0-9]{64}", self.certificate_sha256):
            raise ValueError("exact pretrusted worker certificate pin required")
        if not self.server_name or self.expires_at.utcoffset() is None:
            raise ValueError("private worker name and aware expiry required")


class PinnedWorkerTLS:
    """Independent connections; no client queue shared by control and data IO."""

    MAX_REQUEST = 512 * 1024
    MAX_RESPONSE = 23 * 1024 * 1024

    def __init__(
        self,
        target: PinnedWorkerTarget,
        context: ssl.SSLContext,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self._require_context(context)
        # The original host owns client certificate loading and provisioned CA.
        # The peer pin is mandatory even if an SSLContext has broader trust.
        self.target, self.context, self.clock = target, context, clock

    @staticmethod
    def _require_context(context):
        if (
            not isinstance(context, ssl.SSLContext)
            or not context.check_hostname
            or context.verify_mode != ssl.CERT_REQUIRED
            or context.minimum_version < ssl.TLSVersion.TLSv1_3
        ):
            raise ResourceUnsupported("worker TLS requires verified hostname and TLS1.3")

    async def exchange(
        self,
        request: bytes,
        *,
        admission: AbstractAsyncContextManager[Callable[[], None]],
        timeout_seconds: float = 20,
    ) -> bytes:
        """Send an existing bounded private worker newline frame exactly once.

        The admission is the original retained host policy, never a checker
        reconstructed from scope/IDs. After write, any transport failure is an
        unknown result to reconcile with original action/transfer IDs. Caller
        cancellation remains cancellation and preserves that original custody.
        """
        if (
            not isinstance(request, bytes)
            or not request.endswith(b"\n")
            or b"\n" in request[:-1]
            or len(request) > self.MAX_REQUEST
        ):
            raise ResourceUnsupported("invalid bounded private worker request")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not 0 < timeout_seconds <= 30
        ):
            raise ResourceUnsupported("invalid private worker timeout")
        async with admission as original_check:

            def check():
                self._require_context(self.context)
                if original_check() is not None:
                    raise ResourceUnsupported("invalid current worker checker")
                now = self.clock()
                if now.utcoffset() is None or now >= self.target.expires_at:
                    raise ResourceUnsupported("private worker connection expired")

            check()
            writer = None
            sent = False
            try:
                async with asyncio.timeout(timeout_seconds):
                    reader, writer = await asyncio.open_connection(
                        self.target.address,
                        self.target.port,
                        ssl=self.context,
                        server_hostname=self.target.server_name,
                        ssl_handshake_timeout=timeout_seconds,
                        ssl_shutdown_timeout=min(1.0, float(timeout_seconds)),
                        limit=self.MAX_RESPONSE + 1,
                    )
                    peer = writer.get_extra_info("ssl_object")
                    if peer is None or peer.version() != "TLSv1.3":
                        raise ResourceUnsupported("private worker TLS version is not admitted")
                    certificate = peer.getpeercert(binary_form=True)
                    if not certificate or not hmac.compare_digest(
                        hashlib.sha256(certificate).hexdigest(), self.target.certificate_sha256
                    ):
                        raise ResourceUnsupported("private worker certificate does not match")
                    # No await between the ORIGINAL check and the final write.
                    check()
                    writer.write(request)
                    sent = True
                    await writer.drain()
                    response = await reader.readline()
                    if not response.endswith(b"\n") or len(response) > self.MAX_RESPONSE:
                        raise ResourceOutcomeUnknown("worker response framing is uncertain")
            except Exception as error:
                if sent:
                    raise ResourceOutcomeUnknown(
                        "private worker exchange requires original-ID reconciliation"
                    ) from error
                raise
            finally:
                if writer is not None:
                    writer.close()
                    try:
                        await await_owned_effect(writer.wait_closed())
                    except (ConnectionError, ssl.SSLError):
                        pass  # Close may fail after the peer/transport is already gone.
            # Joining physical socket retirement may await after response read.
            # Retain the original delivery check through that last await too.
            try:
                check()
            except Exception as error:
                raise ResourceOutcomeUnknown(
                    "private worker delivery requires original-ID reconciliation"
                ) from error
            return response
