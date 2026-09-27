"""Request body size limit enforced on the raw ASGI stream.

Why not just check the upload in the endpoint: Starlette parses (and spools to disk) the whole
multipart body *before* the endpoint runs, so an endpoint-level check fires only after a 100 GB
upload has already filled the disk. ``Content-Length`` alone is not enough either - chunked
requests do not send it and a client can lie. This middleware counts the bytes actually received
and aborts the request as soon as the limit is crossed.
"""

from __future__ import annotations

import json

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class _BodyTooLarge(Exception):
    pass


class BodySizeLimitMiddleware:
    def __init__(self, app: ASGIApp, *, max_body_bytes: int, message: str = "Запрос слишком большой") -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.message = message

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        declared = headers.get(b"content-length")
        if declared is not None:
            try:
                too_large = int(declared) > self.max_body_bytes
            except ValueError:
                too_large = True  # malformed header: refuse rather than guess
            if too_large:
                await self._reject(send)
                return

        received = 0
        exceeded = False
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received, exceeded
            if exceeded:
                raise _BodyTooLarge
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    exceeded = True
                    raise _BodyTooLarge
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal response_started
            # The framework may turn our exception into its own 400 ("error parsing the body");
            # whatever it answers, the client must see 413.
            if exceeded:
                if message["type"] == "http.response.start" and not response_started:
                    response_started = True
                    await self._reject(send)
                return
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except _BodyTooLarge:
            if not response_started:
                await self._reject(send)

    async def _reject(self, send: Send) -> None:
        body = json.dumps({"detail": self.message}, ensure_ascii=False).encode()
        headers: list[tuple[bytes, bytes]] = [
            (b"content-type", b"application/json; charset=utf-8"),
            (b"content-length", str(len(body)).encode()),
            (b"connection", b"close"),
        ]
        await send({"type": "http.response.start", "status": 413, "headers": headers})
        await send({"type": "http.response.body", "body": body})
