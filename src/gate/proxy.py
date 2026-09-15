"""Transparent forwarding of an incoming request to the upstream vLLM server.

Thin edge: given a caller that has already decided to forward, forward the
request verbatim (same method/path/query/headers/body) and return the upstream
response. The upstream status code is propagated unmasked (a vLLM 5xx surfaces
as a 5xx, never a 200 or a 429), and ``stream: true`` SSE responses are
streamed back verbatim without buffering the body.

Invariants (see AGENTS.md "Known gotchas" #6):

- The gate is a transparent proxy for the two generation endpoints only.
- Hop-by-hop headers are stripped on the way out (request) and back (response);
  in the stream path ``content-length`` and ``transfer-encoding`` are dropped
  because the body is re-framed.
- No logging here (the app layer logs); no per-request timeouts (the client's
  concern).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping

import httpx
from fastapi import Request, Response
from fastapi.responses import StreamingResponse

#: Lowercased hop-by-hop header names (RFC 7230 §6.1) stripped from the
#: proxied request and response.
HOP_BY_HOP: frozenset[str] = frozenset(
    {
        "host",
        "content-length",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


def _drop_hop_by_hop(headers: Mapping[str, str]) -> dict[str, str]:
    """Return ``headers`` minus hop-by-hop names (case-insensitive)."""
    return {key: value for key, value in headers.items() if key.lower() not in HOP_BY_HOP}


async def proxy_request(
    client: httpx.AsyncClient,
    request: Request,
    base_url: str,
    *,
    stream: bool = False,
) -> Response:
    """Forward ``request`` to ``base_url`` and return the upstream response.

    ``base_url`` is like ``http://vllm:8000`` (no trailing slash; the caller
    guarantees this). The upstream URL is ``base_url + request.url.path`` plus
    the query string when one is present. Hop-by-hop request headers are
    stripped; the body is forwarded as-is (the caller has already decided to
    forward, so it is not re-parsed here).

    With ``stream=False`` the upstream body is buffered and returned as a
    :class:`~fastapi.Response`. With ``stream=True`` the upstream status is
    known before responding (build-request/send pattern) and the body is
    streamed back verbatim as a :class:`~fastapi.responses.StreamingResponse`,
    closing the upstream response when the client finishes.
    """
    url = base_url + request.url.path
    if request.url.query:
        url += "?" + request.url.query

    req_headers = _drop_hop_by_hop(request.headers)
    body = await request.body()

    if not stream:
        resp = await client.post(url, content=body, headers=req_headers)
        return Response(
            status_code=resp.status_code,
            content=resp.content,
            headers=_drop_hop_by_hop(resp.headers),
        )

    req = client.build_request("POST", url, content=body, headers=req_headers)
    upstream = await client.send(req, stream=True)

    async def _stream() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(
        _stream(),
        status_code=upstream.status_code,
        headers=_drop_hop_by_hop(upstream.headers),
    )
