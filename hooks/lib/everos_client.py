#!/usr/bin/env python3
"""Zero-third-party-dependency HTTP client for EverOS memory routes.

Port of examples/dsh/src/everos-client.ts. Uses only the standard library (urllib) — no
`pip install` beyond a bare Python 3.12+ interpreter is needed to run these hooks, since the
containers this plugin gets baked into aren't guaranteed to have httpx or any other
third-party package preinstalled. Synchronous rather than async — each hook invocation is a
short-lived one-shot process with no persistent event loop, so callers that want two searches
"in parallel" (user + agent track) use a thread pool instead of asyncio (see recall.py).
"""

from __future__ import annotations

import json as _json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal

_PATH_SAFE = re.compile(r"^[a-zA-Z0-9_.@+-]+$")

ApiVersion = Literal["auto", "v1", "v2"]


class EverosError(Exception):
    def __init__(
        self,
        status: int,
        code: str | None,
        message: str,
        request_id: str | None = None,
        path: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.request_id = request_id
        self.path = path


def _assert_scope_id(value: str | None, field: str) -> None:
    if value is None:
        return
    if value in (".", "..") or len(value) > 128 or not _PATH_SAFE.match(value):
        raise EverosError(0, "INVALID_SCOPE_ID", f"invalid {field}: {value!r}")


@dataclass
class EverosClient:
    base_url: str
    api_version: ApiVersion = "auto"
    timeout_s: float = 15.0
    _negotiated: str | None = None

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if self.api_version != "auto":
            self._negotiated = self.api_version

    def _call(
        self, method: str, path: str, body: dict[str, Any] | None, timeout_s: float | None = None
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = _json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"content-type": "application/json"} if data is not None else {}
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout_s or self.timeout_s) as response:
                status = response.status
                text = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            status = exc.code
            text = exc.read().decode("utf-8") if exc.fp else ""
        except TimeoutError as exc:
            # Read/connect timeouts sometimes surface as a bare TimeoutError (socket.timeout
            # is a TimeoutError alias since Python 3.10) instead of being wrapped in
            # URLError — must be caught explicitly or it bypasses EverosError entirely and
            # skips the retry/fallback logic in recall.py.
            raise EverosError(0, "TIMEOUT", f"{method} {path} timed out: {exc}", path=path) from exc
        except urllib.error.URLError as exc:
            raise EverosError(0, "NETWORK_ERROR", f"{method} {path} failed: {exc}", path=path) from exc

        try:
            parsed = _json.loads(text) if text else {}
        except ValueError as exc:
            raise EverosError(
                status, "BAD_RESPONSE", f"{method} {path} returned non-JSON (HTTP {status})", path=path
            ) from exc
        return {"status": status, "ok": 200 <= status < 300, "parsed": parsed}

    def _enveloped(
        self, path: str, body: dict[str, Any] | None, timeout_s: float | None = None
    ) -> dict[str, Any]:
        result = self._call("POST", path, body, timeout_s)
        parsed = result["parsed"]
        if result["ok"] and isinstance(parsed, dict) and "data" in parsed:
            return parsed["data"]
        if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
            error = parsed["error"]
            raise EverosError(
                result["status"],
                error.get("code"),
                error.get("message") or f"{path} failed (HTTP {result['status']})",
                parsed.get("request_id"),
                error.get("path") or path,
            )
        raise EverosError(
            result["status"],
            None,
            f"{path} returned an unexpected response (HTTP {result['status']})",
            path=path,
        )

    def _memory_call(
        self, route: str, body: dict[str, Any], timeout_s: float | None = None
    ) -> dict[str, Any]:
        if self._negotiated:
            return self._enveloped(f"/api/{self._negotiated}/memory/{route}", body, timeout_s)
        try:
            value = self._enveloped(f"/api/v2/memory/{route}", body, timeout_s)
            self._negotiated = "v2"
            return value
        except EverosError as exc:
            if exc.status != 404:
                raise
        value = self._enveloped(f"/api/v1/memory/{route}", body, timeout_s)
        self._negotiated = "v1"
        return value

    def health(self, timeout_s: float | None = None) -> dict[str, Any]:
        result = self._call("GET", "/health", None, timeout_s)
        parsed = result["parsed"]
        if result["ok"] and isinstance(parsed, dict) and parsed.get("status") == "ok":
            return parsed
        raise EverosError(
            result["status"],
            None,
            f"/health returned an unexpected response (HTTP {result['status']})",
            path="/health",
        )

    def add(self, request: dict[str, Any], timeout_s: float | None = None) -> dict[str, Any]:
        _assert_scope_id(request.get("app_id"), "app_id")
        _assert_scope_id(request.get("project_id"), "project_id")
        return self._memory_call("add", request, timeout_s)

    def search(self, request: dict[str, Any], timeout_s: float | None = None) -> dict[str, Any]:
        has_user = request.get("user_id") is not None
        has_agent = request.get("agent_id") is not None
        if has_user == has_agent:
            raise EverosError(0, "INVALID_OWNER", "exactly one of user_id / agent_id must be provided")
        _assert_scope_id(request.get("app_id"), "app_id")
        _assert_scope_id(request.get("project_id"), "project_id")
        return self._memory_call("search", request, timeout_s)

    def get(
        self, request: dict[str, Any], timeout_s: float | None = None
    ) -> dict[str, Any]:
        has_user = request.get("user_id") is not None
        has_agent = request.get("agent_id") is not None
        if has_user == has_agent:
            raise EverosError(
                0,
                "INVALID_OWNER",
                "exactly one of user_id / agent_id must be provided",
            )
        _assert_scope_id(request.get("app_id"), "app_id")
        _assert_scope_id(request.get("project_id"), "project_id")
        return self._memory_call("get", request, timeout_s)

    def flush(self, request: dict[str, Any], timeout_s: float | None = None) -> dict[str, Any]:
        _assert_scope_id(request.get("app_id"), "app_id")
        _assert_scope_id(request.get("project_id"), "project_id")
        return self._memory_call("flush", request, timeout_s)
