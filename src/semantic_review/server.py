"""
Standard-library HTTP server for the blind semantic review.

Security model (shared login node): binds to loopback only, accepts only
Host: localhost / 127.0.0.1, and requires a per-run access token (exchanged once for an
HttpOnly SameSite=Strict cookie). Images are addressed by review position only, never
by a client-supplied path. The JSON API exposes only blind fields (BLIND_ITEM_FIELDS).
"""

from __future__ import annotations

import hmac
import json
import re
import secrets
from functools import lru_cache
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from src.semantic_review import render
from src.semantic_review.store import (
    MODE_CONFIDENCE_PILOT,
    MODE_POSITIVE_QC,
    MODE_UNMATCHED,
    LabelConflict,
    ReviewStore,
)

WEB_DIR = Path(__file__).resolve().parent / "web"
STATIC_FILES = {
    "/static/semantic_review.css": ("semantic_review.css", "text/css; charset=utf-8"),
    "/static/semantic_review.js": ("semantic_review.js", "application/javascript; charset=utf-8"),
}
ALLOWED_HOSTS = {"localhost", "127.0.0.1"}
LOOPBACK_BIND = {"127.0.0.1", "localhost"}
COOKIE_NAME = "semantic_review_token"
IMAGE_ROUTE = re.compile(r"^/image/(context|crop|crop_raw)/(\d{1,6})\.jpg$")
MAX_BODY_BYTES = 4096

BLIND_ITEM_FIELDS = ("position", "total", "display_id", "human_label", "completed",
                     "next_unlabeled", "first_unlabeled")
# Neutral titles: the page must not tell the reviewer which GT class a set came from.
MODE_TITLES = {
    MODE_UNMATCHED: "Primary review",
    MODE_POSITIVE_QC: "QC review",
    MODE_CONFIDENCE_PILOT: "Pilot review",
}


class ReviewApp:
    def __init__(self, store: ReviewStore, token: str | None = None, image_cache_size: int = 48) -> None:
        self.store = store
        self.token = token or secrets.token_urlsafe(24)
        self._render = lru_cache(maxsize=image_cache_size)(self._render_uncached)

    def _render_uncached(self, position: int, kind: str) -> bytes:
        item = self.store.items[position]
        return render.render(item.image_path, item.bbox, kind)

    def image(self, position: int, kind: str) -> bytes:
        return self._render(position, kind)

    def state(self) -> dict[str, object]:
        return {
            "mode_title": MODE_TITLES[self.store.mode],
            "reviewer": self.store.reviewer,
            "total": len(self.store.items),
            "completed": self.store.completed(),
            "first_unlabeled": self.store.first_unlabeled(),
        }

    def item(self, position: int) -> dict[str, object]:
        item = self.store.items[position]
        payload = {
            "position": position,
            "total": len(self.store.items),
            "display_id": item.display_id,
            "human_label": self.store.labels[position],
            "completed": self.store.completed(),
            "next_unlabeled": self.store.next_unlabeled(position),
            "first_unlabeled": self.store.first_unlabeled(),
        }
        assert tuple(payload) == BLIND_ITEM_FIELDS
        return payload

    def check_token(self, candidate: str | None) -> bool:
        return bool(candidate) and hmac.compare_digest(candidate, self.token)


def make_handler(app: ReviewApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "SemanticReview/1"
        sys_version = ""

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            if len(args) > 1 and str(args[1]).startswith(("4", "5")):
                super().log_message(format, *args)

        def _send(self, status: int, body: bytes, content_type: str, headers: dict[str, str] | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, status: int, payload: dict) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json; charset=utf-8",
                       {"Cache-Control": "no-store"})

        def _error(self, status: int, message: str) -> None:
            self._json(status, {"error": message})

        def _host_ok(self) -> bool:
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
            return host in ALLOWED_HOSTS

        def _authorized(self) -> bool:
            cookie = SimpleCookie(self.headers.get("Cookie") or "")
            value = cookie[COOKIE_NAME].value if COOKIE_NAME in cookie else None
            return app.check_token(value) or app.check_token(self.headers.get("X-Review-Token"))

        def _position(self, text: str) -> int | None:
            if not text.isdigit():
                return None
            position = int(text)
            return position if 0 <= position < len(app.store.items) else None

        def do_GET(self) -> None:  # noqa: N802
            if not self._host_ok():
                return self._error(HTTPStatus.FORBIDDEN, "bad Host header")
            url = urlsplit(self.path)
            if url.path == "/":
                query_token = parse_qs(url.query).get("token", [None])[0]
                if app.check_token(query_token):
                    return self._send(HTTPStatus.SEE_OTHER, b"", "text/plain", {
                        "Location": "/",
                        "Set-Cookie": f"{COOKIE_NAME}={app.token}; Path=/; HttpOnly; SameSite=Strict",
                    })
                if not self._authorized():
                    return self._send(HTTPStatus.FORBIDDEN,
                                      b"Open the URL with ?token=... printed by the server.",
                                      "text/plain; charset=utf-8")
                body = (WEB_DIR / "semantic_review.html").read_bytes()
                return self._send(HTTPStatus.OK, body, "text/html; charset=utf-8", {"Cache-Control": "no-store"})
            if url.path in STATIC_FILES:
                name, content_type = STATIC_FILES[url.path]
                return self._send(HTTPStatus.OK, (WEB_DIR / name).read_bytes(), content_type,
                                  {"Cache-Control": "no-store"})
            if not self._authorized():
                return self._error(HTTPStatus.FORBIDDEN, "missing or invalid token")
            if url.path == "/api/state":
                return self._json(HTTPStatus.OK, app.state())
            if url.path == "/api/item":
                position = self._position(parse_qs(url.query).get("position", [""])[0])
                if position is None:
                    return self._error(HTTPStatus.NOT_FOUND, "no such position")
                return self._json(HTTPStatus.OK, app.item(position))
            match = IMAGE_ROUTE.match(url.path)
            if match:
                position = self._position(match.group(2))
                if position is None:
                    return self._error(HTTPStatus.NOT_FOUND, "no such position")
                return self._send(HTTPStatus.OK, app.image(position, match.group(1)), "image/jpeg",
                                  {"Cache-Control": "private, max-age=3600"})
            return self._error(HTTPStatus.NOT_FOUND, "not found")

        def do_POST(self) -> None:  # noqa: N802
            if not self._host_ok():
                return self._error(HTTPStatus.FORBIDDEN, "bad Host header")
            if not self._authorized():
                return self._error(HTTPStatus.FORBIDDEN, "missing or invalid token")
            if urlsplit(self.path).path != "/api/label":
                return self._error(HTTPStatus.NOT_FOUND, "not found")
            if not (self.headers.get("Content-Type") or "").startswith("application/json"):
                return self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "JSON required")
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= MAX_BODY_BYTES:
                return self._error(HTTPStatus.BAD_REQUEST, "bad body length")
            try:
                body = json.loads(self.rfile.read(length))
                position = int(body["position"])
                result = app.store.set_label(
                    position=position,
                    display_id=str(body["display_id"]),
                    label=str(body["label"]),
                    expected_current=str(body["expected_current"]),
                    confirm_change=bool(body.get("confirm_change", False)),
                )
            except LabelConflict as error:
                return self._error(HTTPStatus.CONFLICT, str(error))
            except (ValueError, KeyError, TypeError) as error:
                return self._error(HTTPStatus.BAD_REQUEST, f"bad request: {error}")
            return self._json(HTTPStatus.OK, {**result, **app.item(position)})

        do_HEAD = do_GET

    return Handler


def make_server(app: ReviewApp, host: str, port: int) -> ThreadingHTTPServer:
    if host not in LOOPBACK_BIND:
        raise ValueError(f"refusing to bind to {host!r}; use 127.0.0.1 and SSH port forwarding")
    server = ThreadingHTTPServer((host, port), make_handler(app))
    server.daemon_threads = True
    return server
