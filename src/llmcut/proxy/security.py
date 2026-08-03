from __future__ import annotations

from urllib.parse import urljoin, urlparse

from llmcut.errors import ConfigurationError

HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


def filtered_headers(headers: dict[str, str]) -> dict[str, str]:
    return {key: value for key, value in headers.items() if key.lower() not in HOP_BY_HOP}


def upstream_url(base_url: str, path: str) -> str:
    base = urlparse(base_url)
    if base.scheme not in {"http", "https"} or not base.netloc or base.username or base.password:
        raise ConfigurationError(
            "configured upstream must be an http(s) origin without credentials"
        )
    if path.startswith("//") or "://" in path:
        raise ConfigurationError("request cannot override the configured upstream")
    joined = urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
    target = urlparse(joined)
    if (target.scheme, target.netloc) != (base.scheme, base.netloc):
        raise ConfigurationError("upstream origin escape rejected")
    return joined


def external_bind_warning(host: str) -> str | None:
    if host not in {"127.0.0.1", "::1", "localhost"}:
        return "Warning: proxy is externally reachable; use network access controls and TLS."
    return None
