"""Compliance-enforcing HTTP client wrapper with host allow-list.

In compliance with Competition Rules §2.1 & §2.2:
Only authorized hosts (CDSE, ohsome, Copernicus DEM S3, model weight host)
are permitted. Any request to forbidden domains (e.g. EMSR927, UNOSAT, live OSM)
raises ComplianceViolationError immediately.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)


class ComplianceViolationError(RuntimeError):
    """Raised when pipeline code attempts network access to an unapproved host."""


ALLOWED_HOSTS: set[str] = {
    # Copernicus Data Space Ecosystem (CDSE)
    "identity.dataspace.copernicus.eu",
    "dataspace.copernicus.eu",
    "stac.dataspace.copernicus.eu",
    "sh.dataspace.copernicus.eu",
    "catalogue.dataspace.copernicus.eu",
    "zipper.dataspace.copernicus.eu",
    "eodata.dataspace.copernicus.eu",
    # Ohsome API for pre-event OSM snapshots
    "api.ohsome.org",
    # Copernicus DEM 30m AWS bucket
    "copernicus-dem-30m.s3.amazonaws.com",
    "copernicus-dem-30m.s3.eu-central-1.amazonaws.com",
    # Official model weights / benchmarks repositories
    "huggingface.co",
    "cdn-lfs.huggingface.co",
    "raw.githubusercontent.com",
    "github.com",
}

FORBIDDEN_KEYWORDS: set[str] = {
    "mapping.emergency.copernicus.eu",
    "emsr927",
    "unosat",
    "overpass-api.de",
    "api.openstreetmap.org",
    "geofabrik",
}


def is_host_allowed(url: str) -> bool:
    """Verify whether a target URL belongs to the allowed host list."""
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()

    # Check forbidden keywords first
    for kw in FORBIDDEN_KEYWORDS:
        if kw in url.lower():
            return False

    return hostname in ALLOWED_HOSTS


def assert_url_allowed(url: str) -> None:
    """Raise ComplianceViolationError if URL is not on the host allow-list."""
    if not is_host_allowed(url):
        raise ComplianceViolationError(
            f"Compliance Violation: Access to host for URL '{url}' is forbidden by Competition Rules §2.2. "
            f"Allowed hosts: {sorted(ALLOWED_HOSTS)}"
        )


class CompliantHttpClient:
    """HTTP client wrapper that enforces domain-level allowlist filtering."""

    def __init__(self, timeout: float = 30.0, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.Client(timeout=timeout)

    def get(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        assert_url_allowed(url)
        logger.debug(f"[HTTP GET ALLOWED] {url}")
        return self._client.get(url, params=params, headers=headers, **kwargs)

    def post(
        self,
        url: str,
        data: Any = None,
        json: Any = None,
        headers: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        assert_url_allowed(url)
        logger.debug(f"[HTTP POST ALLOWED] {url}")
        return self._client.post(url, data=data, json=json, headers=headers, **kwargs)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> CompliantHttpClient:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
