"""Supabase Storage client for invoice images (private bucket, short-lived signed URLs)."""

from __future__ import annotations

import re
from typing import Any

import httpx

SIGNED_URL_SECONDS = 300
_SAFE_PATH = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)*")


class StorageError(RuntimeError):
    """Storage call failed; the message never contains the service key."""


class SupabaseStorage:
    def __init__(self, url: str, service_key: str, bucket: str, client: httpx.Client | None = None,
                 timeout_seconds: float = 30.0) -> None:
        self._base = url.rstrip("/") + "/storage/v1"
        self._key = service_key
        self.bucket = bucket
        self._client = client or httpx.Client(timeout=timeout_seconds)

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._key}", "apikey": self._key, **(extra or {})}

    @staticmethod
    def _check_path(path: str) -> str:
        if not path or ".." in path.split("/") or not _SAFE_PATH.fullmatch(path):
            raise ValueError(f"Unsafe storage object path: {path!r}")
        return path

    def _raise(self, action: str, response: httpx.Response) -> None:
        text = response.text.replace(self._key, "***")[:200]
        raise StorageError(f"Storage {action} failed: HTTP {response.status_code} {text}")

    def ensure_bucket(self) -> None:
        response = self._client.post(f"{self._base}/bucket", headers=self._headers(),
                                     json={"id": self.bucket, "name": self.bucket, "public": False})
        if response.status_code == 200:
            return
        body: dict[str, Any] = {}
        try:
            body = response.json()
        except ValueError:
            pass
        if str(body.get("statusCode")) == "409" or response.status_code == 409:
            return
        self._raise("create bucket", response)

    def upload(self, path: str, data: bytes, content_type: str) -> None:
        path = self._check_path(path)
        response = self._client.post(f"{self._base}/object/{self.bucket}/{path}", content=data,
                                     headers=self._headers({"Content-Type": content_type, "x-upsert": "true"}))
        if response.status_code != 200:
            self._raise("upload", response)

    def signed_url(self, path: str, expires_in: int = SIGNED_URL_SECONDS) -> str:
        path = self._check_path(path)
        response = self._client.post(f"{self._base}/object/sign/{self.bucket}/{path}",
                                     headers=self._headers(), json={"expiresIn": expires_in})
        if response.status_code != 200:
            self._raise("sign", response)
        return self._base + response.json()["signedURL"]
