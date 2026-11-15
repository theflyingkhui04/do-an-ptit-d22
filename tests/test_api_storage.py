"""[M5-01 web demo] Supabase Storage client (private bucket, signed URLs), with a fake HTTP transport."""

from __future__ import annotations

import json

import httpx
import pytest

from src.api.storage import StorageError, SupabaseStorage

KEY = "service-key-secret-xyz"


def make(handler) -> SupabaseStorage:
    return SupabaseStorage("https://proj.supabase.co/", KEY, "invoice-images",
                           client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_upload_is_authenticated_and_upserts():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(method=request.method, url=str(request.url), headers=dict(request.headers),
                    body=request.content)
        return httpx.Response(200, json={"Key": "invoice-images/uploads/up_1.jpg"})

    make(handler).upload("uploads/up_1.jpg", b"\xff\xd8jpeg", "image/jpeg")
    assert seen["method"] == "POST"
    assert seen["url"] == "https://proj.supabase.co/storage/v1/object/invoice-images/uploads/up_1.jpg"
    assert seen["headers"]["authorization"] == f"Bearer {KEY}"
    assert seen["headers"]["x-upsert"] == "true"
    assert seen["headers"]["content-type"] == "image/jpeg"
    assert seen["body"] == b"\xff\xd8jpeg"


def test_signed_url_expires_after_five_minutes_by_default():
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://proj.supabase.co/storage/v1/object/sign/invoice-images/devset/a.jpg"
        assert json.loads(request.content) == {"expiresIn": 300}
        return httpx.Response(200, json={"signedURL": "/object/sign/invoice-images/devset/a.jpg?token=t1"})

    url = make(handler).signed_url("devset/a.jpg")
    assert url == "https://proj.supabase.co/storage/v1/object/sign/invoice-images/devset/a.jpg?token=t1"


def test_private_bucket_is_created_once():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(200, json={"name": "invoice-images"})
        return httpx.Response(400, json={"statusCode": "409", "error": "Duplicate", "message": "exists"})

    storage = make(handler)
    storage.ensure_bucket()
    storage.ensure_bucket()
    assert calls[0] == {"id": "invoice-images", "name": "invoice-images", "public": False}


@pytest.mark.parametrize("path", ["../x.jpg", "/abs.jpg", "a/../../b.jpg", ""])
def test_unsafe_object_paths_are_refused(path):
    with pytest.raises(ValueError):
        make(lambda r: httpx.Response(200)).upload(path, b"x", "image/jpeg")


def test_errors_never_contain_the_key():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"message": f"bad token {KEY}"})

    with pytest.raises(StorageError) as info:
        make(handler).upload("uploads/up_1.jpg", b"x", "image/jpeg")
    assert KEY not in str(info.value)
    assert "403" in str(info.value)
