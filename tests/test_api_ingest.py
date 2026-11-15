"""[M5-01 web demo] Tasks 4.2-4.3: upload job, Node 1 on full OCR text, lane map and lane comparison."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.api.compare import compare_fields
from src.api.grounding import grounding_report
from src.api.jobs import NODE1_MAX_TOKENS, node1_system_prompt
from src.api.llm import LLMError
from src.pipeline.ocr import OcrError
from src.pipeline.text_normalize import has_vietnamese_marks
from tests.api_fakes import (
    EXAMPLE,
    HEADERS,
    LANE_IDS,
    OCR_TEXT,
    OCR_TEXT_PLAIN,
    FakeCaller,
    FakeOcrReader,
    FakeStore,
    build,
    jpeg_bytes,
    lane_callers,
    node1_answer,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def upload(client, data=None, name="a.jpg", mime="image/jpeg", **form):
    form.setdefault("endpoint_ids", LANE_IDS["adaptive"])
    return client.post("/api/invoices", headers=HEADERS, files={"file": (name, data or jpeg_bytes(), mime)},
                       data=form)


def finished_job(client, job_id):
    return client.get(f"/api/jobs/{job_id}", headers=HEADERS).json()


# ---------------------------------------------------------------- 4.2 upload job
def test_valid_upload_runs_to_done_and_stores_the_adaptive_extraction():
    app, store, storage = build()
    with TestClient(app) as client:
        r = upload(client)
        assert r.status_code == 202
        job = finished_job(client, r.json()["job_id"])
    assert job["status"] == "done" and job["error"] is None
    result = job["result"]
    assert result["ocr"]["text"] == OCR_TEXT_PLAIN and not has_vietnamese_marks(result["ocr"]["text"])
    assert result["ocr"]["engine"] == "A" and result["ocr"]["marks_stripped"] is True
    assert result["ocr"]["kind"] == "library" and "finish_reason" not in result["ocr"]
    assert result["stored_endpoint"] == LANE_IDS["adaptive"] and result["stored_from_fallback"] is False
    assert result["primary"] == LANE_IDS["adaptive"] and result["baseline"] == LANE_IDS["vanilla"]
    assert job["invoice_id"] == result["invoice_id"] == f"{job['job_id']}:{job['job_id']}"
    for lane in ("adaptive", "vanilla"):
        report = result["runs"][LANE_IDS[lane]]
        assert report["valid"] is True
        for key in ("latency_ms", "ttft_ms", "tpot_ms", "output_tokens_per_s"):
            assert report[key] is not None
    assert result["grounding"]["ungrounded_fields"] == []
    payload, run = store.saved[0]
    assert run.lane is None and run.model_name == "model-adaptive" and run.prompt_version == "node1_extractor.v2"
    assert payload["source"]["raw_text"] == OCR_TEXT_PLAIN
    assert storage.uploads == {f"uploads/{job['job_id']}.jpg": storage.uploads[f"uploads/{job['job_id']}.jpg"]}


def test_node1_request_uses_prompt_v2_and_3072_tokens():
    seen = []

    def remember(request):
        seen.append(request)
        return node1_answer(request)

    lanes = lane_callers(adaptive=FakeCaller("adaptive", content=remember))
    app, _, _ = build(lanes=lanes)
    with TestClient(app) as client:
        upload(client)
    request = seen[0]
    assert request["max_tokens"] == NODE1_MAX_TOKENS == 3072
    assert request["temperature"] == 0.0 and request["seed"] == 42
    assert request["messages"][0]["content"] == node1_system_prompt()
    package_prompt = (REPO_ROOT / "experiments/configs/w1_v2/system_prompt.md").read_text(encoding="utf-8")
    assert node1_system_prompt() == package_prompt.rstrip("\n")
    ocr_part = request["messages"][1]["content"].split("OCR_TEXT:\n", 1)[1]
    assert OCR_TEXT_PLAIN in ocr_part and not has_vietnamese_marks(ocr_part)


def test_node1_input_matches_the_w1_v3_package_construction():
    """Same text path as experiments/configs/w1_v3: strip marks, then build_chat_request normalises lines."""
    import json

    from src.api.jobs import node1_request_row
    from src.pipeline.text_normalize import strip_vietnamese_marks

    package = REPO_ROOT / "experiments/configs/w1_v3/node1_chat_requests.jsonl"
    first = json.loads(package.read_text(encoding="utf-8").splitlines()[0])
    raw = first["expected_source"]["raw_text"]
    row = node1_request_row("up_x", "uploads/up_x.jpg", strip_vietnamese_marks(raw), first["request"]["model"])
    assert row["request"]["messages"][0] == first["request"]["messages"][0]
    assert row["expected_source"]["raw_text"] == raw
    ocr_part = row["request"]["messages"][1]["content"].split("OCR_TEXT:\n", 1)[1]
    assert ocr_part == first["request"]["messages"][1]["content"].split("OCR_TEXT:\n", 1)[1]


def test_pdf_and_non_image_bytes_are_415_without_a_job():
    app, store, _ = build()
    with TestClient(app) as client:
        assert upload(client, data=b"%PDF-1.7 ...", name="a.pdf", mime="application/pdf").status_code == 415
        assert upload(client, data=b"\xff\xd8\xff" + b"not really a jpeg").status_code == 415
    assert store.jobs == {}


def test_file_over_10_mb_is_413_without_a_job():
    app, store, _ = build()
    with TestClient(app) as client:
        assert upload(client, data=jpeg_bytes() + b"\0" * (10 * 1024 * 1024)).status_code == 413
    assert store.jobs == {}


def test_no_text_or_ocr_down_fails_at_ocr_without_calling_node1():
    cases = (FakeOcrReader(error=OcrError("no text detected")), FakeOcrReader("  \n "),
             FakeOcrReader(error=LLMError("ocr: cannot reach https://ocr.test/v1/ocr: ConnectError")))
    for ocr in cases:
        called = []
        lanes = lane_callers(adaptive=FakeCaller("adaptive", content=lambda r, called=called: called.append(1) or node1_answer(r)))
        app, store, _ = build(ocr=ocr, lanes=lanes)
        with TestClient(app) as client:
            job = finished_job(client, upload(client).json()["job_id"])
        assert job["status"] == "failed" and job["error"].startswith("ocr:")
        assert called == [] and store.saved == [] and len(ocr.calls) == 1


def test_a_source_copied_wrong_is_restored_and_reported():
    def wrong_source(request):
        return node1_answer(request, mutate=lambda p: p["source"].update(image_path="uploads/up_f6_x.jpg"))

    lanes = lane_callers(adaptive=FakeCaller("adaptive", content=wrong_source),
                         vanilla=FakeCaller("vanilla", content=wrong_source))
    app, store, _ = build(lanes=lanes)
    with TestClient(app) as client:
        job = finished_job(client, upload(client).json()["job_id"])
    assert job["status"] == "done" and len(store.saved) == 1
    run = job["result"]["runs"][LANE_IDS["adaptive"]]
    assert run["valid"] is True and run["source_restored"] == ["image_path"]
    assert job["result"]["invoice"]["source"]["image_path"] == f"uploads/{job['job_id']}.jpg"


def test_unknown_job_is_404():
    app, _, _ = build()
    with TestClient(app) as client:
        assert client.get("/api/jobs/up_0123456789ab", headers=HEADERS).status_code == 404
        assert client.get("/api/jobs/../../etc", headers=HEADERS).status_code == 404


def test_jobs_running_at_restart_become_interrupted():
    store = FakeStore()
    store.create_job("up_aaaaaaaaaaaa")
    store.update_job("up_aaaaaaaaaaaa", "node1")
    app, _, _ = build(store=store)
    with TestClient(app) as client:
        job = finished_job(client, "up_aaaaaaaaaaaa")
    assert job["status"] == "interrupted"


def test_a_crash_never_leaves_a_job_running():
    class BrokenStorage:
        def upload(self, *a):
            raise RuntimeError("storage down")

    app, _store, _ = build(storage=BrokenStorage())
    with TestClient(app) as client:
        job = finished_job(client, upload(client).json()["job_id"])
    assert job["status"] == "failed" and "storage down" in job["error"]


# ---------------------------------------------------------------- 4.3 lanes
def test_identical_extraction_counts_every_field_equal():
    comparison = compare_fields(EXAMPLE, EXAMPLE)
    assert comparison["equal_fields"] == comparison["total_fields"] > 0
    assert comparison["differing_fields"] == []


def test_one_different_total_is_named():
    other = {**EXAMPLE, "totals": {**EXAMPLE["totals"],
                                   "total_amount": {**EXAMPLE["totals"]["total_amount"], "value": 42001}}}
    comparison = compare_fields(other, EXAMPLE)
    assert comparison["equal_fields"] == comparison["total_fields"] - 1
    assert comparison["differing_fields"] == ["totals.total_amount.value"]


def test_comparison_is_reported_per_endpoint_against_the_baseline():
    app, _, _ = build()
    with TestClient(app) as client:
        result = finished_job(client, upload(client).json()["job_id"])["result"]
    adaptive = result["runs"][LANE_IDS["adaptive"]]
    assert adaptive["equal_fields"] == adaptive["total_fields"] and adaptive["name"] == "adaptive"
    assert round(adaptive["speedup_tpot_vs_baseline"], 3) == round(20.0 / 14.0, 3)
    baseline = result["runs"][LANE_IDS["vanilla"]]
    assert baseline["is_baseline"] is True and baseline["speedup_tpot_vs_baseline"] is None


def test_several_endpoints_run_together():
    app, _, _ = build()
    with TestClient(app) as client:
        ids = ",".join(LANE_IDS.values())
        result = finished_job(client, upload(client, endpoint_ids=ids).json()["job_id"])["result"]
    assert set(result["runs"]) == set(LANE_IDS.values())


def test_endpoint_choice_is_validated_and_the_baseline_joins():
    app, _, _ = build()
    with TestClient(app) as client:
        assert upload(client, endpoint_ids="").status_code == 422
        assert upload(client, endpoint_ids="ep_0000ffff").status_code == 404
        assert upload(client, endpoint_ids=",".join(["ep_0000000a"] * 7 + list(LANE_IDS.values()))).status_code == 202
        body = upload(client, endpoint_ids=LANE_IDS["static"]).json()
    assert body["endpoint_ids"] == [LANE_IDS["static"], LANE_IDS["vanilla"]]


def test_without_a_baseline_there_is_no_comparison():
    from tests.api_fakes import lane_endpoints

    app, _, _ = build(endpoints=lane_endpoints(baseline=None))
    with TestClient(app) as client:
        result = finished_job(client, upload(client).json()["job_id"])["result"]
    assert result["baseline"] is None and list(result["runs"]) == [LANE_IDS["adaptive"]]
    assert result["runs"][LANE_IDS["adaptive"]]["speedup_tpot_vs_baseline"] is None


def test_baseline_is_stored_and_labelled_when_the_first_endpoint_fails():
    lanes = lane_callers(adaptive=FakeCaller("adaptive", content="not json"))
    app, store, _ = build(lanes=lanes)
    with TestClient(app) as client:
        job = finished_job(client, upload(client).json()["job_id"])
    assert job["status"] == "done"
    assert job["result"]["stored_endpoint"] == LANE_IDS["vanilla"] and job["result"]["stored_from_fallback"] is True
    assert store.saved[0][1].model_name == "model-vanilla"
    assert job["result"]["runs"][LANE_IDS["adaptive"]]["valid"] is False


def test_grounding_flags_text_not_in_the_ocr():
    payload = {**EXAMPLE, "seller": {**EXAMPLE["seller"], "phone": "0901 234 567"}}
    report = grounding_report(payload, OCR_TEXT)
    assert report["ungrounded_fields"] == ["seller.phone"]
    assert grounding_report(EXAMPLE, OCR_TEXT.lower())["ungrounded_fields"] == []


def test_ungrounded_value_is_reported_on_the_job():
    def invented_phone(request):
        return node1_answer(request, mutate=lambda p: p["seller"].update(phone="0901 234 567"))

    app, _, _ = build(lanes=lane_callers(adaptive=FakeCaller("adaptive", content=invented_phone)))
    with TestClient(app) as client:
        result = finished_job(client, upload(client).json()["job_id"])["result"]
    assert result["grounding"]["ungrounded_fields"] == ["seller.phone"]


def test_invoice_detail_has_a_signed_image_url():
    app, _store, _ = build()
    with TestClient(app) as client:
        job = finished_job(client, upload(client).json()["job_id"])
        detail = client.get(f"/api/invoices/{job['invoice_id']}", headers=HEADERS).json()
        assert client.get("/api/invoices/nope", headers=HEADERS).status_code == 404
    assert detail["image_url"].startswith("https://storage.test/uploads/")
