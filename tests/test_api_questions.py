"""[M5-01 web demo] Tasks 5.1 and 5.3: grounded Node 2 prompt, question flow, number check."""

from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.questions import (
    NO_DATA_ANSWER,
    QuestionService,
    build_node2_prompt,
    numbers_are_grounded,
)
from tests.api_fakes import HEADERS, FakeCaller, FakeStore, build

REPO_ROOT = Path(__file__).resolve().parents[1]
W2_PROMPT = REPO_ROOT / "experiments/configs/w2_v2/system_prompt.md"
W2_PROMPT_SHA256 = "40a0bff9f8a9701e992efeac38f60d48094bc5376649990ba5b22975e413605b"  # frozen w2_v2 prompt, LF bytes
CRLF, LF = bytes([13, 10]), bytes([10])


class QAStore(FakeStore):
    def __init__(self, results: list) -> None:
        super().__init__()
        self.results = list(results)
        self.executed: list[str] = []

    def categorical_values(self):
        return {"extraction_status": ["partial"], "document_type": ["retail_receipt"], "currency": ["VND"],
                "payment_method": [], "tax_name": ["VAT"]}

    def run_readonly(self, sql):
        self.executed.append(sql)
        outcome = self.results.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def scripted(*replies: str) -> FakeCaller:
    queue = list(replies)
    return FakeCaller("static", content=lambda request: queue.pop(0))


def service(store, sql_caller, answer_caller=None) -> QuestionService:
    # SQL on lane static, short answer on lane vanilla, so the two scripted callers stay apart.
    callers = {"static": sql_caller, "vanilla": answer_caller or scripted("Không dùng.")}
    return QuestionService(store, lane_caller=callers.__getitem__,
                           lane_map={"node1": "adaptive", "node2": "static", "answer": "vanilla", "report": "eagle"})


# ---------------------------------------------------------------- 5.1 prompt
def test_prompt_lists_the_values_present_in_the_database():
    prompt = build_node2_prompt({"extraction_status": ["partial"], "currency": ["VND"]})
    assert "- extraction_status: 'partial'" in prompt
    assert "'success'" not in prompt
    assert "v_invoice_facts(invoice_id, run_id," in prompt and "v_tax_facts(" in prompt
    assert "{{" not in prompt


def test_frozen_w2_prompt_is_unchanged():
    build_node2_prompt({"extraction_status": ["partial"]})
    data = W2_PROMPT.read_bytes().replace(CRLF, LF)
    assert hashlib.sha256(data).hexdigest() == W2_PROMPT_SHA256


# ---------------------------------------------------------------- 5.3 flow
def test_answerable_question():
    store = QAStore([(["total"], [[4930094.0]])])
    qa = service(store, scripted("SELECT SUM(total_amount_value) AS total FROM v_invoice_facts "
                                 "WHERE issued_date LIKE '2020-08-%'"),
                 scripted("Tổng tiền các hoá đơn tháng 8 năm 2020 là 4.930.094 đồng."))
    result = qa.answer("Tổng tiền các hoá đơn tháng 8 năm 2020 là bao nhiêu?")
    assert result["status"] == "ok" and result["warning"] is None
    assert result["sql"].startswith("SELECT SUM(total_amount_value)") and result["sql"].endswith("LIMIT 200")
    assert result["rows"] == [[4930094.0]] and result["answer"].endswith("4.930.094 đồng.")
    assert result["repairs"] == 0
    assert [c["step"] for c in result["calls"]] == ["node2", "answer"]
    assert result["prompt_versions"] == {"node2": "node2_text_to_sql_demo.v1", "answer": "node3_answer_demo.v1"}


def test_rejected_sql_is_repaired_once_and_never_executed():
    store = QAStore([(["n"], [[29]])])
    qa = service(store, scripted("SELECT status FROM v_invoice_facts", "SELECT COUNT(*) AS n FROM v_invoice_facts"),
                 scripted("Có 29 hoá đơn."))
    result = qa.answer("Có bao nhiêu hoá đơn?")
    assert result["status"] == "ok" and result["repairs"] == 1
    assert result["first_error"].startswith("UNKNOWN_COLUMN")
    assert store.executed == ["SELECT COUNT(*) AS n FROM v_invoice_facts LIMIT 200"]


def test_database_error_is_repaired_once():
    store = QAStore([RuntimeError("operator does not exist: text > integer"), (["n"], [[3]])])
    qa = service(store, scripted("SELECT COUNT(*) AS n FROM v_invoice_facts WHERE issued_date > 5",
                                 "SELECT COUNT(*) AS n FROM v_invoice_facts WHERE issued_date > '2020-08-05'"),
                 scripted("Có 3 hoá đơn."))
    result = qa.answer("Có bao nhiêu hoá đơn sau ngày 5/8/2020?")
    assert result["status"] == "ok" and result["repairs"] == 1 and len(store.executed) == 2


def test_still_failing_after_repair_is_an_error_without_answer():
    store = QAStore([RuntimeError("syntax error"), RuntimeError("syntax error again")])
    qa = service(store, scripted("SELECT COUNT(*) AS n FROM v_invoice_facts", "SELECT COUNT(*) AS n FROM v_invoice_facts"))
    result = qa.answer("Có bao nhiêu hoá đơn?")
    assert result["status"] == "error" and result["answer"] is None
    assert "syntax error again" in result["error"]


def test_zero_rows_is_no_data_without_calling_the_answer_model():
    answer = scripted()
    store = QAStore([(["seller_name"], [])])
    qa = service(store, scripted("SELECT seller_name FROM v_invoice_facts WHERE issued_date LIKE '2030-%'"), answer)
    result = qa.answer("Hoá đơn năm 2030 của ai?")
    assert result["status"] == "no_data" and result["answer"] == NO_DATA_ANSWER
    assert answer.calls == []


def test_fabricated_number_withholds_the_answer():
    store = QAStore([(["total"], [[4930094.0]])])
    qa = service(store, scripted("SELECT SUM(total_amount_value) AS total FROM v_invoice_facts"),
                 scripted("Tổng là 4.930.095 đồng."))
    result = qa.answer("Tổng tiền là bao nhiêu?")
    assert result["status"] == "ok" and result["answer"] is None
    assert result["warning"] == "fabricated_number" and result["rows"] == [[4930094.0]]


def test_number_grounding_rules():
    rows = [[4930094.0, "2020-08-10", 170003.2413]]
    assert numbers_are_grounded("Tổng 4.930.094 đồng ngày 10/08/2020", rows, "câu hỏi")
    assert numbers_are_grounded("Trung bình 170.003 đồng", rows, "")
    assert numbers_are_grounded("Trung bình 170.003,24 đồng", rows, "")
    assert numbers_are_grounded("Có 1 dòng", rows, "")                    # row count
    assert numbers_are_grounded("Trong tháng 8 năm 2020", [[1]], "tháng 8 năm 2020")  # from the question
    assert not numbers_are_grounded("Tổng 4.930.095 đồng", rows, "")
    assert not numbers_are_grounded("Khoảng 5 triệu", rows, "")


def test_question_endpoint_validation_and_wiring():
    store = QAStore([(["n"], [[29]])])
    qa = service(store, scripted("SELECT COUNT(*) AS n FROM v_invoice_facts"), scripted("Có 29 hoá đơn."))
    app, _, _ = build(store=store, questions=qa)
    with TestClient(app) as client:
        assert client.post("/api/questions", headers=HEADERS, json={"question": ""}).status_code == 422
        assert client.post("/api/questions", headers=HEADERS, json={"question": "x" * 501}).status_code == 422
        body = client.post("/api/questions", headers=HEADERS, json={"question": " Có bao nhiêu hoá đơn? "}).json()
    assert body["status"] == "ok" and body["answer"] == "Có 29 hoá đơn."
    assert body["question"] == "Có bao nhiêu hoá đơn?"
