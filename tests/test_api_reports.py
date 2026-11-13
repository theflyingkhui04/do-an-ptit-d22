"""[M5-01 web console] Node 3: Q1–Q6 through Node 2 on one endpoint, then the grounded report."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from src.api.questions import QuestionService
from src.api.report_data import TABLE_KEYS
from src.api.reports import ReportService
from src.pipeline.node2_questions import QUESTIONS
from tests.api_fakes import HEADERS, FakeCaller, FakeStore, build
from tests.test_api_report_data import sqlite_runner

SQL = {
    "Q1": ("SELECT SUM(total_amount_value) AS tong FROM v_invoice_facts WHERE issued_date LIKE '2020-08-%'", (["tong"], [[3447578.0]])),
    "Q2": ("SELECT description, SUM(quantity_value) AS sl FROM v_item_facts GROUP BY description ORDER BY sl DESC LIMIT 5", (["description", "sl"], [["Sua tuoi", 48]])),
    "Q3": ("SELECT seller_name, SUM(total_amount_value) AS tong FROM v_invoice_facts GROUP BY seller_name ORDER BY tong DESC LIMIT 5", (["seller_name", "tong"], [["MINIMART ANAN", 2904594.0]])),
    "Q4": ("SELECT COUNT(*) AS n FROM v_invoice_facts WHERE vat_amount_value > 0", (["n"], [[1]])),
    "Q5": ("SELECT issued_date, SUM(total_amount_value) AS tong FROM v_invoice_facts GROUP BY issued_date ORDER BY issued_date", (["issued_date", "tong"], [["2020-08-16", 1145548.0]])),
    "Q6": ("SELECT tax_rate, SUM(tax_amount_value) AS thue FROM v_tax_facts GROUP BY tax_rate", (["tax_rate", "thue"], [])),
}


class ReportStore(FakeStore):
    def __init__(self) -> None:
        super().__init__()
        self.answers = {sql: result for sql, result in SQL.values()}
        self.executed: list[str] = []

    def run_readonly(self, sql):
        self.executed.append(sql)
        if "COUNT(*) AS n FROM v_invoice_facts" in sql and "vat" not in sql:
            return ["n"], [[25]]
        key = next(k for k in self.answers if sql.startswith(k.split(" LIMIT")[0]))
        return self.answers[key]


def report_caller(report_text: str) -> FakeCaller:
    replies = iter([SQL[q.id][0] for q in QUESTIONS] + [report_text])
    return FakeCaller("flexsds-serve", content=lambda request: next(replies))


GOOD = ("(1) Tổng quan: tháng 08/2020 tổng chi tiêu 3.447.578 đ trên 25 hoá đơn. (2) MINIMART ANAN 2.904.594 đ. "
        "(3) 1 hoá đơn có VAT. (4) Q6 chưa có dữ liệu.")


def test_report_runs_q1_to_q6_on_the_endpoint_then_writes_from_their_rows():
    store = ReportStore()
    caller = report_caller(GOOD)
    out = ReportService(store, QuestionService(store)).run(caller)
    assert [q["question_id"] for q in out["questions"]] == [q.id for q in QUESTIONS]
    assert [q["status"] for q in out["questions"]] == ["ok", "ok", "ok", "ok", "ok", "no_data"]
    assert out["invoice_count"] == 25 and out["report"]["status"] == "ok" and out["report"]["text"] == GOOD
    assert out["number_check"] == {"numbers": 11, "ungrounded": []}
    system = caller.calls[-1]["messages"][0]["content"]
    user = caller.calls[-1]["messages"][1]["content"]
    assert "Node 3" in system and "3447578" in user and "KHÔNG CÓ DỮ LIỆU" in user
    assert set(out["timings_ms"]) == {"node2", "node3"}


def test_invented_numbers_in_the_report_are_reported():
    store = ReportStore()
    out = ReportService(store, QuestionService(store)).run(report_caller("Tổng chi tiêu là 5.000.000 đ, tăng 12%."))
    assert out["number_check"]["ungrounded"] == ["5.000.000", "12"]


def test_no_report_when_node2_failed_everywhere():
    store = ReportStore()
    caller = FakeCaller("flexsds-serve", content="DROP TABLE invoices")
    out = ReportService(store, QuestionService(store)).run(caller)
    assert {q["status"] for q in out["questions"]} == {"error"}
    assert out["report"]["status"] == "skipped_no_data" and out["report"]["text"] == ""


# ---------------------------------------------------------------- Node 3 on its own: request + fixed data
REQUEST = "Báo cáo chi tiêu và dashboard của 2 hoá đơn gần nhất"
PLAN = json.dumps({"charts": [
    {"table": "kpis", "type": "kpi", "title": "Số tổng"},
    {"table": "by_store", "type": "bar", "value": "tong_tien", "title": "Theo cửa hàng"},
    {"table": "by_date", "type": "column", "title": "Theo ngày"},
]}, ensure_ascii=False)
REPORT = "Hai hoá đơn gần nhất có tổng 117000 đ; NHA SACH CAM PHA 75000 đ, MINIMART ANAN 42000 đ."


class SnapshotStore(FakeStore):
    def __init__(self, empty: bool = False) -> None:
        super().__init__()
        self.conn, self.run = sqlite_runner()
        if empty:
            self.conn.execute("DELETE FROM invoices")
        self.executed: list[str] = []

    def run_readonly(self, sql):
        self.executed.append(sql)
        return self.run(sql)


def replies(*texts):
    it = iter(texts)
    return FakeCaller("flexsds-serve", content=lambda request: next(it))


def service(store):
    return ReportService(store, QuestionService(store))


def test_request_report_reads_fixed_tables_then_plans_charts_and_writes():
    store = SnapshotStore()
    caller = replies(f"```json\n{PLAN}\n```", REPORT)
    out = service(store).run_request(REQUEST, 2, caller)
    assert out["request"] == REQUEST and out["scope"] == {"limit": 2, "invoice_count": 2,
                                                          "label": "2 hoá đơn có ngày lập gần nhất"}
    assert [t["key"] for t in out["tables"]] == list(TABLE_KEYS)
    assert not any("v_" not in sql or "INSERT" in sql for sql in store.executed)
    assert out["dashboard"]["status"] == "ok" and out["dashboard"]["error"] is None
    assert out["dashboard"]["charts"] == [
        {"table": "kpis", "type": "kpi", "value": None, "title": "Số tổng"},
        {"table": "by_store", "type": "bar", "value": "tong_tien", "title": "Theo cửa hàng"},
        {"table": "by_date", "type": "column", "value": "tong_tien", "title": "Theo ngày"},
    ]
    assert out["report"]["status"] == "ok" and out["report"]["text"] == REPORT
    assert out["number_check"] == {"numbers": 3, "ungrounded": []}
    plan_call, report_call = caller.calls
    assert "dashboard" in plan_call["messages"][0]["content"] and REQUEST in plan_call["messages"][1]["content"]
    assert "Node 3" in report_call["messages"][0]["content"]
    user = report_call["messages"][1]["content"]
    assert REQUEST in user and "NHA SACH CAM PHA" in user and "2 hoá đơn có ngày lập gần nhất" in user
    assert "117000" in user and ".0" not in user
    assert set(out["timings_ms"]) == {"data", "dashboard", "report"}
    assert out["dashboard"]["metrics"]["tpot_ms"] == 20.0 and out["report"]["metrics"]["tpot_ms"] == 20.0


def test_bad_chart_choices_are_dropped_and_a_broken_plan_falls_back_to_default_charts():
    store = SnapshotStore()
    mixed = json.dumps({"charts": [
        {"table": "nope", "type": "bar", "title": "x"},
        {"table": "by_store", "type": "pie", "title": "x"},
        {"table": "by_store", "type": "kpi", "title": "nhiều dòng"},
        {"table": "by_store", "type": "bar", "value": "seller_name", "title": "Cửa hàng"},
    ]})
    out = service(store).run_request(REQUEST, 2, replies(mixed, REPORT))
    assert out["dashboard"]["status"] == "ok"
    assert out["dashboard"]["charts"] == [{"table": "by_store", "type": "bar", "value": "tong_tien", "title": "Cửa hàng"}]

    out = service(store).run_request(REQUEST, 2, replies("Tôi sẽ vẽ biểu đồ cột.", REPORT))
    assert out["dashboard"]["status"] == "fallback" and "JSON" in out["dashboard"]["error"]
    assert [c["table"] for c in out["dashboard"]["charts"]] == ["kpis", "by_date", "by_store", "top_items"]
    assert out["report"]["status"] == "ok"


def test_invented_numbers_in_a_request_report_are_reported():
    out = service(SnapshotStore()).run_request(REQUEST, 2, replies(PLAN, "Tổng 117000 đ, tăng 37% so với tháng trước."))
    assert out["number_check"]["ungrounded"] == ["37"]


def test_no_model_call_when_the_scope_has_no_invoice():
    caller = replies(PLAN, REPORT)
    out = service(SnapshotStore(empty=True)).run_request(REQUEST, None, caller)
    assert caller.calls == []
    assert out["scope"]["invoice_count"] == 0 and out["dashboard"]["charts"] == []
    assert out["report"]["status"] == "skipped_no_data" and out["report"]["text"] == ""


def test_endpoint_errors_are_reported_without_the_key():
    caller = FakeCaller("flexsds-serve", content="", error="HTTP 503 for token sk-secret-123")
    svc = ReportService(SnapshotStore(), QuestionService(SnapshotStore()), redact=lambda s: s.replace("sk-secret-123", "***"))
    out = svc.run_request(REQUEST, 2, caller)
    assert out["dashboard"]["status"] == "fallback" and "***" in out["dashboard"]["error"]
    assert out["report"]["status"] == "transport_error" and "sk-secret-123" not in out["report"]["error"]


class RecordingReports:
    def __init__(self) -> None:
        self.calls = []

    def run_request(self, request, limit, caller):
        self.calls.append((request, limit, caller.lane))
        return {"request": request}


def test_report_route_takes_a_request_and_a_scope():
    reports = RecordingReports()
    app, _, _ = build(reports=reports)
    with TestClient(app) as client:
        body = {"endpoint_id": "ep_0000000b", "request": f"  {REQUEST} ", "limit": 10}
        response = client.post("/api/reports", headers=HEADERS, json=body)
        assert response.status_code == 200
        assert response.json()["endpoint"] == {"endpoint_id": "ep_0000000b", "name": "static"}
        assert client.post("/api/reports", headers=HEADERS, json={**body, "limit": None}).status_code == 200
        for bad in ({**body, "request": " "}, {**body, "limit": 0}, {**body, "limit": 51}, {"endpoint_id": "ep_0000000b"}):
            assert client.post("/api/reports", headers=HEADERS, json=bad).status_code == 422, bad
    assert reports.calls == [(REQUEST, 10, "static"), (REQUEST, None, "static")]
