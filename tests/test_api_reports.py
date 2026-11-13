"""[M5-01 web console] Node 3: Q1–Q6 through Node 2 on one endpoint, then the grounded report."""

from __future__ import annotations

from src.api.questions import QuestionService
from src.api.reports import ReportService
from src.pipeline.node2_questions import QUESTIONS
from tests.api_fakes import FakeCaller, FakeStore

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
