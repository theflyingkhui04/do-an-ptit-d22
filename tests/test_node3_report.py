"""Tests for the Node 3 VAT report writer (fake LLM client)."""

from __future__ import annotations

import unittest

from src.pipeline.openai_compatible import OpenAICompatibleError
from src.pipeline.report_writer import (
    W3_SYSTEM_PROMPT_PATH,
    Node3Writer,
    build_user_message,
    markdown_table,
    render_report_md,
)
from src.pipeline.text_to_sql import Node2Result, load_system_prompt


class FakeClient:
    def __init__(self, content: str | Exception = "Báo cáo thử.") -> None:
        self.content = content
        self.payloads: list[dict] = []

    def chat_completions(self, payload: dict) -> dict:
        self.payloads.append(payload)
        if isinstance(self.content, Exception):
            raise self.content
        return {
            "choices": [{"message": {"content": self.content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 300},
        }


def result(qid: str, status: str = "ok") -> Node2Result:
    r = Node2Result(question_id=qid, question=f"Câu {qid}?", sql=f"SELECT 1 FROM v_invoice_facts -- {qid}", status=status)
    if status == "ok":
        r.columns = ["seller_name", "total"]
        r.rows = [["MINIMART ANAN", 42000.0]]
        r.row_count = 1
    return r


def six_results() -> list[Node2Result]:
    statuses = {"Q4": "empty", "Q6": "sql_error"}
    return [result(f"Q{i}", statuses.get(f"Q{i}", "ok")) for i in range(1, 7)]


class MarkdownTableTests(unittest.TestCase):
    def test_markdown_table_escapes(self) -> None:
        table = markdown_table(["a", "b"], [["x|y", None], ["Bánh mì", 1.5]])
        lines = table.splitlines()
        self.assertEqual(len(lines), 4)
        self.assertIn("x\\|y", table)
        self.assertIn("NULL", table)
        self.assertIn("Bánh mì", table)
        self.assertIn("1.5", table)

    def test_markdown_table_flattens_newlines(self) -> None:
        table = markdown_table(["a"], [["dòng 1\ndòng 2"]])
        self.assertEqual(len(table.splitlines()), 3)

    def test_markdown_table_truncates(self) -> None:
        table = markdown_table(["n"], [[i] for i in range(25)], limit=20)
        self.assertIn("(còn 5 dòng)", table)
        self.assertNotIn("| 20 |", table)


class BuildUserMessageTests(unittest.TestCase):
    def test_user_message_marks_failures(self) -> None:
        message = build_user_message(six_results(), invoice_count=30)
        self.assertIn("30 hóa đơn", message)
        self.assertIn("### Q1.", message)
        self.assertIn("### Q6.", message)
        self.assertIn("KHÔNG CÓ DỮ LIỆU", message)
        self.assertIn("LỖI TRUY VẤN: sql_error", message)
        self.assertIn("MINIMART ANAN", message)
        self.assertNotIn("{question_blocks}", message)
        self.assertNotIn("{invoice_count}", message)

    def test_user_message_treats_zero_as_result_and_forbids_speculation(self) -> None:
        # M3-06 run L4: a COUNT of 0 was reported as "no data" and the model
        # guessed causes ("tables not updated") that are not in the data.
        message = build_user_message(six_results(), invoice_count=30)
        self.assertIn("Giá trị 0 là kết quả hợp lệ", message)
        self.assertIn("không suy đoán nguyên nhân", message)

    def test_missing_sql_rendered(self) -> None:
        r = Node2Result(question_id="Q1", question="Câu?", status="transport_error")
        message = build_user_message([r], invoice_count=1)
        self.assertIn("không có", message)
        self.assertIn("LỖI TRUY VẤN: transport_error", message)


class Node3WriterTests(unittest.TestCase):
    def test_write_ok_payload(self) -> None:
        client = FakeClient()
        system = load_system_prompt(W3_SYSTEM_PROMPT_PATH)
        out = Node3Writer(client, "m", system).write(six_results(), invoice_count=30)
        self.assertEqual(out.status, "ok")
        self.assertEqual(out.report_text, "Báo cáo thử.")
        self.assertEqual(out.input_question_ids, ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"])
        self.assertEqual(out.prompt_tokens, 900)
        self.assertEqual(out.finish_reason, "stop")
        payload = client.payloads[0]
        self.assertEqual(payload["max_tokens"], 768)
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["seed"], 42)
        self.assertEqual(payload["messages"][0]["content"], system)
        self.assertTrue(system.startswith("You are Node 3 business-analysis writer"))

    def test_skipped_when_no_data(self) -> None:
        client = FakeClient()
        results = [result(f"Q{i}", "sql_error") for i in range(1, 7)]
        out = Node3Writer(client, "m", "S").write(results, invoice_count=30)
        self.assertEqual(out.status, "skipped_no_data")
        self.assertEqual(client.payloads, [])

    def test_transport_error(self) -> None:
        out = Node3Writer(FakeClient(OpenAICompatibleError("down")), "m", "S").write(six_results(), 30)
        self.assertEqual(out.status, "transport_error")
        self.assertIsNone(out.report_text)

    def test_raw_network_error_is_transport_error(self) -> None:
        out = Node3Writer(FakeClient(TimeoutError("read timed out")), "m", "S").write(six_results(), 30)
        self.assertEqual(out.status, "transport_error")

    def test_empty_content_is_not_ok(self) -> None:
        out = Node3Writer(FakeClient("   "), "m", "S").write(six_results(), 30)
        self.assertEqual(out.status, "empty_response")

    def test_render_report_md(self) -> None:
        out = Node3Writer(FakeClient(), "m", "S").write(six_results(), 30)
        md = render_report_md(out, six_results(), {"run_id": "r1", "lane": "vanilla", "model": "m"})
        self.assertIn("Báo cáo thử.", md)
        self.assertIn("## Phụ lục: kết quả Q1–Q6", md)
        self.assertIn("r1", md)
        self.assertIn("MINIMART ANAN", md)


if __name__ == "__main__":
    unittest.main()
