"""Tests for the Node 2 Text-to-SQL runner (fake LLM client, real SQLite)."""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from init_invoice_sqlite import connect, init_schema, insert_extraction  # noqa: E402

from src.pipeline.node2_questions import QUESTIONS, Node2Question  # noqa: E402
from src.pipeline.openai_compatible import OpenAICompatibleError  # noqa: E402
from src.pipeline.text_to_sql import (  # noqa: E402
    W2_SYSTEM_PROMPT_PATH,
    Node2Runner,
    load_system_prompt,
    open_read_only,
)

EXAMPLE_JSON = REPO_ROOT / "src/pipeline/examples/node1_extractor_output_m0_04_001.json"
SCHEMA_SQL = REPO_ROOT / "src/pipeline/sqlite/schema.sql"


def build_example_db(db_path: Path, run_id: str = "t") -> None:
    conn = connect(db_path)
    try:
        init_schema(conn, SCHEMA_SQL)
        payload = json.loads(EXAMPLE_JSON.read_text(encoding="utf-8"))
        insert_extraction(conn, payload, run_id, replace=False)
        conn.commit()
    finally:
        conn.close()


class FakeClient:
    """Returns queued assistant contents (or raises queued exceptions)."""

    def __init__(self, contents: list) -> None:
        self.contents = list(contents)
        self.payloads: list[dict] = []

    def chat_completions(self, payload: dict) -> dict:
        self.payloads.append(payload)
        item = self.contents.pop(0)
        if isinstance(item, Exception):
            raise item
        return {
            "choices": [{"message": {"role": "assistant", "content": item}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }


Q = Node2Question(id="QX", question="Câu hỏi thử?", preferred_views=("v_invoice_facts",))


class Node2RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "invoices.sqlite"
        build_example_db(self.db_path)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def runner(self, contents: list, **kwargs) -> tuple[Node2Runner, FakeClient]:
        client = FakeClient(contents)
        return Node2Runner(client, "m", "SYSTEM", self.db_path, **kwargs), client

    def test_questions_catalog(self) -> None:
        self.assertEqual([q.id for q in QUESTIONS], ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"])
        self.assertEqual(QUESTIONS[0].question, "Tổng chi tiêu trong tháng 08/2020 là bao nhiêu?")
        self.assertEqual(QUESTIONS[3].preferred_views, ("v_invoice_facts", "v_tax_facts"))

    def test_ok_with_fence(self) -> None:
        runner, _ = self.runner(
            ["```sql\nSELECT seller_name, total_amount_value FROM v_invoice_facts;\n```"]
        )
        result = runner.run_question(Q)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.row_count, 1)
        self.assertEqual(result.views_used, ["v_invoice_facts"])
        self.assertEqual(result.columns, ["seller_name", "total_amount_value"])
        self.assertIsNone(result.explanation)
        self.assertEqual(result.sql, "SELECT seller_name, total_amount_value FROM v_invoice_facts")
        self.assertEqual(result.prompt_tokens, 100)
        self.assertEqual(result.completion_tokens, 20)
        self.assertEqual(result.finish_reason, "stop")
        self.assertIsNotNone(result.latency_ms)

    def test_payload_shape(self) -> None:
        runner, client = self.runner(["SELECT 1 FROM v_invoice_facts"])
        runner.run_question(Q)
        payload = client.payloads[0]
        self.assertEqual(payload["model"], "m")
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["seed"], 42)
        self.assertEqual(payload["max_tokens"], 256)
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "SYSTEM"})
        self.assertEqual(payload["messages"][1], {"role": "user", "content": Q.question})

    def test_empty_result(self) -> None:
        runner, _ = self.runner(["SELECT * FROM v_tax_facts"])
        result = runner.run_question(Q)
        self.assertEqual(result.status, "empty")
        self.assertEqual(result.row_count, 0)

    def test_aggregate_over_no_rows_is_empty(self) -> None:
        # SUM over zero matching rows returns one all-NULL row (M3-06 run L1).
        runner, _ = self.runner(
            ["SELECT SUM(total_amount_value) FROM v_invoice_facts WHERE extraction_status = 'Success'"]
        )
        result = runner.run_question(Q)
        self.assertEqual(result.status, "empty")
        self.assertEqual(result.rows, [[None]])

    def test_zero_count_stays_ok(self) -> None:
        runner, _ = self.runner(["SELECT COUNT(*) FROM v_tax_facts"])
        result = runner.run_question(Q)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.rows, [[0]])

    def test_row_with_some_nulls_stays_ok(self) -> None:
        runner, _ = self.runner(["SELECT seller_name, NULL AS x FROM v_invoice_facts"])
        self.assertEqual(runner.run_question(Q).status, "ok")

    def test_guard_rejected(self) -> None:
        runner, _ = self.runner(["DELETE FROM invoices"])
        result = runner.run_question(Q)
        self.assertEqual(result.status, "guard_rejected")
        self.assertIn("not_select", result.error)
        conn = sqlite3.connect(self.db_path)
        try:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM invoices").fetchone()[0], 1)
        finally:
            conn.close()

    def test_sql_error(self) -> None:
        runner, _ = self.runner(["SELECT nope FROM v_invoice_facts"])
        result = runner.run_question(Q)
        self.assertEqual(result.status, "sql_error")
        self.assertIn("nope", result.error)

    def test_transport_error(self) -> None:
        runner, _ = self.runner([OpenAICompatibleError("HTTP 500")])
        result = runner.run_question(Q)
        self.assertEqual(result.status, "transport_error")
        self.assertIsNone(result.sql)

    def test_raw_network_errors_are_transport_errors(self) -> None:
        for exc in (TimeoutError("read timed out"), ConnectionResetError("reset")):
            with self.subTest(exc=type(exc).__name__):
                runner, _ = self.runner([exc])
                result = runner.run_question(Q)
                self.assertEqual(result.status, "transport_error")
                self.assertIn(type(exc).__name__, result.error)

    def test_row_limit(self) -> None:
        runner, _ = self.runner(
            ["SELECT 1 FROM v_invoice_facts UNION ALL SELECT 2 FROM v_invoice_facts"], row_limit=1
        )
        result = runner.run_question(Q)
        self.assertEqual(result.row_count, 1)
        self.assertTrue(result.row_count_truncated)

    def test_timeout(self) -> None:
        slow = (
            "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) "
            "SELECT COUNT(*) FROM c, v_invoice_facts"
        )
        runner, _ = self.runner([slow], query_timeout_s=0.05)
        result = runner.run_question(Q)
        self.assertEqual(result.status, "timeout")

    def test_db_is_read_only(self) -> None:
        conn = open_read_only(self.db_path)
        try:
            with self.assertRaises(sqlite3.OperationalError):
                conn.execute("DELETE FROM invoices")
        finally:
            conn.close()

    def test_run_all_continues_after_failure(self) -> None:
        runner, _ = self.runner([OpenAICompatibleError("down"), "SELECT 1 FROM v_invoice_facts"])
        results = runner.run_all([Q, Q])
        self.assertEqual([r.status for r in results], ["transport_error", "ok"])

    def test_to_dict_keeps_unicode(self) -> None:
        runner, _ = self.runner(["SELECT seller_name FROM v_invoice_facts"])
        data = runner.run_question(Q).to_dict()
        self.assertEqual(data["question_id"], "QX")
        self.assertIn("Câu hỏi", json.dumps(data, ensure_ascii=False))

    def test_load_w2_prompt(self) -> None:
        self.assertTrue(load_system_prompt(W2_SYSTEM_PROMPT_PATH).startswith("You are Node 2 Text-to-SQL"))


if __name__ == "__main__":
    unittest.main()
