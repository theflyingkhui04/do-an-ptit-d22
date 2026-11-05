"""Tests for the Node 2 read-only SQL guard."""

from __future__ import annotations

import unittest

from src.pipeline.sql_guard import SqlGuardError, check_read_only, clean_sql


class CleanSqlTests(unittest.TestCase):
    def test_clean_sql_strips_fence_and_trailing_semicolon(self) -> None:
        raw = "```sql\nSELECT 1 FROM v_invoice_facts;\n```"
        self.assertEqual(clean_sql(raw), "SELECT 1 FROM v_invoice_facts")

    def test_clean_sql_plain_fence_and_whitespace(self) -> None:
        self.assertEqual(clean_sql("  ```\nSELECT 1 FROM v_tax_facts\n```  "), "SELECT 1 FROM v_tax_facts")


class CheckReadOnlyTests(unittest.TestCase):
    def test_accepts_select_on_views(self) -> None:
        sql = "SELECT seller_name, SUM(total_amount_value) FROM v_invoice_facts GROUP BY seller_name"
        self.assertEqual(check_read_only(sql), ("v_invoice_facts",))

    def test_accepts_cte_and_join(self) -> None:
        sql = (
            "WITH t AS (SELECT invoice_id FROM v_tax_facts) "
            "SELECT COUNT(*) FROM v_invoice_facts f JOIN t ON t.invoice_id = f.invoice_id"
        )
        self.assertEqual(check_read_only(sql), ("v_invoice_facts", "v_tax_facts"))

    def test_accepts_quoted_view_name(self) -> None:
        self.assertEqual(check_read_only('SELECT * FROM "v_item_facts"'), ("v_item_facts",))

    def test_keyword_inside_literal_is_allowed(self) -> None:
        sql = "SELECT * FROM v_item_facts WHERE description LIKE '%DROP%'"
        self.assertEqual(check_read_only(sql), ("v_item_facts",))

    def test_literal_with_escaped_quote_and_semicolon(self) -> None:
        sql = "SELECT * FROM v_item_facts WHERE description = 'it''s; DELETE'"
        self.assertEqual(check_read_only(sql), ("v_item_facts",))

    def test_keyword_inside_comment_is_ignored(self) -> None:
        self.assertEqual(check_read_only("SELECT 1 FROM v_item_facts -- delete later"), ("v_item_facts",))
        self.assertEqual(check_read_only("SELECT 1 /* drop */ FROM v_item_facts"), ("v_item_facts",))

    def test_rejects(self) -> None:
        cases = [
            ("", "empty"),
            ("   ", "empty"),
            ("SELECT 1 FROM v_item_facts; DELETE FROM invoices", "multiple_statements"),
            ("UPDATE invoices SET x=1", "not_select"),
            (
                "SELECT * FROM v_item_facts WHERE 1=1 UNION SELECT name FROM sqlite_master",
                "forbidden_relation",
            ),
            ("SELECT raw_json FROM v_invoice_facts", "forbidden_relation"),
            ("SELECT * FROM invoices", "forbidden_relation"),
            ("SELECT * FROM v_item_facts, invoice_items", "forbidden_relation"),
            ("SELECT * FROM some_other_table", "forbidden_relation"),
            ("WITH x AS (SELECT 1) DELETE FROM invoices", "forbidden_keyword"),
            ("SELECT load_extension('x') FROM v_item_facts", "forbidden_keyword"),
        ]
        for sql, code in cases:
            with self.subTest(sql=sql):
                with self.assertRaises(SqlGuardError) as ctx:
                    check_read_only(sql)
                self.assertEqual(ctx.exception.code, code)


if __name__ == "__main__":
    unittest.main()
