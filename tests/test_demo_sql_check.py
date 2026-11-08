"""[M5-01 web demo] Task 5.2: Postgres SQL check for free-form questions (no database needed)."""

from __future__ import annotations

import pytest

from src.api.questions import SqlCheckError, check_demo_sql


def code_of(sql: str) -> str:
    with pytest.raises(SqlCheckError) as info:
        check_demo_sql(sql)
    return info.value.code


def test_aggregate_over_one_view_gets_limit_200():
    sql = check_demo_sql("SELECT SUM(total_amount_value) AS t FROM v_invoice_facts WHERE issued_date LIKE '2020-08-%'")
    assert sql.endswith("LIMIT 200")


def test_fences_and_trailing_semicolon_are_removed():
    sql = check_demo_sql("```sql\nSELECT COUNT(*) AS n FROM v_invoice_facts;\n```")
    assert sql == "SELECT COUNT(*) AS n FROM v_invoice_facts LIMIT 200"


def test_existing_limit_is_capped():
    assert check_demo_sql("SELECT seller_name FROM v_invoice_facts LIMIT 5").endswith("LIMIT 5")
    assert check_demo_sql("SELECT seller_name FROM v_invoice_facts LIMIT 5000").endswith("LIMIT 200")


def test_cte_aliases_and_joins_are_allowed():
    sql = ("WITH s AS (SELECT seller_name, SUM(total_amount_value) AS t FROM v_invoice_facts GROUP BY seller_name) "
           "SELECT s.seller_name, s.t FROM s ORDER BY t DESC")
    assert "LIMIT 200" in check_demo_sql(sql)
    join = ("SELECT f.seller_name, COUNT(i.line_number) AS n FROM v_invoice_facts AS f "
            "JOIN v_item_facts AS i ON i.invoice_id = f.invoice_id GROUP BY f.seller_name")
    assert check_demo_sql(join)


def test_unknown_column_is_rejected():
    assert code_of("SELECT v_invoice_facts.status FROM v_invoice_facts") == "UNKNOWN_COLUMN"
    assert code_of("SELECT description FROM v_invoice_facts") == "UNKNOWN_COLUMN"


@pytest.mark.parametrize("sql", [
    "SELECT 1; DROP VIEW v_invoice_facts",
    "DELETE FROM invoices",
    "UPDATE v_invoice_facts SET seller_name = 'x'",
    "CREATE TABLE x AS SELECT 1",
    "SELECT seller_name INTO stolen FROM v_invoice_facts",
    "SELECT seller_name FROM v_invoice_facts FOR UPDATE",
])
def test_write_attempts_are_rejected(sql):
    assert code_of(sql) in {"MULTIPLE_STATEMENTS", "NOT_SELECT", "FORBIDDEN_CLAUSE"}


@pytest.mark.parametrize("sql", [
    "SELECT * FROM pg_catalog.pg_roles",
    "SELECT table_name FROM information_schema.tables",
    "SELECT raw_json FROM invoices",
    "SELECT * FROM demo.invoices",
])
def test_catalogs_and_base_tables_are_rejected(sql):
    assert code_of(sql) == "FORBIDDEN_RELATION"


@pytest.mark.parametrize("sql", [
    "SELECT pg_sleep(6)",
    "SELECT current_setting('is_superuser') AS s FROM v_invoice_facts",
    "SELECT * FROM v_invoice_facts, generate_series(1, 1000000) AS g",
])
def test_functions_outside_the_allow_list_are_rejected(sql):
    assert code_of(sql) == "FORBIDDEN_FUNCTION"


def test_unparseable_sql_is_rejected():
    assert code_of("SELEC seller_name FRM") in {"PARSE_ERROR", "NOT_SELECT"}
    assert code_of("") == "EMPTY"


def test_ordinary_predicates_and_operators_are_allowed():
    # Regression: sqlglot models AND/OR/EXISTS as function nodes.
    sql = ("SELECT seller_name FROM v_invoice_facts WHERE issued_date LIKE '2020-08-%' AND total_amount_value > 100000 "
           "OR (seller_name ILIKE '%ANAN%' AND NOT total_amount_value IS NULL) AND item_count IN (1, 2) "
           "AND total_amount_value BETWEEN 1 AND 900000 AND seller_name NOT LIKE 'x%' "
           "AND EXISTS (SELECT 1 FROM v_item_facts WHERE v_item_facts.invoice_id = v_invoice_facts.invoice_id)")
    assert check_demo_sql(sql).endswith("LIMIT 200")
