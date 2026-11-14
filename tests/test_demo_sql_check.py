"""[M5-01 web demo] Task 5.2: Postgres SQL check for free-form questions (no database needed)."""

from __future__ import annotations

import pytest

from src.api.questions import SqlCheckError, check_demo_sql, relax_text_filters


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


def test_demo_upload_view_is_allowed_with_its_own_columns():
    sql = check_demo_sql("SELECT COUNT(*) AS n FROM v_upload_facts WHERE source = 'upload' AND uploaded_date = '2026-11-08'")
    assert sql.endswith("LIMIT 200")
    join = ("SELECT f.seller_name, u.uploaded_at FROM v_invoice_facts AS f "
            "JOIN v_upload_facts AS u ON u.invoice_id = f.invoice_id WHERE u.source = 'upload'")
    assert check_demo_sql(join)
    assert code_of("SELECT uploaded_by FROM v_upload_facts") == "UNKNOWN_COLUMN"


def test_ordinary_predicates_and_operators_are_allowed():
    # Regression: sqlglot models AND/OR/EXISTS as function nodes.
    sql = ("SELECT seller_name FROM v_invoice_facts WHERE issued_date LIKE '2020-08-%' AND total_amount_value > 100000 "
           "OR (seller_name ILIKE '%ANAN%' AND NOT total_amount_value IS NULL) AND item_count IN (1, 2) "
           "AND total_amount_value BETWEEN 1 AND 900000 AND seller_name NOT LIKE 'x%' "
           "AND EXISTS (SELECT 1 FROM v_item_facts WHERE v_item_facts.invoice_id = v_invoice_facts.invoice_id)")
    assert check_demo_sql(sql).endswith("LIMIT 200")


# [M5-01 canonical flow] fan-out joins
A2_SQL = ("SELECT SUM(vi.total_amount_value) AS tong_tien FROM v_invoice_facts AS vi JOIN v_item_facts AS fi "
          "ON vi.invoice_id = fi.invoice_id WHERE vi.seller_name ILIKE '%MINIMART ANAN%' AND vi.issued_date LIKE '2020-08-12%'")


@pytest.mark.parametrize("sql", [
    A2_SQL,
    "SELECT AVG(total_amount_value) FROM v_invoice_facts JOIN v_tax_facts ON v_tax_facts.invoice_id = v_invoice_facts.invoice_id",
    "SELECT COUNT(f.extraction_status) FROM v_invoice_facts AS f JOIN v_item_facts AS i ON i.invoice_id = f.invoice_id",
])
def test_invoice_level_aggregates_over_item_or_tax_joins_are_rejected(sql):
    assert code_of(sql) == "JOIN_FANOUT"


@pytest.mark.parametrize("sql", [
    "SELECT SUM(i.line_total_value) FROM v_invoice_facts AS f JOIN v_item_facts AS i ON i.invoice_id = f.invoice_id",
    "SELECT SUM(total_amount_value) FROM v_invoice_facts",
    "SELECT COUNT(DISTINCT f.total_amount_value) FROM v_invoice_facts AS f JOIN v_item_facts AS i ON i.invoice_id = f.invoice_id",
    "SELECT COUNT(*) FROM v_invoice_facts AS f JOIN v_item_facts AS i ON i.invoice_id = f.invoice_id",
    "SELECT SUM(f.total_amount_value) FROM v_invoice_facts AS f JOIN v_upload_facts AS u ON u.invoice_id = f.invoice_id",
    ("SELECT SUM(t) AS s FROM (SELECT f.total_amount_value AS t FROM v_invoice_facts AS f WHERE EXISTS "
     "(SELECT 1 FROM v_item_facts AS i WHERE i.invoice_id = f.invoice_id)) AS x"),
])
def test_aggregates_without_fanout_stay_allowed(sql):
    assert check_demo_sql(sql)


# ---------------------------------------------------------------- 14/11: c12 and c02/c06/c08 of canonical_questions_v1
@pytest.mark.parametrize("op", ["UNION", "UNION ALL", "INTERSECT", "EXCEPT"])
def test_compound_selects_go_back_for_one_select(op):
    sql = (f"SELECT COUNT(*) AS n FROM v_invoice_facts WHERE issued_date LIKE '2020-08-10%' {op} "
           "SELECT COUNT(*) AS n FROM v_invoice_facts WHERE issued_date LIKE '2020-08-11%'")
    with pytest.raises(SqlCheckError) as info:
        check_demo_sql(sql)
    assert info.value.code == "COMPOUND_SELECT" and "BETWEEN" in str(info.value)


KNOWN = {"seller_name": ["NHA SACH GD-TC CAM PHA", "THUC COFFEE", "MINIMART ANAN", "MINIMART"]}


def test_text_filters_lose_their_marks_and_match_word_by_word():
    sql, notes = relax_text_filters("SELECT SUM(total_amount_value) AS t FROM v_invoice_facts "
                                    "WHERE seller_name ILIKE '%Nhà sách Cẩm Phả%' LIMIT 200", KNOWN)
    assert "seller_name ILIKE '%nha%sach%cam%pha%'" in sql and sql.endswith("LIMIT 200")
    assert notes == [{"column": "seller_name", "from": "%Nhà sách Cẩm Phả%", "to": "%nha%sach%cam%pha%"}]


def test_a_store_name_keeps_only_the_words_found_in_a_real_name():
    sql, notes = relax_text_filters("SELECT seller_address FROM v_invoice_facts "
                                    "WHERE seller_name ILIKE '%Quán Thức Coffee%'", KNOWN)
    assert "seller_name ILIKE '%thuc%coffee%'" in sql
    assert notes[0]["to"] == "%thuc%coffee%"


def test_item_text_has_no_known_list_and_only_loses_marks():
    sql, _ = relax_text_filters("SELECT COUNT(*) AS n FROM v_item_facts WHERE description LIKE '%bánh%'", KNOWN)
    assert "description ILIKE '%banh%'" in sql


def test_equality_on_a_name_becomes_a_word_match():
    sql, _ = relax_text_filters("SELECT COUNT(*) AS n FROM v_invoice_facts WHERE seller_name = 'Nhà sách Cẩm Phả'", KNOWN)
    assert "seller_name ILIKE '%nha%sach%cam%pha%'" in sql


def test_dates_numbers_and_plain_filters_are_left_untouched():
    sql = ("SELECT COUNT(*) AS n FROM v_invoice_facts WHERE issued_date LIKE '2020-08-1%' "
           "AND seller_name ILIKE '%minimart%' AND currency = 'VND' LIMIT 200")
    assert relax_text_filters(sql, KNOWN) == (sql, [])


def test_a_name_with_no_known_match_keeps_all_its_words():
    sql, _ = relax_text_filters("SELECT COUNT(*) AS n FROM v_invoice_facts WHERE seller_name ILIKE '%Bách Hoá Xanh%'", KNOWN)
    assert "'%bach%hoa%xanh%'" in sql
