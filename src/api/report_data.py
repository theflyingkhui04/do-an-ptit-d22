"""Node 3 input data for the web console: fixed read-only queries over the demo views for a chosen scope.

Node 3 is its own task (W3: a business request plus the facts to answer it), so it never calls Node 2.
These queries are written here, not by a model; they run on the read-only role and work on both the
SQLite v2 views and their Postgres mirror. The model then chooses charts from these tables and writes the
report; every number on the web comes from the rows below.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

MAX_SCOPE = 50
TOP_ITEMS = 10

Runner = Callable[[str], tuple[list[str], list[list[Any]]]]

# key -> (title shown on the web and given to the model, query over ``scope``)
TABLES: dict[str, tuple[str, str]] = {
    "kpis": ("Số tổng của phạm vi",
             ("SELECT COUNT(*) AS so_hoa_don, SUM(f.total_amount_value) AS tong_tien, "
             "ROUND(AVG(f.total_amount_value)) AS trung_binh, "
             "SUM(CASE WHEN f.vat_amount_value > 0 THEN 1 ELSE 0 END) AS hoa_don_co_vat "
             "FROM v_invoice_facts f JOIN scope s ON s.invoice_id = f.invoice_id")),
    "invoices": ("Danh sách hoá đơn",
                 ("SELECT f.issued_date, f.seller_name, f.total_amount_value AS tong_tien, f.item_count AS so_mat_hang "
                 "FROM v_invoice_facts f JOIN scope s ON s.invoice_id = f.invoice_id "
                 "ORDER BY f.issued_date DESC NULLS LAST, f.invoice_id")),
    "by_store": ("Tổng tiền theo cửa hàng",
                 ("SELECT f.seller_name, COUNT(*) AS so_hoa_don, SUM(f.total_amount_value) AS tong_tien "
                 "FROM v_invoice_facts f JOIN scope s ON s.invoice_id = f.invoice_id "
                 "GROUP BY f.seller_name ORDER BY tong_tien DESC NULLS LAST, f.seller_name")),
    "by_date": ("Tổng tiền theo ngày lập",
                ("SELECT f.issued_date, COUNT(*) AS so_hoa_don, SUM(f.total_amount_value) AS tong_tien "
                "FROM v_invoice_facts f JOIN scope s ON s.invoice_id = f.invoice_id "
                "WHERE f.issued_date IS NOT NULL GROUP BY f.issued_date ORDER BY f.issued_date")),
    "top_items": ("Mặt hàng mua nhiều nhất",
                  ("SELECT i.description, SUM(i.quantity_value) AS tong_so_luong, SUM(i.line_total_value) AS tong_thanh_tien "
                  "FROM v_item_facts i JOIN scope s ON s.invoice_id = i.invoice_id "
                  "GROUP BY i.description ORDER BY tong_so_luong DESC NULLS LAST, tong_thanh_tien DESC NULLS LAST, "
                  f"i.description LIMIT {TOP_ITEMS}")),
    "taxes": ("Thuế theo loại và thuế suất",
              ("SELECT t.tax_name, t.tax_rate, COUNT(DISTINCT t.invoice_id) AS so_hoa_don, SUM(t.tax_amount_value) AS tong_thue "
              "FROM v_tax_facts t JOIN scope s ON s.invoice_id = t.invoice_id "
              "GROUP BY t.tax_name, t.tax_rate ORDER BY tong_thue DESC NULLS LAST, t.tax_rate")),
}
TABLE_KEYS = tuple(TABLES)


def check_limit(limit: int | None) -> int | None:
    if limit is None:
        return None
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_SCOPE:
        raise ValueError(f"scope limit must be 1..{MAX_SCOPE} or None, got {limit!r}")
    return limit


def scope_label(limit: int | None) -> str:
    return "tất cả hoá đơn đã lưu" if limit is None else f"{limit} hoá đơn có ngày lập gần nhất"


def _scope_cte(limit: int | None) -> str:
    tail = "" if limit is None else f" LIMIT {limit}"
    return ("WITH scope AS (SELECT invoice_id FROM v_invoice_facts "
            f"ORDER BY issued_date DESC NULLS LAST, invoice_id{tail}) ")


def build_snapshot(run: Runner, limit: int | None) -> dict[str, Any]:
    """Tables of the scope (``limit`` latest invoices by issue date, or all), in ``TABLE_KEYS`` order."""
    limit = check_limit(limit)
    cte = _scope_cte(limit)
    tables = []
    for key, (title, sql) in TABLES.items():
        columns, rows = run(cte + sql)
        tables.append({"key": key, "title": title, "columns": columns, "rows": rows})
    kpis = tables[0]["rows"]
    count = int(kpis[0][0]) if kpis and kpis[0][0] is not None else 0
    return {"limit": limit, "invoice_count": count, "tables": tables}
