"""[M5-01 web console] Node 3 input data: fixed read-only queries over the demo views for a chosen scope.

Node 3 does not depend on Node 2: the backend fetches the numbers with these fixed queries, the model only
chooses charts and writes the report. The same queries must give the same rows on SQLite and Postgres.
"""

from __future__ import annotations

import sqlite3

import pytest

from src.api.report_data import TABLE_KEYS, build_snapshot, scope_label
from tests.test_demo_store import (  # noqa: F401  (pg is a fixture)
    DB_URL,
    RUN,
    loader,
    needs_db,
    pg,
    sqlite_conn,
    writer,
)
from tests.test_invoice_rows import payload_with_items_and_tax


def invoice(sample_id: str, issued_date: str | None, seller: str, total: float, vat: float | None,
            with_items: bool = True) -> dict:
    payload = payload_with_items_and_tax()
    payload["source"]["sample_id"] = sample_id
    payload["source"]["img_id"] = f"{sample_id}.jpg"
    payload["invoice"]["issued_date"] = issued_date
    payload["invoice"]["issued_time"] = "17:12:00" if issued_date else None
    payload["invoice"]["issued_at"] = f"{issued_date}T17:12:00" if issued_date else None
    payload["seller"]["name"] = seller
    payload["totals"]["total_amount"]["value"] = total
    payload["totals"]["vat_amount"]["value"] = vat
    if not with_items:
        payload["items"] = []
        payload["taxes"] = []
    return payload


INVOICES = [
    invoice("s_a", "2020-08-10", "MINIMART ANAN", 42000, 2000),
    invoice("s_b", "2020-08-12", "NHA SACH CAM PHA", 75000, None, with_items=False),
    invoice("s_c", None, "CUA HANG NAM OANH", 16500, None, with_items=False),
]


def sqlite_runner() -> tuple[sqlite3.Connection, callable]:
    conn = sqlite_conn()
    for payload in INVOICES:
        loader.insert_extraction(conn, payload, "r1", replace=False, latency_ms=10.0)

    def run(sql: str):
        cur = conn.execute(sql)
        return [d[0] for d in cur.description], [list(row) for row in cur.fetchall()]

    return conn, run


def table(snapshot: dict, key: str) -> dict:
    return next(t for t in snapshot["tables"] if t["key"] == key)


def test_latest_scope_keeps_the_most_recent_issue_dates():
    _, run = sqlite_runner()
    snap = build_snapshot(run, 2)
    assert snap["limit"] == 2 and snap["invoice_count"] == 2
    assert [t["key"] for t in snap["tables"]] == list(TABLE_KEYS)
    assert table(snap, "kpis")["columns"] == ["so_hoa_don", "tong_tien", "trung_binh", "hoa_don_co_vat"]
    assert table(snap, "kpis")["rows"] == [[2, 117000, 58500, 1]]
    assert [r[:3] for r in table(snap, "invoices")["rows"]] == [["2020-08-12", "NHA SACH CAM PHA", 75000],
                                                               ["2020-08-10", "MINIMART ANAN", 42000]]
    assert table(snap, "by_store")["rows"] == [["NHA SACH CAM PHA", 1, 75000], ["MINIMART ANAN", 1, 42000]]
    assert table(snap, "by_date")["rows"] == [["2020-08-10", 1, 42000], ["2020-08-12", 1, 75000]]
    assert table(snap, "top_items")["rows"] == [["Sữa tươi", 2, 20000], ["Bánh mì", None, 22000]]
    assert table(snap, "taxes")["rows"] == [["VAT", 10, 1, 2000]]
    assert all(t["title"] for t in snap["tables"])


def test_all_scope_counts_every_invoice_but_dates_skip_missing_ones():
    _, run = sqlite_runner()
    snap = build_snapshot(run, None)
    assert snap["invoice_count"] == 3
    assert table(snap, "kpis")["rows"][0][:2] == [3, 133500]
    assert [r[0] for r in table(snap, "by_date")["rows"]] == ["2020-08-10", "2020-08-12"]
    assert table(snap, "invoices")["rows"][-1][1] == "CUA HANG NAM OANH"


@pytest.mark.parametrize("limit", [0, -1, 51, True])
def test_scope_limit_is_checked(limit):
    with pytest.raises(ValueError):
        build_snapshot(lambda sql: ([], []), limit)


def test_scope_label():
    assert scope_label(10) == "10 hoá đơn có ngày lập gần nhất"
    assert scope_label(None) == "tất cả hoá đơn đã lưu"


@needs_db
def test_same_snapshot_on_postgres_as_on_sqlite(pg):  # noqa: F811  (pytest fixture)
    from psycopg.conninfo import make_conninfo

    from src.api.store import PostgresStore

    for payload in INVOICES:
        writer.insert_extraction(pg, payload, RUN, source="devset")
    store = PostgresStore(DB_URL, make_conninfo(DB_URL, user="demo_reader"))
    _, lite = sqlite_runner()
    for limit in (2, None):
        assert build_snapshot(store.run_readonly, limit) == build_snapshot(lite, limit)


def test_whole_numbers_come_back_as_integers():
    """SQLite and Postgres return 117000.0 for sums of money; the report prompt must read 117000, not 117000.0."""
    _, run = sqlite_runner()
    snap = build_snapshot(run, 2)
    assert [type(v) for v in table(snap, "kpis")["rows"][0]] == [int, int, int, int]
    assert [type(v) for v in table(snap, "by_store")["rows"][0]] == [str, int, int]
