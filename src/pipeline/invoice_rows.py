"""Validated Node 1 payload -> table rows (contract ``invoice_extraction.v1``).

Shared by the SQLite loader (``scripts/init_invoice_sqlite.py``) and the
Postgres writer of the web demo, so both databases store the same values.
Validation is the caller's job: this module only maps fields to columns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

INVOICE_COLUMNS: tuple[str, ...] = (
    "invoice_id", "run_id", "schema_version", "document_type",
    "sample_id", "img_id", "image_path", "raw_text", "raw_json",
    "seller_name", "seller_address", "seller_tax_code", "seller_phone",
    "buyer_name", "buyer_address", "buyer_tax_code", "buyer_phone",
    "invoice_number", "invoice_series", "issued_at", "issued_date", "issued_time", "currency", "payment_method",
    "subtotal_raw", "subtotal_value", "subtotal_currency",
    "discount_raw", "discount_value", "discount_currency",
    "vat_amount_raw", "vat_amount_value", "vat_amount_currency",
    "total_amount_raw", "total_amount_value", "total_amount_currency",
    "extraction_status", "extraction_confidence", "missing_fields_json", "warnings_json",
    "latency_ms",
)
ITEM_COLUMNS: tuple[str, ...] = (
    "invoice_id", "line_number", "description", "sku",
    "quantity_raw", "quantity_value", "quantity_unit",
    "unit_price_raw", "unit_price_value", "unit_price_currency",
    "line_total_raw", "line_total_value", "line_total_currency",
    "vat_rate",
)
TAX_COLUMNS: tuple[str, ...] = (
    "invoice_id", "tax_name", "tax_rate",
    "taxable_amount_raw", "taxable_amount_value", "taxable_amount_currency",
    "tax_amount_raw", "tax_amount_value", "tax_amount_currency",
)
EVIDENCE_COLUMNS: tuple[str, ...] = ("invoice_id", "field_path", "raw_text", "source_label")
# JSON text columns (TEXT in SQLite, jsonb in Postgres).
JSON_COLUMNS: frozenset[str] = frozenset({"raw_json", "missing_fields_json", "warnings_json"})


@dataclass(frozen=True)
class InvoiceRows:
    invoice: dict[str, Any]
    items: list[dict[str, Any]] = field(default_factory=list)
    taxes: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)


def money_value(data: dict, name: str, key: str) -> object:
    return data.get(name, {}).get(key)


def make_invoice_id(run_id: str, payload: dict) -> str:
    source = payload.get("source", {})
    stable_id = source.get("sample_id") or source.get("img_id")
    if not stable_id:
        raise ValueError("source.sample_id or source.img_id is required to build invoice_id")
    return f"{run_id}:{stable_id}"


def build_invoice_rows(payload: dict, run_id: str, latency_ms: float | None = None) -> InvoiceRows:
    """Rows of ``invoices``, ``invoice_items``, ``invoice_taxes`` and ``field_evidence``."""
    invoice_id = make_invoice_id(run_id, payload)
    source = payload.get("source", {})
    seller = payload.get("seller", {})
    buyer = payload.get("buyer", {})
    invoice = payload.get("invoice", {})
    totals = payload.get("totals", {})
    extraction = payload.get("extraction", {})

    invoice_row: dict[str, Any] = {
        "invoice_id": invoice_id,
        "run_id": run_id,
        "schema_version": payload["schema_version"],
        "document_type": payload["document_type"],
        "sample_id": source.get("sample_id"),
        "img_id": source.get("img_id"),
        "image_path": source.get("image_path"),
        "raw_text": source.get("raw_text"),
        "raw_json": json.dumps(payload, ensure_ascii=False),
        "seller_name": seller.get("name"),
        "seller_address": seller.get("address"),
        "seller_tax_code": seller.get("tax_code"),
        "seller_phone": seller.get("phone"),
        "buyer_name": buyer.get("name"),
        "buyer_address": buyer.get("address"),
        "buyer_tax_code": buyer.get("tax_code"),
        "buyer_phone": buyer.get("phone"),
        "invoice_number": invoice.get("invoice_number"),
        "invoice_series": invoice.get("invoice_series"),
        "issued_at": invoice.get("issued_at"),
        "issued_date": invoice.get("issued_date"),
        "issued_time": invoice.get("issued_time"),
        "currency": invoice.get("currency"),
        "payment_method": invoice.get("payment_method"),
        "extraction_status": extraction.get("status", "failed"),
        "extraction_confidence": extraction.get("confidence"),
        "missing_fields_json": json.dumps(extraction.get("missing_fields", []), ensure_ascii=False),
        "warnings_json": json.dumps(extraction.get("warnings", []), ensure_ascii=False),
        "latency_ms": latency_ms,
    }
    for name in ("subtotal", "discount", "vat_amount", "total_amount"):
        for key in ("raw", "value", "currency"):
            invoice_row[f"{name}_{key}"] = money_value(totals, name, key)

    items = []
    for item in payload.get("items", []):
        row = {"invoice_id": invoice_id, "line_number": item["line_number"],
               "description": item.get("description"), "sku": item.get("sku"),
               "vat_rate": item.get("vat_rate")}
        quantity = item.get("quantity", {})
        row.update(quantity_raw=quantity.get("raw"), quantity_value=quantity.get("value"),
                   quantity_unit=quantity.get("unit"))
        for name in ("unit_price", "line_total"):
            for key in ("raw", "value", "currency"):
                row[f"{name}_{key}"] = item.get(name, {}).get(key)
        items.append({column: row[column] for column in ITEM_COLUMNS})

    taxes = []
    for tax in payload.get("taxes", []):
        row = {"invoice_id": invoice_id, "tax_name": tax.get("tax_name"), "tax_rate": tax.get("tax_rate")}
        for name in ("taxable_amount", "tax_amount"):
            for key in ("raw", "value", "currency"):
                row[f"{name}_{key}"] = tax.get(name, {}).get(key)
        taxes.append({column: row[column] for column in TAX_COLUMNS})

    evidence = [{"invoice_id": invoice_id, "field_path": e["field_path"], "raw_text": e["raw_text"],
                 "source_label": e.get("source_label")} for e in extraction.get("field_evidence", [])]
    return InvoiceRows(invoice={column: invoice_row[column] for column in INVOICE_COLUMNS},
                       items=items, taxes=taxes, evidence=evidence)
