PRAGMA foreign_keys = ON;

BEGIN;

CREATE TABLE IF NOT EXISTS extraction_runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    completed_at TEXT,
    lane TEXT CHECK (lane IS NULL OR lane IN ('vanilla', 'static', 'adaptive', 'eagle')),
    model_name TEXT,
    prompt_version TEXT,
    schema_version TEXT NOT NULL DEFAULT 'invoice_extraction.v1',
    dataset_manifest_path TEXT,
    notes TEXT,
    latency_ms_median REAL CHECK (latency_ms_median IS NULL OR latency_ms_median >= 0)
);

CREATE TABLE IF NOT EXISTS invoices (
    invoice_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES extraction_runs(run_id) ON DELETE CASCADE,
    schema_version TEXT NOT NULL CHECK (schema_version = 'invoice_extraction.v1'),
    document_type TEXT NOT NULL CHECK (document_type IN ('retail_receipt', 'vat_invoice', 'unknown')),

    sample_id TEXT NOT NULL CHECK (length(sample_id) > 0),
    img_id TEXT,
    image_path TEXT,
    raw_text TEXT NOT NULL,
    raw_json TEXT NOT NULL CHECK (json_valid(raw_json) AND json_type(raw_json) = 'object'),

    seller_name TEXT,
    seller_address TEXT,
    seller_tax_code TEXT,
    seller_phone TEXT,

    buyer_name TEXT,
    buyer_address TEXT,
    buyer_tax_code TEXT,
    buyer_phone TEXT,

    invoice_number TEXT,
    invoice_series TEXT,
    issued_at TEXT,
    issued_date TEXT,
    issued_time TEXT,
    currency TEXT CHECK (currency IS NULL OR currency GLOB '[A-Z][A-Z][A-Z]'),
    payment_method TEXT,

    subtotal_raw TEXT,
    subtotal_value REAL CHECK (subtotal_value IS NULL OR subtotal_value >= 0),
    subtotal_currency TEXT CHECK (subtotal_currency IS NULL OR subtotal_currency GLOB '[A-Z][A-Z][A-Z]'),
    discount_raw TEXT,
    discount_value REAL CHECK (discount_value IS NULL OR discount_value >= 0),
    discount_currency TEXT CHECK (discount_currency IS NULL OR discount_currency GLOB '[A-Z][A-Z][A-Z]'),
    vat_amount_raw TEXT,
    vat_amount_value REAL CHECK (vat_amount_value IS NULL OR vat_amount_value >= 0),
    vat_amount_currency TEXT CHECK (vat_amount_currency IS NULL OR vat_amount_currency GLOB '[A-Z][A-Z][A-Z]'),
    total_amount_raw TEXT,
    total_amount_value REAL CHECK (total_amount_value IS NULL OR total_amount_value >= 0),
    total_amount_currency TEXT CHECK (total_amount_currency IS NULL OR total_amount_currency GLOB '[A-Z][A-Z][A-Z]'),

    extraction_status TEXT NOT NULL CHECK (extraction_status IN ('ok', 'partial', 'failed')),
    extraction_confidence REAL CHECK (
        extraction_confidence IS NULL OR (extraction_confidence >= 0 AND extraction_confidence <= 1)
    ),
    missing_fields_json TEXT NOT NULL DEFAULT '[]' CHECK (
        json_valid(missing_fields_json) AND json_type(missing_fields_json) = 'array'
    ),
    warnings_json TEXT NOT NULL DEFAULT '[]' CHECK (
        json_valid(warnings_json) AND json_type(warnings_json) = 'array'
    ),
    latency_ms REAL CHECK (latency_ms IS NULL OR latency_ms >= 0),

    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),

    UNIQUE (run_id, sample_id),
    UNIQUE (run_id, img_id)
);

CREATE TABLE IF NOT EXISTS invoice_items (
    invoice_item_id INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_id TEXT NOT NULL REFERENCES invoices(invoice_id) ON DELETE CASCADE,
    line_number INTEGER NOT NULL CHECK (line_number >= 1),
    description TEXT,
    sku TEXT,

    quantity_raw TEXT,
    quantity_value REAL CHECK (quantity_value IS NULL OR quantity_value >= 0),
    quantity_unit TEXT,

    unit_price_raw TEXT,
    unit_price_value REAL CHECK (unit_price_value IS NULL OR unit_price_value >= 0),
    unit_price_currency TEXT CHECK (unit_price_currency IS NULL OR unit_price_currency GLOB '[A-Z][A-Z][A-Z]'),

    line_total_raw TEXT,
    line_total_value REAL CHECK (line_total_value IS NULL OR line_total_value >= 0),
    line_total_currency TEXT CHECK (line_total_currency IS NULL OR line_total_currency GLOB '[A-Z][A-Z][A-Z]'),

    vat_rate REAL CHECK (vat_rate IS NULL OR (vat_rate >= 0 AND vat_rate <= 100)),

    UNIQUE (invoice_id, line_number)
);

CREATE TABLE IF NOT EXISTS invoice_taxes (
    invoice_tax_id INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_id TEXT NOT NULL REFERENCES invoices(invoice_id) ON DELETE CASCADE,
    tax_name TEXT,
    tax_rate REAL CHECK (tax_rate IS NULL OR (tax_rate >= 0 AND tax_rate <= 100)),

    taxable_amount_raw TEXT,
    taxable_amount_value REAL CHECK (taxable_amount_value IS NULL OR taxable_amount_value >= 0),
    taxable_amount_currency TEXT CHECK (
        taxable_amount_currency IS NULL OR taxable_amount_currency GLOB '[A-Z][A-Z][A-Z]'
    ),

    tax_amount_raw TEXT,
    tax_amount_value REAL CHECK (tax_amount_value IS NULL OR tax_amount_value >= 0),
    tax_amount_currency TEXT CHECK (tax_amount_currency IS NULL OR tax_amount_currency GLOB '[A-Z][A-Z][A-Z]')
);

CREATE TABLE IF NOT EXISTS field_evidence (
    evidence_id INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_id TEXT NOT NULL REFERENCES invoices(invoice_id) ON DELETE CASCADE,
    field_path TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    source_label TEXT
);

CREATE INDEX IF NOT EXISTS idx_invoices_run_id ON invoices(run_id);
CREATE INDEX IF NOT EXISTS idx_invoices_sample_id ON invoices(sample_id);
CREATE INDEX IF NOT EXISTS idx_invoices_img_id ON invoices(img_id);
CREATE INDEX IF NOT EXISTS idx_invoices_issued_date ON invoices(issued_date);
CREATE INDEX IF NOT EXISTS idx_invoices_seller_name ON invoices(seller_name);
CREATE INDEX IF NOT EXISTS idx_invoice_items_invoice_id ON invoice_items(invoice_id);
CREATE INDEX IF NOT EXISTS idx_invoice_taxes_invoice_id ON invoice_taxes(invoice_id);
CREATE INDEX IF NOT EXISTS idx_field_evidence_invoice_id ON field_evidence(invoice_id);

DROP VIEW IF EXISTS v_invoice_facts;
DROP VIEW IF EXISTS v_item_facts;
DROP VIEW IF EXISTS v_tax_facts;

CREATE VIEW v_invoice_facts AS
SELECT
    i.invoice_id,
    i.run_id,
    i.sample_id,
    i.img_id,
    i.document_type,
    i.invoice_number,
    i.invoice_series,
    i.issued_at,
    i.issued_date,
    i.issued_time,
    i.seller_name,
    i.seller_address,
    i.seller_tax_code,
    i.seller_phone,
    i.buyer_name,
    i.buyer_address,
    i.buyer_tax_code,
    i.buyer_phone,
    i.currency,
    i.payment_method,
    i.subtotal_value,
    i.subtotal_currency,
    i.discount_value,
    i.discount_currency,
    i.vat_amount_value,
    i.vat_amount_currency,
    i.total_amount_value,
    i.total_amount_currency,
    i.extraction_status,
    i.extraction_confidence,
    (SELECT COUNT(*) FROM invoice_items AS ii WHERE ii.invoice_id = i.invoice_id) AS item_count,
    (SELECT COUNT(*) FROM invoice_taxes AS it WHERE it.invoice_id = i.invoice_id) AS tax_line_count
FROM invoices AS i;

CREATE VIEW v_item_facts AS
SELECT
    i.invoice_id,
    i.run_id,
    i.sample_id,
    i.img_id,
    i.issued_date,
    i.seller_name,
    ii.line_number,
    ii.description,
    ii.sku,
    ii.quantity_value,
    ii.quantity_unit,
    ii.unit_price_value,
    ii.unit_price_currency,
    ii.line_total_value,
    ii.line_total_currency,
    ii.vat_rate
FROM invoice_items AS ii
JOIN invoices AS i ON i.invoice_id = ii.invoice_id;

CREATE VIEW v_tax_facts AS
SELECT
    i.invoice_id,
    i.run_id,
    i.sample_id,
    i.img_id,
    i.issued_date,
    i.seller_name,
    i.buyer_name,
    it.tax_name,
    it.tax_rate,
    it.taxable_amount_value,
    it.taxable_amount_currency,
    it.tax_amount_value,
    it.tax_amount_currency
FROM invoice_taxes AS it
JOIN invoices AS i ON i.invoice_id = it.invoice_id;

PRAGMA user_version = 2;

COMMIT;
