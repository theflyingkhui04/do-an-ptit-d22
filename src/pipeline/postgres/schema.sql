-- [M5-01 web demo] Postgres (Supabase) mirror of src/pipeline/sqlite/schema.sql (user_version 2).
-- Same table names, column names and view columns as SQLite v2 (tests/test_demo_store.py).
-- Demo-only additions: invoices.source, invoices.storage_path, table demo_jobs, view v_upload_facts.
-- Lives in schema "demo" (not "public", which Supabase exposes through its REST API).
-- Roles and grants: src/pipeline/postgres/roles.sql (run after this file).

CREATE SCHEMA IF NOT EXISTS demo;
SET search_path = demo;

CREATE TABLE IF NOT EXISTS extraction_runs (
    run_id TEXT PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    lane TEXT CHECK (lane IS NULL OR lane IN ('vanilla', 'static', 'adaptive', 'eagle')),
    model_name TEXT,
    prompt_version TEXT,
    schema_version TEXT NOT NULL DEFAULT 'invoice_extraction.v1',
    dataset_manifest_path TEXT,
    notes TEXT,
    latency_ms_median DOUBLE PRECISION CHECK (latency_ms_median IS NULL OR latency_ms_median >= 0)
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
    raw_json JSONB NOT NULL CHECK (jsonb_typeof(raw_json) = 'object'),

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
    currency TEXT CHECK (currency IS NULL OR currency ~ '^[A-Z]{3}$'),
    payment_method TEXT,

    subtotal_raw TEXT,
    subtotal_value DOUBLE PRECISION CHECK (subtotal_value IS NULL OR subtotal_value >= 0),
    subtotal_currency TEXT CHECK (subtotal_currency IS NULL OR subtotal_currency ~ '^[A-Z]{3}$'),
    discount_raw TEXT,
    discount_value DOUBLE PRECISION CHECK (discount_value IS NULL OR discount_value >= 0),
    discount_currency TEXT CHECK (discount_currency IS NULL OR discount_currency ~ '^[A-Z]{3}$'),
    vat_amount_raw TEXT,
    vat_amount_value DOUBLE PRECISION CHECK (vat_amount_value IS NULL OR vat_amount_value >= 0),
    vat_amount_currency TEXT CHECK (vat_amount_currency IS NULL OR vat_amount_currency ~ '^[A-Z]{3}$'),
    total_amount_raw TEXT,
    total_amount_value DOUBLE PRECISION CHECK (total_amount_value IS NULL OR total_amount_value >= 0),
    total_amount_currency TEXT CHECK (total_amount_currency IS NULL OR total_amount_currency ~ '^[A-Z]{3}$'),

    extraction_status TEXT NOT NULL CHECK (extraction_status IN ('ok', 'partial', 'failed')),
    extraction_confidence DOUBLE PRECISION CHECK (
        extraction_confidence IS NULL OR (extraction_confidence >= 0 AND extraction_confidence <= 1)
    ),
    missing_fields_json JSONB NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(missing_fields_json) = 'array'),
    warnings_json JSONB NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(warnings_json) = 'array'),
    latency_ms DOUBLE PRECISION CHECK (latency_ms IS NULL OR latency_ms >= 0),

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- demo-only columns
    source TEXT NOT NULL DEFAULT 'upload' CHECK (source IN ('devset', 'upload')),
    storage_path TEXT,

    UNIQUE (run_id, sample_id),
    UNIQUE (run_id, img_id)
);

CREATE TABLE IF NOT EXISTS invoice_items (
    invoice_item_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    invoice_id TEXT NOT NULL REFERENCES invoices(invoice_id) ON DELETE CASCADE,
    line_number INTEGER NOT NULL CHECK (line_number >= 1),
    description TEXT,
    sku TEXT,

    quantity_raw TEXT,
    quantity_value DOUBLE PRECISION CHECK (quantity_value IS NULL OR quantity_value >= 0),
    quantity_unit TEXT,

    unit_price_raw TEXT,
    unit_price_value DOUBLE PRECISION CHECK (unit_price_value IS NULL OR unit_price_value >= 0),
    unit_price_currency TEXT CHECK (unit_price_currency IS NULL OR unit_price_currency ~ '^[A-Z]{3}$'),

    line_total_raw TEXT,
    line_total_value DOUBLE PRECISION CHECK (line_total_value IS NULL OR line_total_value >= 0),
    line_total_currency TEXT CHECK (line_total_currency IS NULL OR line_total_currency ~ '^[A-Z]{3}$'),

    vat_rate DOUBLE PRECISION CHECK (vat_rate IS NULL OR (vat_rate >= 0 AND vat_rate <= 100)),

    UNIQUE (invoice_id, line_number)
);

CREATE TABLE IF NOT EXISTS invoice_taxes (
    invoice_tax_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    invoice_id TEXT NOT NULL REFERENCES invoices(invoice_id) ON DELETE CASCADE,
    tax_name TEXT,
    tax_rate DOUBLE PRECISION CHECK (tax_rate IS NULL OR (tax_rate >= 0 AND tax_rate <= 100)),

    taxable_amount_raw TEXT,
    taxable_amount_value DOUBLE PRECISION CHECK (taxable_amount_value IS NULL OR taxable_amount_value >= 0),
    taxable_amount_currency TEXT CHECK (taxable_amount_currency IS NULL OR taxable_amount_currency ~ '^[A-Z]{3}$'),

    tax_amount_raw TEXT,
    tax_amount_value DOUBLE PRECISION CHECK (tax_amount_value IS NULL OR tax_amount_value >= 0),
    tax_amount_currency TEXT CHECK (tax_amount_currency IS NULL OR tax_amount_currency ~ '^[A-Z]{3}$')
);

CREATE TABLE IF NOT EXISTS field_evidence (
    evidence_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    invoice_id TEXT NOT NULL REFERENCES invoices(invoice_id) ON DELETE CASCADE,
    field_path TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    source_label TEXT
);

CREATE TABLE IF NOT EXISTS demo_jobs (
    job_id TEXT PRIMARY KEY,
    status TEXT NOT NULL CHECK (status IN ('queued', 'ocr', 'node1', 'saving', 'done', 'failed', 'interrupted')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    error TEXT,
    result JSONB NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(result) = 'object'),
    invoice_id TEXT REFERENCES invoices(invoice_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_invoices_run_id ON invoices(run_id);
CREATE INDEX IF NOT EXISTS idx_invoices_sample_id ON invoices(sample_id);
CREATE INDEX IF NOT EXISTS idx_invoices_img_id ON invoices(img_id);
CREATE INDEX IF NOT EXISTS idx_invoices_issued_date ON invoices(issued_date);
CREATE INDEX IF NOT EXISTS idx_invoices_seller_name ON invoices(seller_name);
CREATE INDEX IF NOT EXISTS idx_invoices_created_at ON invoices(created_at);
CREATE INDEX IF NOT EXISTS idx_invoice_items_invoice_id ON invoice_items(invoice_id);
CREATE INDEX IF NOT EXISTS idx_invoice_taxes_invoice_id ON invoice_taxes(invoice_id);
CREATE INDEX IF NOT EXISTS idx_field_evidence_invoice_id ON field_evidence(invoice_id);

CREATE OR REPLACE VIEW v_invoice_facts AS
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

CREATE OR REPLACE VIEW v_item_facts AS
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

CREATE OR REPLACE VIEW v_tax_facts AS
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

-- Demo-only: when and how each invoice entered the demo (Node 2 questions about uploads).
-- uploaded_date is the calendar date in Vietnam (UTC+7) as text YYYY-MM-DD.
CREATE OR REPLACE VIEW v_upload_facts AS
SELECT
    i.invoice_id,
    i.source,
    i.created_at AS uploaded_at,
    to_char(i.created_at AT TIME ZONE 'Asia/Ho_Chi_Minh', 'YYYY-MM-DD') AS uploaded_date
FROM invoices AS i;
