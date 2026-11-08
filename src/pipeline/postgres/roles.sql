-- [M5-01 web demo] Roles for the demo database; run as the project owner after schema.sql.
-- Passwords are NOT in this file. Set them once in the SQL editor, then store them only
-- in the backend environment (Render):
--   ALTER ROLE demo_writer PASSWORD '...';
--   ALTER ROLE demo_reader PASSWORD '...';

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'demo_writer') THEN
        CREATE ROLE demo_writer LOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'demo_reader') THEN
        CREATE ROLE demo_reader LOGIN;
    END IF;
END
$$;

-- Writer: the backend's ingest path (tables of schema demo only).
GRANT USAGE ON SCHEMA demo TO demo_writer;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA demo TO demo_writer;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA demo TO demo_writer;
ALTER ROLE demo_writer SET search_path = demo;

-- Reader: runs Node 2 SQL. SELECT on the three views and nothing else; the views read
-- the base tables with their owner's rights, so the reader never needs table access.
REVOKE ALL ON ALL TABLES IN SCHEMA demo FROM demo_reader;
REVOKE ALL ON SCHEMA public FROM demo_reader;
GRANT USAGE ON SCHEMA demo TO demo_reader;
GRANT SELECT ON demo.v_invoice_facts, demo.v_item_facts, demo.v_tax_facts TO demo_reader;
ALTER ROLE demo_reader SET search_path = demo;
ALTER ROLE demo_reader SET default_transaction_read_only = on;
ALTER ROLE demo_reader SET statement_timeout = '5s';
