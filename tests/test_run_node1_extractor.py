"""Regression tests for deterministic Node 1 output canonicalization."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_node1_extractor import normalize_canonical_payload  # noqa: E402


class NormalizeCanonicalPayloadTests(unittest.TestCase):
    def test_composes_issued_at_without_changing_existing_business_values(self) -> None:
        payload = {
            "invoice": {
                "issued_date": "2020-08-10",
                "issued_time": "17:12",
                "issued_at": None,
            },
            "totals": {
                "subtotal": {"raw": "10.000", "value": 10000, "currency": "VND"},
                "discount": {"raw": None, "value": None, "currency": None},
                "vat_amount": {"raw": None, "value": None, "currency": None},
                "total_amount": {"raw": "10.000", "value": 10000, "currency": "VND"},
            },
        }

        notes = normalize_canonical_payload(payload, {"raw_text": "10.000"})

        self.assertEqual(payload["invoice"]["issued_at"], "2020-08-10T17:12:00")
        self.assertEqual(payload["totals"]["subtotal"]["value"], 10000)
        self.assertEqual(len(notes), 1)

    def test_replaces_null_total_containers_with_empty_money_objects(self) -> None:
        payload = {
            "totals": {
                "subtotal": None,
                "discount": None,
                "vat_amount": None,
                "total_amount": None,
            }
        }

        notes = normalize_canonical_payload(payload, {"raw_text": ""})

        empty_money = {"raw": None, "value": None, "currency": None}
        self.assertTrue(all(value == empty_money for value in payload["totals"].values()))
        self.assertEqual(len(notes), 4)

    def test_defaults_source_label_and_removes_unsupported_evidence(self) -> None:
        payload = {
            "extraction": {
                "field_evidence": [
                    {"field_path": "seller.name", "raw_text": "SHOP ABC"},
                    {
                        "field_path": "totals.total_amount",
                        "raw_text": "invented total",
                        "source_label": None,
                    },
                ]
            }
        }

        notes = normalize_canonical_payload(payload, {"raw_text": "SHOP ABC\n10.000"})

        self.assertEqual(
            payload["extraction"]["field_evidence"],
            [{"field_path": "seller.name", "raw_text": "SHOP ABC", "source_label": None}],
        )
        self.assertEqual(len(notes), 2)


if __name__ == "__main__":
    unittest.main()
