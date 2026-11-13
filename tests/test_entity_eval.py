"""Entity scoring of Node 1 JSON against MC-OCR labels."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.pipeline.entity_eval import (
    BLANK,
    MATCH,
    MISMATCH,
    NO_LABEL,
    UNSCORED,
    normalize_text,
    parse_amounts,
    parse_date,
    parse_time,
    score_sample,
    token_f1,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEVSET = REPO_ROOT / "data/processed/m0_04_mcocr_devset/samples.jsonl"


def pairs(**labels: list[str]) -> list[dict]:
    return [{"label": label, "text": text} for label, texts in labels.items() for text in texts]


def payload(name=None, address=None, date=None, time=None, total=None) -> dict:
    return {
        "seller": {"name": name, "address": address},
        "invoice": {"issued_date": date, "issued_time": time},
        "totals": {"total_amount": {"value": total}},
    }


class TextFieldTests(unittest.TestCase):
    def test_normalize_keeps_diacritics_and_drops_case_and_punctuation(self) -> None:
        self.assertEqual(normalize_text("  Chợ   Sủi, Gia-Lâm. "), "chợ sủi gia lâm")

    def test_token_f1(self) -> None:
        self.assertEqual(token_f1("a b", "a b"), 1.0)
        self.assertEqual(token_f1("a b", "c"), 0.0)
        self.assertEqual(token_f1("a b", ""), 0.0)
        self.assertAlmostEqual(token_f1("a b c d", "a b"), 2 / 3)

    def test_exact_match_after_normalization(self) -> None:
        score = score_sample(payload(name="minimart  ANAN"), pairs(SELLER=["MINIMART ANAN"]))["SELLER"]
        self.assertEqual((score.status, score.f1), (MATCH, 1.0))

    def test_multiline_seller_is_joined(self) -> None:
        score = score_sample(
            payload(name="Saigon Co.op"), pairs(SELLER=["Saigon Co.op", "Co.op Food HN THE K-PARK"])
        )["SELLER"]
        self.assertEqual(score.status, MISMATCH)
        self.assertAlmostEqual(score.f1, 0.46, places=2)

    def test_address_prefix_is_removed(self) -> None:
        for prefix in ("ĐC:", "Đ/c:", "Địa chỉ:", "đc :"):
            score = score_sample(
                payload(address="212 Đường Trần Phú-Cẩm Phả"),
                pairs(ADDRESS=[f"{prefix} 212 Đường Trần Phú-Cẩm Phả"]),
            )["ADDRESS"]
            self.assertEqual(score.status, MATCH, prefix)

    def test_extra_words_in_prediction_stay_partial(self) -> None:
        score = score_sample(
            payload(address="212 Đường Trần Phú-Cẩm Phả, Cẩm Phả"),
            pairs(ADDRESS=["ĐC: 212 Đường Trần Phú-Cẩm Phả"]),
        )["ADDRESS"]
        self.assertEqual(score.status, MISMATCH)
        self.assertGreater(score.f1, 0.8)

    def test_diacritics_are_significant(self) -> None:
        score = score_sample(payload(address="Cho Sui"), pairs(ADDRESS=["Chợ Sủi"]))["ADDRESS"]
        self.assertEqual(score.status, MISMATCH)

    def test_null_prediction_is_blank(self) -> None:
        for value in (None, "", "   "):
            score = score_sample(payload(address=value), pairs(ADDRESS=["Chợ Sủi Phú Thị Gia Lâm"]))["ADDRESS"]
            self.assertEqual((score.status, score.f1), (BLANK, 0.0))

    def test_field_without_label_is_not_scored(self) -> None:
        self.assertEqual(score_sample(payload(address="x"), pairs(SELLER=["a"]))["ADDRESS"].status, NO_LABEL)


class TimestampTests(unittest.TestCase):
    def test_parse_date_formats(self) -> None:
        cases = {
            "Ngày: 10/08/2020 - 17:12": "2020-08-10",
            "Ngày: 11-08-2020": "2020-08-11",
            "Ngày: 21.04.2019 Giờ vào: 10.30": "2019-04-21",
            "Thời gian: 21:43:39 - 14/08/2020": "2020-08-14",
            "Ngày: 12/08/2020-12:03": "2020-08-12",
        }
        for text, expected in cases.items():
            self.assertEqual(parse_date(text), expected, text)

    def test_parse_date_rejects_impossible_or_absent_dates(self) -> None:
        self.assertIsNone(parse_date("Ngày: 45/13/2020"))
        self.assertIsNone(parse_date("Ngày hôm nay"))

    def test_parse_time_formats(self) -> None:
        cases = {
            "Ngày: 10/08/2020 - 17:12": "17:12",
            "Ngày: 12/08/2020-12:03": "12:03",
            "Ngày: 11-08-2020": None,
            "Ngày: 21.04.2019 Giờ vào: 10.30 Giờ ra: 10.31": "10:30",
            "Thời gian: 21:43:39 - 14/08/2020": "21:43",
            "Ngày: 23/09/2019 10:58:13": "10:58",
        }
        for text, expected in cases.items():
            self.assertEqual(parse_time(text), expected, text)

    def test_date_and_time_match(self) -> None:
        score = score_sample(
            payload(date="2020-08-10", time="17:12:00"), pairs(TIMESTAMPS=["Ngày: 10/08/2020 - 17:12"])
        )["TIMESTAMPS"]
        self.assertEqual((score.status, score.date_ok, score.time_ok), (MATCH, True, True))

    def test_label_without_time_ignores_time(self) -> None:
        score = score_sample(payload(date="2020-08-11"), pairs(TIMESTAMPS=["Ngày: 11-08-2020"]))["TIMESTAMPS"]
        self.assertEqual((score.status, score.date_ok, score.time_ok), (MATCH, True, None))

    def test_dotted_date_across_lines(self) -> None:
        score = score_sample(
            payload(date="2019-04-21", time="10:30:00"),
            pairs(TIMESTAMPS=["Ngày:", "21.04.2019", "Giờ vào:", "10.30", "Giờ ra:", "10.31"]),
        )["TIMESTAMPS"]
        self.assertEqual(score.status, MATCH)

    def test_wrong_time_is_mismatch(self) -> None:
        score = score_sample(
            payload(date="2020-08-10", time="17:13:00"), pairs(TIMESTAMPS=["Ngày: 10/08/2020 - 17:12"])
        )["TIMESTAMPS"]
        self.assertEqual((score.status, score.date_ok, score.time_ok), (MISMATCH, True, False))

    def test_missing_time_when_label_has_one_is_mismatch(self) -> None:
        score = score_sample(
            payload(date="2020-08-10"), pairs(TIMESTAMPS=["Ngày: 10/08/2020 - 17:12"])
        )["TIMESTAMPS"]
        self.assertEqual((score.status, score.time_ok), (MISMATCH, False))

    def test_missing_date_is_blank(self) -> None:
        score = score_sample(payload(), pairs(TIMESTAMPS=["Ngày: 10/08/2020"]))["TIMESTAMPS"]
        self.assertEqual(score.status, BLANK)

    def test_unparseable_label_is_unscored(self) -> None:
        score = score_sample(payload(date="2020-08-10"), pairs(TIMESTAMPS=["Hôm qua"]))["TIMESTAMPS"]
        self.assertEqual(score.status, UNSCORED)


class AmountTests(unittest.TestCase):
    def test_parse_amounts(self) -> None:
        cases = {
            "42,000": [42000.0],
            "52 000": [52000.0],
            "75.000đ": [75000.0],
            "27,200.00": [27200.0],
            "1,161,500": [1161500.0],
            "16500": [16500.0],
            "Tổng tiền": [],
            "Tổng số thanh toán:": [],
        }
        for text, expected in cases.items():
            self.assertEqual(parse_amounts(text), expected, text)

    def test_total_matches_any_number_in_label(self) -> None:
        for lines, value in (
            (["Tổng tiền", "42,000"], 42000),
            (["335,500", "Tổng cộng"], 335500),
            (["Tiên Thanh Toán:", "52 000"], 52000),
            (["Tổng số thanh toán:", "75.000đ"], 75000),
            (["Tong so tien thanh toan:", "27,200.00"], 27200),
        ):
            self.assertEqual(score_sample(payload(total=value), pairs(TOTAL_COST=lines))["TOTAL_COST"].status, MATCH)

    def test_wrong_total_is_mismatch(self) -> None:
        score = score_sample(payload(total=4200), pairs(TOTAL_COST=["42,000"]))["TOTAL_COST"]
        self.assertEqual(score.status, MISMATCH)

    def test_half_unit_tolerance(self) -> None:
        self.assertEqual(score_sample(payload(total=42000.4), pairs(TOTAL_COST=["42,000"]))["TOTAL_COST"].status, MATCH)
        self.assertEqual(
            score_sample(payload(total=42000.6), pairs(TOTAL_COST=["42,000"]))["TOTAL_COST"].status, MISMATCH
        )

    def test_null_total_is_blank(self) -> None:
        self.assertEqual(score_sample(payload(), pairs(TOTAL_COST=["42,000"]))["TOTAL_COST"].status, BLANK)

    def test_label_without_number_is_unscored(self) -> None:
        self.assertEqual(score_sample(payload(total=1), pairs(TOTAL_COST=["Tổng tiền"]))["TOTAL_COST"].status, UNSCORED)


class MissingPayloadTests(unittest.TestCase):
    def test_missing_sample_is_blank_on_labelled_fields(self) -> None:
        scores = score_sample(None, pairs(SELLER=["A"], TOTAL_COST=["42,000"]))
        self.assertEqual(scores["SELLER"].status, BLANK)
        self.assertEqual(scores["TOTAL_COST"].status, BLANK)
        self.assertEqual(scores["ADDRESS"].status, NO_LABEL)


class RealLabelTests(unittest.TestCase):
    def test_every_devset_label_can_be_parsed(self) -> None:
        for line in DEVSET.read_text(encoding="utf-8").splitlines():
            sample = json.loads(line)
            by_label: dict[str, list[str]] = {}
            for pair in sample["key_text_pairs"]:
                by_label.setdefault(pair["label"], []).append(pair["text"])
            self.assertIsNotNone(parse_date(" ".join(by_label["TIMESTAMPS"])), sample["sample_id"])
            self.assertTrue(
                any(parse_amounts(text) for text in by_label["TOTAL_COST"]), sample["sample_id"]
            )


if __name__ == "__main__":
    unittest.main()
