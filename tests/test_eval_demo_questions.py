"""[M5-01 web demo] Task 5.4: question set demo_questions.v1 and its scoring."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import eval_demo_questions as ev

from src.api.questions import NODE2_PROMPT, check_demo_sql
from tests.test_demo_store import (  # noqa: F401
    DB_URL,
    needs_db,
    pg,
)


def test_question_set_is_well_formed_and_separate_from_the_prompt_examples():
    data = ev.load_questions()
    prompt = NODE2_PROMPT.read_text(encoding="utf-8")
    assert len(data["questions"]) == 15
    assert sum(q["expected_status"] == "no_data" for q in data["questions"]) == 2
    for question in data["questions"]:
        assert check_demo_sql(question["reference_sql"])
        assert question["question"] not in prompt


def test_results_match_ignores_extra_columns_and_rounding():
    assert ev.results_match([["MINIMART ANAN"]], [["MINIMART ANAN", 3918594.0]])
    assert ev.results_match([[178903.357]], [[178903.0]])
    assert ev.results_match([["a"], ["b"]], [["b"], ["a"]])
    assert not ev.results_match([["a"]], [["a"], ["b"]])
    assert not ev.results_match([[840000.0]], [[840001.0]])


def test_scoring_no_data_fabrication_and_writes():
    no_data_q = {"id": "q14", "question": "Liệt kê các hoá đơn năm 2021.", "expected_status": "no_data"}
    assert ev.score(no_data_q, [], {"status": "no_data", "rows": []})["correct"]
    assert ev.score(no_data_q, [], {"status": "ok", "rows": [[None]]})["correct"]
    ok_q = {"id": "q04", "question": "Tổng tiền?", "expected_status": "ok"}
    entry = ev.score(ok_q, [[4930094.0]], {"status": "ok", "rows": [[4930094.0]], "answer": "Tổng 4.930.095 đồng",
                                           "sql": "SELECT 1"})
    assert entry["correct"] and entry["fabricated_answer"] and not entry["executed_write"]
    assert ev.is_single_select("SELECT 1") and not ev.is_single_select("DELETE FROM invoices")


@needs_db
def test_reference_answers_on_the_seeded_database(pg):  # noqa: F811  (fixture)
    import seed_demo_db as seed

    from src.api.store import PostgresStore

    # The v1 question set's reference rows were frozen on the W1_v2 seed: keep that seed for it.
    seed.seed(pg, seed.load_validated(REPO_ROOT / seed.W1_V2_VALIDATED), notes=seed.W1_V2_SEED_NOTES)
    store = PostgresStore(DB_URL)
    report = ev.run(store, None, ev.load_questions())
    rows = {e["id"]: e["reference_rows"] for e in report["entries"]}
    assert rows["q01"] == [[4]] and rows["q06"] == [[22]] and rows["q11"] == [[84]] and rows["q12"] == [[17]]
    assert rows["q05"] == [["MINIMART ANAN"]] and rows["q07"] == [["CỬA HÀNG NAM OÁNH", 840000.0]]
    assert rows["q09"] == [["2020-08-12"]] and rows["q10"] == [["Sữa bột Ensure gold 400g"]]
    assert rows["q13"] == [[1011500.0]] and rows["q14"] == [] and rows["q15"] == []

    class Oracle:
        """Answers with the reference rows: the scorer must give 15/15."""

        def answer(self, question):
            item = next(q for q in ev.load_questions()["questions"] if q["question"] == question)
            _, result = store.run_readonly(check_demo_sql(item["reference_sql"]))
            return {"status": "ok" if result else "no_data", "rows": result, "sql": item["reference_sql"],
                    "answer": None}

    full = ev.run(store, Oracle(), ev.load_questions())
    assert full["summary"]["correct"] == 15 and full["summary"]["executed_writes"] == 0
