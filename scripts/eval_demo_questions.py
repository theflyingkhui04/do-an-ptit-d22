"""[M5-01 web demo] Score the free-form question flow against demo_questions.v1.

For each question: run its reference SQL (checked, read-only) and the full
QuestionService flow (Node 2 on the lane map's node2 lane, answer lane), then
record whether the result matches the reference, whether a number in the
answer was invented, and whether any executed SQL was not a single SELECT.

    python scripts/eval_demo_questions.py --reference-only      # DB only: check the reference answers
    python scripts/eval_demo_questions.py                       # full run (Modal lanes + DB)

[M5-01 canonical flow] The locked set runs once, on the W1_v3 seed database:

    python scripts/eval_demo_questions.py --questions experiments/configs/canonical_questions_v1/questions.json

Its questions carry ``reference_rows`` frozen when the set was locked; the run stops if the
database gives different reference rows (not the locked seed). The report goes to the set's
``report`` path and records the SHA-256 of the question file.

Environment: DEMO_DATABASE_URL, DEMO_READER_DATABASE_URL, MODAL_PROXY_TOKEN_ID/SECRET,
PTIT_APP_NAME (default ptit-vllm-demo), PTIT_MODAL_WORKSPACE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import sqlglot
from sqlglot import exp

from src.api.questions import check_demo_sql, numbers_are_grounded

QUESTIONS = Path("experiments/configs/demo_questions_v1/questions.json")
DEFAULT_OUTPUT = Path("experiments/results/m5_01_demo_questions_eval.json")
CANONICAL_QUESTIONS = Path("experiments/configs/canonical_questions_v1/questions.json")


def load_questions(path: Path = REPO_ROOT / QUESTIONS) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    ids = [q["id"] for q in data["questions"]]
    if len(ids) != len(set(ids)) or len(ids) != 15:
        raise SystemExit(f"{path}: expected 15 questions with unique ids")
    return data


def report_path(data: dict[str, Any]) -> Path:
    return Path(data["report"]) if data.get("report") else DEFAULT_OUTPUT


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(float(a) - float(b)) <= 0.5
    if a is None or b is None:
        return a is b
    return str(a).strip().casefold() == str(b).strip().casefold()


def results_match(reference: list[list[Any]], model: list[list[Any]]) -> bool:
    """Same row count, and every reference row is contained (extra columns allowed) in a distinct model row."""
    if len(reference) != len(model):
        return False
    unused = list(range(len(model)))
    for ref_row in reference:
        for index in unused:
            if all(any(_same(value, cell) for cell in model[index]) for value in ref_row):
                unused.remove(index)
                break
        else:
            return False
    return True


def is_single_select(sql: str) -> bool:
    try:
        statements = [s for s in sqlglot.parse(sql, read="postgres") if s is not None]
    except sqlglot.errors.ParseError:
        return False
    return len(statements) == 1 and isinstance(statements[0], (exp.Select, exp.Union, exp.Intersect, exp.Except))


def score(question: dict[str, Any], reference_rows: list[list[Any]], outcome: dict[str, Any]) -> dict[str, Any]:
    rows = outcome.get("rows") or []
    if question["expected_status"] == "no_data":
        correct = outcome.get("status") == "no_data" or (
            outcome.get("status") == "ok" and all(cell in (None, 0) for row in rows for cell in row))
    else:
        correct = outcome.get("status") == "ok" and results_match(reference_rows, rows)
    answer = outcome.get("answer")
    fabricated = bool(answer) and outcome.get("status") == "ok" and not numbers_are_grounded(
        answer, rows, question["question"])
    executed = [outcome["sql"]] if outcome.get("sql") and outcome.get("status") in ("ok", "no_data") else []
    return {"id": question["id"], "question": question["question"], "expected_status": question["expected_status"],
            "status": outcome.get("status"), "correct": correct, "fabricated_answer": fabricated,
            "answer_withheld": outcome.get("warning") == "fabricated_number",
            "executed_write": any(not is_single_select(sql) for sql in executed),
            "repairs": outcome.get("repairs", 0), "first_error": outcome.get("first_error"),
            "sql": outcome.get("sql"), "reference_rows": reference_rows, "rows": rows[:20],
            "answer": answer, "error": outcome.get("error"), "calls": outcome.get("calls", [])}


def summarize(entries: list[dict[str, Any]]) -> dict[str, int]:
    return {"questions": len(entries), "correct": sum(e["correct"] for e in entries),
            "fabricated_answers": sum(e["fabricated_answer"] for e in entries),
            "answers_withheld": sum(e["answer_withheld"] for e in entries),
            "executed_writes": sum(e["executed_write"] for e in entries),
            "repairs": sum(e["repairs"] for e in entries)}


def run(store: Any, service: Any | None, data: dict[str, Any], *,
        questions_sha256: str | None = None) -> dict[str, Any]:
    entries = []
    for question in data["questions"]:
        _, reference_rows = store.run_readonly(check_demo_sql(question["reference_sql"]))
        if (question["expected_status"] == "no_data") != (not reference_rows):
            raise SystemExit(f"{question['id']}: reference result does not match expected_status")
        locked = question.get("reference_rows")
        if locked is not None and not results_match(locked, reference_rows):
            raise SystemExit(f"{question['id']}: reference rows {reference_rows} differ from the locked {locked}; "
                             "run on the seed database the set was locked on")
        if service is None:
            entries.append({"id": question["id"], "reference_rows": reference_rows})
            continue
        entries.append(score(question, reference_rows, service.answer(question["question"])))
    return {"version": data["version"], "questions_sha256": questions_sha256,
            "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "summary": summarize(entries) if service is not None else None, "entries": entries}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--output", type=Path, default=None, help="default: the set's report path")
    args = parser.parse_args(argv)

    from src.api.store import PostgresStore

    store = PostgresStore(os.environ["DEMO_DATABASE_URL"], os.environ.get("DEMO_READER_DATABASE_URL"))
    service = None
    if not args.reference_only:
        import httpx

        from src.api.llm import LaneCaller
        from src.api.questions import QuestionService
        from src.api.settings import DEFAULT_LANE_MAP
        from src.client.modal_client import LANE_MODELS, lane_base_url, load_token

        os.environ.setdefault("PTIT_APP_NAME", "ptit-vllm-demo")
        token = load_token()
        http = httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0))
        service = QuestionService(store, lane_map=dict(DEFAULT_LANE_MAP), lane_caller=lambda lane: LaneCaller(
            lane, lane_base_url(lane), token, LANE_MODELS[lane], client=http))
    questions_file = REPO_ROOT / args.questions
    data = load_questions(questions_file)
    report = run(store, service, data, questions_sha256=file_sha256(questions_file))
    text = json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n"
    if args.reference_only:
        print(text)
        return 0
    output = REPO_ROOT / (args.output or report_path(data))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8", newline="\n")
    print(json.dumps(report["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
