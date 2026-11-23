"""[M5-01 web console] Live log of a Node 1 job: timestamped lines kept in the job result."""

from __future__ import annotations

import copy
import threading

from src.api.job_log import JobLog


class Recorder:
    def __init__(self) -> None:
        self.saves: list[tuple[str, list[dict]]] = []

    def update_job(self, job_id, status, *, result=None, error=None, invoice_id=None) -> None:
        self.saves.append((status, copy.deepcopy(result["events"])))


def test_lines_are_timestamped_and_saved_with_the_current_status():
    store, result = Recorder(), {}
    log = JobLog(store, "up_1", result, clock=iter([10.0, 11.5, 12.0]).__next__)
    log.add("system", "info", "Nhận ảnh")
    log.set_status("ocr")
    log.add("ocr", "ok", "Đọc xong")
    assert result["events"] == [{"t": 10.0, "src": "system", "level": "info", "msg": "Nhận ảnh"},
                                {"t": 11.5, "src": "ocr", "level": "ok", "msg": "Đọc xong"}]
    assert [status for status, _ in store.saves] == ["queued", "ocr", "ocr"]
    assert store.saves[-1][1] == result["events"]


def test_a_keyed_line_is_updated_in_place_and_moves_to_the_end():
    store, result = Recorder(), {}
    log = JobLog(store, "up_1", result, clock=iter([1.0, 2.0, 3.0]).__next__)
    log.add("ep_a", "warn", "Chưa sẵn sàng, đã thử 1 lần", key="ep_a-wait")
    log.add("ocr", "ok", "Đọc xong")
    log.add("ep_a", "warn", "Chưa sẵn sàng, đã thử 2 lần", key="ep_a-wait")
    assert [(e["src"], e["msg"], e["t"]) for e in result["events"]] == [
        ("ocr", "Đọc xong", 2.0), ("ep_a", "Chưa sẵn sàng, đã thử 2 lần", 3.0)]
    assert result["events"][-1]["key"] == "ep_a-wait"


def test_lines_from_endpoints_running_in_parallel_are_all_kept():
    result: dict = {}
    log = JobLog(Recorder(), "up_1", result)
    def worker(n: int) -> None:
        for i in range(50):
            log.add(f"ep_{n}", "info", str(i))

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(result["events"]) == 200


def test_a_line_can_carry_numbers_for_the_console():
    result: dict = {}
    log = JobLog(Recorder(), "up_1", result, clock=lambda: 5.0)
    log.add("ep_a", "info", "Đang nhận 1.234 ký tự", key="ep_a-recv", data={"chars": 1234})
    log.add("ocr", "ok", "Đọc xong")
    assert result["events"][0]["data"] == {"chars": 1234} and "data" not in result["events"][1]
