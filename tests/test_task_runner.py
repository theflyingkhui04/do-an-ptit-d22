"""Unit tests for FlexSDS TaskRunner with Ctrl+O interactive detail view."""
import sys
import time
import pytest
from flexsds.cli.task_runner import Task, TaskRunner, TaskStatus


def test_task_runner_add_and_lifecycle():
    runner = TaskRunner(interactive=False)
    task = runner.add_task("Kiểm tra phần cứng")
    assert task.status == TaskStatus.PENDING
    assert task.name == "Kiểm tra phần cứng"
    assert len(task.logs) == 0


def test_task_runner_run_success():
    runner = TaskRunner(interactive=False)
    task = runner.add_task("Cấu hình mô hình")

    def dummy_action(task_ref):
        task_ref.log("Đang kiểm tra tensor...")
        task_ref.log("Tensor hợp lệ.")
        return 42

    result = runner.run_task(task, dummy_action)
    assert result == 42
    assert task.status == TaskStatus.COMPLETED
    assert len(task.logs) == 2
    assert "Tensor hợp lệ." in task.logs[1]
    assert task.duration >= 0.0


def test_task_runner_run_failure():
    runner = TaskRunner(interactive=False)
    task = runner.add_task("Tải trọng số lỗi")

    def failing_action(task_ref):
        task_ref.log("Kết nối tới máy chủ...")
        raise RuntimeError("Mất kết nối mạng.")

    with pytest.raises(RuntimeError, match="Mất kết nối mạng."):
        runner.run_task(task, failing_action)

    assert task.status == TaskStatus.FAILED
    assert task.error == "Mất kết nối mạng."
    assert len(task.logs) == 1


def test_task_runner_toggle_details():
    runner = TaskRunner(interactive=False)
    assert runner.show_details is False
    runner.toggle_details()
    assert runner.show_details is True
    runner.toggle_details()
    assert runner.show_details is False


def test_task_runner_handle_key_ctrl_o():
    runner = TaskRunner(interactive=False)
    assert runner.show_details is False
    # Phím Ctrl+O có mã ASCII \x0f
    handled = runner.handle_key("\x0f")
    assert handled is True
    assert runner.show_details is True


def test_task_runner_render_collapsed_vs_expanded():
    runner = TaskRunner(interactive=False)
    task1 = runner.add_task("Tác vụ 1")
    task1.status = TaskStatus.COMPLETED
    task1.logs.append("Chi tiết dòng 1 của tác vụ 1")

    task2 = runner.add_task("Tác vụ 2")
    task2.status = TaskStatus.RUNNING
    task2.logs.append("Đang thực thi dòng 2...")

    # Chế độ thu gọn
    runner.show_details = False
    collapsed = runner.render_view()
    assert "Tác vụ 1" in collapsed
    assert "Tác vụ 2" in collapsed
    assert "Chi tiết dòng 1" not in collapsed
    assert "Ctrl+O" in collapsed

    # Chế độ mở rộng
    runner.show_details = True
    expanded = runner.render_view()
    assert "Tác vụ 1" in expanded
    assert "Đang thực thi dòng 2..." in expanded
