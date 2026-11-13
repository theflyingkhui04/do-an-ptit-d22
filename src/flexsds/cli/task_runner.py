"""Interactive task manager with compact view and Ctrl+O toggled details."""
from __future__ import annotations

import enum
import os
import select
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

try:
    from flexsds.cli.output import ascii_text, supports_unicode
except ModuleNotFoundError:
    from src.flexsds.cli.output import ascii_text, supports_unicode


class TaskStatus(enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """Represents a discrete setup or serving execution task."""
    name: str
    status: TaskStatus = TaskStatus.PENDING
    logs: List[str] = field(default_factory=list)
    start_time: float = 0.0
    duration: float = 0.0
    error: Optional[str] = None

    def log(self, message: str) -> None:
        """Appends a log line to task buffer without polluting terminal."""
        clean_msg = message.rstrip("\r\n")
        if clean_msg:
            self.logs.append(clean_msg)


class TaskRunner:
    """Orchestrates tasks, renders interactive status, and handles Ctrl+O log toggle."""

    SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, interactive: Optional[bool] = None) -> None:
        self.tasks: List[Task] = []
        self.show_details: bool = False
        self.is_interactive = interactive if interactive is not None else sys.stdout.isatty()
        self._current_frame_idx = 0
        self._listener_thread: Optional[threading.Thread] = None
        self._stop_listener = threading.Event()

    def add_task(self, name: str) -> Task:
        """Enqueues a new task."""
        task = Task(name=name)
        self.tasks.append(task)
        return task

    def toggle_details(self) -> None:
        """Toggles between compact task list and expanded log details."""
        self.show_details = not self.show_details

    def handle_key(self, key: str) -> bool:
        """Handles keypress events (Ctrl+O is ASCII 0x0F)."""
        if key == "\x0f" or key == "\x0e":  # Ctrl+O (or Ctrl+N alternative)
            self.toggle_details()
            return True
        return False

    def render_view(self) -> str:
        """Renders current status of all tasks with optional log detail box."""
        lines: List[str] = []
        active_task: Optional[Task] = None

        for idx, t in enumerate(self.tasks, 1):
            if t.status == TaskStatus.COMPLETED:
                icon = "\033[32m✔\033[0m"
                dur_str = f" \033[90m({t.duration:.1f}s)\033[0m"
                lines.append(f"  {icon} [{idx}/{len(self.tasks)}] {t.name}{dur_str}")
            elif t.status == TaskStatus.RUNNING:
                active_task = t
                spinner = self.SPINNER_FRAMES[self._current_frame_idx % len(self.SPINNER_FRAMES)]
                icon = f"\033[36m{spinner}\033[0m"
                elapsed = max(time.perf_counter() - t.start_time, 0.0) if t.start_time > 0 else 0.0
                dur_str = f" \033[33m({elapsed:.1f}s)\033[0m"
                lines.append(f"  {icon} [{idx}/{len(self.tasks)}] \033[1m{t.name}\033[0m{dur_str}")
            elif t.status == TaskStatus.FAILED:
                active_task = t
                icon = "\033[31m✖\033[0m"
                lines.append(f"  {icon} [{idx}/{len(self.tasks)}] \033[31m{t.name}\033[0m")
            else:
                icon = "\033[90m○\033[0m"
                lines.append(f"  {icon} [{idx}/{len(self.tasks)}] \033[90m{t.name}\033[0m")

        # Footer hướng dẫn phím tắt
        if self.show_details:
            footer = "  \033[90m[\033[36mCtrl+O\033[90m: Thu gọn chi tiết log | \033[36mCtrl+C\033[90m: Dừng]\033[0m"
        else:
            footer = "  \033[90m[\033[36mCtrl+O\033[90m: Xem chi tiết tác vụ | \033[36mCtrl+C\033[90m: Dừng]\033[0m"

        lines.append("")
        lines.append(footer)

        # Nếu chế độ mở rộng đang bật, hiển thị khung log của tác vụ đang chạy hoặc vừa chạy
        if self.show_details and active_task and active_task.logs:
            lines.append("")
            lines.append(f"  \033[90m┌─── \033[1;37mNhật ký tác vụ: {active_task.name}\033[0m \033[90m" + "─" * 30 + "┐\033[0m")
            recent_logs = active_task.logs[-8:]  # Hiển thị 8 dòng gần nhất
            for log_line in recent_logs:
                lines.append(f"  \033[90m│\033[0m \033[90m{log_line[:76]}\033[0m")
            lines.append("  \033[90m└" + "─" * 60 + "┘\033[0m")

        view = "\n".join(lines)
        return view if supports_unicode(sys.stdout) else ascii_text(view)

    def _start_key_listener(self) -> None:
        """Starts background non-blocking key listener for Ctrl+O."""
        if not self.is_interactive or not sys.stdin.isatty():
            return

        def _listen() -> None:
            try:
                import termios
                import tty

                old_settings = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
                try:
                    while not self._stop_listener.is_set():
                        # Kiểm tra xem có dữ liệu stdin trong 0.1 giây không
                        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                        if rlist:
                            ch = sys.stdin.read(1)
                            if ch:
                                self.handle_key(ch)
                finally:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            except Exception:
                pass

        self._stop_listener.clear()
        self._listener_thread = threading.Thread(target=_listen, daemon=True)
        self._listener_thread.start()

    def _stop_key_listener(self) -> None:
        """Stops background key listener."""
        self._stop_listener.set()
        if self._listener_thread and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=0.3)

    def run_task(self, task: Task, action: Callable[[Task], Any]) -> Any:
        """Executes a single task with live status updates."""
        task.status = TaskStatus.RUNNING
        task.start_time = time.perf_counter()
        self._start_key_listener()

        try:
            result = action(task)
            task.duration = max(time.perf_counter() - task.start_time, 0.0)
            task.status = TaskStatus.COMPLETED
            return result
        except Exception as e:
            task.duration = max(time.perf_counter() - task.start_time, 0.0)
            task.status = TaskStatus.FAILED
            task.error = str(e)
            raise
        finally:
            self._stop_key_listener()
