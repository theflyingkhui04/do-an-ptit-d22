"""Interactive terminal chat session with real-time streaming and performance box."""
from __future__ import annotations

import sys

try:
    from flexsds.cli.output import print_output as print, write_output
except ModuleNotFoundError:
    from src.flexsds.cli.output import print_output as print, write_output
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from flexsds.client.client import FlexSDSClient, EndpointUnreachableError
    from flexsds.cli.wizard import render_box
except ModuleNotFoundError:
    from src.flexsds.client.client import FlexSDSClient, EndpointUnreachableError
    from src.flexsds.cli.wizard import render_box


@dataclass
class PerformanceMetrics:
    """Performance measurements for one generation turn."""
    total_chunks: int
    elapsed_seconds: float

    @property
    def chunks_per_second(self) -> float:
        if self.elapsed_seconds <= 0 or self.total_chunks == 0:
            return 0.0
        return round(self.total_chunks / self.elapsed_seconds, 2)

    @property
    def ms_per_chunk(self) -> float:
        """Elapsed milliseconds per SSE content chunk; not TPOT."""
        if self.total_chunks == 0 or self.elapsed_seconds <= 0:
            return 0.0
        return round((self.elapsed_seconds / self.total_chunks) * 1000.0, 2)


class InteractiveChatSession:
    """Manages multi-turn streaming chat on the terminal."""

    def __init__(
        self,
        client: FlexSDSClient,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        self.client = client
        self.model = model or "qwen2.5-7b-adaptive"
        self.temperature = temperature
        self.history: List[Dict[str, str]] = []

    def start(self) -> int:
        """Runs the interactive chat loop."""
        if not self.client.check_health():
            print(f"\n\033[31m[Lỗi kết nối]\033[0m Không thể kết nối tới máy chủ tại: {self.client.base_url}")
            print("  Vui lòng đảm bảo máy chủ đã được bật bằng lệnh: flexsds serve")
            return 1

        print(f"\n\033[1;32m✔ Đã kết nối tới FlexSDS Serving Endpoint:\033[0m \033[36m{self.client.base_url}\033[0m")
        print(f"  Mô hình: \033[1;37m{self.model}\033[0m | Đầu cơ: \033[33mEAGLE-3.1 (Adaptive k động)\033[0m")
        print("  \033[90m(Gõ tin nhắn để trò chuyện, '/exit' để thoát, '/clear' để xóa ngữ cảnh)\033[0m\n")

        while True:
            try:
                user_input = input("\033[1;32mYou >\033[0m ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\033[90mPhiên trò chuyện kết thúc.\033[0m")
                return 0

            if not user_input:
                continue

            if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
                print("\033[90mPhiên trò chuyện kết thúc.\033[0m")
                return 0

            if user_input.lower() == "/clear":
                self.history.clear()
                print("\033[32m✔ Đã xóa lịch sử ngữ cảnh trò chuyện.\033[0m\n")
                continue

            self.history.append({"role": "user", "content": user_input})
            write_output("\033[1;36mFlexSDS >\033[0m ")
            sys.stdout.flush()

            start_t = time.perf_counter()
            chunk_count = 0
            full_response: List[str] = []

            try:
                for token in self.client.stream_chat(
                    self.history,
                    model=self.model,
                    temperature=self.temperature,
                ):
                    write_output(token)
                    sys.stdout.flush()
                    full_response.append(token)
                    chunk_count += 1
                write_output("\n")
                sys.stdout.flush()
            except KeyboardInterrupt:
                write_output("\n\033[90m(Hủy lượt trả lời hiện tại)\033[0m\n")
                sys.stdout.flush()
                continue
            except EndpointUnreachableError as e:
                print(f"\n\033[31m[Lỗi đường truyền]\033[0m Mất kết nối tới server: {e}")
                continue

            elapsed = max(time.perf_counter() - start_t, 0.001)
            metrics = PerformanceMetrics(total_chunks=chunk_count, elapsed_seconds=elapsed)
            self.history.append({"role": "assistant", "content": "".join(full_response)})

            # Render box thống kê hiệu năng
            items = [
                ("SSE content chunks", f"{metrics.total_chunks} chunks", "\033[1;37m"),
                ("Thời gian xử lý", f"{metrics.elapsed_seconds:.2f} s", "\033[36m"),
                ("Tốc độ trung bình", f"{metrics.chunks_per_second} chunks/s", "\033[32m"),
                ("Elapsed per chunk", f"{metrics.ms_per_chunk} ms/chunk", "\033[33m"),
            ]
            box = render_box("Thống kê suy luận", items, min_width=44)
            print(f"\n{box}\n")


def run_interactive_chat(
    endpoint: str = "http://localhost:8000/v1",
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> int:
    """Entry point for CLI flexsds chat command."""
    client = FlexSDSClient(base_url=endpoint)
    session = InteractiveChatSession(client, model=model, temperature=temperature)
    return session.start()
