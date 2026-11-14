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
    ttft_seconds: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

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

    @property
    def tokens_per_second(self) -> Optional[float]:
        if not isinstance(self.completion_tokens, (int, float)) or self.completion_tokens <= 0:
            return None
        if self.elapsed_seconds <= 0:
            return None
        gen_time = (self.elapsed_seconds - (self.ttft_seconds or 0.0)) if self.ttft_seconds else self.elapsed_seconds
        gen_time = max(gen_time, 0.001)
        return round(self.completion_tokens / gen_time, 2)

    @property
    def ms_per_token(self) -> Optional[float]:
        """Time per output token (TPOT)."""
        if not isinstance(self.completion_tokens, (int, float)) or self.completion_tokens <= 0:
            return None
        if self.elapsed_seconds <= 0:
            return None
        gen_time = (self.elapsed_seconds - (self.ttft_seconds or 0.0)) if self.ttft_seconds else self.elapsed_seconds
        gen_time = max(gen_time, 0.001)
        return round((gen_time / self.completion_tokens) * 1000.0, 2)


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
            print(f"\n\033[31m[Connection Error]\033[0m Cannot connect to server at: {self.client.base_url}")
            print("  Please ensure the serving engine is active (e.g. flexsds serve)\n")
            return 1

        if self.model == "qwen2.5-7b-adaptive":
            try:
                available = self.client.get_models()
                if available:
                    self.model = available[0]
            except Exception:
                pass

        print(f"\n\033[1;32m✔ Connected to FlexSDS Serving Endpoint:\033[0m \033[36m{self.client.base_url}\033[0m")
        print(f"  Model: \033[1;37m{self.model}\033[0m | Speculation: \033[33mEAGLE-3.1 (HardwareAdaptive)\033[0m")
        print("  \033[90m(Type a message to chat, '/exit' to quit, '/clear' to reset context)\033[0m\n")

        while True:
            try:
                user_input = input("\033[1;32mYou >\033[0m ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\033[90mChat session ended.\033[0m")
                return 0

            if not user_input:
                continue

            if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
                print("\033[90mChat session ended.\033[0m")
                return 0

            if user_input.lower() == "/clear":
                self.history.clear()
                print("\033[32m✔ Conversation history cleared.\033[0m\n")
                continue

            self.history.append({"role": "user", "content": user_input})
            write_output("\033[1;36mFlexSDS >\033[0m ")
            sys.stdout.flush()

            start_t = time.perf_counter()
            first_chunk_t: Optional[float] = None
            chunk_count = 0
            full_response: List[str] = []

            try:
                for token in self.client.stream_chat(
                    self.history,
                    model=self.model,
                    temperature=self.temperature,
                ):
                    if first_chunk_t is None:
                        first_chunk_t = time.perf_counter()
                    write_output(token)
                    sys.stdout.flush()
                    full_response.append(token)
                    chunk_count += 1
                write_output("\n")
                sys.stdout.flush()
            except KeyboardInterrupt:
                write_output("\n\033[90m(Response canceled)\033[0m\n")
                sys.stdout.flush()
                continue
            except EndpointUnreachableError as e:
                print(f"\n\033[31m[Network Error]\033[0m Lost connection to server: {e}")
                continue

            elapsed = max(time.perf_counter() - start_t, 0.001)
            ttft = (first_chunk_t - start_t) if first_chunk_t else None

            raw_usage = getattr(self.client, "last_usage", None)
            usage = raw_usage if isinstance(raw_usage, dict) else {}
            p_tokens = usage.get("prompt_tokens")
            c_tokens = usage.get("completion_tokens")
            if not isinstance(p_tokens, int):
                p_tokens = None
            if not isinstance(c_tokens, int):
                c_tokens = None

            metrics = PerformanceMetrics(
                total_chunks=chunk_count,
                elapsed_seconds=elapsed,
                ttft_seconds=ttft,
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
            )
            self.history.append({"role": "assistant", "content": "".join(full_response)})

            # Render performance metrics box
            items = [
                ("SSE content chunks", f"{metrics.total_chunks} chunks", "\033[1;37m"),
            ]
            if metrics.prompt_tokens is not None:
                items.append(("Prompt tokens", f"{metrics.prompt_tokens} tokens", "\033[90m"))
            if metrics.completion_tokens is not None:
                items.append(("Generated tokens", f"{metrics.completion_tokens} tokens", "\033[1;37m"))

            if metrics.ttft_seconds is not None:
                items.append(("Time to first token", f"{metrics.ttft_seconds * 1000.0:.1f} ms", "\033[35m"))

            items.append(("Processing time", f"{metrics.elapsed_seconds:.2f} s", "\033[36m"))

            if metrics.tokens_per_second is not None:
                items.append(("Generation speed", f"{metrics.tokens_per_second} tokens/s", "\033[32m"))
            else:
                items.append(("Average speed", f"{metrics.chunks_per_second} chunks/s", "\033[32m"))

            if metrics.ms_per_token is not None:
                items.append(("Elapsed per token (TPOT)", f"{metrics.ms_per_token} ms/token", "\033[33m"))
            else:
                items.append(("Elapsed per chunk", f"{metrics.ms_per_chunk} ms/chunk", "\033[33m"))

            box = render_box("Inference Telemetry", items, min_width=44)
            print(f"\n{box}\n")


def run_interactive_chat(
    endpoint: str = "http://localhost:8000/v1",
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> int:
    """Entry point for CLI flexsds chat command."""
    import json
    from pathlib import Path

    if endpoint == "http://localhost:8000/v1":
        cfg_file = Path("serving_config.json")
        if cfg_file.exists():
            try:
                with open(cfg_file, "r", encoding="utf-8") as f:
                    cfg_data = json.load(f)
                saved_ep = cfg_data.get("endpoint") or cfg_data.get("serving", {}).get("endpoint")
                if saved_ep:
                    endpoint = saved_ep
            except Exception:
                pass

    client = FlexSDSClient(base_url=endpoint)
    session = InteractiveChatSession(client, model=model, temperature=temperature)
    return session.start()
