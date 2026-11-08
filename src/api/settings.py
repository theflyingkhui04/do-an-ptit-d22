"""Backend settings, read from environment variables only (secrets never live in files)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

LANES = ("vanilla", "static", "adaptive", "eagle")
# Lane per step: fastest lane at c=1 in the M3-05 campaign
# (experiments/results/m3_sweep_4lanes_summary.json): W1 adaptive 1.45x,
# W2 static 1.35x, W3 eagle 1.24x (adaptive 0.75x on W3).
DEFAULT_LANE_MAP = {"node1": "adaptive", "node2": "static", "answer": "static", "report": "eagle"}
MIN_DEMO_KEY_LENGTH = 8


@dataclass(frozen=True)
class Settings:
    demo_key: str
    frontend_origins: tuple[str, ...] = ("http://localhost:3000",)
    writer_url: str | None = None
    reader_url: str | None = None
    supabase_url: str | None = None
    supabase_service_key: str | None = None
    image_bucket: str = "invoice-images"
    modal_token: str | None = None
    lane_map: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_LANE_MAP))
    gpu_label: str = "A10G"
    upload_limit: int = 10
    question_limit: int = 30
    rate_window_s: float = 600.0
    max_upload_bytes: int = 10 * 1024 * 1024

    def __post_init__(self) -> None:
        if len(self.demo_key or "") < MIN_DEMO_KEY_LENGTH:
            raise ValueError(f"DEMO_KEY must be set (at least {MIN_DEMO_KEY_LENGTH} characters)")
        for step, lane in self.lane_map.items():
            if step not in DEFAULT_LANE_MAP or lane not in LANES:
                raise ValueError(f"Invalid lane map entry {step}={lane!r}")

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> Settings:
        token = None
        if environ.get("MODAL_PROXY_TOKEN_ID") and environ.get("MODAL_PROXY_TOKEN_SECRET"):
            token = f"{environ['MODAL_PROXY_TOKEN_ID']}.{environ['MODAL_PROXY_TOKEN_SECRET']}"
        lane_map = dict(DEFAULT_LANE_MAP)
        for step in DEFAULT_LANE_MAP:
            if environ.get(f"DEMO_LANE_{step.upper()}"):
                lane_map[step] = environ[f"DEMO_LANE_{step.upper()}"]
        origins = tuple(o.strip().rstrip("/") for o in environ.get("DEMO_FRONTEND_ORIGINS", "").split(",")
                        if o.strip())
        return cls(
            demo_key=environ.get("DEMO_KEY", ""),
            frontend_origins=origins or ("http://localhost:3000",),
            writer_url=environ.get("DEMO_DATABASE_URL") or None,
            reader_url=environ.get("DEMO_READER_DATABASE_URL") or None,
            supabase_url=environ.get("SUPABASE_URL") or None,
            supabase_service_key=environ.get("SUPABASE_SERVICE_KEY") or None,
            image_bucket=environ.get("DEMO_IMAGE_BUCKET") or "invoice-images",
            modal_token=token,
            lane_map=lane_map,
            gpu_label=environ.get("DEMO_GPU_LABEL") or "A10G",
        )

    def secret_values(self) -> list[str]:
        values = [self.demo_key, self.writer_url, self.reader_url, self.supabase_service_key, self.modal_token]
        if self.modal_token:
            values.extend(self.modal_token.split("."))
        return [v for v in values if v and len(v) >= 6]
