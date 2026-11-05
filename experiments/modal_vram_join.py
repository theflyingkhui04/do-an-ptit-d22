"""[M3-01] Fill run_vram_peak_mb from the server's own NVML samples.

In remote-server mode (profile m3-l4-modal-v1) the harness cannot read the
Modal L4 GPU, so each repeat records its UTC window (run_started_at_utc,
run_finished_at_utc) and leaves run_vram_peak_mb empty. The server container
writes vram.jsonl ({ts_utc, used_mb, total_mb}); this module sets each repeat's
peak to the largest sample inside its window (± epsilon for clock skew).

    python -m experiments.modal_vram_join --raw raw.jsonl --vram vram.jsonl --out raw.vram.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


class VramJoinError(ValueError):
    """A repeat cannot be matched to server VRAM samples."""


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def join_vram(
    records: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    epsilon_s: float = 0.5,
) -> list[dict[str, Any]]:
    """Return copies of ``records`` with run_vram_peak_mb taken from ``samples``."""
    points = sorted((_ts(s["ts_utc"]), float(s["used_mb"])) for s in samples)
    eps = timedelta(seconds=epsilon_s)
    peaks: dict[str, tuple[float, int]] = {}
    missing: list[str] = []
    out: list[dict[str, Any]] = []

    for record in records:
        run_id = record.get("run_id")
        if not record.get("run_started_at_utc") or not record.get("run_finished_at_utc"):
            raise VramJoinError(f"record of run {run_id!r} has no run_started_at_utc/run_finished_at_utc")
        if run_id not in peaks:
            start = _ts(record["run_started_at_utc"]) - eps
            end = _ts(record["run_finished_at_utc"]) + eps
            inside = [used for ts, used in points if start <= ts <= end]
            if not inside:
                missing.append(run_id)
                peaks[run_id] = (float("nan"), 0)
            else:
                peaks[run_id] = (round(max(inside), 2), len(inside))
        peak, count = peaks[run_id]
        out.append({**record, "run_vram_peak_mb": peak, "vram_join_samples": count,
                    "vram_source": "server_nvml"})

    if missing:
        raise VramJoinError(f"no server VRAM sample inside the window of run(s): {', '.join(sorted(set(missing)))}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Join server VRAM samples into harness raw JSONL")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--vram", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epsilon-s", type=float, default=0.5)
    args = parser.parse_args(argv)

    base = Path.cwd().resolve()
    paths = {}
    for name in ("raw", "vram", "out"):
        resolved = getattr(args, name).resolve()
        if not resolved.is_relative_to(base):
            print(f"[FAIL] --{name} {resolved} is outside the working directory {base}", file=sys.stderr)
            return 2
        paths[name] = resolved
    args.raw, args.vram, args.out = paths["raw"], paths["vram"], paths["out"]

    if args.out == args.raw:
        print("[FAIL] --out must differ from --raw (the original raw file is never overwritten)", file=sys.stderr)
        return 2
    try:
        joined = join_vram(load_jsonl(args.raw), load_jsonl(args.vram), args.epsilon_s)
    except (OSError, ValueError, KeyError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in joined), encoding="utf-8")
    print(f"Joined {len(joined)} records -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
