"""Join server-side VRAM samples into harness raw records by UTC repeat window."""

from __future__ import annotations

import json

import pytest

from experiments import modal_vram_join as vj


def rec(run_id, start, end, **extra):
    return {"run_id": run_id, "phase": "measured", "run_vram_peak_mb": None,
            "run_started_at_utc": f"2026-10-25T{start}+00:00",
            "run_finished_at_utc": f"2026-10-25T{end}+00:00", **extra}


def sample(t, used):
    return {"ts_utc": f"2026-10-25T{t}+00:00", "used_mb": used, "total_mb": 23034.0}


SAMPLES = [sample("09:59:58", 11000), sample("10:00:00", 12000), sample("10:00:10", 12500),
           sample("10:00:29", 12300), sample("10:01:00", 13000)]


def test_peak_is_max_inside_window_spec_example():
    out = vj.join_vram([rec("r1", "10:00:00", "10:00:30")], SAMPLES)

    assert out[0]["run_vram_peak_mb"] == 12500
    assert out[0]["vram_join_samples"] == 3
    assert out[0]["vram_source"] == "server_nvml"


def test_epsilon_includes_sample_just_outside_window():
    samples = [sample("10:00:30.400000", 14000)]

    out = vj.join_vram([rec("r1", "10:00:00", "10:00:30")], samples, epsilon_s=0.5)

    assert out[0]["run_vram_peak_mb"] == 14000


def test_epsilon_excludes_sample_beyond_tolerance():
    samples = [sample("10:00:31", 14000), sample("10:00:05", 12000)]

    out = vj.join_vram([rec("r1", "10:00:00", "10:00:30")], samples, epsilon_s=0.5)

    assert out[0]["run_vram_peak_mb"] == 12000


def test_no_sample_in_window_is_an_error_naming_run():
    with pytest.raises(vj.VramJoinError, match="r2"):
        vj.join_vram([rec("r1", "10:00:00", "10:00:30"), rec("r2", "11:00:00", "11:00:05")], SAMPLES)


def test_other_fields_untouched_and_input_not_mutated():
    original = rec("r1", "10:00:00", "10:00:30", tpot_ms=21.5, lane="vanilla")

    out = vj.join_vram([original], SAMPLES)

    assert original["run_vram_peak_mb"] is None
    assert out[0]["tpot_ms"] == 21.5
    assert out[0]["lane"] == "vanilla"


def test_records_without_window_are_rejected():
    bad = {"run_id": "r1", "phase": "measured"}

    with pytest.raises(vj.VramJoinError, match="run_started_at_utc"):
        vj.join_vram([bad], SAMPLES)


def test_cli_writes_new_file_and_never_overwrites_raw(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "raw.jsonl"
    vram = tmp_path / "vram.jsonl"
    raw.write_text(json.dumps(rec("r1", "10:00:00", "10:00:30")) + "\n", encoding="utf-8")
    vram.write_text("".join(json.dumps(s) + "\n" for s in SAMPLES), encoding="utf-8")
    out = tmp_path / "raw.vram.jsonl"

    assert vj.main(["--raw", str(raw), "--vram", str(vram), "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8").splitlines()[0])["run_vram_peak_mb"] == 12500
    assert json.loads(raw.read_text(encoding="utf-8"))["run_vram_peak_mb"] is None
    assert vj.main(["--raw", str(raw), "--vram", str(vram), "--out", str(raw)]) == 2


def test_cli_refuses_paths_outside_working_directory(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    raw = work / "raw.jsonl"
    raw.write_text(json.dumps(rec("r1", "10:00:00", "10:00:30")) + "\n", encoding="utf-8")
    (work / "vram.jsonl").write_text(json.dumps(SAMPLES[2]) + "\n", encoding="utf-8")
    monkeypatch.chdir(work)

    assert vj.main(["--raw", "raw.jsonl", "--vram", "vram.jsonl", "--out", str(tmp_path / "escape.jsonl")]) == 2
    assert not (tmp_path / "escape.jsonl").exists()
