"""CPU-only forensic audit regressions; no network or model execution."""
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("m203_evidence_analyzer", ROOT / "scripts/m2_03_analyze_evidence.py")
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)


def records(ids=(10, 20), finish="stop"):
    identity = dict(workload="W1", sample_id="synthetic", dataset_version="v1",
                    prompt_version="p1", request_sha256="a" * 64)
    common = dict(identity, ok=True, http_status=200, prompt_token_ids=[1, 2],
                  token_ids=list(ids), completion_tokens_reported=len(ids),
                  finish_reason=finish, content="diagnostic")
    return identity, dict(common, lane="vanilla", response_id="v"), dict(common, lane="adaptive", response_id="a")


@pytest.mark.parametrize("left,right,prefix,div", [
    ([10, 20], [10, 20], 2, None),
    ([10, 20], [11, 20], 0, (10, 11)),
    ([10, 20], [10, 21], 1, (20, 21)),
    ([10, 20], [10], 1, (20, None)),
    ([10], [10, 20], 1, (None, 20)),
])
def test_sequence_boundaries(left, right, prefix, div):
    r = a.sequence_comparison(left, right)
    assert r["equal_prefix_length"] == prefix
    assert r["exact_token_match"] == (div is None)
    if div is None:
        assert r["first_mismatch"] is None
    else:
        assert r["first_mismatch"] == dict(position_zero_based=prefix,
                                           vanilla_token_id=div[0], adaptive_token_id=div[1])


@pytest.mark.parametrize("mismatch", [False, True])
def test_truncation_is_independent_of_equality(mismatch):
    f, v, d = records(finish="length")
    d["token_ids"] = [10, 21] if mismatch else [10, 20]
    r = a.compare_pair(v, d, f)
    assert r["truncated"] is True
    assert r["passed"] is (not mismatch)


@pytest.mark.parametrize("field,value", [
    ("token_ids", [True]), ("token_ids", [-1]), ("token_ids", []),
    ("completion_tokens_reported", 99), ("completion_tokens_reported", True),
    ("sample_id", "other"), ("request_sha256", "b" * 64),
    ("prompt_token_ids", [3]), ("lane", "vanilla"), ("ok", False),
    ("finish_reason", "error"), ("response_id", None),
])
def test_bad_pair_rejected(field, value):
    f, v, d = records()
    d[field] = value
    with pytest.raises(a.AuditError):
        a.compare_pair(v, d, f)


@pytest.mark.parametrize("raw", [b'{', b'[]', b'{"x":1,"x":2}', b'{"x":NaN}', b'\xff'])
def test_malformed_json_rejected(raw):
    with pytest.raises(a.AuditError):
        a.parse_json(raw)


def test_blank_jsonl_not_filtered():
    with pytest.raises(a.AuditError):
        a.parse_jsonl(b'{}\n\n{}\n')


def test_duplicate_missing_extra_records():
    row = dict(workload="W1", sample_id="one")
    for rows, expected in [([row, row], None), ([], {("W1", "one")}),
                           ([row], {("W1", "other")})]:
        with pytest.raises(a.AuditError):
            a.index_records(rows, expected)


def test_checksum_uses_exact_bytes_and_never_normalizes():
    lf, crlf = b'{"x":1}\n', b'{"x":1}\r\n'
    assert a.checked_bytes(lf, a.digest(lf), "fixture") == lf
    with pytest.raises(a.AuditError, match="checksum mismatch"):
        a.checked_bytes(crlf, a.digest(lf), "fixture")
    assert crlf == b'{"x":1}\r\n'


@pytest.mark.parametrize("mode", ["unique", "missing", "ambiguous", "duplicate", "extra"])
def test_telemetry_identity_join(mode):
    _, _, d = records()
    rows = [{"request_id": "a-1234abcd", "lane": "adaptive"}]
    if mode == "missing":
        rows = []
    elif mode == "ambiguous":
        rows.append({"request_id": "a-8765abcd", "lane": "adaptive"})
    elif mode == "duplicate":
        rows *= 2
    elif mode == "extra":
        rows.append({"request_id": "other-8765abcd", "lane": "adaptive"})
    report = a.link_telemetry({("W1", "synthetic"): d}, rows)
    assert report["status"] == ("PASS" if mode == "unique" else "FAIL")


def test_telemetry_reordered_rows():
    _, _, d = records()
    e = dict(d, response_id="b")
    rows = [{"request_id": "b-1234abcd", "lane": "adaptive"},
            {"request_id": "a-1234abcd", "lane": "adaptive"}]
    result = a.link_telemetry({("W1", "one"): d, ("W1", "two"): e}, rows)
    assert result["status"] == "PASS"
    assert result["links"][0]["request_id"] == "a-1234abcd"


@pytest.mark.parametrize("valid,gate,strict,expected", [
    (True, False, False, 0), (True, False, True, 1),
    (True, True, True, 0), (False, False, False, 2), (False, False, True, 2),
])
def test_exit_semantics(valid, gate, strict, expected):
    assert a.exit_status(dict(analysis_valid=valid, token_id_gate_passed=gate), strict) == expected


@pytest.fixture(scope="module")
def checkpoint():
    # Missing Git checkpoint is a failure, never a skip or working-tree fallback.
    reader = a.GitBlobs(ROOT)
    before = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
              for p in reader.git("ls-tree", "-r", "--name-only", a.CHECKPOINT,
                                  "experiments/runs/m2_03", "experiments/runs/m2_03_cacheoff").decode().splitlines()}
    report = a.analyze(ROOT)
    yield report, reader
    assert before == {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in before}


def test_immutable_checkpoint_totals_and_boundaries(checkpoint):
    report, _ = checkpoint
    run = report["runs"]["cacheoff"]
    assert report["analysis_valid"] is True
    assert report["token_id_gate_passed"] is False
    assert (run["denominator"], run["valid_pairs"], run["exact_sequence_matches"], run["mismatches"], run["truncated_pairs"]) == (54, 54, 47, 7, 7)
    assert {(p["workload"], p["sample_id"]): p["first_mismatch"]["position_zero_based"]
            for p in run["pairs"] if not p["exact_token_match"]} == {
                ("W1", "m0_04_005"): 548, ("W1", "m0_04_021"): 219,
                ("W2", "w2_sql_v2_003"): 46, ("W3", "w3_business_v2_001"): 385,
                ("W3", "w3_business_v2_003"): 60, ("W3", "w3_business_v2_008"): 183,
                ("W3", "w3_business_v2_010"): 601}
    assert {p["sample_id"] for p in run["pairs"] if p["truncated"]} == {
        "w2_sql_v2_003", "w2_sql_v2_005", "w2_sql_v2_006", "w2_sql_v2_007",
        "w2_sql_v2_009", "w2_sql_v2_010", "w2_sql_v2_011"}
    assert run["telemetry_linkage"]["status"] == "PASS"
    assert len(run["telemetry_linkage"]["links"]) == 54
    assert run["checkpoint_sha"] == a.CHECKPOINT
    assert run["execution_sha"] == a.RUNS["cacheoff"][1] != a.CHECKPOINT


def test_cross_run_sequences(checkpoint):
    report, _ = checkpoint
    w1, w3 = [r["comparisons"] for r in report["cross_run"]]
    assert w1["canonical_adaptive_vs_cacheoff_vanilla"]["exact_token_match"]
    assert w1["canonical_vanilla_vs_cacheoff_adaptive"]["equal_prefix_length"] == 537
    for lane in ("vanilla", "adaptive"):
        assert w1[lane + "_across_runs"]["equal_prefix_length"] == 219
        assert w3[lane + "_across_runs"]["exact_token_match"]
    for run, expected in (("canonical", (11, 756)), ("cacheoff", (756, 11))):
        rows = {(p["workload"], p["sample_id"]): p for p in report["runs"][run]["pairs"]}
        d = rows["W1", "m0_04_021"]["first_mismatch"]
        assert (d["vanilla_token_id"], d["adaptive_token_id"]) == expected
        assert rows["W3", "w3_business_v2_001"]["first_mismatch"] == {
            "position_zero_based": 385, "vanilla_token_id": 128360, "adaptive_token_id": 128317}


class AlteredBlobs:
    def __init__(self, reader, path, transform):
        self.reader, self.path, self.transform = reader, path, transform

    @property
    def hashes(self):
        return self.reader.hashes

    def read(self, path, sha=a.CHECKPOINT):
        raw = self.reader.read(path, sha)
        return self.transform(raw) if path == self.path and sha == a.CHECKPOINT else raw


@pytest.mark.parametrize("kind", ["checksum", "summary", "plan", "provenance", "missing"])
def test_tampered_checkpoint_inputs_fail_closed(checkpoint, kind):
    _, reader = checkpoint
    base = a.RUNS["cacheoff"][0]
    file = "vanilla.jsonl" if kind == "checksum" else "reference_meta.json" if kind in ("plan", "provenance") else "summary.json"

    def mutate(raw):
        if kind == "checksum":
            return raw + b'\n'
        obj = json.loads(raw)
        if kind == "summary":
            obj["exact_sequence_matches"] = 54
        elif kind == "plan":
            obj["plan_sha256"] = "0" * 64
        elif kind == "provenance":
            obj["runtime"]["git_sha"] = "0" * 40
        else:
            obj["comparisons"].pop()
        return json.dumps(obj).encode()

    with pytest.raises(a.AuditError):
        a.audit_run(AlteredBlobs(reader, base + "/" + file, mutate), "cacheoff")


def test_plan_hash_recomputed_not_only_compared_between_metadata(checkpoint):
    _, reader = checkpoint
    meta = a.parse_json(reader.read(a.RUNS["cacheoff"][0] + "/reference_meta.json"))
    meta["plan_sha256"] = "0" * 64
    with pytest.raises(a.AuditError, match="plan checksum mismatch"):
        a.frozen_plan(reader, meta, a.RUNS["cacheoff"][1])


def test_blob_failure_never_reads_working_tree(checkpoint):
    _, reader = checkpoint
    with pytest.raises(a.AuditError, match="no working-tree fallback"):
        reader.read("this/path/is/not/in/the/checkpoint.json")


def test_malformed_blob_cli_is_exit_2(checkpoint):
    _, reader = checkpoint
    altered = AlteredBlobs(reader, a.RUNS["cacheoff"][0] + "/summary.json", lambda raw: b'{')
    with patch.object(a, "GitBlobs", return_value=altered), redirect_stdout(io.StringIO()) as out:
        assert a.main([]) == 2
    report = json.loads(out.getvalue())
    assert report["analysis_valid"] is False and report["token_id_gate_passed"] is None


def test_cli_exit_modes_deterministic_stdout(checkpoint):
    expected, _ = checkpoint
    outputs = []
    for flags in ([], ["--require-gate-pass"]):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/m2_03_analyze_evidence.py"), *flags],
                                cwd=ROOT, capture_output=True, check=False)
        assert result.returncode == (1 if flags else 0)
        assert not result.stderr
        assert json.loads(result.stdout) == expected
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]


def test_git_failure_is_exit_2_no_fallback(tmp_path):
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        rc = a.main(["--repo", str(tmp_path)])
    report = json.loads(stdout.getvalue())
    assert rc == 2 and report["analysis_valid"] is False
    assert report["token_id_gate_passed"] is None
    assert "no working-tree fallback" in report["errors"][0]


def test_telemetry_failure_separate_from_gate(checkpoint):
    _, reader = checkpoint
    path = a.RUNS["cacheoff"][0] + "/adaptive_telemetry.jsonl"

    def add_ambiguous_record(raw):
        rows = a.parse_jsonl(raw)
        extra = dict(rows[0], request_id=rows[0]["request_id"][:-8] + "00000000")
        return raw + json.dumps(extra).encode() + b'\n'

    altered = AlteredBlobs(reader, path, add_ambiguous_record)
    with patch.object(a, "GitBlobs", return_value=altered), redirect_stdout(io.StringIO()) as out:
        assert a.main([]) == 2
    report = json.loads(out.getvalue())
    assert report["analysis_valid"] is False
    assert report["token_id_gate_passed"] is False
    assert report["telemetry_linkage"]["status"] == "FAIL"


@pytest.mark.parametrize("flags", [[], ["--require-gate-pass"]])
def test_wrong_telemetry_lane_preserves_verified_gate(checkpoint, flags):
    _, reader = checkpoint
    path = a.RUNS["cacheoff"][0] + "/adaptive_telemetry.jsonl"

    def wrong_lane(raw):
        rows = a.parse_jsonl(raw)
        rows[0]["lane"] = "vanilla"
        return b''.join(json.dumps(row).encode() + b'\n' for row in rows)

    altered = AlteredBlobs(reader, path, wrong_lane)
    with patch.object(a, "GitBlobs", return_value=altered), redirect_stdout(io.StringIO()) as out:
        rc = a.main(flags)
    report = json.loads(out.getvalue())
    assert rc == 2
    assert report["analysis_valid"] is False
    assert report["token_id_gate_passed"] is False
    assert report["telemetry_linkage"] == {"status": "FAIL", "reason": "wrong telemetry lane"}
    assert report["runs"]["cacheoff"]["exact_sequence_matches"] == 47


@pytest.mark.parametrize("vanilla_finish,adaptive_finish", [("stop", "length"), ("length", "stop")])
def test_equal_tokens_different_finish_summary_contract(checkpoint, vanilla_finish, adaptive_finish):
    """Synthetic 54-pair run: raw equality is not the historical pair-pass count."""
    _, reader = checkpoint
    base, execution = a.RUNS["cacheoff"]
    meta = a.parse_json(reader.read(base + "/reference_meta.json"))
    frozen = a.frozen_plan(reader, meta, execution)
    summary = a.parse_json(reader.read(base + "/summary.json"))
    vanilla, adaptive, comparisons, telemetry = [], [], [], []
    for i, identity in enumerate(frozen):
        _, v, d = records()
        v.update(identity, response_id=f"v-{i}")
        d.update(identity, response_id=f"a-{i}")
        if i == 0:
            v["finish_reason"], d["finish_reason"] = vanilla_finish, adaptive_finish
        vanilla.append(v)
        adaptive.append(d)
        telemetry.append({"request_id": d["response_id"] + "-1234abcd", "lane": "adaptive"})
        comparisons.append({
            "workload": identity["workload"], "sample_id": identity["sample_id"],
            "passed": i != 0, "failures": ["finish_reason_mismatch"] if i == 0 else [],
            "first_mismatch": None, "vanilla_token_count": 2, "adaptive_token_count": 2,
            "finish_reason_vanilla": v["finish_reason"], "finish_reason_adaptive": d["finish_reason"],
            "truncated": i == 0, "content_equal_diagnostic_only": True,
        })
    encode = lambda rows: b''.join(json.dumps(row).encode() + b'\n' for row in rows)
    vraw, draw = encode(vanilla), encode(adaptive)
    meta["reference_jsonl_sha256"] = a.digest(vraw)
    summary.update(valid_pairs=54, exact_sequence_matches=53, truncated_pairs=1,
                   exact_sequence_match_rate=53 / 54, token_id_gate_passed=False,
                   full_completion_verified=False, comparisons=comparisons,
                   vanilla_jsonl_sha256=a.digest(vraw), adaptive_jsonl_sha256=a.digest(draw))
    replacements = {
        base + "/vanilla.jsonl": vraw, base + "/adaptive.jsonl": draw,
        base + "/adaptive_telemetry.jsonl": encode(telemetry),
        base + "/summary.json": json.dumps(summary).encode(),
        base + "/reference_meta.json": json.dumps(meta).encode(),
    }

    class SyntheticRun:
        def read(self, path, sha=a.CHECKPOINT):
            if sha == a.CHECKPOINT and path in replacements:
                return replacements[path]
            return reader.read(path, sha)

    result, _ = a.audit_run(SyntheticRun(), "cacheoff")
    assert result["exact_sequence_matches"] == summary["exact_sequence_matches"] == 53
    assert result["exact_sequence_match_rate"] == summary["exact_sequence_match_rate"]
    assert result["mismatches"] == 1
    assert result["exact_token_sequence_matches"] == 54
    assert result["token_id_gate_passed"] is False
    assert result["truncated_pairs"] == 1
    assert sum(p["passed"] for p in result["pairs"]) == 53
    assert all(p["exact_token_match"] for p in result["pairs"])
