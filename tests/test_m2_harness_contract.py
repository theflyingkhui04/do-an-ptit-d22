"""Exercise the existing harness CLI and warm-up with no GPU or HTTP calls."""
import argparse
import asyncio
import sys
from unittest.mock import AsyncMock, patch

import pytest

from experiments import benchmark_harness as harness
from experiments import run_m2_campaign as campaign


def test_existing_cli_supports_adaptive_without_duplicate_engine_flag():
    with patch.object(sys, "argv", ["harness", "--model", "test", "--lane", "adaptive"]):
        args = harness.parse_args()
    assert args.lane == "adaptive"
    assert args.repeats == 5
    assert harness.resolve_concurrencies(args) == [1, 2, 4, 8, 16, 32]
    assert not hasattr(args, "engine")


def test_all_planned_commands_are_valid_harness_cli(tmp_path):
    for item in campaign.build_plan(tmp_path / "attempt"):
        with patch.object(sys, "argv", item["harness"][1:]):
            args = harness.parse_args()
        assert args.lane in ("vanilla", "static", "adaptive")
        assert args.repeats >= 5
        assert args.workload_file.is_file()
        assert not args.overwrite and not args.skip_env_check


@pytest.mark.parametrize("concurrency,expected", [(1, 3), (2, 3), (4, 4), (8, 8), (16, 16), (32, 32)])
def test_warmup_is_concurrency_matched(concurrency, expected):
    args = argparse.Namespace(warmup_requests=3)
    request = AsyncMock(return_value=[{"success": True}])
    with patch.object(harness, "run_requests", request):
        assert asyncio.run(harness.warm_up_cell(
            session=None, args=args, env={}, workload_rows=[], workload_integrity=None,
            sample_policy="cycle-rotate", experiment_id="cpu-test", cell_id="test",
            concurrency=concurrency,
        )) == 0
    assert request.call_args.kwargs["count"] == expected
    assert request.call_args.kwargs["phase"] == "warmup"
    assert request.call_args.kwargs["sample_offset"] == 0


def test_cell_warms_once_before_five_measured_repeats():
    calls = []

    async def warm(**kwargs):
        calls.append("warmup")
        return 0

    async def repeat(**kwargs):
        calls.append(kwargs["repeat_index"])
        return 0

    args = argparse.Namespace(lane="adaptive", workload="W3", repeats=5)
    with patch.object(harness, "warm_up_cell", side_effect=warm), \
         patch.object(harness, "run_repeat", side_effect=repeat):
        assert asyncio.run(harness.run_cell(
            session=None, args=args, env={}, workload_rows=[], workload_integrity=None,
            sample_policy="cycle-rotate", gpu_handle=None, raw_writer=None,
            experiment_id="cpu-test", concurrency=32,
        )) == 0
    assert calls == ["warmup", 1, 2, 3, 4, 5]


def test_client_keepalive_is_shorter_than_the_server_keepalive():
    # vLLM (uvicorn) closes idle connections after 5 s (VLLM_HTTP_TIMEOUT_KEEP_ALIVE); aiohttp kept them
    # for 15 s, so the first request of a repeat could reuse a closed connection and fail with
    # ServerDisconnectedError (A100 static W1_V3 c8, 27/11). Connections are still reused within a repeat.
    assert harness.CLIENT_KEEPALIVE_S < harness.SERVER_KEEPALIVE_S == 5.0

    async def build():
        connector = harness.make_connector()
        try:
            return connector._keepalive_timeout, connector.force_close
        finally:
            await connector.close()

    keepalive, force_close = asyncio.run(build())
    assert keepalive == harness.CLIENT_KEEPALIVE_S
    assert force_close is False
