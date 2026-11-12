"""Read-only Modal allocation snapshot, no function invocation or GPU launch."""
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from modal.client import _Client
from modal_proto import api_pb2


async def main():
    client = await _Client.from_env()
    layout = await client._stub.AppGetLayout(api_pb2.AppGetLayoutRequest(app_id="ap-Zf9l4KDUxWPZUWyCmELjtI"))
    result = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), functions={})
    for name, fid in layout.app_layout.function_ids.items():
        stats = await client._stub.FunctionGetCurrentStats(api_pb2.FunctionGetCurrentStatsRequest(function_id=fid))
        result["functions"][name] = dict(backlog=stats.backlog, containers=stats.num_total_tasks,
                                       running_inputs=stats.num_running_inputs)
    path = Path(__file__).with_name("allocation-snapshots.jsonl")
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(result) + "\n")
    print(json.dumps(result, indent=2))


asyncio.run(asyncio.wait_for(main(), timeout=25))
