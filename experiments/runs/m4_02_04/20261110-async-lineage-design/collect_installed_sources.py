"""Read-only CPU inspection of the exact project-pinned vLLM image."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import modal
from experiments.modal_runner import image, SOURCE_COMMIT, SOURCE_ROOT

app = modal.App("m40204-lineage-source-inspection")
image = image.add_local_dir(SOURCE_ROOT / "experiments", "/root/experiments", ignore=["**/__pycache__/**"])


@app.function(image=image, timeout=180, retries=0)
def collect():
    distribution = importlib.metadata.distribution("vllm")
    assert distribution.version == "0.30.0"
    files = {}
    for entry in distribution.files:
        name = str(entry).replace("\\", "/")
        if not name.endswith(".py") or not name.startswith("vllm/"):
            continue
        # Entire V1 tree permits checking identity candidates beyond named files.
        if name.startswith("vllm/v1/") or name in ("vllm/config/scheduler.py", "vllm/config/speculative.py"):
            files[name] = distribution.locate_file(entry).read_text()
    return {"vllm": distribution.version, "project_sha": SOURCE_COMMIT, "files": files}


@app.local_entrypoint()
def main():
    root = Path(__file__).resolve().parent
    call = collect.spawn()
    (root / "inspection_call.json").write_text(json.dumps({"call_id": call.object_id}))
    result = call.get(timeout=240)
    records = []
    for name, content in result.pop("files").items():
        path = root / "installed" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode())
        records.append({"path": name, "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    result.update(call_id=call.object_id, files=records)
    (root / "installed_source_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"vllm": result["vllm"], "source_files": len(records), "call_id": call.object_id}))
