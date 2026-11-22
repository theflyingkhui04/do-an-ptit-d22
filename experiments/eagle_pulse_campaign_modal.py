"""Explicit-live-only adapter. Importing this module performs no external action.

The single-use generator waits for a queue command between slots. The driver
sends the next command only after durable local recording of the prior result.
"""
from contextlib import ExitStack
from pathlib import Path

from experiments.eagle_pulse_campaign import read_json, require, verify_source


class ModalExecutor:
    synthetic = False

    def __init__(self, root):
        self.root = Path(root)

    def allocate(self, plan, index):
        # This method is reachable only from the explicit live run command,
        # after the source, state, raw and single-writer gates have passed.
        package = verify_source(self.root, plan)
        import modal
        stack = ExitStack()
        try:
            app = modal.App("secp-pulse-2r-dual-campaign")
            image = (modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu22.04", add_python="3.11")
                     .entrypoint([]).uv_pip_install("vllm==0.30.0").pip_install("psutil").apt_install("git")
                     .add_local_file(str(self.root / "source/reviewed-source.bundle"), "/root/reviewed-source.bundle"))
            volume = modal.Volume.from_name("ptit-model-cache", create_if_missing=False)

            @app.function(image=image, gpu="A10G", max_containers=1, retries=0,
                          timeout=3600, single_use_containers=True, serialized=True,
                          include_source=False,
                          volumes={"/root/.cache/huggingface": volume})
            def allocation(plan, package, commands):
                import hashlib
                from pathlib import Path
                import subprocess
                import sys
                bundle = Path("/root/reviewed-source.bundle")
                if hashlib.sha256(bundle.read_bytes()).hexdigest() != package["bundle_sha256"]:
                    raise RuntimeError("remote_bundle_hash")
                project = Path("/root/reviewed-project")
                subprocess.run(["git", "clone", "--quiet", str(bundle), str(project)], check=True)
                def git(*args):
                    return subprocess.check_output(["git", "-C", str(project), *args]).decode().strip()
                if git("rev-parse", "HEAD") != plan["source_git_sha"] or git("status", "--short"):
                    raise RuntimeError("remote_source_checkout")
                subprocess.run(["git", "-C", str(project), "fsck", "--full"], check=True)
                if (project / ".git/objects/info/alternates").exists():
                    raise RuntimeError("remote_alternates")
                for name, expected in package["files"].items():
                    if hashlib.sha256((project / name).read_bytes()).hexdigest() != expected:
                        raise RuntimeError("remote_source_file:" + name)
                sys.path[:0] = [str(project), str(project / "src")]
                from experiments.eagle_pulse_campaign_worker import LiveWorker
                worker = LiveWorker.__new__(LiveWorker)
                try:
                    worker.__init__(plan, project)
                    yield dict(kind="identity", identity=worker.identity)
                    # A finite maximum of 150 scientific slots per product.
                    for _ in range(151):
                        command = commands.get(timeout=120)
                        if command["kind"] == "stop":
                            break
                        if command["kind"] != "slot":
                            raise RuntimeError("unsupported_worker_command")
                        yield dict(kind="slot", attempt_id=command["attempt_id"],
                            result=worker.execute(command["product"], command["cell"], command["slot"], command["attempt_id"]))
                    else:
                        raise RuntimeError("worker_slot_limit")
                finally:
                    worker.close()

            stack.enter_context(app.run())
            commands = stack.enter_context(modal.Queue.ephemeral())
            generator = allocation.remote_gen(plan, package, commands)
            first = next(generator)
            require(first["kind"] == "identity", "worker_identity_envelope")
            return dict(stack=stack, commands=commands, generator=generator, identity=first["identity"])
        except BaseException:
            stack.close()
            raise

    def observe_identity(self, worker):
        return worker["identity"]

    def available(self, worker):
        return True

    def execute_slot(self, worker, plan, product, cell, slot, attempt_id):
        worker["commands"].put(dict(kind="slot", product=product, cell=cell, slot=slot, attempt_id=attempt_id), timeout=30)
        reply = next(worker["generator"])
        require(reply["kind"] == "slot" and reply["attempt_id"] == attempt_id, "worker_slot_envelope")
        return reply["result"]

    def close(self, worker):
        try:
            worker["commands"].put(dict(kind="stop"), timeout=30)
            require(list(worker["generator"]) == [], "unexpected_worker_output_after_stop")
        finally:
            worker["stack"].close()
