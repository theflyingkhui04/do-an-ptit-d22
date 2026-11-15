"""Google Colab serving runner and Cloudflare tunnel integration for FlexSDS."""
from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional


class ColabServingRunner:
    """Manages FlexSDS serving instructions and Cloudflare tunnel verification on Google Colab."""

    COLAB_NOTEBOOK_URL = (
        "https://colab.research.google.com/github/theflyingkhui04/do-an-ptit-d22/blob/develop/notebooks/flexsds_colab_server.ipynb"
    )

    def get_colab_instructions(self, target_gpu: str = "T4") -> Dict[str, Any]:
        """Returns notebook URL and instructions for running FlexSDS on Google Colab."""
        gpu_upper = target_gpu.upper()
        if "T4" in gpu_upper:
            profile = "profile_sm75_t4.json"
            runtime_desc = "T4 GPU (Free on Google Colab)"
        elif "L4" in gpu_upper:
            profile = "profile_sm89_l4.json"
            runtime_desc = "L4 GPU (Colab Pro)"
        elif "A100" in gpu_upper:
            profile = "profile_sm80_a100.json"
            runtime_desc = "A100 GPU (Colab Pro+)"
        else:
            profile = "profile_sm75_t4.json"
            runtime_desc = f"{target_gpu} GPU"

        return {
            "notebook_url": self.COLAB_NOTEBOOK_URL,
            "recommended_runtime": runtime_desc,
            "cost_profile": profile,
            "tunnel_method": "cloudflared",
            "steps": [
                "1. Open Google Colab and upload 'notebooks/flexsds_colab_server.ipynb' (or open via notebook URL if repo is authorized).",
                f"2. Select Runtime -> Change runtime type -> Choose {runtime_desc.split(' ')[0]} GPU.",
                "3. Click 'Run all' to set up the environment and launch vLLM + Cloudflare tunnel.",
                "4. Copy the public *.trycloudflare.com URL and paste it into this prompt.",
            ],
        }

    def _probe_endpoint(self, url: str, timeout: int = 5) -> bool:
        """Sends a probe request to verify endpoint health."""
        health_url = f"{url.rstrip('/')}/health"
        req = urllib.request.Request(
            health_url,
            headers={"User-Agent": "FlexSDS-TunnelVerifier/0.1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status == 200
        except Exception:
            return False

    def verify_tunnel_health(self, public_url: str, timeout: int = 5) -> bool:
        """Checks if the Cloudflare tunnel from Google Colab is reachable."""
        if not public_url or not public_url.startswith("http"):
            return False
        return self._probe_endpoint(public_url, timeout=timeout)

    def resolve_notebook_path(self, notebook_path: Optional[str] = None) -> Path:
        """Finds the absolute path to the local notebook file."""
        from pathlib import Path
        if notebook_path:
            return Path(notebook_path).resolve()

        # 1. Inside repository checkout
        repo_nb = Path(__file__).resolve().parents[3] / "notebooks" / "flexsds_colab_server.ipynb"
        if repo_nb.exists():
            return repo_nb.resolve()

        # 2. In current working directory
        cwd_nb = Path.cwd() / "notebooks" / "flexsds_colab_server.ipynb"
        if cwd_nb.exists():
            return cwd_nb.resolve()

        cwd_local = Path.cwd() / "flexsds_colab_server.ipynb"
        if cwd_local.exists():
            return cwd_local.resolve()

        # 3. Inside package directory assets
        pkg_nb = Path(__file__).resolve().parents[1] / "assets" / "flexsds_colab_server.ipynb"
        if pkg_nb.exists():
            return pkg_nb.resolve()

        return (Path.cwd() / "notebooks" / "flexsds_colab_server.ipynb").resolve()

    def open_in_file_manager(self, path: Path) -> bool:
        """Attempts to open the file manager to reveal the notebook file."""
        import subprocess
        import sys
        abs_path = str(path.resolve())
        try:
            if sys.platform == "win32":
                subprocess.Popen(["explorer", f"/select,{abs_path}"])
                return True
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", abs_path])
                return True
            else:
                folder = str(path.parent.resolve())
                subprocess.Popen(["xdg-open", folder])
                return True
        except Exception:
            return False

    def sync_notebook_config(
        self,
        config: Dict[str, Any],
        notebook_path: Optional[str] = None,
    ) -> bool:
        """Updates the local Colab notebook file with the exact models and parameters configured."""
        import json

        nb_file = self.resolve_notebook_path(notebook_path)
        if not nb_file.exists():
            return False

        try:
            with open(nb_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            serving_cfg = config.get("serving", config)
            target = serving_cfg.get("target_model", "Qwen/Qwen2.5-7B-Instruct-AWQ")
            draft = serving_cfg.get("draft_model", "duckhoi/qwen2.5-7b-eagle3.1-speculator")
            method = serving_cfg.get("draft_method", "eagle3")
            adaptive = serving_cfg.get("enable_adaptive", True)

            target_lower = target.lower()
            if "awq" in target_lower:
                quant = "awq"
            elif "gptq" in target_lower:
                quant = "gptq"
            else:
                quant = "none"

            for cell in data.get("cells", []):
                if cell.get("cell_type") == "code":
                    new_lines = []
                    for line in cell.get("source", []):
                        if line.startswith("TARGET_MODEL ="):
                            new_lines.append(f'TARGET_MODEL = "{target}"  #@param {{type:"string"}}\n')
                        elif line.startswith("DRAFT_MODEL ="):
                            new_lines.append(f'DRAFT_MODEL = "{draft}"  #@param {{type:"string"}}\n')
                        elif line.startswith("DRAFT_METHOD ="):
                            new_lines.append(f'DRAFT_METHOD = "{method}"  #@param ["eagle", "eagle3", "draft_model"]\n')
                        elif line.startswith("QUANTIZATION ="):
                            new_lines.append(f'QUANTIZATION = "{quant}"  #@param ["awq", "gptq", "none"]\n')
                        elif line.startswith("ENABLE_ADAPTIVE ="):
                            new_lines.append(f'ENABLE_ADAPTIVE = {adaptive}  #@param {{type:"boolean"}}\n')
                        else:
                            new_lines.append(line)
                    cell["source"] = new_lines

            with open(nb_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=1, ensure_ascii=False)
            return True
        except Exception:
            return False
