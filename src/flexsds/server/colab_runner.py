"""Google Colab serving runner and Cloudflare tunnel integration for FlexSDS."""
from __future__ import annotations

import urllib.error
import urllib.request
from typing import Any, Dict


class ColabServingRunner:
    """Quản lý thông tin khởi chạy FlexSDS trên Google Colab qua Cloudflare Tunnel."""

    COLAB_NOTEBOOK_URL = (
        "https://colab.research.google.com/github/theflyingkhui04/do-an-ptit-d22/blob/develop/notebooks/flexsds_colab_server.ipynb"
    )

    def get_colab_instructions(self, target_gpu: str = "T4") -> Dict[str, Any]:
        """Trả về đường dẫn notebook và hướng dẫn kết nối máy chủ Google Colab."""
        gpu_upper = target_gpu.upper()
        if "T4" in gpu_upper:
            profile = "profile_sm75_t4.json"
            runtime_desc = "T4 GPU (Miễn phí trên Google Colab)"
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
                "1. Nhấp vào liên kết Google Colab để mở notebook FlexSDS Server.",
                "2. Chọn Runtime -> Change runtime type -> Chọn T4 hoặc GPU mong muốn.",
                "3. Nhấp 'Run all' để tiến hành cài đặt môi trường và khởi chạy vLLM + Cloudflare tunnel.",
                "4. Sao chép đường dẫn công khai kết thúc bằng .trycloudflare.com và dán vào CLI này.",
            ],
        }

    def _probe_endpoint(self, url: str, timeout: int = 5) -> bool:
        """Gửi yêu cầu thăm dò tình trạng hoạt động (health check) của endpoint."""
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
        """Kiểm tra xem đường hầm Cloudflare từ Google Colab đã sẵn sàng hay chưa."""
        if not public_url or not public_url.startswith("http"):
            return False
        return self._probe_endpoint(public_url, timeout=timeout)
