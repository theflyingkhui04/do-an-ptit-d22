"""Hardware cost model loader and dynamic break-even solver for FlexSDS."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from flexsds.core.hardware import HardwareInfo
    from flexsds.core.tree_topology import (
        TreeTopology,
        CANONICAL_TOPOLOGIES,
        get_tree_topology,
    )
except ModuleNotFoundError:
    from src.flexsds.core.hardware import HardwareInfo
    from src.flexsds.core.tree_topology import (
        TreeTopology,
        CANONICAL_TOPOLOGIES,
        get_tree_topology,
    )

PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"
USER_PROFILES_DIR = Path.home() / ".flexsds" / "profiles"


@dataclass(frozen=True)
class SolverResult:
    """Represents the outcome of a break-even acceptance rate solver."""
    alpha_star: Optional[float]
    is_feasible: bool
    status: str  # "converged" | "always_profitable" | "infeasible"
    cost_ratio: float
    residual: float
    iterations: int
    topology_name: str
    batch_size: int


class FlexCostModel:
    """Hardware cost model evaluating forward latencies and dynamic break-even alpha*(T, B)."""

    def __init__(
        self,
        cost_matrix: Optional[Dict[Tuple[Union[str, int], int], float]] = None,
        forward_target: Optional[Dict[int, float]] = None,
        forward_draft: Optional[Dict[int, float]] = None,
        metadata: Optional[dict] = None,
        provenance: str = "analytical",
    ):
        self.matrix: Dict[Tuple[Union[str, int], int], float] = dict(cost_matrix) if cost_matrix else {}
        self.forward_target: Dict[int, float] = forward_target or {}
        self.forward_draft: Dict[int, float] = forward_draft or {}
        self.metadata: Dict[str, Any] = metadata or {}
        self.provenance: str = provenance
        self._cache_alpha_star: Dict[Tuple[str, int], SolverResult] = {}

    @classmethod
    def from_json(cls, path_or_dict: Union[str, Path, dict]) -> FlexCostModel:
        """Loads cost model from a profile JSON file."""
        if isinstance(path_or_dict, (str, Path)):
            with open(path_or_dict, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = path_or_dict

        matrix: Dict[Tuple[Union[str, int], int], float] = {}
        for key_str, val in data.get("cost_matrix_ms", {}).items():
            parts = key_str.split("_")
            # Parse topo_{name}_b{b}
            if len(parts) == 3 and parts[0] == "topo" and parts[2].startswith("b"):
                topo_name = parts[1].upper()
                b_val = int(parts[2][1:])
                matrix[(topo_name, b_val)] = float(val)
            # Parse k{k}_b{b}
            elif len(parts) == 2 and parts[0].startswith("k") and parts[1].startswith("b"):
                k_val = int(parts[0][1:])
                b_val = int(parts[1][1:])
                matrix[(k_val, b_val)] = float(val)
                # Alias canonical linear chains or base T0
                if k_val == 0:
                    matrix[("T0", b_val)] = float(val)
                elif f"T{k_val}" in CANONICAL_TOPOLOGIES:
                    matrix[(f"T{k_val}", b_val)] = float(val)

        c_target: Dict[int, float] = {}
        f_costs = data.get("forward_costs_median_ms", {})
        for b_str, val in f_costs.get("target_c_t", {}).items():
            c_target[int(b_str)] = float(val)
        if not c_target and "forward_target" in data:
            for b_str, val in data["forward_target"].items():
                c_target[int(b_str)] = float(val)

        c_draft: Dict[int, float] = {}
        for b_str, val in f_costs.get("draft_c_d", {}).items():
            c_draft[int(b_str)] = float(val)
        if not c_draft and "forward_draft" in data:
            for b_str, val in data["forward_draft"].items():
                c_draft[int(b_str)] = float(val)

        return cls(
            cost_matrix=matrix,
            forward_target=c_target,
            forward_draft=c_draft,
            metadata=data.get("metadata", {}),
            provenance="measured",
        )

    def _resolve_topology(self, topo: Union[str, TreeTopology, int]) -> TreeTopology:
        """Normalizes string name, TreeTopology instance, or integer budget k into a TreeTopology."""
        if isinstance(topo, TreeTopology):
            return topo

        if isinstance(topo, int):
            if topo <= 0:
                return get_tree_topology("T0")
            canonical_name = f"T{topo}"
            if canonical_name in CANONICAL_TOPOLOGIES:
                return get_tree_topology(canonical_name)
            # Ephemeral linear chain for non-canonical integer k (e.g. k=2, 3)
            return TreeTopology(
                name=canonical_name,
                num_tokens=topo,
                depth=topo,
                layer_distribution=[1] * (topo + 1),
                parent_indices=[-1] + list(range(topo)),
            )

        if isinstance(topo, str):
            clean_name = topo.strip().upper()
            return get_tree_topology(clean_name)

        raise TypeError(f"Invalid topology specification: {topo} (expected str, TreeTopology, or int)")

    def get_forward_costs(self, batch_size: int = 1) -> Tuple[float, float]:
        """Returns (C_T, C_D) in ms for the closest matching batch size."""
        if not self.forward_target:
            return 20.0, 5.0
        closest_b = min(self.forward_target.keys(), key=lambda b: abs(b - batch_size))
        c_t = self.forward_target[closest_b]
        c_d = self.forward_draft.get(closest_b, c_t * 0.25)
        return c_t, c_d

    def get_step_cost(self, topo: Union[str, TreeTopology, int] = "T0", batch_size: int = 1) -> float:
        """Gets step execution latency C(T, B) in milliseconds."""
        resolved = self._resolve_topology(topo)

        # 1. Exact lookup in matrix
        if (resolved.name, batch_size) in self.matrix:
            return self.matrix[(resolved.name, batch_size)]
        if isinstance(topo, int) and (topo, batch_size) in self.matrix:
            return self.matrix[(topo, batch_size)]
        if resolved.name == "T0" and (0, batch_size) in self.matrix:
            return self.matrix[(0, batch_size)]

        if self.metadata.get("scalar_only"):
            raise ValueError("Measured scalar profile has no exact cost for this topology/batch")

        # 2. Analytical Fallback calculation
        c_t, c_d = self.get_forward_costs(batch_size=batch_size)
        if resolved.num_tokens == 0:
            return c_t

        n_tokens = resolved.num_tokens
        attn_verify_factor = float(self.metadata.get("attn_verify_factor", 0.008))
        c_v = c_t * (1.0 + attn_verify_factor * max(0, n_tokens - 1))
        t_draft = n_tokens * c_d
        return t_draft + c_v

    def evaluate_speedup(
        self,
        topo: Union[str, TreeTopology, int],
        batch_size: int = 1,
        alpha: float = 0.0,
    ) -> float:
        """Calculates theoretical predicted speedup S(T, B, alpha)."""
        resolved = self._resolve_topology(topo)
        c_vanilla = self.get_step_cost("T0", batch_size=batch_size)
        c_spec = self.get_step_cost(resolved, batch_size=batch_size)
        e_tau = resolved.expected_acceptance(alpha)
        return (e_tau * c_vanilla) / max(c_spec, 1e-4)

    def solve_alpha_star(
        self,
        topo: Union[str, TreeTopology, int],
        batch_size: int = 1,
        max_iter: int = 50,
    ) -> SolverResult:
        """Solves break-even acceptance rate alpha*(T, B) where Speedup == 1.0."""
        resolved = self._resolve_topology(topo)
        cache_key = (resolved.name, batch_size)

        if cache_key in self._cache_alpha_star:
            return self._cache_alpha_star[cache_key]

        c_vanilla = self.get_step_cost("T0", batch_size=batch_size)
        c_spec = self.get_step_cost(resolved, batch_size=batch_size)
        cost_ratio = c_spec / max(c_vanilla, 1e-4)
        depth = resolved.depth

        # Boundary condition 1: R <= 1.0 (always profitable, even at alpha=0)
        if cost_ratio <= 1.0:
            result = SolverResult(
                alpha_star=0.0,
                is_feasible=True,
                status="always_profitable",
                cost_ratio=round(cost_ratio, 4),
                residual=abs(1.0 - cost_ratio),
                iterations=0,
                topology_name=resolved.name,
                batch_size=batch_size,
            )
            self._cache_alpha_star[cache_key] = result
            return result

        # Boundary condition 2 (Review R2 S08): R > 1 + d (infeasible across all alpha in [0, 1])
        if cost_ratio > 1.0 + float(depth):
            result = SolverResult(
                alpha_star=None,
                is_feasible=False,
                status="infeasible",
                cost_ratio=round(cost_ratio, 4),
                residual=float("inf"),
                iterations=0,
                topology_name=resolved.name,
                batch_size=batch_size,
            )
            self._cache_alpha_star[cache_key] = result
            return result

        # Converged region: 1.0 < R <= 1 + d (unique root via IVT)
        low, high = 0.0, 1.0
        for _ in range(max_iter):
            mid = (low + high) / 2.0
            y = resolved.expected_acceptance(mid)
            if y < cost_ratio:
                low = mid
            else:
                high = mid

        alpha_star = (low + high) / 2.0
        expected_tokens = resolved.expected_acceptance(alpha_star)
        residual = abs(expected_tokens - cost_ratio)

        result = SolverResult(
            alpha_star=round(alpha_star, 4),
            is_feasible=True,
            status="converged",
            cost_ratio=round(cost_ratio, 4),
            residual=residual,
            iterations=max_iter,
            topology_name=resolved.name,
            batch_size=batch_size,
        )
        self._cache_alpha_star[cache_key] = result
        return result

    def select_optimal_topology(
        self,
        alpha: float,
        batch_size: int = 1,
        candidate_topologies: Optional[List[Union[str, TreeTopology]]] = None,
    ) -> Tuple[str, float]:
        """Finds topology maximizing predicted speedup with catalog-wide fallback gate to T0."""
        candidates = candidate_topologies or ["T0", "T4", "T8", "T16", "T32"]
        best_topo = "T0"
        best_speedup = 1.0

        for cand in candidates:
            resolved = self._resolve_topology(cand)
            if resolved.name == "T0":
                continue
            speedup = self.evaluate_speedup(resolved, batch_size=batch_size, alpha=alpha)
            if speedup > best_speedup:
                best_speedup = speedup
                best_topo = resolved.name

        # Catalog-wide fallback gate (Review R2 S09)
        if best_speedup <= 1.0:
            return "T0", 1.0

        return best_topo, round(best_speedup, 3)

    # --------------------------------------------------------------------------
    # Backward Compatibility Wrappers (for v0.1 baseline and controllers)
    # --------------------------------------------------------------------------

    def calculate_alpha_star(
        self,
        k: Union[int, str] = 1,
        batch_size: int = 1,
        iterations: int = 50,
    ) -> float:
        """Solves break-even acceptance rate (backward-compatible float return)."""
        res = self.solve_alpha_star(topo=k, batch_size=batch_size, max_iter=iterations)
        if res.alpha_star is not None:
            return res.alpha_star
        return 1.0

    @staticmethod
    def estimate_expected_tokens(k: int, alpha: float) -> float:
        """Calculates expected accepted tokens per step (linear chain v0.1)."""
        if k <= 0:
            return 1.0
        alpha_clamped = min(max(alpha, 0.0), 1.0)
        gain = 1.0
        term = 1.0
        for _ in range(k):
            term *= alpha_clamped
            gain += term
        return gain

    def select_optimal_budget(
        self,
        alpha: float,
        batch_size: int = 1,
        candidate_budgets: Optional[List[int]] = None,
    ) -> Tuple[int, float]:
        """Finds k* maximizing speedup (backward-compatible integer budget return)."""
        candidates = candidate_budgets or [0, 2, 3, 4]
        c_vanilla = self.get_step_cost(0, batch_size=batch_size)
        best_k = 0
        best_speedup = 1.0

        for cand_k in candidates:
            if cand_k == 0:
                continue
            c_cand = self.get_step_cost(cand_k, batch_size=batch_size)
            e_tau = self.estimate_expected_tokens(k=cand_k, alpha=alpha)
            speedup = (e_tau * c_vanilla) / max(c_cand, 1e-4)
            if speedup > best_speedup:
                best_speedup = speedup
                best_k = cand_k

        return best_k, round(best_speedup, 3)


def save_hardware_profile(
    profile_data: dict,
    sm_architecture: str,
    device_name: str = "",
    target_dir: Optional[Path] = None,
) -> Path:
    """Saves a newly calibrated hardware latency profile to disk."""
    save_dir = target_dir or USER_PROFILES_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    slug = (
        device_name.lower().replace(" ", "_").replace("-", "_")
        if device_name
        else sm_architecture
    )
    filename = f"profile_{sm_architecture}_{slug}.json" if device_name else f"profile_{sm_architecture}.json"
    file_path = save_dir / filename
    import os
    import tempfile
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=save_dir, delete=False, encoding='utf-8') as f:
            temporary = Path(f.name)
            json.dump(profile_data, f, indent=2, allow_nan=False)
        os.replace(temporary, file_path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()
    return file_path


def resolve_profile_path(name: str) -> Path:
    for directory in (USER_PROFILES_DIR, PROFILES_DIR):
        path = directory / name
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"Profile artifact missing: {name}")


def inspect_cost_matrix_monotonicity(cost_matrix_ms: Dict[str, float]) -> List[Dict[str, Any]]:
    """Inspect scalar k{k}_b{b} cost matrix for non-monotonic cells without mutating data."""
    by_batch: Dict[int, Dict[int, float]] = {}
    for key_str, raw_val in cost_matrix_ms.items():
        parts = key_str.split("_")
        if len(parts) == 2 and parts[0].startswith("k") and parts[1].startswith("b"):
            try:
                k_val = int(parts[0][1:])
                b_val = int(parts[1][1:])
            except ValueError:
                continue
            by_batch.setdefault(b_val, {})[k_val] = float(raw_val)

    anomalies: List[Dict[str, Any]] = []
    for batch in sorted(by_batch):
        entries = by_batch[batch]
        baseline_k0 = entries.get(0)
        prev_k: Optional[int] = None
        prev_cost: Optional[float] = None
        for k_val in sorted(entries):
            cost = entries[k_val]
            below_k0 = bool(k_val > 0 and baseline_k0 is not None and cost < baseline_k0)
            below_prev_k = bool(prev_cost is not None and cost < prev_cost)
            if below_k0 or below_prev_k:
                anomalies.append(
                    {
                        "batch": batch,
                        "k": k_val,
                        "cost_ms": cost,
                        "baseline_k0_ms": baseline_k0,
                        "prev_k": prev_k,
                        "prev_cost_ms": prev_cost,
                        "below_k0": below_k0,
                        "below_prev_k": below_prev_k,
                    }
                )
            prev_k = k_val
            prev_cost = cost
    return anomalies


def validate_measured_profile(data, hardware=None, target_model=None, draft_model=None):
    """Reproduce scalar matrix from preserved worker samples, not a measured label."""
    from flexsds.core.calibration import build_profile, SCHEMA
    meta = data.get('metadata', {})
    if meta.get('schema') != SCHEMA or meta.get('measurement_status') != 'measured':
        raise ValueError('Profile lacks verified empirical evidence; recalibrate')
    evidence = data.get('evidence', {})
    config = evidence.get('config', {})
    try:
        rebuilt = build_profile(config, evidence['groups'], data['grid']['batches'])
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError('Malformed calibration evidence') from exc
    for field in ('cost_matrix_ms', 'models', 'draft_method', 'grid'):
        if data.get(field) != rebuilt[field]:
            raise ValueError(f'Profile {field} differs from worker evidence')
    for field in ('raw_sha256', 'device', 'sm_architecture', 'cost_semantics', 'vllm_version'):
        if meta.get(field) != rebuilt['metadata'][field]:
            raise ValueError(f'Profile {field} identity mismatch')
    if hardware and meta['sm_architecture'] != hardware.sm_architecture:
        raise ValueError('Profile GPU architecture mismatch')
    if hardware and hardware.is_cuda_available:
        from flexsds.core.hardware import matches_gpu_label
        expected_label = getattr(hardware, "gpu_match_label", None) or hardware.device_name.removeprefix("Remote ")
        if not matches_gpu_label(expected_label, meta['device']):
            raise ValueError('Profile GPU model mismatch')
    if target_model and data['models']['target'] != target_model:
        raise ValueError('Profile target model mismatch')
    if draft_model and data['models']['draft'] != draft_model:
        raise ValueError('Profile draft model mismatch')
    return data


def _measured_for_card(path: Path, hw_info: HardwareInfo) -> bool:
    """False for a measured (flexsds-component-v2) profile recorded on a different GPU model."""
    try:
        meta = json.loads(path.read_text(encoding="utf-8")).get("metadata", {})
    except (OSError, ValueError):
        return False
    if meta.get("schema") != "flexsds-component-v2" or not meta.get("device"):
        return True
    from flexsds.core.hardware import matches_gpu_label
    label = getattr(hw_info, "gpu_match_label", None) or hw_info.device_name.removeprefix("Remote ")
    return matches_gpu_label(label, meta["device"])


def load_profile_for_hardware(
    hw_info: Optional[HardwareInfo] = None,
    custom_profile_path: Optional[Union[str, Path]] = None,
    strict: bool = False,
    target_model: Optional[str] = None,
    draft_model: Optional[str] = None,
) -> Tuple[Optional[FlexCostModel], str]:
    """Loads hardware cost profile using a hierarchical resolution strategy.

    Resolution order:
    1. Explicit custom profile path.
    2. User profile repository (~/.flexsds/profiles/).
    3. Built-in vendor profiles.
    4. Conservative analytical fallback for non-A100 hardware when strict=False.

    A100/sm80 is fail-closed: it requires a measured scalar profile_sm80.json
    with the exact frozen M4 B/K grid and never falls back to L4/analytical data.
    """
    sm_arch = hw_info.sm_architecture if hw_info else "sm_89"
    if strict and sm_arch == "cpu":
        return None, ""

    def load_checked(path: Path) -> Tuple[FlexCostModel, str]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if strict and data.get("metadata", {}).get("schema") == "flexsds-component-v2":
            validate_measured_profile(data, hw_info, target_model, draft_model)
        elif strict:
            raise ValueError("Profile has no verified worker measurements; recalibrate or use static mode")
        if sm_arch == "sm_80" and data.get("metadata", {}).get("schema") != "flexsds-component-v2":
            meta = data.get("metadata", {})
            matrix = data.get("cost_matrix_ms", {})

            expected = {
                f"k{k}_b{b}"
                for k in range(5)
                for b in (1, 2, 4, 8, 16, 32)
            }

            if (
                meta.get("sm_architecture") != "sm_80"
                or meta.get("scalar_only") is not True
                or meta.get("measurement_status") != "measured"
                or set(matrix) != expected
                or any(
                    type(v) not in (int, float)
                    or not math.isfinite(v)
                    or v <= 0
                    for v in matrix.values()
                )
            ):
                raise ValueError("Malformed/unmeasured A100 scalar profile")

        return FlexCostModel.from_json(path), path.name

    # Tier 1: explicit custom profile.
    if custom_profile_path:
        custom_p = Path(custom_profile_path)
        if custom_p.exists():
            return load_checked(custom_p)

    # Tier 2: user profile store.
    if USER_PROFILES_DIR.exists():
        user_direct = USER_PROFILES_DIR / f"profile_{sm_arch}.json"
        if user_direct.exists():
            try:
                return load_checked(user_direct)
            except ValueError:
                if not strict:
                    raise

        matches = sorted(USER_PROFILES_DIR.glob(f"profile_{sm_arch}*.json"),
                         key=lambda path: (path.stem.endswith("_fast"), path.name))
        for match in matches:
            try:
                return load_checked(match)
            except ValueError:
                if not strict:
                    raise

    # M4 invariant: A100 must use a measured sm_80 profile, never L4/analytical fallback.
    if sm_arch == "sm_80":
        if strict:
            for path in sorted(PROFILES_DIR.glob("profile_sm*80*.json")):
                try:
                    return load_checked(path)
                except ValueError:
                    continue
            return None, ""

        profile_path = PROFILES_DIR / "profile_sm80.json"

        if profile_path.exists():
            return load_checked(profile_path)

        raise FileNotFoundError(
            "A100 requires a measured profile_sm80.json; "
            "no L4/analytical substitution"
        )

    # Packaged measured artifacts use the same evidence checks as user profiles. An architecture is shared
    # by several cards (sm_89: L4, RTX 40xx, L40S), so a measured profile of another card is skipped.
    for path in sorted(PROFILES_DIR.glob(f"profile_{sm_arch}*.json")):
        if hw_info is not None and not _measured_for_card(path, hw_info):
            continue
        try:
            return load_checked(path)
        except ValueError:
            continue

    # Tier 3: built-in vendor profiles for non-A100 hardware.
    arch_map = {
        "sm_89": "profile_sm89_l4.json",
        "sm_75": "profile_sm75_t4.json",
        "cpu": "profile_sm89_l4.json",
    }

    built_in_filename = arch_map.get(sm_arch)

    if built_in_filename:
        built_in_path = PROFILES_DIR / built_in_filename
        if built_in_path.exists():
            try:
                return load_checked(built_in_path)
            except ValueError:
                if not strict:
                    raise

    # Develop behavior: strict mode fails closed.
    if strict:
        return None, ""

    # Conservative analytical fallback for non-A100 hardware only.
    default_target = {
        1: 20.0,
        2: 20.5,
        4: 21.0,
        8: 22.0,
        16: 25.0,
        32: 32.0,
    }

    default_draft = {
        1: 3.6,
        2: 3.8,
        4: 4.0,
        8: 4.5,
        16: 5.2,
        32: 6.0,
    }

    fallback_meta = {
        "fallback_mode": True,
        "k_max_safe": 3,
        "reason": (
            f"No profile found for '{sm_arch}'. "
            "Conservative analytical fallback applied."
        ),
    }

    return (
        FlexCostModel(
            forward_target=default_target,
            forward_draft=default_draft,
            metadata=fallback_meta,
            provenance="conservative_analytical_fallback",
        ),
        "conservative_analytical_fallback",
    )


def run_hardware_micro_benchmark(
    hw_info: HardwareInfo,
    target_model: str = "Qwen/Qwen2.5-7B-Instruct-AWQ",
    draft_model: str = "duckhoi/qwen2.5-7b-eagle3.1-speculator",
    batch_sizes: Optional[List[int]] = None,
    target_dir: Optional[Path] = None,
    backend: str = "local",
    measurement_mode: str = "full",
    gpu_memory_utilization: Optional[float] = None,
) -> Tuple[FlexCostModel, Path]:
    """Measure actual vLLM worker components; never substitute analytical costs.

    Remote hardware must be dispatched explicitly with backend='modal'.
    """
    from flexsds.core.calibration import BATCHES, KS, build_profile, run_local_group
    from flexsds.core.compatibility import check_compatibility
    from flexsds.core.calibration import measurement_settings
    measurement_settings({'measurement_mode': measurement_mode})
    batches = BATCHES if batch_sizes is None else sorted(set(batch_sizes))
    if not batches or any(type(b) is not int or b not in BATCHES for b in batches):
        raise ValueError("Invalid calibration batch grid")
    if backend == "local" and hw_info.device_name.startswith("Remote "):
        raise ValueError("Remote GPU calibration requires backend='modal'")
    if backend == "local" and not hw_info.is_cuda_available:
        raise RuntimeError("Calibration requires a real CUDA GPU; use --backend modal --gpu A10")
    report = check_compatibility(target_model, draft_model, hw_info.free_vram_mb)
    if report.draft_method == "unsupported" or not report.is_compatible:
        raise ValueError("Calibration blocked: " + "; ".join(report.blocking_reasons))
    config = dict(target_model=target_model, draft_model=draft_model,
                  draft_method=report.draft_method, sm_architecture=hw_info.sm_architecture,
                  measurement_mode=measurement_mode)
    if gpu_memory_utilization is not None:  # only when overridden, so default evidence keeps its shape
        config['gpu_memory_utilization'] = float(gpu_memory_utilization)
    for key in ('target_revision', 'draft_revision'):
        revision = getattr(report, key, None)
        if revision and revision != 'local':
            config[key] = revision
    if backend == "modal":
        from flexsds.server.modal_runner import ModalServingRunner
        from flexsds.server.modal_calibration import calibrate
        if not ModalServingRunner().check_credentials():
            raise RuntimeError("Modal credentials missing; run flexsds auth login --provider modal")
        config['target_gpu'] = hw_info.device_name.removeprefix('Remote ')
        groups = calibrate(config, batches)
    elif backend == "local":
        groups = [run_local_group(config, k, batches) for k in KS]
    else:
        raise ValueError("Calibration supports local CUDA or Modal, not Colab guidance")
    data = build_profile(config, groups, batches)
    import hashlib
    identity = hashlib.sha256(json.dumps(data['models'], sort_keys=True).encode()).hexdigest()[:12]
    saved_path = save_hardware_profile(data, hw_info.sm_architecture,
                                       f"{hw_info.device_name}_{identity}_{measurement_mode}", target_dir)
    return FlexCostModel.from_json(saved_path), saved_path
