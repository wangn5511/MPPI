from __future__ import annotations

import os
import sys
import types

os.environ.setdefault("MPPI_URDF_PATH", "/home/wangning/MPPI/pyproject.toml")

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    nn_stub = types.ModuleType("torch.nn")

    class _Module:
        pass

    nn_stub.Module = _Module
    nn_stub.Linear = lambda *_args, **_kwargs: object()
    torch_stub.nn = nn_stub
    sys.modules["torch"] = torch_stub
    sys.modules["torch.nn"] = nn_stub

from mppi.comm.ws_server_async_pcl import (
    _JOINT_SOLVERS,
    _cuda_runtime_available,
    _effective_pointworld_enabled,
    _effective_use_curobo_collision,
    _effective_use_pointworld_cost,
    _get_joint_solver,
    _request_needs_camera_decode,
    _tracker_device_summary,
)


class _MultiTrackerLike:
    devices = ("cuda:0", "cuda:1")


class _SingleTrackerLike:
    _device = "cuda:0"


def test_tracker_device_summary_handles_single_and_multi_device_trackers() -> None:
    assert _tracker_device_summary(_MultiTrackerLike()) == "cuda:0,cuda:1"
    assert _tracker_device_summary(_SingleTrackerLike()) == "cuda:0"


def test_cpu_only_effective_flags_disable_gpu_perception(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("MPPI_USE_CUROBO_COLLISION", "1")
    monkeypatch.setenv("MPPI_SCENE_FROM_PCD_BACK_CAM", "1")
    monkeypatch.setenv("MPPI_PW_ENABLE", "1")
    monkeypatch.setenv("MPPI_USE_POINTWORLD_COST", "1")
    monkeypatch.delenv("MPPI_ALLOW_CPU_CUROBO", raising=False)
    monkeypatch.delenv("MPPI_ALLOW_CPU_POINTWORLD", raising=False)
    monkeypatch.delenv("MPPI_PCL_SAVE_PCD", raising=False)
    _cuda_runtime_available.cache_clear()

    assert _cuda_runtime_available() is False
    assert _effective_use_curobo_collision() is False
    assert _effective_pointworld_enabled() is False
    assert _effective_use_pointworld_cost() is False
    assert _request_needs_camera_decode("mppi_joint", None) is False


def test_cpu_only_solver_uses_cpu_samples_and_disables_gpu_costs(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("MPPI_CPU_NUM_SAMPLES", "17")
    monkeypatch.setenv("MPPI_USE_CUROBO_COLLISION", "1")
    monkeypatch.setenv("MPPI_USE_POINTWORLD_COST", "1")
    monkeypatch.delenv("MPPI_NUM_SAMPLES", raising=False)
    monkeypatch.delenv("MPPI_ALLOW_CPU_CUROBO", raising=False)
    monkeypatch.delenv("MPPI_ALLOW_CPU_POINTWORLD", raising=False)
    _cuda_runtime_available.cache_clear()
    _JOINT_SOLVERS.clear()

    solver = _get_joint_solver(3)

    assert solver.cfg.num_samples == 17
    assert solver.cfg.use_curobo_collision is False
    assert solver.cfg.use_pointworld_cost is False

    _JOINT_SOLVERS.clear()
