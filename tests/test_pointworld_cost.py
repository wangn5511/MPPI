from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mppi.costs.pointworld_cost import (
    PointWorldCostConfig,
    reduce_pointworld_cost,
    reduce_pointworld_cost_torch,
)


def test_reduce_pointworld_cost_ignores_t0_and_weights_confidence() -> None:
    rel = np.zeros((1, 3, 2, 3), dtype=np.float32)
    rel[0, 1, 0] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    rel[0, 2, 1] = np.asarray([2.0, 0.0, 0.0], dtype=np.float32)

    exists = np.ones((1, 3, 2), dtype=bool)
    model_conf = np.ones((1, 3, 2), dtype=np.float32)
    track_conf = np.asarray([[[1.0, 1.0], [1.0, 0.5], [1.0, 0.5]]], dtype=np.float32)

    out = reduce_pointworld_cost(
        scene_relative=rel,
        scene_exists=exists,
        model_confidence=model_conf,
        track_confidence=track_conf,
        cfg=PointWorldCostConfig(mode="flow_l2"),
    )

    expected = (1.0 + (4.0 * 0.5)) / (1.0 + 0.5 + 1.0 + 0.5)
    assert np.allclose(out, np.asarray([expected], dtype=np.float32))


def test_reduce_pointworld_cost_final_mode() -> None:
    rel = np.zeros((2, 4, 1, 3), dtype=np.float32)
    rel[0, -1, 0] = np.asarray([3.0, 0.0, 0.0], dtype=np.float32)
    rel[1, -1, 0] = np.asarray([4.0, 0.0, 0.0], dtype=np.float32)
    exists = np.ones((2, 4, 1), dtype=bool)

    out = reduce_pointworld_cost(
        scene_relative=rel,
        scene_exists=exists,
        cfg=PointWorldCostConfig(mode="final_flow_l2"),
    )

    assert np.allclose(out, np.asarray([9.0, 16.0], dtype=np.float32))


def test_reduce_pointworld_cost_torch_matches_numpy() -> None:
    rng = np.random.default_rng(7)
    rel = rng.normal(size=(3, 5, 7, 3)).astype(np.float32)
    exists = rng.random((3, 5, 7)) > 0.2
    model_conf = rng.random((3, 5, 7)).astype(np.float32)
    track_conf = rng.random((3, 5, 7)).astype(np.float32)
    cfg = PointWorldCostConfig(
        mode="flow_l2",
        use_model_confidence=True,
        use_track_confidence=True,
        min_confidence=0.15,
        ignore_t0=True,
    )

    expected = reduce_pointworld_cost(
        scene_relative=rel,
        scene_exists=exists,
        model_confidence=model_conf,
        track_confidence=track_conf,
        cfg=cfg,
    )

    out = reduce_pointworld_cost_torch(
        scene_relative=torch.as_tensor(rel),
        scene_exists=torch.as_tensor(exists),
        model_confidence=torch.as_tensor(model_conf),
        track_confidence=torch.as_tensor(track_conf),
        cfg=cfg,
    )

    assert out.device.type == "cpu"
    assert np.allclose(out.numpy(), expected, atol=1e-6)
