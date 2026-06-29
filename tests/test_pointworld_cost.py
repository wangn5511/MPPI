from __future__ import annotations

import numpy as np

from mppi.costs.pointworld_cost import PointWorldCostConfig, reduce_pointworld_cost


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
