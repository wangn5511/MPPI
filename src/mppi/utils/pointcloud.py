from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float32).reshape(4, 4)
    p = np.asarray(pts, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"Expected pts shape (N,3), got {p.shape}")
    ones = np.ones((p.shape[0], 1), dtype=np.float32)
    ph = np.concatenate([p, ones], axis=1)
    out = (ph @ T.T)[:, :3]
    return out.astype(np.float32)


def crop_aabb(pts: np.ndarray, xyz_min: Tuple[float, float, float], xyz_max: Tuple[float, float, float]) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float32)
    mn = np.asarray(xyz_min, dtype=np.float32).reshape(1, 3)
    mx = np.asarray(xyz_max, dtype=np.float32).reshape(1, 3)
    keep = np.all((p >= mn) & (p <= mx), axis=1)
    return p[keep]


def voxel_downsample(pts: np.ndarray, voxel_size: float) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float32)
    vs = float(voxel_size)
    if p.shape[0] == 0:
        return p
    if vs <= 0:
        return p
    g = np.floor(p / vs).astype(np.int32)
    keys = g[:, 0].astype(np.int64) * 73856093 ^ g[:, 1].astype(np.int64) * 19349663 ^ g[:, 2].astype(np.int64) * 83492791
    order = np.argsort(keys, kind="mergesort")
    keys_s = keys[order]
    p_s = p[order]
    _, idx = np.unique(keys_s, return_index=True)
    return p_s[idx].astype(np.float32)


def remove_points_in_spheres(pts: np.ndarray, spheres: np.ndarray, margin: float) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float32)
    s = np.asarray(spheres, dtype=np.float32)
    if p.shape[0] == 0:
        return p
    if s.ndim != 2 or s.shape[1] != 4:
        raise ValueError(f"Expected spheres shape (M,4), got {s.shape}")
    centers = s[:, :3]
    radii = s[:, 3] + float(margin)
    keep = np.ones((p.shape[0],), dtype=bool)
    for c, r in zip(centers, radii):
        d2 = np.sum((p - c[None, :]) ** 2, axis=1)
        keep &= d2 > float(r * r)
    return p[keep]


@dataclass(frozen=True)
class AABB:
    xyz_min: Tuple[float, float, float]
    xyz_max: Tuple[float, float, float]

    @property
    def volume(self) -> float:
        mn = np.asarray(self.xyz_min, dtype=np.float32)
        mx = np.asarray(self.xyz_max, dtype=np.float32)
        d = np.maximum(mx - mn, 0.0)
        return float(d[0] * d[1] * d[2])

    def to_cuboid(self, padding: float) -> Dict[str, List[float]]:
        mn = np.asarray(self.xyz_min, dtype=np.float32)
        mx = np.asarray(self.xyz_max, dtype=np.float32)
        pad = float(padding)
        mn = mn - pad
        mx = mx + pad
        center = ((mn + mx) * 0.5).tolist()
        dims = (mx - mn).tolist()
        return {"center": [float(x) for x in center], "dims": [float(x) for x in dims]}


def _union_find_clusters(voxels: np.ndarray) -> List[np.ndarray]:
    if voxels.shape[0] == 0:
        return []
    vox = np.asarray(voxels, dtype=np.int32)
    key = vox[:, 0].astype(np.int64) * 73856093 ^ vox[:, 1].astype(np.int64) * 19349663 ^ vox[:, 2].astype(np.int64) * 83492791
    order = np.argsort(key, kind="mergesort")
    vox = vox[order]
    key = key[order]

    parents = np.arange(vox.shape[0], dtype=np.int32)

    def find(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = int(parents[i])
        return i

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parents[rb] = ra

    idx_by_tuple: Dict[Tuple[int, int, int], int] = {}
    for i, v in enumerate(vox):
        idx_by_tuple[(int(v[0]), int(v[1]), int(v[2]))] = i

    neigh = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    for i, v in enumerate(vox):
        x, y, z = int(v[0]), int(v[1]), int(v[2])
        for dx, dy, dz in neigh:
            j = idx_by_tuple.get((x + dx, y + dy, z + dz))
            if j is not None:
                union(i, j)

    buckets: Dict[int, List[int]] = {}
    for i in range(vox.shape[0]):
        r = find(i)
        buckets.setdefault(r, []).append(i)

    return [vox[np.asarray(idxs, dtype=np.int32)] for idxs in buckets.values()]


def cluster_points_to_aabbs(
    pts: np.ndarray,
    *,
    voxel_size: float,
    min_cluster_voxels: int,
) -> List[AABB]:
    p = np.asarray(pts, dtype=np.float32)
    if p.shape[0] == 0:
        return []
    vs = float(voxel_size)
    if vs <= 0:
        mn = p.min(axis=0)
        mx = p.max(axis=0)
        return [AABB(tuple(mn.tolist()), tuple(mx.tolist()))]

    vox = np.floor(p / vs).astype(np.int32)
    uniq = np.unique(vox, axis=0)
    clusters = _union_find_clusters(uniq)

    aabbs: List[AABB] = []
    for c in clusters:
        if c.shape[0] < int(min_cluster_voxels):
            continue
        mn = c.min(axis=0).astype(np.float32) * vs
        mx = (c.max(axis=0).astype(np.float32) + 1.0) * vs
        aabbs.append(AABB(tuple(mn.tolist()), tuple(mx.tolist())))

    return aabbs