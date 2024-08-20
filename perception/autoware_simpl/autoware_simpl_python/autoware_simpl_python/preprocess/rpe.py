from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ("relative_pose_encode",)


def relative_pose_encode(
    agent_ctr: NDArray,
    agent_vec: NDArray,
    lane_ctr: NDArray,
    lane_vec: NDArray,
    *,
    return_mask: bool = False,
    num_rpe_stack: int = 10,
    radius: float = 100.0,
) -> tuple[NDArray, NDArray | None]:
    centers = np.concatenate((agent_ctr, lane_ctr), axis=0)  # (N+K, 2)
    vectors = np.concatenate((agent_vec, lane_vec), axis=0)  # (N+K, 2)

    v_pos = centers[None, ...] - centers[:, None, ...]  # (N+K, N+K, 2)

    # RPE angeles
    cos_a1 = _calculate_cos(vectors[None, ...], vectors[:, None, ...])
    sin_a1 = _calculate_sin(vectors[None, ...], vectors[:, None, ...])

    cos_a2 = _calculate_cos(vectors[None, ...], v_pos)
    sin_a2 = _calculate_sin(vectors[None, ...], v_pos)

    ang_rpe = np.stack((cos_a1, sin_a1, cos_a2, sin_a2), axis=-1)

    # RPE distances
    d_pos = np.linalg.norm(v_pos, axis=-1)  # (N, N)
    pos_rpe: NDArray
    if return_mask:
        rpe_mask = d_pos >= radius
        d_pos = d_pos * 2.0 / radius
        pos_rpe = np.concatenate(
            [
                [np.sin(2**l_pos * np.pi * d_pos), np.cos(2**l_pos * np.pi * d_pos)]
                for l_pos in range(num_rpe_stack)
            ],
            axis=0,
        )  # (2S, N+K, N+K)
        pos_rpe = pos_rpe.transpose(1, 2, 0)  # (N+K, N+K, 2S)
    else:
        rpe_mask = None
        d_pos = d_pos * 2.0 / radius
        pos_rpe = d_pos[..., None]  # (N, N, 1)

    rpe: NDArray = np.concatenate((ang_rpe, pos_rpe), axis=-1, dtype=np.float32)  # (N+K, N+K, Dr)
    return rpe, rpe_mask


def _calculate_cos(v1: NDArray, v2: NDArray) -> NDArray:
    v1_norm: NDArray = np.linalg.norm(v1, axis=-1)
    v2_norm: NDArray = np.linalg.norm(v2, axis=-1)
    v1_x, v1_y = v1[..., 0], v1[..., 1]
    v2_x, v2_y = v2[..., 0], v2[..., 1]
    return (v1_x * v2_x + v1_y * v2_y) / (v1_norm * v2_norm + 1e-10)


def _calculate_sin(v1: NDArray, v2: NDArray) -> NDArray:
    v1_norm = np.linalg.norm(v1, axis=-1)
    v2_norm = np.linalg.norm(v2, axis=-1)
    v1_x, v1_y = v1[..., 0], v1[..., 1]
    v2_x, v2_y = v2[..., 0], v2[..., 1]
    return (v1_x * v2_y - v1_y * v2_x) / (v1_norm * v2_norm + 1e-10)
