from autoware_simpl_python.dataclass import AgentState
from autoware_simpl_python.dataclass import LaneSegment
from autoware_simpl_python.geometry import rotate_along_z
import numpy as np
from numpy.typing import NDArray

from .lane import _polyline_break_idx

__all__ = ("embed_polyline",)


def embed_polyline(
    polylines: list[LaneSegment],
    current_ego: AgentState,
    num_polyline: int = 300,
    num_point: int = 11,
    break_distance: float = np.inf,
) -> tuple[NDArray, NDArray, NDArray]:
    """Embed polyline attributes.

    Args:
        polylines (list[LaneSegment]): List of polyline segments.
        current_ego (AgentState): Current ego state.
        num_polyline (int, optional): Max number of polylines. Defaults to 300.
        num_point (int, optional): Max number of points in a single polyline.
            Defaults to 11.
        break_distance (float, optional): Distance threshold to break points.
            Defaults to np.inf.

    Returns:
        tuple[NDArray, NDArray, NDArray]: Polyline embedding in the shape of (K, P-1, C)
          , and polyline center and vector in the shape of (K, 2).

    Todo:
        use all polyline segments instead of `list[LaneSegment]`.
    """
    polyline = np.concatenate([lane.as_array(full=True, as_3d=False) for lane in polylines], axis=0)

    all_polyline, all_polyline_mask = _batch_polyline(
        polyline=polyline,
        num_point=num_point,
        break_distance=break_distance,
    )

    if len(all_polyline) > num_polyline:
        lane_ctr_xy: NDArray = all_polyline[..., :2].sum(axis=1) / np.clip(
            all_polyline_mask.sum(axis=-1, dtype=np.float32, keepdims=True),
            a_min=1.0,
            a_max=None,
        )
        distances = np.linalg.norm(lane_ctr_xy - current_ego.xy[None, :], axis=-1)
        topk_indices = np.argsort(distances, axis=-1)[:num_polyline]
        all_polyline = all_polyline[topk_indices]
        all_polyline_mask = all_polyline_mask[topk_indices]

    xy: NDArray = all_polyline[..., :2]
    node_ctr, node_vec, polyline_ctr, polyline_vec = _extract_polyline_node(xy, current_ego)

    polyline_embed: NDArray = np.concatenate(
        (
            node_ctr,  # (K, P-1, 2)
            node_vec,  # (K, P-1, 2)
        ),
        axis=-1,
        dtype=np.float32,
    )

    # filter polylines which all points are invalid
    tmp_polyline_mask: NDArray = all_polyline_mask.any(axis=1)
    polyline_embed = polyline_embed[tmp_polyline_mask]  # (K', P-1, D)
    polyline_ctr = polyline_ctr[tmp_polyline_mask]  # (K', 2)
    polyline_vec = polyline_vec[tmp_polyline_mask]  # (K', 2)

    # polyline_mask = all_polyline_mask[tmp_polyline_mask, :-1]  # (K', P-1)

    return polyline_embed, polyline_ctr, polyline_vec


def _batch_polyline(
    polyline: NDArray,
    num_point: int,
    break_distance: float,
) -> tuple[NDArray, NDArray]:
    break_idx: NDArray = _polyline_break_idx(polyline, break_distance)
    polyline_list: list[NDArray] = np.array_split(polyline, break_idx, axis=0)

    ret_polyline_list: list[NDArray] = []
    ret_polyline_mask_list: list[NDArray] = []
    for points in polyline_list:
        num_pts = len(points)
        if num_pts <= 0:
            continue
        for idx in range(0, num_pts, num_point):
            _append_single_polyline(
                points[idx : idx + num_point],
                num_point,
                ret_polyline_list,
                ret_polyline_mask_list,
            )

    ret_polyline: NDArray = np.stack(ret_polyline_list, axis=0)
    ret_polyline_mask: NDArray = np.stack(ret_polyline_mask_list, axis=0)

    return ret_polyline, ret_polyline_mask


def _append_single_polyline(
    new_polyline: NDArray,
    num_point: int,
    ret_polyline_list: list[NDArray],
    ret_polyline_mask_list: list[NDArray],
) -> None:
    """
    Append a single polyline info to a `ret_*`.

    Args:
    ----
        new_polyline (NDArray): Polyline array to be appended.
        num_point (int): Max number of points contained in a single polyline.
        ret_polyline_list (list[NDArray]): A container to append new polyline.
        ret_polyline_mask_list (NDArray): A container to append new polyline mask.

    """
    num_new_polyline, point_dim = new_polyline.shape

    cur_polyline: NDArray = np.zeros((num_point, point_dim), dtype=np.float32)
    cur_polyline_mask: NDArray = np.zeros(num_point, dtype=np.bool_)
    cur_polyline[:num_new_polyline] = new_polyline
    cur_polyline_mask[:num_new_polyline] = True

    ret_polyline_list.append(cur_polyline)
    ret_polyline_mask_list.append(cur_polyline_mask)


def _extract_polyline_node(
    polyline_xy: NDArray,
    current_ego: AgentState,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Extract polyline node information.

    Args:
    ----
        polyline_xy (NDArray): Polyline xy coordinates.
        current_ego (AgentState): Current ego state.

    Returns:
    -------
        tuple[NDArray, NDArray, NDArray, NDArray]: Node center and vector, and polyline center and vector.

    """
    # transform ego centric coords: (K, 2)
    polyline_xy -= current_ego.xy
    polyline_xy = rotate_along_z(points=polyline_xy, angle=current_ego.yaw)  # (K, P, 2)
    polyline_ctr = polyline_xy.mean(axis=1, dtype=np.float32)  # (K, 2)
    polyline_vec = polyline_xy[:, -1, :] - polyline_xy[:, 0, :]  # (K, 2)
    polyline_vec_norm = np.linalg.norm(polyline_vec, axis=-1, keepdims=True)
    polyline_vec = np.divide(polyline_vec, polyline_vec_norm, where=polyline_vec_norm != 0)
    polyline_angle = np.arctan2(polyline_vec[..., 1], polyline_vec[..., 0])

    # transform to instance centric coords: (K, P-1, 2)
    polyline_xy -= polyline_ctr[:, None, :]
    polyline_xy = rotate_along_z(points=polyline_xy, angle=polyline_angle)
    node_ctr = 0.5 * (polyline_xy[:, :-1, :] + polyline_xy[:, 1:, :])  # (K, P-1, 2)
    node_vec = polyline_xy[:, 1:, :] - polyline_xy[:, :-1, :]  # (K, P-1, 2)

    return node_ctr, node_vec, polyline_ctr, polyline_vec
