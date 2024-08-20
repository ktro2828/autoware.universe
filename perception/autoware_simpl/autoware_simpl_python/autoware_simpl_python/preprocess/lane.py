from autoware_simpl_python.dataclass import AgentState
from autoware_simpl_python.dataclass import LaneSegment
from autoware_simpl_python.datatype import T4Lane
from autoware_simpl_python.geometry import rotate_along_z
import numpy as np
from numpy.typing import NDArray

__all__ = ("embed_lane",)


def embed_lane(
    lane_segments: list[LaneSegment],
    current_ego: AgentState,
    num_polyline: int = 300,
    num_point: int = 11,
    break_distance: float = np.inf,
) -> tuple[NDArray, NDArray, NDArray]:
    """Embed lane attributes.

    Args:
        lane_segments (list[LaneSegment]): List of lane segments.
        current_ego (AgentState): Current ego state.
        num_polyline (int, optional): Max number of polylines. Defaults to 300.
        num_point (int, optional): Max number of point in a single polyline.
            Defaults to 11.
        break_distance (float, optional): Distance threshold to break points.
            Defaults to np.inf.

    Returns:
        tuple[NDArray, NDArray, NDArray]: Lane embedding in the shape of (K, P-1, C),
            and lane center and vector in the shape of (K, 2).
    """
    (
        lane_polyline,
        lane_polyline_mask,
        is_intersection,
        is_lr_crossable,
        has_lr_neighbor,
    ) = _batch_lane(
        lane_segments=lane_segments,
        num_point=num_point,
        break_distance=break_distance,
        full=True,
        as_3d=False,
    )

    if len(lane_polyline) > num_polyline:
        lane_ctr_xy: NDArray = lane_polyline[..., :2].sum(axis=1) / np.clip(
            lane_polyline_mask.sum(axis=-1, dtype=np.float64, keepdims=True),
            a_min=1.0,
            a_max=None,
        )
        distances = np.linalg.norm(lane_ctr_xy - current_ego.xy[None, :], axis=-1)
        topk_indices = np.argsort(distances, axis=-1)[:num_polyline]
        lane_polyline = lane_polyline[topk_indices]
        lane_polyline_mask = lane_polyline_mask[topk_indices]
        is_intersection = is_intersection[topk_indices]
        is_lr_crossable = is_lr_crossable[topk_indices]
        has_lr_neighbor = has_lr_neighbor[topk_indices]
    else:
        num_polyline = len(lane_polyline)

    node_ctr, node_vec, lane_ctr, lane_vec = _extract_lane_node(
        lane_xy=lane_polyline[..., :2],
        current_ego=current_ego,
    )

    lane_label_id: NDArray = lane_polyline[..., -1]
    num_type = len(T4Lane)
    type_onehot = np.zeros((num_polyline, num_point, num_type))
    for i, label in enumerate(T4Lane):
        type_onehot[lane_label_id == label.value, i] = 1

    lane_embed = np.concatenate(
        (
            node_ctr,  # (K, P-1, 2)
            node_vec,  # (K, P-1, 2)
            is_intersection[:, :-1, None],
            type_onehot[:, :-1],  # (K, P-1, L)
            is_lr_crossable[:, :-1, :, 0],  # left: (K, P-1, 3)
            is_lr_crossable[:, :-1, :, 1],  # right: (K, P-1, 3)
            has_lr_neighbor[:, :-1],  # (K, P-1, 2)
        ),
        axis=-1,
        dtype=np.float32,
    )

    tmp_lane_mask = lane_polyline_mask.any(axis=1)
    lane_embed = lane_embed[tmp_lane_mask]
    lane_ctr = lane_ctr[tmp_lane_mask]
    lane_vec = lane_vec[tmp_lane_mask]

    return lane_embed, lane_ctr, lane_vec


def _extract_lane_node(
    lane_xy: NDArray,
    current_ego: AgentState,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Extract lane node.

    Args:
        lane_xy (NDArray): Lane xy positions.
        current_ego (AgentState): Current ego state.

    Returns:
        tuple[NDArray, NDArray, NDArray, NDArray]: Node center and vector
            in the shape of (K, P-1, 2), and lane center and vector in the shape of (K, 2).
    """
    lane_xy -= current_ego.xy
    lane_xy = rotate_along_z(lane_xy, current_ego.yaw)
    lane_ctr = lane_xy.mean(axis=1)  # (K, 2)
    lane_vec = lane_xy[:, -1, :] - lane_xy[:, 0, :]  # (K, 2)
    lane_vec_norm = np.linalg.norm(lane_vec, axis=-1, keepdims=True)
    lane_vec = np.divide(lane_vec, lane_vec_norm, where=lane_vec_norm != 0)
    lane_angle = np.arctan2(lane_vec[..., 1], lane_vec[..., 0])

    lane_xy -= lane_ctr[:, None, :]
    lane_xy = rotate_along_z(lane_xy, lane_angle)
    node_ctr = 0.5 * (lane_xy[:, :-1, :] + lane_xy[:, 1:, :])  # (K, P-1, 2)
    node_vec = lane_xy[:, 1:, :] - lane_xy[:, :-1, :]  # (K, P-1, 2)

    return node_ctr, node_vec, lane_ctr, lane_vec


def _batch_lane(
    lane_segments: list[LaneSegment],
    num_point: int,
    *,
    break_distance: float = np.inf,
    full: bool = False,
    as_3d: bool = True,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """
    Generate batch lane polyline from (L, D) to (K, P, D).

    Args:
    ----
        lane_segments (list[LaneSegment]): List of `LaneSegment` instances.
        num_point (int): The number of points contained in a single polyline.
        break_distance (float): Distance threshold to break a polyline into two.
        full (bool, optional): Indicates whether to convert polyline as full array.
            Defaults to False.
        as_3d (bool, optional): Indicates whether to convert polyline as 3d array.
            Defaults to True.

    Returns:
    -------
        tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
            - NDArray: Batch polyline, in shape (K, P, D).
            - NDArray: Batch polyline mask, in shape (K, P).
            - NDArray: Batch mask that each element indicates if the lane is
                intersection, in shape (K, P).
            - NDArray: Batch mask that each element indicates if the lane boundary is
                crossable, in shape (K, P, 3, 2).
            - NDArray: Batch mask that each element indicates if the lane has neighbor
                in the order of (left, right), in shape (K, P, 2).

    """
    polyline_list: list[NDArray] = []
    is_intersection_list: list[NDArray] = []
    is_lr_crossable_list: list[NDArray] = []
    has_lr_neighbor_list: list[NDArray] = []
    for lane in lane_segments:
        cur_points = lane.as_array(full=full, as_3d=as_3d)
        num_cur_pts = cur_points.shape[0]
        polyline_list.append(cur_points)
        is_intersection_list.append(np.tile(lane.is_intersection, num_cur_pts))
        is_lr_crossable_list.append(np.tile((_crossable_lane_onehot(lane)), (num_cur_pts, 1, 1)))
        has_lr_neighbor_list.append(
            np.tile((lane.has_left_neighbor(), lane.has_right_neighbor()), (num_cur_pts, 1))
        )

    polyline: NDArray = np.concatenate(polyline_list, axis=0, dtype=np.float32)
    is_intersection: NDArray = np.concatenate(is_intersection_list, axis=0, dtype=np.bool_)
    is_lr_crossable: NDArray = np.concatenate(is_lr_crossable_list, axis=0, dtype=np.bool_)
    has_lr_neighbor: NDArray = np.concatenate(has_lr_neighbor_list, axis=0, dtype=np.bool_)

    # break polyline
    break_idx: NDArray = _polyline_break_idx(polyline, break_distance)

    break_polyline_list: list[NDArray] = np.array_split(polyline, break_idx, axis=0)
    break_is_intersection_list: list[NDArray] = np.array_split(is_intersection, break_idx, axis=0)
    break_is_lr_crossable_list: list[NDArray] = np.array_split(is_lr_crossable, break_idx, axis=0)
    break_has_lr_neighbor_list: list[NDArray] = np.array_split(has_lr_neighbor, break_idx, axis=0)

    ret_polyline_list: list[NDArray] = []
    ret_polyline_mask_list: list[NDArray] = []
    ret_is_intersection_list: list[NDArray] = []
    ret_is_lr_crossable_list: list[NDArray] = []
    ret_has_lr_neighbor_list: list[NDArray] = []

    for points, is_inter, is_lr_cross, has_lr in zip(
        break_polyline_list,
        break_is_intersection_list,
        break_is_lr_crossable_list,
        break_has_lr_neighbor_list,
        strict=True,
    ):
        num_pts = len(points)
        if num_pts <= 0:
            continue
        for idx in range(0, num_pts, num_point):
            _append_single_lane(
                points[idx : idx + num_point],
                is_inter[idx : idx + num_point],
                is_lr_cross[idx : idx + num_point],
                has_lr[idx : idx + num_point],
                num_point,
                ret_polyline_list,
                ret_polyline_mask_list,
                ret_is_intersection_list,
                ret_is_lr_crossable_list,
                ret_has_lr_neighbor_list,
            )

    ret_polyline: NDArray = np.stack(ret_polyline_list, axis=0)
    ret_polyline_mask: NDArray = np.stack(ret_polyline_mask_list, axis=0)
    ret_is_intersection: NDArray = np.stack(ret_is_intersection_list, axis=0)
    ret_is_lr_crossable: NDArray = np.stack(ret_is_lr_crossable_list, axis=0)
    ret_has_lr_neighbor: NDArray = np.stack(ret_has_lr_neighbor_list, axis=0)

    return (
        ret_polyline,
        ret_polyline_mask,
        ret_is_intersection,
        ret_is_lr_crossable,
        ret_has_lr_neighbor,
    )


def _polyline_break_idx(polyline: NDArray, break_distance: float) -> NDArray:
    """
    Return indices to break a single polyline into multiple.

    Args:
    ----
        polyline (NDArray): Polyline array in shape (L, D).
        break_distance (float): Distance threshold to break a polyline into two.

    Returns:
    -------
        NDArray: Indices of polyline groups.

    """
    polyline_shifts = np.roll(polyline, shift=1, axis=0)
    buffer = np.concatenate((polyline[:, 0:2], polyline_shifts[:, 0:2]), axis=-1)
    buffer[0, 2:4] = buffer[0, 0:2]
    return (np.linalg.norm(buffer[:, 0:2] - buffer[:, 2:4], axis=-1) > break_distance).nonzero()[0]


def _append_single_lane(
    new_polyline: NDArray,
    new_is_intersection: NDArray,
    new_is_lr_crossable: NDArray,
    new_has_lr_neighbor: NDArray,
    num_point: int,
    ret_polyline_list: list[NDArray],
    ret_polyline_mask_list: list[NDArray],
    ret_is_intersection_list: list[NDArray],
    ret_is_lr_crossable_list: list[NDArray],
    ret_has_lr_neighbor_list: list[NDArray],
) -> None:
    """
    Append single lane polyline info to `ret_*`.

    Args:
    ----
        new_polyline (NDArray): Polyline array to be appended.
        new_is_intersection (NDArray): Array indicates corresponding lane is an
            intersection.
        new_is_lr_crossable (NDArray): Boolean array indicates the lane boundaries
            are crossable.
        new_has_lr_neighbor (NDArray): Boolean array indicates the lane has
            neighbors.
        num_point (int): Max number of points contained in a polyline.
        ret_polyline_list (list[NDArray]): A container to append new polyline.
        ret_polyline_mask_list (list[NDArray]): A container to append new polyline
            mask.
        ret_is_intersection_list (list[NDArray]): A container to append flag if the
            lane is intersection.
        ret_is_lr_crossable_list (list[NDArray]): A container to append flag if the
            lane boundaries are crossable.
        ret_has_lr_neighbor_list (list[NDArray]): A container to append flag if the
            lane has neighbors.

    """
    num_new_polyline, point_dim = new_polyline.shape

    cur_polyline: NDArray = np.zeros((num_point, point_dim), dtype=np.float32)
    cur_polyline_mask: NDArray = np.zeros(num_point, dtype=np.bool_)
    cur_is_intersection: NDArray = np.zeros(num_point, dtype=np.bool_)
    cur_is_lr_crossable: NDArray = np.zeros((num_point, 3, 2), dtype=np.bool_)
    cur_has_lr_neighbor: NDArray = np.zeros((num_point, 2), dtype=np.bool_)

    cur_polyline[:num_new_polyline] = new_polyline
    cur_polyline_mask[:num_new_polyline] = True
    cur_is_intersection[:num_new_polyline] = new_is_intersection
    cur_is_lr_crossable[:num_new_polyline] = new_is_lr_crossable
    cur_has_lr_neighbor[:num_new_polyline] = new_has_lr_neighbor

    ret_polyline_list.append(cur_polyline)
    ret_polyline_mask_list.append(cur_polyline_mask)
    ret_is_intersection_list.append(cur_is_intersection)
    ret_is_lr_crossable_list.append(cur_is_lr_crossable)
    ret_has_lr_neighbor_list.append(cur_has_lr_neighbor)


def _crossable_lane_onehot(lane: LaneSegment) -> NDArray:
    """
    Return an onehot mask whether the boundary is crossable or not.

    The value of an element is the order of (crossable, non-crossable, unknown).

    Args:
    ----
        lane (LaneSegment): Lane segment.

    Returns:
    -------
        NDArray: Onehot mask of left and right in the shape of (3, 2).

    """
    left_onehot: list[bool] = [False, False, False]
    if lane.is_left_virtual():
        left_onehot[2] = True
    elif lane.is_left_crossable():
        left_onehot[0] = True
    else:
        left_onehot[1] = True
    right_onehot: list[bool] = [False, False, False]
    if lane.is_right_virtual():
        right_onehot[2] = True
    elif lane.is_right_crossable():
        right_onehot[0] = True
    else:
        right_onehot[1] = True

    return np.stack([left_onehot, right_onehot], axis=1)
