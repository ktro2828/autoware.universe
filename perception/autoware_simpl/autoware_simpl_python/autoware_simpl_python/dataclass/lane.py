from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .polyline import Polyline
from autoware_simpl_python.datatype import LaneType, BoundaryType

__all__ = ("LaneSegment", "BoundarySegment")

# TODO(ktro2828): Type definition


@dataclass
class LaneSegment:
    """
    Represents a lane segment.

    Attributes
    ----------
        lane_id (int): Unique ID associated with this lane.
        lane_type (LaneType): `LaneType` instance.
        polyline (Polyline): `Polyline` instance.
        is_intersection (bool): Flag indicating if this lane is intersection.
        left_boundaries (list[BoundarySegment]): List of `BoundarySegment` instances.
        right_boundaries (list[BoundarySegment]): List of `BoundarySegment` instances.
        left_neighbor_ids (list[int]): List of left neighbor ids on left side.
        right_neighbor_ids (list[int]): List of neighbor ids on right side.
        speed_limit_mph (float | None, optional): Lane speed limit in [miles/h].

    """

    lane_id: int
    lane_type: LaneType
    polyline: Polyline
    is_intersection: bool
    left_boundaries: list[BoundarySegment]
    right_boundaries: list[BoundarySegment]
    left_neighbor_ids: list[int]
    right_neighbor_ids: list[int]
    speed_limit_mph: float | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.lane_type, LaneType), "Expected LaneType."
        assert isinstance(self.polyline, Polyline), "Expected Polyline."
        assert isinstance(self.left_boundaries, list) and all(
            isinstance(b, BoundarySegment) for b in self.left_boundaries
        ), "Expected list of BoundarySegment for left."
        assert isinstance(self.right_boundaries, list) and all(
            isinstance(b, BoundarySegment) for b in self.right_boundaries
        ), "Expected list of BoundarySegment for right."

    def is_drivable(self) -> bool:
        """
        Whether the lane is allowed to drive by car like vehicle.

        Returns
        -------
            bool: Return True if the lane is allowed to drive.

        """
        return self.lane_type.is_drivable()

    def as_array(self, *, full: bool = False, as_3d: bool = True) -> NDArray:
        """
        Return polyline containing all points on the road segment.

        Args:
        ----
            full (bool, optional): Indicates whether to return `(x, y, z, dx, dy, dz, type_id)`.
                If `False`, returns `(x, y, z)`. Defaults to False.
            as_3d (bool, optional): If `True` returns array containing 3D coordinates.
                Otherwise, 2D coordinates. Defaults to True.

        Returns:
        -------
            NDArray: Polyline of the road segment in shape (N, D).

        """
        all_polyline: list[NDArray] = [self.polyline.as_array(full=full, as_3d=as_3d)]
        duplicate_boundary_ids: list[int] = []

        def _append_boundaries(boundaries: list[BoundarySegment]) -> None:
            for bound in boundaries:
                if bound.boundary_id in duplicate_boundary_ids:
                    continue
                duplicate_boundary_ids.append(bound.boundary_id)
                all_polyline.append(bound.as_array(full=full, as_3d=as_3d))

        _append_boundaries(self.left_boundaries)
        _append_boundaries(self.right_boundaries)

        return np.concatenate(all_polyline, axis=0, dtype=np.float32)

    def is_left_crossable(self) -> bool:
        """
        Whether all left boundaries of lane are allowed to cross.

        Returns
        -------
            bool: Return True if all boundaries are allowed to cross.

        """
        if len(self.left_boundaries) == 0:
            return False
        return all(bound.is_crossable() for bound in self.left_boundaries)

    def is_right_crossable(self) -> bool:
        """
        Whether all right boundaries of this lane are allowed to cross.

        Returns
        -------
            bool: Return True if all boundaries are allowed to cross.

        """
        if len(self.right_boundaries) == 0:
            return False
        return all(bound.is_crossable() for bound in self.right_boundaries)

    def is_left_virtual(self) -> bool:
        """
        Whether all left boundaries of this lane are virtual (=unknown).

        Returns
        -------
            bool: Return True if all boundaries are virtual.

        """
        if len(self.left_boundaries) == 0:
            return False
        return all(bound.is_virtual() for bound in self.left_boundaries)

    def is_right_virtual(self) -> bool:
        """
        Whether all right boundaries of this lane are virtual (=unknown).

        Returns
        -------
            bool: Return True if all boundaries are virtual.

        """
        if len(self.right_boundaries) == 0:
            return False
        return all(bound.is_virtual() for bound in self.right_boundaries)

    def has_left_neighbor(self) -> bool:
        """
        Whether the lane segment has the neighbor lane on its left side.

        Returns
        -------
            bool: Return True if it has at least one `left_neighbor_ids`.

        """
        return len(self.left_neighbor_ids) > 0

    def has_right_neighbor(self) -> bool:
        """
        Whether the lane segment has the neighbor lane on its right side.

        Returns
        -------
            bool: Return True if it has at least one `right_neighbor_ids`.

        """
        return len(self.right_neighbor_ids) > 0


@dataclass
class BoundarySegment:
    """
    Represents a boundary segment which is RoadLine or RoadEdge.

    Attributes
    ----------
        boundary_id (int): Unique ID associated with this boundary.
        boundary_type (BoundaryType): `BoundaryType` instance.
        polyline (Polyline): `Polyline` instance.

    """

    boundary_id: int
    boundary_type: BoundaryType
    polyline: Polyline

    def __post_init__(self) -> None:
        assert isinstance(self.boundary_type, BoundaryType), "Expected BoundaryType."
        assert isinstance(self.polyline, Polyline), "Expected Polyline."

    def is_crossable(self) -> bool:
        """
        Indicate whether the boundary is allowed to cross or not.

        Return value depends on the `BoundaryType` definition.

        Returns
        -------
            bool: Return True if the boundary is allowed to cross.

        """
        return self.boundary_type.is_crossable()

    def is_virtual(self) -> bool:
        """
        Indicate whether the boundary is virtual(or Unknown) or not.

        Returns
        -------
            bool: Return True if the boundary is virtual.

        """
        return self.boundary_type.is_virtual()

    def as_array(self, *, full: bool = False, as_3d: bool = True) -> NDArray:
        """
        Return the polyline as `NDArray`.

        Args:
        ----
            full (bool, optional): Indicates whether to return `(x, y, z, dx, dy, dz, type_id)`.
                If `False`, returns `(x, y, z)`. Defaults to False.
            as_3d (bool, optional): If `True` returns array containing 3D coordinates.
                Otherwise, 2D coordinates. Defaults to True.

        Returns:
        -------
            NDArray: Polyline array.

        """
        return self.polyline.as_array(full=full, as_3d=as_3d)
