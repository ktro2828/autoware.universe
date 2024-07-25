from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray

__all__ = ("AgentState", "AgentTrajectory")


@dataclass(frozen=True)
class AgentState:
    """A class represents agent state at the specific time."""

    uuid: str
    timestamp: float = -np.inf
    label_id: int = -1
    xyz: NDArray = np.zeros(3)
    size: NDArray = np.zeros(3)
    yaw: float = 0.0
    vxy: NDArray = np.zeros(2)
    is_valid: bool = False

    @property
    def xy(self) -> NDArray:
        return self.xyz[:2]

    @xy.setter
    def xy(self, xy: ArrayLike) -> None:
        self.xyz[:2] = xy


@dataclass
class AgentTrajectory:
    """
    A class represents agent trajectory.

    Attributes
    ----------
        waypoints (NDArray): Trajectory waypoints in shape (..., D).

    """

    waypoints: NDArray
    label_ids: NDArray

    # NOTE: For the 1DArray indices must be a list.
    XYZ_IDX: ClassVar[list[int]] = [0, 1, 2]
    XY_IDX: ClassVar[list[int]] = [0, 1]
    SIZE_IDX: ClassVar[list[int]] = [3, 4, 5]
    YAW_IDX: ClassVar[int] = 6
    VEL_IDX: ClassVar[list[int]] = [7, 8]
    IS_VALID_IDX: ClassVar[int] = 9

    num_dim: ClassVar[int] = 10

    def __post_init__(self) -> None:
        assert self.waypoints.shape[-1] == self.num_dim
        assert len(self.waypoints) == len(self.label_ids)

    @property
    def xyz(self) -> NDArray:
        return self.waypoints[..., self.XYZ_IDX]

    @xyz.setter
    def xyz(self, xyz: ArrayLike) -> None:
        self.waypoints[..., self.XYZ_IDX] = xyz

    @property
    def xy(self) -> NDArray:
        return self.waypoints[..., self.XY_IDX]

    @xy.setter
    def xy(self, xy: ArrayLike) -> None:
        self.waypoints[..., self.XY_IDX] = xy

    @property
    def size(self) -> NDArray:
        return self.waypoints[..., self.SIZE_IDX]

    @size.setter
    def size(self, size: ArrayLike) -> None:
        self.waypoints[..., self.SIZE_IDX] = size

    @property
    def yaw(self) -> NDArray:
        return self.waypoints[..., self.YAW_IDX]

    @yaw.setter
    def yaw(self, yaw: ArrayLike) -> None:
        self.waypoints[..., self.YAW_IDX] = yaw

    @property
    def vxy(self) -> NDArray:
        return self.waypoints[..., self.VEL_IDX]

    @vxy.setter
    def vxy(self, velocity: ArrayLike) -> None:
        self.waypoints[..., self.VEL_IDX] = velocity

    @property
    def is_valid(self) -> NDArray:
        return self.waypoints[..., self.IS_VALID_IDX] == 1

    @property
    def shape(self) -> ArrayLike:
        return self.waypoints.shape

    def as_array(self) -> NDArray:
        return self.waypoints.copy()
