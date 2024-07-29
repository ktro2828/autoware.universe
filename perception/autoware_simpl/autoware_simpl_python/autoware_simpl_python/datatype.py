from enum import IntEnum, EnumMeta
from abc import ABCMeta, abstractmethod
import sys

if sys.version_info < (3, 11):

    from typing_extensions import Self

else:
    from typing import Self


__all__ = ("PolylineType", "LaneType", "BoundaryType", "T4Lane", "T4RoadLine", "T4RoadEdge")


class ABCEnumMeta(EnumMeta, ABCMeta):
    pass


class _TypeBase(IntEnum, metaclass=ABCEnumMeta):
    UNKNOWN = -1

    @classmethod
    def from_str(cls, name: str) -> Self:
        """
        Construct from the name of member.

        Args:
        ----
            name (str): Name of an enum member.

        Returns:
        -------
            Self: Constructed member.

        """
        name = name.upper()
        if name not in cls.__members__:
            return cls.UNKNOWN
        else:
            return cls.__members__[name]

    @classmethod
    def contains(cls, name: str) -> bool:
        """
        Check whether the input name is contained in members.

        Args:
        ----
            name (str): Name of enum member.

        Returns:
        -------
            bool: Whether it is contained.

        """
        return name.upper() in cls.__members__


class PolylineType(_TypeBase, metaclass=ABCEnumMeta):
    """A base enum of Polyline."""

    def is_dynamic(self) -> bool:
        """
        Whether the item is dynamic.

        Returns
        -------
            bool: Return always False.

        """
        return False


class LaneType(_TypeBase, metaclass=ABCEnumMeta):
    """A base enum of Lane."""

    def is_dynamic(self) -> bool:
        """
        Whether the item is dynamic.

        Returns
        -------
            bool: Return always False.

        """
        return False

    @abstractmethod
    def is_drivable(self) -> bool:
        """
        Indicate whether the lane is drivable.

        Returns
        -------
            bool: Return `True` if drivable.

        """


class BoundaryType(_TypeBase, metaclass=ABCEnumMeta):
    """A base enum of RoadLine and RoadEdge."""

    def is_dynamic(self) -> bool:
        """
        Indicate whether the lane is drivable.

        Returns
        -------
            bool: Return always False.

        """
        return False

    @abstractmethod
    def is_virtual(self) -> bool:
        """
        Whether the boundary is virtual or not.

        Returns
        -------
            bool: Return `True` if the boundary is virtual.

        """

    @abstractmethod
    def is_crossable(self) -> bool:
        """
        Whether the boundary is allowed to cross or not.

        Returns
        -------
            bool: Return `True` if the boundary is allowed to cross.

        """


class T4Polyline(PolylineType):
    """Polyline types in T4."""

    # for lane
    ROAD = 0
    HIGHWAY = 1
    ROAD_SHOULDER = 2
    BICYCLE_LANE = 3
    PEDESTRIAN_LANE = 4
    WALKWAY = 5

    # for road line
    DASHED = 6
    SOLID = 7
    DASHED_DASHED = 8
    VIRTUAL = 9

    # for road edge
    ROAD_BORDER = 10

    # for crosswalk
    CROSSWALK = 11

    # for stop sign
    TRAFFIC_SIGN = 12

    # for speed bump
    SPEED_BUMP = 13

    # catch otherwise
    UNKNOWN = -1


class T4Lane(LaneType):
    """Lane types in T4."""

    ROAD = 0
    HIGHWAY = 1
    ROAD_SHOULDER = 2
    BICYCLE_LANE = 3
    PEDESTRIAN_LANE = 4
    WALKWAY = 5

    def is_drivable(self) -> bool:
        """
        Indicate whether the lane is drivable.

        Returns
        -------
            bool: True if drivable.

        """
        return self in (T4Lane.ROAD, T4Lane.HIGHWAY, T4Lane.ROAD_SHOULDER)


class T4RoadLine(BoundaryType):
    """Road line types in T4."""

    DASHED = 0
    SOLID = 1
    DASHED_DASHED = 2
    VIRTUAL = 3

    def is_crossable(self) -> bool:
        """
        Whether the boundary is allowed to cross or not.

        Returns
        -------
            bool: Return `True` if the boundary is allowed to cross.

        """
        return self in (T4RoadLine.DASHED, T4RoadLine.DASHED_DASHED)

    def is_virtual(self) -> bool:
        """
        Whether the boundary is virtual or not.

        Returns
        -------
            bool: Return `True` if the boundary is virtual.

        """
        return self == T4RoadLine.VIRTUAL


class T4RoadEdge(BoundaryType):
    """Road edge types in T4."""

    ROAD_BORDER = 0

    def is_crossable(self) -> bool:
        """
        Whether the boundary is allowed to cross or not.

        Returns
        -------
            bool: Return always `False`.

        """
        return False

    def is_virtual(self) -> bool:
        """
        Whether the boundary is virtual or not.

        Returns
        -------
            bool: Return always `False`.

        """
        return False
