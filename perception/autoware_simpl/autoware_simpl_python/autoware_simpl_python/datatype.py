from abc import ABCMeta
from enum import EnumMeta
from enum import IntEnum
import sys

if sys.version_info < (3, 11):

    from typing_extensions import Self

else:
    from typing import Self


__all__ = (
    "AgentLabel",
    "LaneLabel",
    "RoadLineLabel",
    "RoadEdgeLabel",
)


class ABCEnumMeta(EnumMeta, ABCMeta):
    pass


class _TypeBase(IntEnum, metaclass=ABCEnumMeta):

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


class AgentLabel(_TypeBase):
    """Agent types for T4."""

    # Dynamic movers
    VEHICLE = 0
    PEDESTRIAN = 1
    CYCLIST = 2
    # Static objects
    STATIC = 3

    # Catch-all type for other/unknown objects
    UNKNOWN = 4

    def is_dynamic(self) -> bool:
        """
        Whether the object is dynamic movers.

        Returns
        -------
            bool: True if any of (VEHICLE, PEDESTRIAN, MOTORCYCLIST, CYCLIST, BUS).

        """
        return self in (AgentLabel.VEHICLE, AgentLabel.PEDESTRIAN, AgentLabel.CYCLIST)


# class AgentLabel(_TypeBase):
#     """Agent types for AV2."""

#     # Dynamic movers
#     VEHICLE = 0
#     PEDESTRIAN = 1
#     MOTORCYCLIST = 2
#     CYCLIST = 3
#     BUS = 4

#     # Static objects
#     STATIC = 5
#     BACKGROUND = 6
#     CONSTRUCTION = 7
#     RIDERLESS_BICYCLE = 8

#     # Catch-all type for other/unknown objects
#     UNKNOWN = 9

#     def is_dynamic(self) -> bool:
#         """
#         Whether the object is dynamic movers.

#         Returns
#         -------
#             bool: True if any of (VEHICLE, PEDESTRIAN, MOTORCYCLIST, CYCLIST, BUS).

#         """
#         return self in (
#             AgentLabel.VEHICLE,
#             AgentLabel.PEDESTRIAN,
#             AgentLabel.MOTORCYCLIST,
#             AgentLabel.CYCLIST,
#             AgentLabel.BUS,
#         )


class PolylineLabel(_TypeBase):
    """Polyline types."""

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


class LaneLabel(_TypeBase):
    """Lane types."""

    ROAD = 0
    HIGHWAY = 1
    ROAD_SHOULDER = 2
    BICYCLE_LANE = 3
    PEDESTRIAN_LANE = 4
    WALKWAY = 5

    # catch otherwise
    UNKNOWN = -1

    def is_drivable(self) -> bool:
        """
        Indicate whether the lane is drivable.

        Returns
        -------
            bool: True if drivable.

        """
        return self in (LaneLabel.ROAD, LaneLabel.HIGHWAY, LaneLabel.ROAD_SHOULDER)


class RoadLineLabel(_TypeBase):
    """Road line types."""

    DASHED = 0
    SOLID = 1
    DASHED_DASHED = 2
    VIRTUAL = 3

    # catch otherwise
    UNKNOWN = -1

    def is_crossable(self) -> bool:
        """
        Whether the boundary is allowed to cross or not.

        Returns
        -------
            bool: Return `True` if the boundary is allowed to cross.

        """
        return self in (RoadLineLabel.DASHED, RoadLineLabel.DASHED_DASHED)

    def is_virtual(self) -> bool:
        """
        Whether the boundary is virtual or not.

        Returns
        -------
            bool: Return `True` if the boundary is virtual.

        """
        return self == RoadLineLabel.VIRTUAL


class RoadEdgeLabel(_TypeBase):
    """Road edge types."""

    ROAD_BORDER = 0

    # catch otherwise
    UNKNOWN = -1

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
