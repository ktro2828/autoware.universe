from autoware_simpl_python.dataclass import BoundarySegment
from autoware_simpl_python.dataclass import LaneSegment
from autoware_simpl_python.dataclass import Polyline
from autoware_simpl_python.datatype import BoundaryType
from autoware_simpl_python.datatype import T4Lane
from autoware_simpl_python.datatype import T4Polyline
from autoware_simpl_python.datatype import T4RoadEdge
from autoware_simpl_python.datatype import T4RoadLine
import lanelet2
from lanelet2.routing import RoutingGraph
from lanelet2.traffic_rules import Locations
from lanelet2.traffic_rules import Participants
from lanelet2.traffic_rules import create as create_traffic_rules
from lanelet2_extension_python.projection import MGRSProjector
import numpy as np

__all__ = ("convert_lanelet",)


def convert_lanelet(filename: str) -> list[LaneSegment]:
    """Convert lanelet to list of lane segments.

    Args:
        filename (str): Path to osm file.

    Returns:
        list[LaneSegment]: List of converted lane segments.
    """
    lanelet_map = _load_osm(filename)
    traffic_rules = create_traffic_rules(Locations.Germany, Participants.Vehicle)
    routing_graph = RoutingGraph(lanelet_map, traffic_rules)

    lane_segments: list[LaneSegment] = []
    for lanelet in lanelet_map.laneletLayer:
        lanelet_subtype = _get_lanelet_subtype(lanelet)
        if lanelet_subtype == "":
            continue

        if not T4Lane.contains(lanelet_subtype) or lanelet_subtype == "walkway":
            continue

        lane_type = T4Lane.from_str(lanelet_subtype)
        waypoints = np.array([(line.x, line.y, line.z) for line in lanelet.centerline])
        polyline_type = T4Polyline.from_str(lanelet_subtype)
        polyline = Polyline(polyline_type=polyline_type, waypoints=waypoints)
        is_intersection = _is_intersection(lanelet)
        left_neighbor_ids, right_neighbor_ids = _get_left_and_right_neighbor_ids(
            lanelet, routing_graph
        )
        speed_limit_mph = _get_speed_limit_mph(lanelet)

        # boundary segments
        left_linestring, right_linestring = _get_left_and_right_linestring(lanelet)
        left_boundary = _get_boundary_segment(left_linestring)
        right_boundary = _get_boundary_segment(right_linestring)

        lane_segments.append(
            LaneSegment(
                lane_id=lanelet.id,
                lane_type=lane_type,
                polyline=polyline,
                is_intersection=is_intersection,
                left_boundaries=[left_boundary],
                right_boundaries=[right_boundary],
                left_neighbor_ids=left_neighbor_ids,
                right_neighbor_ids=right_neighbor_ids,
                speed_limit_mph=speed_limit_mph,
            )
        )
    return lane_segments


def _load_osm(filename: str) -> lanelet2.core.LaneletMap:
    """Load lanelet map from osm file.

    Args:
        filename (str): Path to osm file.

    Returns:
        lanelet2.core.LaneletMap: Loaded lanelet map.
    """
    projection = MGRSProjector(lanelet2.io.Origin(0.0, 0.0))
    return lanelet2.io.load(filename, projection)


def _get_lanelet_subtype(lanelet: lanelet2.core.Lanelet) -> str:
    """Return subtype name of the lanelet.

    Args:
        lanelet (lanelet2.core.Lanelet): Lanelet instance.

    Returns:
        str: Subtype name. Return "" if it has no attribute named `subtype`.
    """
    if "subtype" in lanelet.attributes:
        return lanelet.attributes["subtype"]
    else:
        return ""


def _is_intersection(lanelet: lanelet2.core.Lanelet) -> bool:
    """Check whether specified lanelet is a part of intersection.

    Args:
        lanelet (lanelet2.core.Lanelet): Lanelet instance.

    Returns:
        bool: Return `True` if the lanelet has an attribute named `turn_direction`.
    """
    return "turn_direction" in lanelet.attributes


def _get_left_and_right_neighbor_ids(
    lanelet: lanelet2.core.Lanelet, routing_graph: RoutingGraph
) -> tuple[list[int], list[int]]:
    """Return the pair of the list of left and right neighbor ids.

    Args:
        lanelet (lanelet2.core.Lanelet): Lanelet instance.
        routing_graph (RoutingGraph): RoutingGraph instance.

    Returns:
        tuple[list[int], list[int]]: List of left and right ids.
    """
    left_lanelet = routing_graph.left(lanelet)
    right_lanelet = routing_graph.right(lanelet)
    left_neighbor_ids = [left_lanelet.id] if left_lanelet is not None else []
    right_neighbor_ids = [right_lanelet.id] if right_lanelet is not None else []
    return left_neighbor_ids, right_neighbor_ids


def _get_speed_limit_mph(lanelet: lanelet2.core.Lanelet) -> float | None:
    """
    Return the lane speed limit in miles per hour (mph).

    Args:
    ----
        lanelet (lanelet2.core.Lanelet): Lanelet instance.

    Returns:
    -------
        float | None: If the lane has the speed limit return float, otherwise None.

    """
    kph2mph = 0.621371
    if "speed_limit" in lanelet.attributes:
        # NOTE: attributes of ["speed_limit"] is str
        return float(lanelet.attributes["speed_limit"]) * kph2mph
    else:
        return None


def _get_left_and_right_linestring(
    lanelet: lanelet2.core.Lanelet,
) -> tuple[lanelet2.core.LineString3d, lanelet2.core.LineString3d]:
    """
    Return the left and right boundaries from lanelet.

    Args:
    ----
        lanelet (lanelet2.core.Lanelet): Lanelet instance.

    Returns:
    -------
        tuple[lanelet2.core.LineString3d, lanelet2.core.LineString3d]: Left and right boundaries.

    """
    return lanelet.leftBound, lanelet.rightBound


def _get_boundary_segment(linestring: lanelet2.core.LineString3d) -> BoundarySegment:
    """
    Return the `BoundarySegment` from linestring.

    Args:
    ----
        linestring (lanelet2.core.LineString3d): LineString instance.

    Returns:
    -------
        BoundarySegment: BoundarySegment instance.

    """
    boundary_type = _get_boundary_type(linestring)
    waypoints = np.array([(line.x, line.y, line.z) for line in linestring])
    polyline_type = T4Polyline.from_str(str(boundary_type))
    polyline = Polyline(polyline_type=polyline_type, waypoints=waypoints)
    return BoundarySegment(linestring.id, boundary_type, polyline)


def _get_linestring_type(linestring: lanelet2.core.LineString3d) -> str:
    """
    Return type name from linestring.

    Args:
    ----
        linestring (lanelet2.core.LineString3d): Linestring instance.

    Returns:
    -------
        str: Type name. Return "" if it has no attribute named type.

    """
    if "type" in linestring.attributes:
        return linestring.attributes["type"]
    else:
        return ""


def _get_linestring_subtype(linestring: lanelet2.core.LineString3d) -> str:
    """
    Return subtype name from linestring.

    Args:
    ----
        linestring (lanelet2.core.LineString3d): Linestring instance.

    Returns:
    -------
        str: Subtype name. Return "" if it has no attribute named subtype.

    """
    if "subtype" in linestring.attributes:
        return linestring.attributes["subtype"]
    else:
        return ""


def _is_virtual_linestring(line_type: str, line_subtype: str) -> bool:
    """
    Indicate whether input linestring type and subtype is virtual.

    Args:
    ----
        line_type (str): Line type name.
        line_subtype (str): Line subtype name.

    Returns:
    -------
        bool: Return True if line type is `virtual` and subtype is `""`.

    """
    return line_type == "virtual" and line_subtype == ""


def _is_roadedge_linestring(line_type: str, _line_subtype: str) -> bool:
    """
    Indicate whether input linestring type and subtype is supported RoadEdge.

    Args:
    ----
        line_type (str): Line type name.
        _line_subtype (str): Line subtype name.

    Returns:
    -------
        bool: Return True if line type is contained in T4RoadEdge.

    Note:
    ----
        Currently `_line_subtype` is not used, but it might be used in the future.

    """
    return line_type.upper() in T4RoadEdge.__members__


def _is_roadline_linestring(_line_type: str, line_subtype: str) -> bool:
    """
    Indicate whether input linestring type and subtype is supported RoadLine.

    Args:
    ----
        _line_type (str): Line type name.
        line_subtype (str): Line subtype name.

    Returns:
    -------
        bool: Return True if line subtype is contained in T4RoadLine.

    Note:
    ----
        Currently `_line_type` is not used, but it might be used in the future.

    """
    return line_subtype.upper() in T4RoadLine.__members__


def _get_boundary_type(linestring: lanelet2.core.LineString3d) -> BoundaryType:
    """
    Return the `BoundaryType` from linestring.

    Args:
    ----
        linestring (lanelet2.core.LineString3d): LineString instance.

    Returns:
    -------
        BoundaryType: BoundaryType instance.

    """
    line_type = _get_linestring_type(linestring)
    line_subtype = _get_linestring_subtype(linestring)
    if _is_virtual_linestring(line_type, line_subtype):
        return T4RoadLine.VIRTUAL
    elif _is_roadedge_linestring(line_type, line_subtype):
        return T4RoadEdge.from_str(line_type)
    elif _is_roadline_linestring(line_type, line_subtype):
        return T4RoadLine.from_str(line_subtype)
    else:
        return T4RoadLine.VIRTUAL
