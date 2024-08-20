from __future__ import annotations

from typing import Sequence
from uuid import UUID as PyUUID

from autoware_perception_msgs.msg import ObjectClassification
from autoware_perception_msgs.msg import TrackedObject
from autoware_perception_msgs.msg import TrackedObjects
from autoware_simpl_python.dataclass import AgentState
from autoware_simpl_python.dataclass import OriginalInfo
from autoware_simpl_python.datatype import T4Agent
import numpy as np
from unique_identifier_msgs.msg import UUID as RosUUID

from .misc import timestamp2ms
from .misc import yaw_from_quaternion

__all__ = ("from_tracked_objects", "sort_object_infos")


def from_tracked_objects(
    msg: TrackedObjects,
) -> tuple[list[AgentState], list[OriginalInfo]]:
    """Convert TrackedObjects msg to AgentTrajectory instance.

    Args:
        msg (TrackedObjects): Tracked objects.

    Returns:
        tuple[list[AgentState], list[OriginalInfo]]: List of converted states and original information.
    """
    states: list[AgentState] = []
    infos: list[OriginalInfo] = []
    timestamp = timestamp2ms(msg.header)
    for obj in msg.objects:
        obj: TrackedObject

        classification = _max_probability_classification(obj.classification)
        label_id = _convert_label(classification.label)

        pose = obj.kinematics.pose_with_covariance.pose
        xyz = np.array((pose.position.x, pose.position.y, pose.position.z))

        dimensions = obj.shape.dimensions
        size = np.array((dimensions.x, dimensions.y, dimensions.z))

        yaw = yaw_from_quaternion(pose.orientation)

        twist = obj.kinematics.twist_with_covariance.twist
        vxy = np.array((twist.linear.x, twist.linear.y))

        states.append(
            AgentState(
                uuid=_uuid_msg_to_str(obj.object_id),
                timestamp=timestamp,
                label_id=label_id,
                xyz=xyz,
                size=size,
                yaw=yaw,
                vxy=vxy,
                is_valid=True,
            )
        )

        infos.append(OriginalInfo.from_msg(obj))

    return states, infos


def sort_object_infos(infos: dict[str, OriginalInfo], uuids: list[str]) -> list[OriginalInfo]:
    """Sort the list of object infos by input uuids.

    Args:
        infos (dict[str, OriginalInfo]): Dict of ObjectInfos history.
        uuids (list[str]): List of uuids.

    Returns:
        list[OriginalInfo]: Sorted ObjectInfos.
    """
    return [infos[uuid] for uuid in uuids]


def _uuid_msg_to_str(uuid_msg: RosUUID) -> str:
    bytes_array = bytes(uuid_msg.uuid)
    uuid_obj = PyUUID(bytes=bytes_array)
    return str(uuid_obj)


def _max_probability_classification(
    classifications: Sequence[ObjectClassification],
) -> ObjectClassification:
    """Return a max probability classification.

    Args:
        classifications (Sequence[ObjectClassification]): Sequence of classifications.

    Returns:
        ObjectClassification: Max probability classification.
    """
    return max(classifications, key=lambda c: c.probability)


def _convert_label(label: int) -> int:
    """Convert the label of ObjectClassification to T4Agent.

    Args:
        label (int): Label id.

    Returns:
        int: T4Agent value.
    """
    if label in (
        ObjectClassification.CAR,
        ObjectClassification.BUS,
        ObjectClassification.TRAILER,
        ObjectClassification.TRUCK,
    ):
        return T4Agent.VEHICLE.value
    elif label in (ObjectClassification.BICYCLE, ObjectClassification.MOTORCYCLE):
        return T4Agent.CYCLIST.value
    elif label == ObjectClassification.PEDESTRIAN:
        return T4Agent.PEDESTRIAN
    elif label == ObjectClassification.UNKNOWN:
        return T4Agent.UNKNOWN.value
    else:
        return T4Agent.STATIC.value
