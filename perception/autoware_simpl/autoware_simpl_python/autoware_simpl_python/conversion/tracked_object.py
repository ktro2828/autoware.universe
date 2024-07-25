from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
from uuid import UUID as PyUUID

from autoware_perception_msgs.msg import ObjectClassification
from autoware_perception_msgs.msg import Shape
from autoware_perception_msgs.msg import TrackedObject
from autoware_perception_msgs.msg import TrackedObjectKinematics
from autoware_perception_msgs.msg import TrackedObjects
from autoware_simpl_python.dataclass import AgentState
import numpy as np
from unique_identifier_msgs.msg import UUID as RosUUID

from .misc import timestamp2ms
from .misc import yaw_from_quaternion

__all__ = ("ObjectInfo", "from_tracked_objects")


@dataclass(frozen=True)
class ObjectInfo:
    uuid: RosUUID
    classification: Sequence[ObjectClassification]
    shape: Shape
    existence_probability: float
    kinematics: TrackedObjectKinematics

    @classmethod
    def from_msg(cls, msg: TrackedObject) -> ObjectInfo:
        return cls(
            uuid=msg.object_id,
            classification=msg.classification,
            shape=msg.shape,
            existence_probability=msg.existence_probability,
            kinematics=msg.kinematics,
        )


def from_tracked_objects(
    msg: TrackedObjects,
) -> tuple[list[AgentState], list[ObjectInfo]]:
    """Convert TrackedObjects msg to AgentTrajectory instance.

    Args:
        msg (TrackedObjects):

    Returns:
        tuple[list[AgentState], list[ObjectInfo]]:
    """
    states: list[AgentState] = []
    infos: list[ObjectInfo] = []
    timestamp = timestamp2ms(msg.header)
    for obj in msg.objects:
        obj: TrackedObject

        classification = _max_probability_classification(obj.classification)
        label_id = classification.label

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

        infos.append(ObjectInfo.from_msg(obj))

    return states, infos


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
