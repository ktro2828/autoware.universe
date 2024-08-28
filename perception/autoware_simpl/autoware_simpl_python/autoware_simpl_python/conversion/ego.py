from autoware_simpl_python.dataclass import AgentState
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
import numpy as np
from numpy.typing import NDArray

from .misc import timestamp2ms
from .misc import yaw_from_quaternion

__all__ = ("convert_odometry", "convert_transform_stamped")


def convert_odometry(
    msg: Odometry,
    uuid: str,
    label_id: int,
    size: NDArray,
) -> AgentState:
    """Convert odometry msg to AgentState.

    Args:
        msg (Odometry): Odometry msg.
        uuid (str): Object uuid.
        label_id (int): Label id.
        size (NDArray): Object size in the order of (length, width, height).

    Returns:
        AgentState: Instanced AgentState.
    """
    timestamp = timestamp2ms(msg.header)
    pose = msg.pose.pose
    xyz = np.array((pose.position.x, pose.position.y, pose.position.z))

    yaw = yaw_from_quaternion(pose.orientation)

    twist = msg.twist.twist
    vxy = np.array((twist.linear.x, twist.linear.y))

    return AgentState(
        uuid=uuid,
        timestamp=timestamp,
        label_id=label_id,
        xyz=xyz,
        size=size,
        yaw=yaw,
        vxy=vxy,
        is_valid=True,
    )


def convert_transform_stamped(
    tf_stamped: TransformStamped,
    uuid: str,
    label_id: int,
    size: NDArray,
    vxy: NDArray,
) -> AgentState:
    timestamp = timestamp2ms(tf_stamped.header)

    translation = tf_stamped.transform.translation
    xyz = np.array((translation.x, translation.y, translation.z))
    yaw = yaw_from_quaternion(tf_stamped.transform.rotation)

    return AgentState(
        uuid=uuid,
        timestamp=timestamp,
        label_id=label_id,
        xyz=xyz,
        size=size,
        yaw=yaw,
        vxy=vxy,
        is_valid=True,
    )
