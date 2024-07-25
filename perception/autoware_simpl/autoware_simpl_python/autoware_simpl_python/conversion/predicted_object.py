from typing import Sequence

from autoware_perception_msgs.msg import PredictedObject
from autoware_perception_msgs.msg import PredictedObjects
from autoware_perception_msgs.msg import PredictedPath
from geometry_msgs.msg import Pose
from numpy.typing import NDArray
from rclpy.duration import Duration
from std_msgs.msg import Header

from .tracked_object import ObjectInfo

__all__ = ("to_predicted_objects",)


def to_predicted_objects(
    header: Header,
    infos: Sequence[ObjectInfo],
    pred_scores: NDArray,
    pred_trajs: NDArray,
) -> PredictedObjects:
    output = PredictedObjects()
    output.header = header
    # convert each object
    for info, cur_scores, cur_trajs in zip(infos, pred_scores, pred_trajs, strict=True):
        pred_obj = _to_predicted_object(info, cur_scores, cur_trajs)
        output.objects.append(pred_obj)

    return output


def _to_predicted_object(
    info: ObjectInfo,
    pred_scores: NDArray,
    pred_trajs: NDArray,
) -> PredictedObject:
    """Convert prediction of a single object to PredictedObject msg.

    Args:
        info (ObjectInfo): Object original info.
        pred_scores (NDArray): Predicted score in the shape of (M,).
        pred_trajs (NDArray): Predicted trajectory in the shape of (M, T, 4).

    Returns:
        PredictedObject: Instanced msg.
    """
    output = PredictedObject()

    output.object_id = info.uuid
    output.classification = info.classification
    output.shape = info.shape
    output.existence_probability = info.existence_probability

    output.kinematics.initial_pose_with_covariance = info.kinematics.pose_with_covariance
    output.kinematics.initial_twist_with_covariance = info.kinematics.twist_with_covariance
    output.kinematics.initial_acceleration_with_covariance = (
        info.kinematics.acceleration_with_covariance
    )

    # convert each mode
    for cur_score, cur_traj in zip(pred_scores, pred_trajs, strict=True):
        cur_mode_path = _to_predicted_path(cur_score, cur_traj)
        output.kinematics.predicted_paths.append(cur_mode_path)

    return output


def _to_predicted_path(
    info: ObjectInfo,
    pred_score: float,
    pred_traj: NDArray,
) -> PredictedPath:
    """Convert prediction of a single mode to PredictedPath msg.

    Args:
        info (ObjectInfo): Object original info.
        pred_score (float): Predicted score.
        pred_traj (NDArray): Predicted waypoints in the shape of (T, 4).

    Returns:
        PredictedPath: Instanced msg.
    """
    output = PredictedPath()
    output.time_step = Duration(seconds=0.1)  # TODO(ktro2828): use specific value?
    output.confidence = pred_score
    for x, y, _, _ in pred_traj:  # (x, y, vx, vy)
        pose = Pose()

        pose.position.x = x
        pose.position.y = y
        pose.position.z = info.kinematics.pose_with_covariance.pose.position.z
        pose.orientation = info.kinematics.pose_with_covariance.pose.orientation

        output.path.append(pose)

    return output
