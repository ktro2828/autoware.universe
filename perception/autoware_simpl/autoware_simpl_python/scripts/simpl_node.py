from dataclasses import dataclass

from autoware_perception_msgs.msg import ObjectClassification
from autoware_perception_msgs.msg import PredictedObjects
from autoware_perception_msgs.msg import TrackedObjects
from autoware_simpl_python.conversion import convert_lanelet
from autoware_simpl_python.conversion import convert_odometry
from autoware_simpl_python.conversion import from_tracked_objects
from autoware_simpl_python.conversion import timestamp2ms
from autoware_simpl_python.conversion import to_predicted_objects
from autoware_simpl_python.dataclass import AgentHistory
from autoware_simpl_python.dataclass import AgentState
from autoware_simpl_python.dataclass import LaneSegment
from autoware_simpl_python.geometry import rotate_along_z
from autoware_simpl_python.preprocess import embed_agent
from autoware_simpl_python.preprocess import embed_lane
from autoware_simpl_python.preprocess import relative_pose_encode
from nav_msgs.msg import Odometry
import numpy as np
from numpy.typing import NDArray
from onnxruntime import InferenceSession
from rcl_interfaces.msg import ParameterDescriptor
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
import rclpy.parameter


@dataclass
class ModelInput:
    agent: NDArray
    lane: NDArray
    rpe: NDArray
    rpe_mask: NDArray | None = None


class SimplNode(Node):
    def __init__(self) -> None:
        super().__init__("simpl_python_node")

        # TODO(ktro2828): QoS settings
        # subscribers
        self._object_sub = self.create_subscription(
            TrackedObjects,
            "~/input/objects",
            self._callback,
            10,
        )
        self._ego_sub = self.create_subscription(Odometry, "~/input/ego", self._on_ego, 10)

        # publisher
        self._object_pub = self.create_publisher(PredictedObjects, "~/output/objects", 10)

        # ROS parameters
        descriptor = ParameterDescriptor(dynamic_typing=True)
        num_timestamp = (
            self.declare_parameter("num_timestamp", descriptor=descriptor)
            .get_parameter_value()
            .integer_value
        )
        self._timestamp_threshold = (
            self.declare_parameter("timestamp_threshold", descriptor=descriptor)
            .get_parameter_value()
            .double_value
        )
        lanelet_file = (
            self.declare_parameter("lanelet_file", descriptor=descriptor)
            .get_parameter_value()
            .string_value
        )
        model_path = (
            self.declare_parameter("model_path", descriptor=descriptor)
            .get_parameter_value()
            .string_value
        )
        build_only = (
            self.declare_parameter("build_only", descriptor=descriptor)
            .get_parameter_value()
            .bool_value
        )

        # input attributes
        self._lane_segments: list[LaneSegment] = convert_lanelet(lanelet_file)
        self._current_ego: AgentState | None = None
        self._history = AgentHistory(max_length=num_timestamp)

        # onnx inference
        self._session = InferenceSession(
            model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self._input_names = [i.name for i in self._session.get_inputs()]
        self._output_names = [o.name for o in self._session.get_outputs()]

        if build_only:
            self.get_logger().info("Onnx runtime session is built and exit.")
            self.destroy_node()
            rclpy.try_shutdown()

    def _on_ego(self, msg: Odometry) -> None:
        # TODO(ktro2828): update values
        uuid = "Ego"
        label_id = ObjectClassification.CAR
        size = np.array((0, 0, 0))

        self._current_ego = convert_odometry(
            msg,
            uuid=uuid,
            label_id=label_id,
            size=size,
        )

    def _callback(self, msg: TrackedObjects) -> None:
        if self._current_ego is None:
            self.get_logger().warning("Ego odometry is not subscribed yet...")
            return

        # remove ancient agent history
        timestamp = timestamp2ms(msg.header)
        self._history.remove_invalid(timestamp, self._timestamp_threshold)

        # update agent history
        # TODO(ktro2828): guarantee the order of agent info and history is same.
        states, infos = from_tracked_objects(msg)
        self._history.update(states)

        # inference
        inputs = self._preprocess()
        # TODO(ktro2828): check model output
        pred_scores, pred_trajs = self._session.run(self._output_names, inputs)
        pred_scores, pred_trajs = self._postprocess(pred_scores, pred_trajs)

        # convert to ROS msg
        pred_objs = to_predicted_objects(
            header=msg.header,
            infos=infos,
            pred_scores=pred_scores,
            pred_trajs=pred_trajs,
        )
        self._object_pub.publish(pred_objs)

    def _preprocess(self) -> dict[str, NDArray]:
        """Run preprocess.

        Returns:
            dict[str, NDArray]: Model inputs, which key is name of input.
        """
        agent, agent_ctr, agent_vec = embed_agent(
            self._history.as_trajectory(),
            self._current_ego,
        )
        lane, lane_ctr, lane_vec = embed_lane(self._lane_segments, self._current_ego)

        rpe, rpe_mask = relative_pose_encode(agent_ctr, agent_vec, lane_ctr, lane_vec)

        inputs = ModelInput(agent, lane, rpe, rpe_mask)
        return {name: getattr(inputs, name) for name in self._input_names}

    def _postprocess(self, pred_scores: NDArray, pred_trajs: NDArray) -> tuple[NDArray, NDArray]:
        """Run postprocess.

        Args:
            pred_scores (NDArray): Predicted scores in the shape of (N, M).
            pred_trajs (NDArray): Predicted trajectories in the shape of (N, M, T, 4).

        Returns:
            tuple[NDArray, NDArray]: Transformed and sorted prediction.
        """
        num_agent, num_mode, num_future, num_feat = pred_trajs.shape
        assert num_feat == 4, f"Expected predicted feature is (x, y, vx, vy), but got {num_feat}"

        # transform from agent centric coords to world coords
        current_agent = self._history.as_trajectory(latest=True)
        pred_trajs[..., :2] = rotate_along_z(
            pred_trajs.reshape(num_agent, -1, num_feat)[..., :2], current_agent.yaw
        ).reshape(num_agent, num_mode, num_future, 2)
        pred_trajs[..., :2] += current_agent.xy[:, None, None, :]

        # sort by score
        pred_scores = np.clip(pred_scores, a_min=0.0, a_max=1.0)
        sort_indices = np.argsort(-pred_scores, axis=1)
        pred_scores = np.take_along_axis(pred_scores, sort_indices, axis=1)
        pred_scores = np.divide(pred_scores, pred_scores.sum(), where=pred_scores != 0)
        pred_trajs = np.take_along_axis(pred_trajs, sort_indices[..., None, None], axis=1)

        return pred_scores, pred_trajs


def main(args=None) -> None:
    rclpy.init(args=args)

    node = SimplNode()
    executor = MultiThreadedExecutor()
    try:
        rclpy.spin(node, executor)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
