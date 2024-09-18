from copy import deepcopy
from typing import Sequence

from autoware_simpl_python.dataclass import AgentState
from autoware_simpl_python.dataclass import AgentTrajectory
from autoware_simpl_python.geometry import rotate_along_z
import numpy as np
from numpy.typing import NDArray

__all__ = ("embed_agent",)


def embed_agent(
    agent: AgentTrajectory,
    current_ego: AgentState,
    target_label_ids: Sequence[int],
) -> tuple[NDArray, NDArray, NDArray]:
    """Embed agent attributes.

    Args:
        agent (AgentTrajectory): Agent past trajectory.
        current_ego (AgentState): Current ego state.
        target_type (Sequence[int]): Sequence of target types.
        agent_type (Sequence[int]): Sequence of agent types in the shape of (N,).

    Returns:
        tuple[NDArray, NDArray, NDArray]: Embedded attributes in the shape of (N, C, T),
            and agent center and vector in the shape of (N, 2).
    """
    num_agent, num_time, _ = agent.shape

    agent, agent_ctr, agent_vec = _transform_agent_coords(agent, current_ego)

    dxy = np.zeros_like(agent.xy)
    dxy[:, 1:] = agent.xy[:, 1:] - agent.xy[:, :-1]

    onehot_dim = len(target_label_ids)
    type_onehot = np.zeros((num_agent, num_time, onehot_dim))
    for i, label_id in enumerate(target_label_ids):
        type_onehot[agent.label_ids == label_id, :, i] = 1

    embedding: NDArray = np.concatenate(
        (
            dxy,
            np.cos(agent.yaw[..., None]),
            np.sin(agent.yaw[..., None]),
            agent.vxy,
            type_onehot,
            agent.is_valid[..., None],
        ),
        axis=-1,
        dtype=np.float32,
    )  # (N, T, C)

    embedding = embedding.transpose(0, 2, 1)  # (N, C, T)

    return embedding, agent_ctr, agent_vec


def _transform_agent_coords(
    agent: AgentTrajectory,
    current_ego: AgentState,
) -> tuple[AgentTrajectory, NDArray, NDArray]:
    """Transform agent coords from world coords to agent centric coords.

    Args:
        agent (AgentTrajectory): Agent past trajectory.
        current_ego (AgentState): Current ego state.

    Returns:
        AgentTrajectory: Transformed agent past trajectory.
    """
    # For RPE
    agent_rpe = deepcopy(agent)
    agent_rpe.xy -= current_ego.xy
    agent_rpe.xy = rotate_along_z(agent_rpe.xy, current_ego.yaw)
    agent_rpe.yaw -= current_ego.yaw
    agent_ctr: NDArray = agent_rpe.xy[:, -1]  # (N, 2)
    ego2agent_yaw: NDArray = agent_rpe.yaw[:, -1]
    agent_vec: NDArray = np.stack(
        (np.cos(ego2agent_yaw), np.sin(ego2agent_yaw)), axis=-1
    )  # (N, 2)

    # Transform from world coords to current agent coords
    agent.xy -= agent.xy[:, -1, :][:, None, :]
    agent.xy = rotate_along_z(agent.xy, agent.yaw[:, -1])
    agent.yaw -= agent.yaw[:, -1][:, None]
    agent.vxy = rotate_along_z(agent.vxy, agent.yaw[:, -1])

    return agent, agent_ctr, agent_vec
