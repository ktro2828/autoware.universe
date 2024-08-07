from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from dataclasses import field
from typing import Sequence

import numpy as np

from .agent import AgentState
from .agent import AgentTrajectory

__all__ = ("AgentHistory",)


@dataclass
class AgentHistory:
    """A class to store agent history data."""

    max_length: int
    histories: dict[str, deque[AgentState]] = field(default_factory=dict, init=False)

    def update(self, states: Sequence[AgentState]) -> None:
        """Update history data.

        Args:
            states (Sequence[AgentState]): Sequence of AgentStates.
        """
        for state in states:
            uuid = state.uuid
            if uuid not in self.histories:
                self.histories[uuid] = deque(
                    [AgentState(uuid)] * self.max_length,
                    maxlen=self.max_length,
                )
            self.histories[uuid].append(state)

    def remove_invalid(self, current_timestamp: float, threshold: float) -> None:
        """Remove agent histories whose the latest state are invalid or ancient.

        Args:
            current_timestamp (float): Current timestamp in [ms].
            threshold (float): Threshold value to filter out ancient history in [ms].
        """
        remove_uuids = []
        for uuid, history in self.histories.items():
            latest = history[-1]
            if (not latest.is_valid) or self.is_ancient(
                latest.timestamp, current_timestamp, threshold
            ):
                remove_uuids.append(uuid)

        for uuid in remove_uuids:
            del self.histories[uuid]

    @staticmethod
    def is_ancient(latest_timestamp: float, current_timestamp: float, threshold: float) -> bool:
        """Check whether the latest state is ancient.

        Args:
            latest_timestamp (float): Latest state timestamp in [ms].
            current_timestamp (float): Current timestamp in [ms].
            threshold (float): Timestamp threshold in [ms].

        Returns:
            bool: Return True if timestamp difference is greater than threshold,
                which means ancient.
        """
        timestamp_diff = abs(current_timestamp - latest_timestamp)
        return timestamp_diff > threshold

    def as_trajectory(self, *, latest: bool = False) -> tuple[AgentTrajectory, list[str]]:
        """Convert agent history to AgentTrajectory.

        Args:
            latest (bool): Whether only to return the latest trajectory,
                in the shape of (N, D). Defaults to False.

        Returns:
            tuple[AgentTrajectory, list[str]]: Instanced AgentTrajectory and the list of their uuids.
        """
        if latest:
            return self._get_latest_trajectory()

        num_agent = len(self.histories)
        waypoints = np.zeros((num_agent, self.max_length, AgentTrajectory.num_dim))
        label_ids = np.zeros(num_agent, dtype=np.int64)
        uuids: list[str] = []
        for n, (uuid, history) in enumerate(self.histories.items()):
            uuids.append(uuid)
            for t, state in enumerate(history):
                waypoints[n, t] = (
                    *state.xyz,
                    *state.size,
                    state.yaw,
                    *state.vxy,
                    state.is_valid,
                )
                label_ids[n] = state.label_id

        return AgentTrajectory(waypoints, label_ids), uuids

    def _get_latest_trajectory(self) -> AgentTrajectory:
        """Return the latest agent state trajectory.

        Returns:
            AgentTrajectory: Instanced AgentTrajectory.
        """
        num_agent = len(self.histories)
        waypoints = np.zeros((num_agent, AgentTrajectory.num_dim))
        label_ids = np.zeros(num_agent, dtype=np.int64)
        for n, (_, history) in enumerate(self.histories.items()):
            state = history[-1]
            waypoints[n] = (
                *state.xyz,
                *state.size,
                state.yaw,
                *state.vxy,
                state.is_valid,
            )
            label_ids[n] = state.label_id

        return AgentTrajectory(waypoints, label_ids)
