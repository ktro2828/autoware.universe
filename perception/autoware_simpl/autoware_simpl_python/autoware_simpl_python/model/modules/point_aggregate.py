import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from .mlp import build_mlps

__all__ = ("PointAggregateBlock",)


class PointAggregateBlock(nn.Module):
    def __init__(self, hidden_size: int, *, agg_out: bool = False) -> None:
        """
        Construct instance.

        Args:
        ----
            hidden_size (int): Hidden layer size.
            agg_out (bool, optional): Whether to aggregate output. Defaults to False.
        """
        super().__init__()
        self.agg_out = agg_out

        self.fc1 = build_mlps(
            hidden_size,
            [hidden_size, hidden_size],
            ret_before_act=False,
            norm_name="LN",
            inplace=True,
        )

        self.fc2 = build_mlps(
            hidden_size * 2,
            [hidden_size, hidden_size],
            ret_before_act=False,
            norm_name="LN",
            inplace=True,
        )

        self.norm = nn.LayerNorm(hidden_size)

    @staticmethod
    def _global_maxpool_aggregate(feat: Tensor) -> Tensor:
        """
        Aggregate global feature by adaptive max-pooling.

        Args:
        ----
            feat (Tensor): Feature tensor, (N, C, L).

        Returns:
        -------
            Tensor: Aggregated tensor (N, 1, C).
        """
        return F.adaptive_max_pool1d(feat.permute(0, 2, 1), 1).permute(0, 2, 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Run forward operation.

        Args:
        ----
            x (Tensor): Input tensor.

        Returns:
        -------
            Tensor: Aggregated tensor.
        """
        x_ret: Tensor = self.fc1(x)  # (K, C, hidden_size)
        x_agg = self._global_maxpool_aggregate(x_ret)
        x_agg = torch.cat([x_ret, x_agg.repeat([1, x_ret.shape[1], 1])], dim=-1)

        out: Tensor = self.norm(x + self.fc2(x_agg))
        return self._global_maxpool_aggregate(out).squeeze() if self.agg_out else out
