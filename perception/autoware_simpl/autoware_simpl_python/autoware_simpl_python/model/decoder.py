import math

import torch
from torch import Tensor
from torch import device
from torch import nn
import torch.nn.functional as F
from typing_extensions import Self

from .modules import build_mlps


class SimplDecoder(nn.Module):
    """Decoder module used in SIMPL."""

    def __init__(
        self,
        in_channels: int = 128,
        n_order: int = 7,
        num_mode: int = 6,
        num_future: int = 60,
        pred_format: str | None = "beizer",
    ) -> None:
        """
        Construct instance.

        Args:
        ----
            in_channels (int): The number of input channels.
            n_order (int): The number of output control points.
            num_future_frames (int): The number of future frames.
            num_motion_modes (int): The number of modes.
            pred_format (str | None, optional): Name of output format. Defaults to None.
            decode_loss (dict | None, optional): Configuration of loss function. Defaults to None.

        """
        super().__init__()
        self.in_channels = in_channels
        self.num_mode = num_mode
        self.num_future = num_future
        self.pred_format = pred_format
        self.n_order = n_order

        dim_mm = self.in_channels * self.num_mode
        dim_inter = dim_mm // 2

        self.multihead_proj = build_mlps(
            self.in_channels, [dim_inter, dim_mm], norm_name="LN", inplace=True
        )

        self.cls = build_mlps(
            self.in_channels,
            [self.in_channels, self.in_channels, 1],
            ret_before_act=True,
            norm_name="LN",
            inplace=True,
        )

        if self.pred_format is None:
            self.reg = build_mlps(
                self.in_channels,
                [self.in_channels, self.in_channels, self.num_future * 2],
                ret_before_act=True,
                norm_name="LN",
                inplace=True,
            )
        elif self.pred_format == "bezier":
            self.mat_T, self.mat_Tp = self._get_matrix_bezier(
                n_order=self.n_order, n_step=self.num_future
            )

            self.reg = build_mlps(
                self.in_channels,
                [self.in_channels, self.in_channels, (self.n_order + 1) * 2],
                ret_before_act=True,
                norm_name="LN",
                inplace=True,
            )
        elif self.pred_format == "monomial":
            self.mat_T, self.mat_Tp = self._get_matrix_monomial(
                n_order=self.n_order, n_step=self.num_future
            )

            self.reg = build_mlps(
                self.in_channels,
                [self.in_channels, self.in_channels, (self.n_order + 1) * 2],
                ret_before_act=True,
                norm_name="LN",
                inplace=True,
            )
        else:
            raise ValueError(f"Unexpected output format: {self.pred_format}")

    @staticmethod
    def _get_matrix_bezier(n_order: int, n_step: int) -> tuple[Tensor, Tensor]:
        """
        Return a matrix to transform bezier control position and velocity points.

        Args:
        ----
            n_order (int): The number of points.
            n_step (int): The number of point steps.

        Returns:
        -------
            tuple[Tensor, Tensor]: Transform matrices, in shape `(n_step, n_order+1)` and `(n_step, n_order)`.
        """
        ts = torch.linspace(0.0, 1.0, n_step)
        mat_t = torch.stack(
            [
                math.comb(n_order, i) * (1.0 - ts) ** (n_order - i) * ts**i
                for i in range(n_order + 1)
            ],
            dim=0,
        ).T

        mat_tp = torch.stack(
            [
                n_order * math.comb(n_order - 1, i) * (1.0 - ts) ** (n_order - 1 - i) * ts**i
                for i in range(n_order)
            ],
            dim=0,
        ).T

        return mat_t, mat_tp

    @staticmethod
    def _get_matrix_monomial(n_order: int, n_step: int) -> tuple[Tensor, Tensor]:
        """
        Return a matrix to transform monomial control position points.

        Args:
        ----
            n_order (int): The number of points.
            n_step (int): The number of point steps.

        Returns:
        -------
            tuple[Tensor, Tensor]: Transform matrices, in shape `(n_step, n_order+1)`, `(n_step, n_order)`.
        """
        ts = torch.linspace(0.0, 1.0, n_step)
        mat_t = torch.stack([ts**i for i in range(n_order + 1)], dim=0).T

        mat_tp = torch.stack([(i + 1) * (ts**i) for i in range(n_order)], dim=0).T

        return mat_t, mat_tp

    def to(self, *args, **kwargs) -> Self:
        if self.pred_format:
            self.mat_T = self.mat_T.to(args, kwargs)
            self.mat_Tp = self.mat_Tp.to(args, kwargs)
        return super().to(args, kwargs)

    def cuda(self, device: int | device | None = None) -> Self:
        if self.pred_format:
            self.mat_T = self.mat_T.cuda(device)
            self.mat_Tp = self.mat_Tp.cuda(device)
        return super().cuda(device)

    def forward(self, embed: Tensor) -> tuple[Tensor, Tensor]:
        """
        Run forward operation.

        Args:
        ----
            embed (Tensor): Embedded agent feature in shape (N,).

        Returns:
        -------
            tuple[Tensor, Tensor]: Return scores, positions and velocities tensors.
        """
        embed = (
            self.multihead_proj(embed).view(-1, self.num_mode, self.in_channels).permute(1, 0, 2)
        )

        pred_cls = self.cls(embed).view(self.num_mode, -1).permute(1, 0)
        pred_cls = F.softmax(pred_cls * 1.0, dim=1)

        param: Tensor
        pred_reg: Tensor
        pred_vel: Tensor
        if self.pred_format == "bezier":
            param = self.reg(embed).view(self.num_mode, -1, self.n_order + 1, 2)
            param = param.permute(1, 0, 2, 3)
            pred_reg = torch.matmul(self.mat_T, param)
            pred_vel = torch.matmul(self.mat_Tp, param[:, :, 1:] - param[:, :, :-1]) / (
                self.num_future * 0.1
            )
        elif self.pred_format == "monomial":
            param = self.reg(embed).view(self.num_mode, -1, self.n_order + 1, 2)
            param = param.permute(1, 0, 2, 3)
            pred_reg = torch.matmul(self.mat_T, param)
            pred_vel = torch.matmul(self.mat_Tp, param[:, :, 1:, :]) / (self.num_future * 0.1)
        else:
            pred_reg = self.reg(embed).view(self.num_mode, -1, self.num_future, 2)
            pred_reg = pred_reg.permute(1, 0, 2, 3)
            pred_vel = torch.gradient(pred_reg, dim=-2)[0] / 0.1

        return self.get_prediction(pred_cls, pred_reg, pred_vel)

    def get_prediction(
        self, pred_cls: Tensor, pred_reg: Tensor, pred_vel: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Return predicted scores and trajectory.

        Args:
        ----
            pred_cls (list[Tensor]): Predicted scores.
            pred_reg (list[Tensor]): Predicted positions.
            pred_vel (list[Tensor]): Predicted velocities.

        Returns:
        -------
            tuple[Tensor, Tensor]: Predicted scores and trajectory in shape (N, M) and (N, M, T, 4).
        """
        pred_traj = torch.cat([pred_reg, pred_vel], dim=-1)
        return pred_cls, pred_traj
