from math import gcd

from torch import Tensor
from torch import nn

__all__ = ("Conv1d", "Res1d")


class Conv1d(nn.Module):
    """A module extending `nn.Conv1d`, which contains a normalization layer and `ReLU` activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        norm: str = "GN",
        ng: int = 32,
        *,
        act: bool = True,
    ) -> None:
        """
        Construct instance.

        Args:
        ----
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            kernel_size (int, optional): The size of kernel. Defaults to 3.
            stride (int, optional): The size of stride. Defaults to 1.
            norm (str, optional): Name of a normalization layer. Defaults to "GN".
            ng (int, optional): The number of groups, which is only used for `"GN"`. Defaults to 32.
            act (bool, optional): Whether to apply activation before output. Defaults to True.
        """
        super().__init__()
        assert norm in ("GN", "BN", "SyncBN"), f"Unexpected norm layer: {norm}"

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            stride=stride,
            bias=False,
        )

        if norm == "GN":
            self.norm = nn.GroupNorm(gcd(ng, out_channels), out_channels)
        elif norm == "BN":
            self.norm = nn.BatchNorm1d(out_channels)
        else:
            self.norm = nn.SyncBatchNorm(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x: Tensor) -> Tensor:
        """
        Run forward operation.

        Args:
        ----
            x (Tensor): Input tensor.

        Returns:
        -------
            Tensor: Output tensor.
        """
        out = self.conv(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out


class Res1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        norm: str = "GN",
        ng: int = 32,
        *,
        act: bool = True,
    ) -> None:
        """
        Construct instance.

        Args:
        ----
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            kernel_size (int, optional): The size of kernel. Defaults to 3.
            stride (int, optional): The size of stride. Defaults to 1.
            norm (str, optional): Name of a normalization layer. Defaults to "GN".
            ng (int, optional): The number of normalization groups. Defaults to 32.
            act (bool, optional): Indicates whether to run activation before output. Defaults to True.
        """
        super().__init__()
        assert norm in ("GN", "BN", "SyncBN"), f"Unexpected norm layer: {norm}"

        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.relu = nn.ReLU(inplace=True)

        # All use name bn1 and bn2 to load ImageNet pretrained weights
        if norm == "GN":
            self.bn1 = nn.GroupNorm(gcd(ng, out_channels), out_channels)
            self.bn2 = nn.GroupNorm(gcd(ng, out_channels), out_channels)
        elif norm == "BN":
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.bn2 = nn.BatchNorm1d(out_channels)
        else:
            self.bn1 = nn.SyncBatchNorm(out_channels)
            self.bn2 = nn.SyncBatchNorm(out_channels)

        if stride != 1 or out_channels != in_channels:
            if norm == "GN":
                self.downsample = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.GroupNorm(gcd(ng, out_channels), out_channels),
                )
            elif norm == "BN":
                self.downsample = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm1d(out_channels),
                )
            else:
                msg = "SyncBN has not been added!"
                raise ValueError(msg)
        else:
            self.downsample = None

        self.act = act

    def forward(self, x: Tensor) -> Tensor:
        """
        Run forward operation.

        Args:
        ----
            x (Tensor): Input tensor.

        Returns:
        -------
            Tensor: Output tensor.
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        if self.act:
            out = self.relu(out)
        return out
