import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from .modules import Conv1d
from .modules import PointAggregateBlock
from .modules import Res1d
from .modules import SymmetricFusionTransformer
from .modules import build_mlps


class SimplEncoder(nn.Module):
    """A encoder module of SIMPL."""

    def __init__(self, actor_net: dict, lane_net: dict, fusion_net: dict) -> None:
        """
        Construct instance.

        Args:
        ----
            actor_net (dict): `SimplActorNet` configuration.
            lane_net (dict): `SimplLaneNet` configuration.
            fusion_net (dict): `SimplFusionNet` configuration.
        """
        super().__init__()
        self.actor_net = SimplActorNet(**actor_net)
        self.lane_net = SimplLaneNet(**lane_net)
        self.fusion_net = SimplFusionNet(**fusion_net)

    def forward(
        self,
        actors: Tensor,
        lanes: Tensor,
        rpe: Tensor,
        rpe_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Run forward operation.

        TODO: Input actor shape must be 16 * n.

        Args:
        ----
            actors (Tensor): Actor tensor in shape (N, Da, T).
            lanes (Tensor): Lane tensor in shape (K, P-1, Dl).
            rpe (Tensor): RPE tensor (N+K, N+K, Dr).
            rpe_mask (Tensor | None, optional): RPE mask tensor in shape (N+K, N+K). Defaults to None.

        Returns:
        -------
            tuple[Tensor, Tensor]: Actor and lane features, in shape (N, F) and (K, F)
        """
        actor_feat = self.actor_net(actors)  # (N, d_model)
        lane_feat = self.lane_net(lanes)

        ret_actor, ret_lane = self.fusion_net(actor_feat, lane_feat, rpe, rpe_mask)

        return ret_actor, ret_lane


class SimplActorNet(nn.Module):
    """Actor feature extractor used in SIMPL."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_size: int = 128,
        fpn_size: int = 4,
    ) -> None:
        """
        Construct instance.

        Args:
        ----
            in_channels (int, optional): Input channel size. Defaults to 3.
            hidden_size (int, optional): Hidden size. Defaults to 128.
            fpn_size (int, optional): The number of FPN layers. Defaults to 4.
        """
        super().__init__()

        if fpn_size <= 0:
            msg = f"Expected FPN size >= 1, but got {fpn_size}"
            raise ValueError(msg)

        self.fpn_size = fpn_size

        n_out: list[int] = [2 ** (5 + s) for s in range(fpn_size)]  # [32, 64, 128]
        num_blocks = [2] * fpn_size

        groups = []
        for i in range(fpn_size):
            group = [
                (
                    Res1d(in_channels, n_out[i], norm="GN", ng=1)
                    if i == 0
                    else Res1d(in_channels, n_out[i], stride=2, norm="GN", ng=1)
                ),
            ]
            for _ in range(1, num_blocks[i]):
                group.append(Res1d(n_out[i], n_out[i], norm="GN", ng=1))
            groups.append(nn.Sequential(*group))
            in_channels = n_out[i]

        self.groups = nn.ModuleList(groups)
        self.lateral = nn.ModuleList(
            [Conv1d(out_ch, hidden_size, norm="GN", ng=1, act=False) for out_ch in n_out],
        )
        self.output = Res1d(hidden_size, hidden_size, norm="GN", ng=1)

    def forward(self, actors: Tensor) -> Tensor:
        """
        Run forward operation.

        Args:
        ----
            actors (Tensor): Input actor tensor.

        Returns:
        -------
            Tensor: Actor feature.
        """
        out = actors

        outputs = []
        for group in self.groups:
            out = group(out)
            outputs.append(out)

        out = self.lateral[-1](outputs[-1])
        for i in range(self.fpn_size - 2, -1, -1):
            out = F.interpolate(out, scale_factor=2, mode="linear", align_corners=False)
            out += self.lateral[i](outputs[i])

        return self.output(out)[:, :, -1]


class SimplLaneNet(nn.Module):
    """Lane feature extractor used in SIMPL."""

    def __init__(self, in_channels: int = 10, hidden_size: int = 128) -> None:
        """
        Construct instance.

        Args:
        ----
            in_channels (int, optional): The number of input channels. Defaults to 10.
            hidden_size (int, optional): The number of hidden size. Defaults to 128.
        """
        super().__init__()

        self.proj = build_mlps(in_channels, hidden_size, norm_name="LN", inplace=True)
        self.aggre1 = PointAggregateBlock(hidden_size, agg_out=False)
        self.aggre2 = PointAggregateBlock(hidden_size, agg_out=True)

    def forward(self, feats: Tensor) -> Tensor:
        """
        Run forward operation.

        Args:
        ----
            feats (Tensor): Lane feature tensor.

        Returns:
        -------
            Tensor: Aggregated lane feature.
        """
        x = self.proj(feats)  # [N_{lane}, 10, hidden_size]
        x = self.aggre1(x)
        return self.aggre2(x)  # [N_{lane}, hidden_size]


class SimplFusionNet(nn.Module):
    """Fusion module used in SIMPL."""

    def __init__(
        self,
        d_actor: int,
        d_lane: int,
        d_rpe_in: int,
        d_rpe: int,
        d_embed: int,
        num_scene_head: int,
        num_scene_layer: int,
        dropout: float,
        *,
        update_edge: bool = True,
    ) -> None:
        """
        Construct instance.

        Args:
        ----
            d_actor (int): The number of actor layer dimensions.
            d_lane (int): The number of lane layer dimensions.
            d_rpe_in (int): The number of RPE input channels.
            d_rpe (int): The number of RPE hidden layer dimensions.
            d_embed (int): The number of embed layer dimensions.
            num_scene_head (int): The number of scene heads.
            num_scene_layer (int): The number of scene layers.
            dropout (float): Dropout ratio.
            update_edge (bool, optional): Indicates whether to update edge feature.
                Defaults to True.
        """
        super().__init__()

        self.proj_actor = build_mlps(d_actor, d_embed, norm_name="LN", inplace=True)
        self.proj_lane = build_mlps(d_lane, d_embed, norm_name="LN", inplace=True)
        self.proj_rpe_scene = build_mlps(d_rpe_in, d_rpe, norm_name="LN", inplace=True)

        self.fuse_scene = SymmetricFusionTransformer(
            d_model=d_embed,
            d_edge=d_rpe,
            n_head=num_scene_head,
            n_layer=num_scene_layer,
            dropout=dropout,
            update_edge=update_edge,
        )

    def forward(
        self,
        actors: Tensor,
        lanes: Tensor,
        rpe: Tensor,
        rpe_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Run forward operation.

        Args:
        ----
            actors (Tensor): Actor tensor.
            lanes (Tensor): Lane tensor.
            rpe (Tensor): RPE tensor in shape (N+K, N+K, Dr).
            rpe_mask (Tensor | None, optional): RPE mask tensor in shape (N+K, N+K). Defaults to None.

        Returns:
        -------
            tuple[Tensor, Tensor]: Fused actor and lane features.
        """
        actor_token = self.proj_actor(actors)
        lane_token = self.proj_lane(lanes)

        tokens = torch.cat((actor_token, lane_token), dim=0)  # (N+K, d_model)
        rpe_proj = self.proj_rpe_scene(rpe)
        fusion = self.fuse_scene(tokens, rpe_proj, rpe_mask)

        num_actor = actors.size(0)
        ret_actor = fusion[:num_actor]
        ret_lane = fusion[num_actor:]

        return ret_actor, ret_lane
