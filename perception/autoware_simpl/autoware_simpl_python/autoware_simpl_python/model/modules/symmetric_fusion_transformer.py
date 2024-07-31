import torch
from torch import Tensor
from torch import nn

from .mlp import build_mlps

__all__ = ("SymmetricFusionTransformer",)


class SymmetricFusionTransformer(nn.Module):
    """SymmetricFusionTransformer module used in SIMPL."""

    def __init__(
        self,
        d_model: int = 128,
        d_edge: int = 128,
        n_head: int = 8,
        n_layer: int = 6,
        dropout: float = 0.1,
        *,
        update_edge: bool = True,
    ) -> None:
        """
        Construct instance.

        Args:
        ----
            d_model (int, optional): The number of input feature dimensions. Defaults to 128.
            d_edge (int, optional): The number of output feature dimensions. Defaults to 128.
            n_head (int, optional): The number of heads. Defaults to 8.
            n_layer (int, optional): The number of fusion layers. Defaults to 6.
            dropout (float, optional): Dropout ratio. Defaults to 0.1.
            update_edge (bool, optional): Indicates whether to update edge feature. Defaults to True.
        """
        super().__init__()

        fusion = []
        for i in range(n_layer):
            need_update_edge = False if i == n_layer - 1 else update_edge
            fusion.append(
                SftLayer(
                    d_edge=d_edge,
                    d_model=d_model,
                    d_ffn=d_model * 2,
                    n_head=n_head,
                    dropout=dropout,
                    update_edge=need_update_edge,
                ),
            )
        self.fusion = nn.ModuleList(fusion)

    def forward(self, x: Tensor, edge: Tensor, edge_mask: Tensor | None) -> Tensor:
        """
        Run forward operation.

        Args:
        ----
            x (Tensor): Concatenated agent and lane feature, in shape (N+K, d_model).
            edge (Tensor): RPE tensor, in shape (N+K, N+K, d_model)
            edge_mask (Tensor | None): RPE mask, in shape (N+K, N+K).

        Returns:
        -------
            Tensor:
        """
        for mod in self.fusion:
            x, edge = mod(x, edge, edge_mask)
        return x


class SftLayer(nn.Module):
    """A module."""

    def __init__(
        self,
        d_edge: int = 128,
        d_model: int = 128,
        d_ffn: int = 2048,
        n_head: int = 8,
        dropout: float = 0.1,
        *,
        update_edge: bool = True,
    ) -> None:
        """
        Construct instance.

        Args:
        ----
            d_edge (int, optional): The number of edge feature dimensions. Defaults to 128.
            d_model (int, optional): The number of . Defaults to 128.
            d_ffn (int, optional): _description_. Defaults to 2048.
            n_head (int, optional): _description_. Defaults to 8.
            dropout (float, optional): _description_. Defaults to 0.1.
            update_edge (bool, optional): Indicates whether to update edge feature. Defaults to True.
        """
        super().__init__()
        self.update_edge = update_edge

        self.proj_memory = build_mlps(
            d_model + d_model + d_edge, d_model, norm_name="LN", inplace=True
        )

        if self.update_edge:
            self.proj_edge = build_mlps(d_model, d_edge, norm_name="LN", inplace=True)
            self.norm_edge = nn.LayerNorm(d_edge)

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_head,
            dropout=dropout,
            batch_first=False,
        )

        # Feedforward model
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

    def forward(
        self, node: Tensor, edge: Tensor, edge_mask: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        """
        Run forward operation.

        Args:
        ----
            node (Tensor): (N, d_model)
            edge (Tensor): (d_model, N, N)
            edge_mask (Tensor | None): (N, N).

        Returns:
        -------
            tuple[Tensor, Tensor]:
        """
        x, edge, memory = self._build_memory(node, edge)
        x_prime = self._mha_block(x, memory, attn_mask=None, key_padding_mask=edge_mask)
        x = self.norm2(x + x_prime).squeeze()
        x = self.norm3(x + self._ff_block(x))
        return x, edge

    def _build_memory(self, node: Tensor, edge: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Build memory.

        Args:
        ----
            node (Tensor): (N, d_model)
            edge (Tensor): (d_model, N, N).

        Returns:
        -------
            tuple[Tensor, Tensor, Tensor]:
        """
        n_token = node.shape[0]

        # 1. build memory
        src_x = node.unsqueeze(0).repeat(n_token, 1, 1)  # (N, N, d_model)
        tar_x = node.unsqueeze(1).repeat(1, n_token, 1)  # (N, N, d_model)
        memory = self.proj_memory(torch.cat((edge, src_x, tar_x), dim=-1))  # (N, N, d_model)
        # 2. (optional) update edge (with residual)
        if self.update_edge:
            edge = self.norm_edge(edge + self.proj_edge(memory))  # (N, N, d_edge)

        return node.unsqueeze(0), edge, memory

    def _mha_block(
        self, x: Tensor, mem: Tensor, attn_mask: Tensor, key_padding_mask: Tensor | None
    ) -> Tensor:
        """
        Run multi-head attention operation.

        Args:
        ----
            x (Tensor): Query tensor.
            mem (Tensor): Key and value tensor.
            attn_mask (Tensor): Attention mask.
            key_padding_mask (Tensor | None): Key padding mask.

        Returns:
        -------
            Tensor: Attention output.
        """
        x, _ = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        return self.dropout2(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        """
        Run feed-forward operation.

        Args:
        ----
            x (Tensor): Input tensor.

        Returns:
        -------
            Tensor: Output tensor.
        """
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)
