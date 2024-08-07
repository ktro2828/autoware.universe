from torch import Tensor
from torch import device
from torch import nn
from typing_extensions import Self

from .decoder import SimplDecoder
from .encoder import SimplEncoder


class Simpl(nn.Module):
    def __init__(self, encoder: dict, decoder: dict) -> None:
        super().__init__()
        self.encoder = SimplEncoder(**encoder)
        self.decoder = SimplDecoder(**decoder)

    def to(self, *args, **kwargs) -> Self:
        """Move and/or the parameters and buffers."""
        self.encoder.to(args, kwargs)
        self.decoder.to(args, kwargs)
        return super().to(args, kwargs)

    def cuda(self, device: int | device | None = None) -> Self:
        """Move all parameters and buffers to the GPU."""
        self.encoder.cuda(device)
        self.decoder.cuda(device)
        return super().cuda(device)

    def forward(
        self,
        actor: Tensor,
        lane: Tensor,
        rpe: Tensor,
        rpe_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Execute forward propagation.

        Args:
            actor (Tensor): Input agent in the shape of (N, Da, T).
            lane (Tensor): Input lane in the shape of (K, P, Dl).
            rpe (Tensor): RPE in the shape of (N+K, N+K, 5).
            rpe_mask (Tensor | None, optional): RPE mask in the shape of (N+K, N+K).
                Defaults to None.

        Returns:
            tuple[Tensor, Tensor]: Predicted score and trajectory,
                in the shape of (N, M) and (N, M, T, 4).
        """
        actor_feature, _ = self.encoder(actor, lane, rpe, rpe_mask)
        return self.decoder(actor_feature)
