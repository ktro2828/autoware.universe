from typing import Sequence

from torch import nn

__all__ = ("build_mlps",)


def _load_norm(name: str) -> nn.Module:
    if name == "BN":
        return nn.BatchNorm1d
    elif name == "LN":
        return nn.LayerNorm
    else:
        raise ValueError(f"Unsupported name: {name}")


def build_mlps(
    c_in: int,
    mlp_channels: int | Sequence[int],
    *,
    ret_before_act: bool = False,
    without_norm: bool = False,
    norm_name: str = "BN",
    inplace: bool = False,
) -> nn.Sequential:
    """
    Return MLP layer.

    Args:
    ----
        c_in (int): Number of input channels.
        mlp_channels (int, Sequence[int]): The Number(s) of hidden channels.
        ret_before_act (bool, optional): _description_. Defaults to False.
        without_norm (bool, optional): _description_. Defaults to False.
        norm_name (str, optional): Name of a normalize layer. Defaults to `"BN"`.
        inplace (bool, optional): Indicates whether to update a given tensor without making a copy.
            Defaults to `False`.

    Returns:
    -------
        nn.Sequential: Return the sequential module.
    """
    if isinstance(mlp_channels, int):
        mlp_channels = [mlp_channels]

    layers = []
    num_layers = len(mlp_channels)

    for k in range(num_layers):
        if k + 1 == num_layers and ret_before_act:
            layers.append(nn.Linear(c_in, mlp_channels[k], bias=True))
        else:
            if without_norm:
                layers.extend(
                    [
                        nn.Linear(c_in, mlp_channels[k], bias=True),
                        nn.ReLU(inplace=inplace),
                    ],
                )
            else:
                norm = _load_norm(norm_name)
                layers.extend(
                    [
                        nn.Linear(c_in, mlp_channels[k], bias=False),
                        norm(mlp_channels[k]),
                        nn.ReLU(inplace=inplace),
                    ],
                )
            c_in = mlp_channels[k]

    return nn.Sequential(*layers)
