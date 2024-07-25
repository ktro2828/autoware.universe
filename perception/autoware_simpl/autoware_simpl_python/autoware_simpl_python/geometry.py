from numbers import Number

import numpy as np
from numpy.typing import NDArray


def rotate_along_z(points: NDArray, angle: Number | NDArray) -> NDArray:
    """Rotate points array along z axis.

    Args:
        points (NDArray): Matrix of coordinates.
        angle (Number | NDArray): Scalar or matrix of angle in [rad].

    Returns:
        NDArray: Rotated points array.
    """
    if not isinstance(angle, np.ndarray):
        angle = np.array(angle)

    cos = np.cos(angle)
    sin = np.sin(angle)

    dim: int = points.shape[-1]
    if dim == 2:
        rotation = np.stack([cos, -sin, sin, cos], axis=1).reshape(-1, 2, 2)
    elif dim == 3:
        zeros = np.zeros_like(angle)
        ones = np.ones_like(angle)

        rotation = np.stack(
            [cos, -sin, zeros, sin, cos, zeros, zeros, zeros, ones],
            axis=1,
        ).reshape(-1, 3, 3)
    else:
        raise ValueError(f"Unexpected point dimension: {dim}")

    return np.matmul(points, rotation)
