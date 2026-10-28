"""Differentiable heat coefficients and sparse three-term recurrence on [0, 2]."""

import math

import torch


def heat_coefficients(heat_time: torch.Tensor, degree: int = 20) -> torch.Tensor:
    """Approximate exp(-tau (x + 1)) = sum c_k T_k(x), x in [-1, 1].

    A fixed 128-node cosine quadrature is ample for the tested tau in [0.05, 4]
    and degree 20. This is not automatic degree selection or a PSD certificate.
    The constant coefficient already includes the usual factor of one half.
    """
    if degree < 0 or degree >= 128:
        raise ValueError("Degree must be between 0 and 127.")
    angles = (torch.arange(128, dtype=heat_time.dtype, device=heat_time.device) + 0.5) * math.pi / 128
    orders = torch.arange(degree + 1, dtype=heat_time.dtype, device=heat_time.device)
    samples = torch.exp(-heat_time * (1 + angles.cos()))
    coefficients = (2 / 128) * (torch.cos(orders[:, None] * angles) @ samples)
    return torch.cat([coefficients[:1] / 2, coefficients[1:]])


def apply_chebyshev(
    laplacian: torch.Tensor, coefficients: torch.Tensor, rhs: torch.Tensor
) -> torch.Tensor:
    """Apply sum c_k T_k(L - I) to a vector or a matrix of right-hand sides.

    L is fixed, symmetric, sparse COO, with spectrum in [0, 2]. No covariance
    matrix is constructed. Forward workspace is proportional to the RHS size;
    autograd may retain recurrence states for the coefficient derivative.
    """
    vector = rhs.ndim == 1
    if vector:
        rhs = rhs[:, None]
    previous = rhs                                    # T_0(L - I) V
    result = coefficients[0] * previous
    if coefficients.numel() > 1:
        current = torch.sparse.mm(laplacian, rhs) - rhs  # T_1(L - I) V
        result = result + coefficients[1] * current
        for order in range(2, coefficients.numel()):
            following = 2 * (torch.sparse.mm(laplacian, current) - current) - previous
            result = result + coefficients[order] * following
            previous = current
            current = following
    if vector:
        return result[:, 0]
    return result
