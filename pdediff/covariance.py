import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor

LinearSolver = Literal["bicgstab", "cg"]


def gaspari_cohn(r: Tensor) -> Tensor:
    r = torch.abs(r)
    w = torch.zeros_like(r)

    m1 = r <= 1
    x = r[m1]
    w[m1] = 1 - (5/3)*x**2 + (5/8)*x**3 + 0.5*x**4 - 0.25*x**5

    m2 = (r > 1) & (r <= 2)
    x = r[m2]
    w[m2] = (
        4 - 5*x + (5/3)*x**2 + (5/8)*x**3
        - 0.5*x**4 + (1/12)*x**5 - 2/(3*x)
    )
    return w


def localize_cov(cov: Tensor, radius: float) -> Tensor:
    n = cov.shape[0]
    idx = torch.arange(n, device=cov.device)
    d = (idx[:, None] - idx[None, :]).abs()

    taper = gaspari_cohn(d / radius)
    return cov * taper


def _safe_divisor(value: Tensor) -> Tensor:
    tiny = torch.finfo(value.dtype).tiny
    return torch.where(value.abs() < tiny, torch.full_like(value, tiny), value)


def conjugate_gradient(
    matvec: Callable[[Tensor], Tensor],
    rhs: Tensor,
    max_iter: int = 3,
    tol: float = 1e-3,
) -> Tensor:
    solution = torch.zeros_like(rhs)
    residual = rhs - matvec(solution)
    direction = residual.clone()
    residual_norm_sq = (residual * residual).sum()
    rhs_norm = rhs.square().sum().sqrt().clamp_min(torch.finfo(rhs.dtype).tiny)

    for _ in range(max_iter):
        matvec_direction = matvec(direction)
        step_size = residual_norm_sq / _safe_divisor((direction * matvec_direction).sum())
        solution = solution + step_size * direction
        residual = residual - step_size * matvec_direction
        if residual.square().sum().sqrt() / rhs_norm < tol:
            break

        next_residual_norm_sq = (residual * residual).sum()
        direction = residual + (next_residual_norm_sq / _safe_divisor(residual_norm_sq)) * direction
        residual_norm_sq = next_residual_norm_sq

    return solution


def bicgstab(
    matvec: Callable[[Tensor], Tensor],
    rhs: Tensor,
    max_iter: int = 20,
    tol: float = 1e-8,
) -> Tensor:
    solution = torch.zeros_like(rhs)
    residual = rhs - matvec(solution)
    shadow_residual = residual.clone()
    direction = torch.zeros_like(rhs)
    matvec_direction = torch.zeros_like(rhs)
    previous_rho = torch.ones((), dtype=rhs.dtype, device=rhs.device)
    step_size = torch.ones((), dtype=rhs.dtype, device=rhs.device)
    smoothing = torch.ones((), dtype=rhs.dtype, device=rhs.device)
    rhs_norm = rhs.square().sum().sqrt().clamp_min(torch.finfo(rhs.dtype).tiny)

    for _ in range(max_iter):
        rho = (shadow_residual * residual).sum()
        beta = (rho / _safe_divisor(previous_rho)) * (step_size / _safe_divisor(smoothing))
        direction = residual + beta * (direction - smoothing * matvec_direction)
        matvec_direction = matvec(direction)
        step_size = rho / _safe_divisor((shadow_residual * matvec_direction).sum())
        intermediate_residual = residual - step_size * matvec_direction
        if intermediate_residual.square().sum().sqrt() / rhs_norm < tol:
            solution = solution + step_size * direction
            break

        matvec_intermediate = matvec(intermediate_residual)
        smoothing = (matvec_intermediate * intermediate_residual).sum() / _safe_divisor(
            (matvec_intermediate * matvec_intermediate).sum()
        )
        solution = solution + step_size * direction + smoothing * intermediate_residual
        residual = intermediate_residual - smoothing * matvec_intermediate
        if residual.square().sum().sqrt() / rhs_norm < tol:
            break
        previous_rho = rho

    return solution


def solve_linear_system(
    matvec: Callable[[Tensor], Tensor],
    rhs: Tensor,
    solver: LinearSolver = "cg",
    max_iter: int = 20,
    tol: float = 1e-8,
) -> Tensor:
    if solver == "bicgstab":
        return bicgstab(matvec, rhs, max_iter=max_iter, tol=tol)
    return conjugate_gradient(matvec, rhs, max_iter=max_iter, tol=tol)


def make_gc_kernel(
    sizes: Sequence[int],
    bandwidths: Sequence[Optional[float]],
    active_axes: Sequence[int],
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    radial: bool = False,
) -> Tensor:
    offsets = []
    for axis in active_axes:
        bandwidth = float("inf") if bandwidths[axis] is None else float(bandwidths[axis])
        if bandwidth <= 0:
            radius = 0
        elif math.isinf(bandwidth):
            radius = sizes[axis] - 1
        else:
            radius = min(math.ceil(2 * bandwidth), sizes[axis] - 1)

        offsets.append(torch.arange(-radius, radius + 1, device=device, dtype=dtype))

    if not offsets:
        return torch.ones((), device=device, dtype=dtype)

    if radial:
        grids = torch.meshgrid(*offsets, indexing="ij")
        q2 = torch.zeros_like(grids[0])
        for grid, axis in zip(grids, active_axes):
            bandwidth = float("inf") if bandwidths[axis] is None else float(bandwidths[axis])
            if bandwidth > 0:
                q2 = q2 + (grid / bandwidth) ** 2
            else:
                q2 = q2 + torch.where(grid == 0, 0.0, float("inf"))
        return gaspari_cohn(torch.sqrt(q2))

    kernel = torch.ones(
        tuple(offset.numel() for offset in offsets),
        device=device,
        dtype=dtype,
    )
    for dim, (axis_offsets, axis) in enumerate(zip(offsets, active_axes)):
        bandwidth = float("inf") if bandwidths[axis] is None else float(bandwidths[axis])
        if bandwidth > 0:
            weights = gaspari_cohn(axis_offsets / bandwidth)
        else:
            weights = (axis_offsets == 0).to(dtype=dtype)

        shape = [1] * len(active_axes)
        shape[dim] = weights.numel()
        kernel = kernel * weights.reshape(shape)

    return kernel


def _pad_for_conv(x: Tensor, kernel_shape: Sequence[int], mode: str = "constant") -> Tensor:
    pads = []
    for kernel_size in reversed(kernel_shape):
        pad = kernel_size // 2
        pads.extend([pad, pad])

    if mode == "constant":
        return F.pad(x, pads, mode="constant", value=0.0)
    return F.pad(x, pads, mode=mode)


def _convolve_active_axes(
    values: Tensor,
    kernel: Tensor,
    active_axes: Sequence[int],
    event_ndim: int,
    boundary: str,
) -> Tensor:
    if not active_axes:
        return values

    leading_ndim = values.ndim - event_ndim
    inactive_axes = [axis for axis in range(event_ndim) if axis not in active_axes]
    permutation = (
        list(range(leading_ndim))
        + [leading_ndim + axis for axis in inactive_axes]
        + [leading_ndim + axis for axis in active_axes]
    )
    work = values.permute(permutation)
    inactive_shape = work.shape[: work.ndim - len(active_axes)]
    active_shape = work.shape[work.ndim - len(active_axes):]
    work = work.reshape(-1, 1, *active_shape)
    weight = kernel.reshape(1, 1, *kernel.shape)
    work = _pad_for_conv(work, kernel.shape, mode=boundary)

    if len(active_axes) == 1:
        smoothed = F.conv1d(work, weight).squeeze(1)
    elif len(active_axes) == 2:
        smoothed = F.conv2d(work, weight).squeeze(1)
    elif len(active_axes) == 3:
        smoothed = F.conv3d(work, weight).squeeze(1)
    else:
        raise ValueError("localized convolution supports at most three active axes")

    smoothed = smoothed.reshape(*inactive_shape, *smoothed.shape[-len(active_axes):])
    inverse_permutation = [0] * len(permutation)
    for dim, permuted_dim in enumerate(permutation):
        inverse_permutation[permuted_dim] = dim
    return smoothed.permute(inverse_permutation)


def _apply_localization(
    values: Tensor,
    event_shape: Sequence[int],
    bandwidths: Sequence[Optional[float]],
    boundary: str,
    radial: bool,
) -> Tensor:
    event_ndim = len(event_shape)
    leading_ndim = values.ndim - event_ndim
    unbounded_axes = [
        axis
        for axis, size in enumerate(event_shape)
        if size > 1 and bandwidths[axis] is None
    ]
    active_axes = [
        axis
        for axis, size in enumerate(event_shape)
        if size > 1 and bandwidths[axis] is not None and float(bandwidths[axis]) > 0
    ]

    localized = values
    for axis in unbounded_axes:
        localized = localized.sum(dim=leading_ndim + axis, keepdim=True)

    if active_axes:
        kernel = make_gc_kernel(
            event_shape,
            bandwidths,
            active_axes,
            device=values.device,
            dtype=values.dtype,
            radial=radial,
        )
        localized = _convolve_active_axes(
            localized,
            kernel,
            active_axes,
            event_ndim,
            boundary,
        )

    if unbounded_axes:
        expanded_shape = list(localized.shape)
        for axis in unbounded_axes:
            expanded_shape[leading_ndim + axis] = event_shape[axis]
        localized = localized.expand(expanded_shape)

    return localized


def localized_correlation_apply(
    v: Tensor,
    bandwidths: Sequence[Optional[float]],
    *,
    boundary: str = "constant",
    radial: bool = False,
) -> Tensor:
    event_ndim = len(bandwidths)
    sizes = tuple(v.shape[-event_ndim:])
    return _apply_localization(
        v,
        sizes,
        bandwidths,
        boundary,
        radial,
    )


def localized_covariance_apply(
    v: Tensor,
    anomalies: Tensor,
    bandwidths: Sequence[Optional[float]],
    *,
    nugget: Union[float, Tensor] = 0.0,
    boundary: str = "constant",
    radial: bool = False,
) -> Tensor:
    event_shape = tuple(anomalies.shape[1:])
    batched = v.ndim == len(event_shape) + 1
    if not batched:
        v = v.unsqueeze(0)

    z = anomalies.unsqueeze(0) * v.unsqueeze(1)
    smoothed = _apply_localization(z, event_shape, bandwidths, boundary, radial)

    if anomalies.shape[0] > 1:
        pv = (anomalies.unsqueeze(0) * smoothed).sum(dim=1) / (anomalies.shape[0] - 1)
    else:
        pv = torch.zeros_like(v)

    if isinstance(nugget, Tensor) or nugget:
        pv = pv + nugget * v

    return pv if batched else pv.squeeze(0)


@dataclass
class LocalizedEmpiricalCovariance:
    anomalies: Tensor
    bandwidths: tuple[Optional[float], ...]
    nugget: float = 0.0
    hybrid_ratio: float = 0.0
    static_std: float = 1.0
    boundary: str = "constant"
    radial: bool = False
    inverse: Optional[Tensor] = None

    @property
    def shape(self) -> torch.Size:
        event_size = math.prod(self.anomalies.shape[1:])
        return torch.Size((event_size, event_size))

    def detach(self) -> "LocalizedEmpiricalCovariance":
        return LocalizedEmpiricalCovariance(
            anomalies=self.anomalies.detach(),
            bandwidths=self.bandwidths,
            nugget=self.nugget,
            hybrid_ratio=self.hybrid_ratio,
            static_std=self.static_std,
            boundary=self.boundary,
            radial=self.radial,
            inverse=self.inverse.detach() if isinstance(self.inverse, Tensor) else self.inverse,
        )

    def to(self, *args, **kwargs) -> "LocalizedEmpiricalCovariance":
        return LocalizedEmpiricalCovariance(
            anomalies=self.anomalies.to(*args, **kwargs),
            bandwidths=self.bandwidths,
            nugget=self.nugget,
            hybrid_ratio=self.hybrid_ratio,
            static_std=self.static_std,
            boundary=self.boundary,
            radial=self.radial,
            inverse=self.inverse.to(*args, **kwargs) if isinstance(self.inverse, Tensor) else self.inverse,
        )

    def with_inverse(self) -> "LocalizedEmpiricalCovariance":
        return LocalizedEmpiricalCovariance(
            anomalies=self.anomalies,
            bandwidths=self.bandwidths,
            nugget=self.nugget,
            hybrid_ratio=self.hybrid_ratio,
            static_std=self.static_std,
            boundary=self.boundary,
            radial=self.radial,
            inverse=torch.linalg.inv(self.to_dense()),
        )

    def to_dense(self) -> Tensor:
        event_shape = self.anomalies.shape[1:]
        event_size = math.prod(event_shape)
        basis = torch.eye(event_size, dtype=self.anomalies.dtype, device=self.anomalies.device)
        return self.matvec(basis, torch.Size((event_size, *event_shape)))

    def matvec(self, vector: Tensor, observation_shape: torch.Size) -> Tensor:
        vector_shape = vector.shape
        vector = vector.reshape(-1, *observation_shape[1:])
        empirical = localized_covariance_apply(
            vector,
            self.anomalies.to(device=vector.device, dtype=vector.dtype),
            self.bandwidths,
            boundary=self.boundary,
            radial=self.radial,
        )
        if self.hybrid_ratio:
            static = localized_correlation_apply(
                vector,
                self.bandwidths,
                boundary=self.boundary,
                radial=self.radial,
            ) * (self.static_std ** 2)
            result = (1 - self.hybrid_ratio) * empirical + self.hybrid_ratio * static
        else:
            result = empirical

        if self.nugget:
            result = result + self.nugget * vector
        return result.reshape(vector_shape)


def empirical_localized_covariance_operator(
    samples: Tensor,
    bandwidth: Optional[int] = None,
    temporal_band: Optional[int] = 0,
    min_std: float = 1e-3,
    hybrid_ratio: float = 0.0,
    static_std: float = 1.0,
    boundary: str = "constant",
    radial: bool = False,
    compute_inverse: bool = True,
) -> LocalizedEmpiricalCovariance:
    # samples: (ensemble, time, channels, *spatial_shape). Channels are
    # correlated globally; localization bandwidth applies only to space.
    event_shape = samples.shape[1:]
    if bandwidth is None:
        bandwidth = max((dim_size // 2 for dim_size in event_shape[2:]), default=0)
    bandwidths = (
        temporal_band,
        *((None,) if len(event_shape) > 1 else ()),
        *((bandwidth,) * max(len(event_shape) - 2, 0)),
    )
    anomalies = samples - samples.mean(dim=0, keepdim=True)
    covariance = LocalizedEmpiricalCovariance(
        anomalies=anomalies,
        bandwidths=bandwidths,
        nugget=min_std ** 2,
        hybrid_ratio=hybrid_ratio,
        static_std=static_std,
        boundary=boundary,
        radial=radial,
    )
    return covariance.with_inverse() if compute_inverse else covariance


def covariance_matvec(
    covariance: Union[Tensor, LocalizedEmpiricalCovariance],
    vector: Tensor,
    observation_shape: torch.Size,
    nugget: Union[float, Tensor] = 0.0,
) -> Tensor:
    if hasattr(covariance, "matvec"):
        result = covariance.matvec(vector, observation_shape)
    else:
        cov = covariance.to(device=vector.device, dtype=vector.dtype)
        if cov.ndim >= 2 and cov.shape[-2:] == (vector.shape[-1], vector.shape[-1]):
            result = vector @ cov.T
        else:
            result = (vector.reshape(observation_shape) * cov).reshape_as(vector)

    if isinstance(nugget, Tensor) or nugget:
        result = result + nugget * vector
    return result


def solve_covariance_system(
    covariance: Union[Tensor, LocalizedEmpiricalCovariance],
    rhs: Tensor,
    observation_shape: torch.Size,
    *,
    nugget: Union[float, Tensor] = 0.0,
    solver: LinearSolver = "cg",
    max_iter: int = 20,
    tol: float = 1e-8,
) -> Tensor:
    if hasattr(covariance, "inverse") and isinstance(covariance.inverse, Tensor):
        has_nugget = bool(nugget.any()) if isinstance(nugget, Tensor) else bool(nugget)
        if not has_nugget:
            inverse = covariance.inverse.to(device=rhs.device, dtype=rhs.dtype)
            return rhs @ inverse.T

    if isinstance(covariance, Tensor):
        cov = covariance.to(device=rhs.device, dtype=rhs.dtype)
        if cov.ndim >= 2 and cov.shape[-2:] == (rhs.shape[-1], rhs.shape[-1]):
            if isinstance(nugget, Tensor) or nugget:
                eye = torch.eye(rhs.shape[-1], dtype=rhs.dtype, device=rhs.device)
                cov = cov + nugget.reshape(-1)[0] * eye if isinstance(nugget, Tensor) else cov + nugget * eye
            return torch.linalg.solve(cov, rhs.unsqueeze(-1)).squeeze(-1)

        diagonal = cov + nugget if isinstance(nugget, Tensor) or nugget else cov
        return (rhs.reshape(observation_shape) / diagonal).reshape_as(rhs)

    def matvec(vector: Tensor) -> Tensor:
        return covariance_matvec(covariance, vector, observation_shape, nugget=nugget)

    return solve_linear_system(matvec, rhs, solver=solver, max_iter=max_iter, tol=tol)


def empirical_localized_covariance(
    samples: Tensor,
    bandwidth: Optional[int] = None,
    temporal_band: Optional[int] = 0,
    min_std: float = 1e-3,
    hybrid_ratio: float = 0.0,
    static_std: float = 1.0,
) -> Tensor:
    # samples has shape (n_samples, time, *space_shape): dim 0 is the
    # empirical sample batch, dim 1 is time, and the remaining dims are
    # periodic spatial axes.
    samples_flat = samples.reshape(samples.shape[0], -1)
    centered = samples_flat - samples_flat.mean(dim=0, keepdim=True)

    if samples_flat.shape[0] > 1:
        cov = centered.T @ centered / (samples_flat.shape[0] - 1)
    else:
        cov = torch.zeros(
            samples_flat.shape[1],
            samples_flat.shape[1],
            dtype=samples.dtype,
            device=samples.device,
        )

    event_size = cov.shape[0]
    event_shape = samples.shape[1:]
    if bandwidth is None:
        bandwidth = max((dim_size // 2 for dim_size in event_shape[1:]), default=0)

    coordinates = torch.stack(
        torch.meshgrid(
            [torch.arange(dim_size, device=samples.device) for dim_size in event_shape],
            indexing="ij",
        ),
        dim=-1,
    ).reshape(event_size, len(event_shape))

    temporal_distances = (coordinates[:, 0, None] - coordinates[None, :, 0]).abs()
    if temporal_band is None:
        temporal_localization = torch.ones_like(temporal_distances, dtype=samples.dtype)
    elif temporal_band == 0:
        temporal_localization = (temporal_distances == 0).to(dtype=samples.dtype)
    else:
        temporal_localization = gaspari_cohn(temporal_distances / temporal_band).to(dtype=samples.dtype)

    if len(event_shape) > 1:
        spatial_distances = (coordinates[:, None, 1:] - coordinates[None, :, 1:]).abs()
        spatial_sizes = torch.tensor(event_shape[1:], device=samples.device)
        periodic_distances = torch.minimum(spatial_distances, spatial_sizes - spatial_distances)

        if bandwidth == 0:
            spatial_taper = (periodic_distances == 0).all(dim=-1).to(dtype=samples.dtype)
        else:
            spatial_distances = periodic_distances.square().sum(dim=-1).sqrt()
            spatial_taper = gaspari_cohn(spatial_distances / bandwidth)

        localization = temporal_localization * spatial_taper
    else:
        localization = temporal_localization

    empirical_cov = cov * localization

    if hybrid_ratio == 0:
        min_var = min_std ** 2
        cov = empirical_cov + torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device) * min_var
    else:
        static_cov = localization * (static_std ** 2)
        cov = hybrid_ratio * static_cov + (1 - hybrid_ratio) * empirical_cov + min_std ** 2 * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
    return cov