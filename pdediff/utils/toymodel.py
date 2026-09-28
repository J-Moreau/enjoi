import torch
from torch import Tensor


def default_two_timestep_mixture_parameters(
    spatial_dim: int = 1,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    weights = torch.tensor([0.75, 0.25], dtype=dtype, device=device)

    base_means = torch.tensor(
        [
            [-1.25, -0.65],
            [1.35, -1.10],
        ],
        dtype=dtype,
        device=device,
    )
    means = base_means[:, :, None].expand(2, 2, spatial_dim).contiguous()

    covariance = torch.zeros(2 * spatial_dim, 2 * spatial_dim, dtype=dtype, device=device)
    spatial_correlation = 0.25
    temporal_correlation = 0.95
    std_x0 = 1.0
    std_x1 = 1.0

    for i in range(spatial_dim):
        for j in range(spatial_dim):
            spatial_scale = spatial_correlation ** abs(i - j)
            covariance[i, j] = std_x0**2 * spatial_scale
            covariance[spatial_dim + i, spatial_dim + j] = std_x1**2 * spatial_scale
            cross_covariance = temporal_correlation * std_x0 * std_x1 * spatial_scale
            covariance[i, spatial_dim + j] = cross_covariance
            covariance[spatial_dim + i, j] = cross_covariance

    return weights, means, covariance


def two_timestep_mixture_density(
    x: Tensor,
    *,
    weights: Tensor | None = None,
    means: Tensor | None = None,
    covariance: Tensor | None = None,
) -> Tensor:
    weights, means, covariance = (
        default_two_timestep_mixture_parameters(
            spatial_dim=x.shape[-1],
            dtype=x.dtype,
            device=x.device,
        )
        if weights is None or means is None or covariance is None
        else (weights, means, covariance)
    )

    event_size = x.shape[-2] * x.shape[-1]
    x_flat = x.reshape(*x.shape[:-2], event_size)
    means_flat = means.reshape(means.shape[0], event_size)
    covariance = covariance.to(dtype=x.dtype, device=x.device)
    weights = weights.to(dtype=x.dtype, device=x.device)

    covariance_cholesky = torch.linalg.cholesky(covariance)
    centered = x_flat.unsqueeze(-2) - means_flat
    solved = torch.cholesky_solve(centered.unsqueeze(-1), covariance_cholesky).squeeze(-1)
    exponent = -0.5 * (centered * solved).sum(dim=-1)
    log_det = 2.0 * covariance_cholesky.diagonal().log().sum()
    log_norm = -0.5 * (event_size * torch.log(torch.tensor(2.0 * torch.pi, dtype=x.dtype, device=x.device)) + log_det)
    log_density = torch.logsumexp(weights.log() + log_norm + exponent, dim=-1)
    return log_density.exp()


def sample_two_timestep_mixture(
    batch_size: int,
    spatial_dim: int = 1,
    *,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Tensor:
    weights, means, covariance = default_two_timestep_mixture_parameters(
        spatial_dim=spatial_dim,
        dtype=dtype,
        device=device,
    )
    mode_idx = torch.multinomial(weights, batch_size, replacement=True, generator=generator)
    noise = torch.randn(batch_size, 2 * spatial_dim, dtype=dtype, device=device, generator=generator)
    covariance_cholesky = torch.linalg.cholesky(covariance)
    samples = means.reshape(2, 2 * spatial_dim)[mode_idx] + noise @ covariance_cholesky.T
    return samples.reshape(batch_size, 2, spatial_dim)


def make_two_timestep_mixture_dataset(
    n_samples: int,
    spatial_dim: int = 1,
    window: int = 2,
    *,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> dict[str, Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    samples = sample_two_timestep_mixture(
        n_samples,
        spatial_dim=spatial_dim,
        generator=generator,
        dtype=dtype,
        device=device,
    )
    return {
        "data": samples[:, :window]
    }
