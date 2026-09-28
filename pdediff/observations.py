from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from pdediff.probes import generate_arz_probe_observations


def generate_observations(
    cfg: DictConfig,
    trajectory_length: int,
    true_x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    observation_shape: Literal[
        "sine", "exponential_full", "sparse", "grid", "probe"
    ] = cfg.observation.type
    mask = torch.zeros_like(true_x)  # (1, traj_len, 1, nx)

    if observation_shape == "sine":
        nx = cfg.data.grid_size[0]
        period = cfg.observation.sine.period
        obs_width = nx / cfg.observation.sine.width_divisor

        t_idx = torch.arange(trajectory_length, dtype=torch.float32)
        center = (nx - 1) / 2 + (nx - 1) * cfg.observation.sine.amplitude_fraction * torch.sin(
            2 * np.pi * t_idx / period
        )
        x_idx = torch.arange(nx, dtype=torch.float32)

        for t in range(trajectory_length):
            # Observe pixels within obs_width of the sine center (with circular wrapping)
            dist = torch.abs(x_idx - center[t])
            dist = torch.min(dist, nx - dist)
            mask[:, t, :, dist <= obs_width] = 1.0

    elif observation_shape == "exponential_full":
        exponential_mean_gap = cfg.observation.exponential_full.mean_gap
        rng = np.random.default_rng(seed=cfg.observation.exponential_full.seed)

        observed_times = [cfg.window]
        current_time = cfg.window
        while current_time < trajectory_length:
            gap = max(1, rng.exponential(scale=exponential_mean_gap))
            current_time += gap
            if current_time < trajectory_length:
                observed_times.append(current_time)

        mask[:, observed_times, :, :] = 1.0
        print(
            f"exponential full-state observation times: {observed_times[:10]}{'...' if len(observed_times) > 10 else ''}"
        )

    elif observation_shape == "sparse":
        generator = torch.Generator(device=true_x.device)
        generator.manual_seed(cfg.observation.sparse.seed)
        probs = torch.ones(true_x.shape[1:], device=true_x.device) * cfg.observation.sparse.perc_obs
        mask = torch.bernoulli(probs, generator=generator).to(true_x.dtype)
        mask = mask.unsqueeze(0).repeat(true_x.shape[0], *([1] * (true_x.ndim - 1)))

    elif observation_shape == "grid":
        points_per_dim = cfg.observation.grid.points_per_dim
        spatial_shape = true_x.shape[3:]
        grid_indices = [
            torch.arange(points_per_dim, device=true_x.device) * size // points_per_dim
            for size in spatial_shape
        ]
        if len(spatial_shape) == 1:
            mask[:, :, :, grid_indices[0]] = 1.0
        elif len(spatial_shape) == 2:
            rows, columns = torch.meshgrid(*grid_indices, indexing="ij")
            mask[:, :, :, rows, columns] = 1.0
        else:
            raise ValueError(
                f"Grid observations support one or two spatial dimensions, got {len(spatial_shape)}"
            )

    elif observation_shape == "probe":
        if cfg.data.name.lower() != "arz":
            raise ValueError("Probe observations are only supported for the ARZ dataset")
        speed = true_x[0, :, 0].detach().cpu().numpy()
        speed = (speed + 1.0) / 2.0 * (cfg.data.scale_max - cfg.data.scale_min)
        speed += cfg.data.scale_min
        probe_mask, probe_values = generate_arz_probe_observations(
            speed=speed,
            dx_meters=cfg.data.dx_meters,
            dt_seconds=cfg.data.dt_seconds,
            scale_min=cfg.data.scale_min,
            scale_max=cfg.data.scale_max,
            n_ramps=cfg.observation.probe.n_ramps,
            rho_max=cfg.observation.probe.rho_max,
            penetration_ratio=cfg.observation.probe.penetration_ratio,
            seed=cfg.observation.probe.seed,
            speed_noise=cfg.observation.probe.speed_noise,
            periodic=cfg.observation.probe.periodic,
        )
        mask = torch.from_numpy(probe_mask).to(device=true_x.device, dtype=true_x.dtype)
        y_obs = torch.from_numpy(probe_values).to(device=true_x.device, dtype=true_x.dtype)
        mask = mask.unsqueeze(0).unsqueeze(2)
        y_obs = y_obs.unsqueeze(0).unsqueeze(2)

    else:
        raise ValueError(f"Unsupported observation_shape: {observation_shape}")

    # Add first state to mask (initial condition is fully observed)
    if observation_shape != "probe":
        y_obs = true_x * mask
        y_obs += torch.randn_like(y_obs) * cfg.observation.noise_std * mask
    initial_condition_frames = cfg.observation.initial_condition_frames
    mask[:, 0:initial_condition_frames, :, :] = 1.0
    if cfg.observation.initial_box_blur > 0:
        k = cfg.observation.initial_box_blur
        initial_condition = true_x[:, :initial_condition_frames]
        flattened = initial_condition.flatten(0, 1)
        left_padding = (k - 1) // 2
        right_padding = k // 2
        spatial_dims = flattened.ndim - 2
        avg_pool = F.avg_pool1d if spatial_dims == 1 else F.avg_pool2d
        padding = (left_padding, right_padding) * spatial_dims
        y_obs[:, :initial_condition_frames] = avg_pool(
            F.pad(flattened, padding, mode="circular"),
            kernel_size=k,
            stride=1,
        ).unflatten(0, initial_condition.shape[:2])
    else:
        y_obs[:, 0:initial_condition_frames, :, :] = true_x[
            :, 0:initial_condition_frames, :, :
        ]  # ensure initial condition is fully observed without noise
    y_obs.clamp_(true_x.min(), true_x.max())
    print(f"Observation shape: {observation_shape}")
    print(f"Observed fraction: {mask.mean():.2%}")
    return mask, y_obs