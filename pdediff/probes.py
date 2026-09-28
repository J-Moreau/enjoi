from dataclasses import dataclass
from math import ceil

import numpy as np


@dataclass
class RoadSimulationParams:
    lmax: float = 20_000.0
    tmax: float = 3_600.0
    dx_meters: float = 100.0
    dt_seconds: float = 1.0


def simulate_probe_observations(
    sim_params: RoadSimulationParams,
    v: np.ndarray,
    rho: np.ndarray,
    ramp_index: np.ndarray,
    penetration_ratio: float,
    rng: np.random.Generator,
    speed_noise: float = 0.0,
    periodic: bool = False,
) -> tuple[float, int, int, np.ndarray, np.ndarray]:
    """Simulate floating-car probe trajectories on a road segment.

    Parameters
    ----------
    sim_params:
        Physical dimensions of the simulation domain.
    v:
        Speed field of shape ``(n_t, n_x)``.
    rho:
        Density field of shape ``(n_t, n_x)``.
    ramp_index:
        Spatial cell indices immediately upstream of each ramp. A ramp at
        index ``r`` is located at the interface between cells ``r`` and
        ``r + 1``.
    penetration_ratio:
        Fraction of vehicles that are instrumented probes.
    rng:
        NumPy random generator used for all stochastic draws.
    periodic:
        If true, vehicles crossing the right boundary re-enter at the left
        boundary. Otherwise, they leave the simulation.

    Returns
    -------
    dt, n_t, n_vehicles, all_positions, all_speeds
        Time step (s), number of time steps, number of probe vehicles, and
        position and speed arrays of shape ``(n_t, n_vehicles)`` (NaN before
        activation or after exit).
    """
    dt = sim_params.dt_seconds
    dx = sim_params.dx_meters
    n_x = int(ceil(sim_params.lmax / dx))
    n_t = int(ceil(sim_params.tmax / dt))
    ramp_index = np.asarray(ramp_index, dtype=int)
    ramp_positions = (ramp_index + 1) * dx

    # Initial probes are already on the road and are distributed across all
    # cells with probability proportional to the initial density field.
    n_initial = int(rho[0].sum() * dx * penetration_ratio)
    initial_cell_probs = rho[0] / rho[0].sum()
    initial_cells = rng.choice(n_x, size=n_initial, p=initial_cell_probs)
    initial_start_positions = (initial_cells + rng.uniform(size=n_initial)) * dx
    initial_start_times = np.zeros(n_initial, dtype=int)

    # Later probes enter only at ramps according to positive input flux.
    upstream_flux = rho[:, ramp_index] * v[:, ramp_index]
    downstream_flux = rho[:, ramp_index + 1] * v[:, ramp_index + 1]
    input_flux = np.clip(downstream_flux - upstream_flux, 0.0, None)

    # Later probe entries are Poisson samples whose rate follows the positive
    # input flux at each ramp and time step.
    expected_entries = input_flux * dt * penetration_ratio
    entry_counts = rng.poisson(expected_entries)
    entry_counts[0] = 0
    entering_start_times, entering_ramps = np.nonzero(entry_counts)
    nonzero_entry_counts = entry_counts[entering_start_times, entering_ramps]
    entering_start_times = np.repeat(entering_start_times, nonzero_entry_counts)
    entering_ramps = np.repeat(entering_ramps, nonzero_entry_counts)
    entering_start_positions = ramp_positions[entering_ramps]

    # Combine initial and later-entering probes.
    start_t_indices = np.concatenate([initial_start_times, entering_start_times])
    positions = np.concatenate([initial_start_positions, entering_start_positions])
    n_vehicles = positions.shape[0]

    # Simulate all vehicles at once
    all_positions = np.full((n_t, n_vehicles), np.nan)
    all_speeds = np.full((n_t, n_vehicles), np.nan)
    active = np.zeros(n_vehicles, dtype=bool)
    random_offset = 1 + rng.normal(scale=speed_noise, size=n_vehicles)  # for time-correlated noise in speeds

    for t_idx in range(n_t):
        starting = start_t_indices == t_idx
        active |= starting
        all_positions[t_idx, active] = positions[active]
        x_indices = np.clip((positions[active] / dx).astype(int), 0, n_x - 1)
        speeds = v[t_idx, x_indices]
        speeds *= random_offset[active]
        speeds = np.clip(speeds, 0, None)  # no backward movement
        all_speeds[t_idx, active] = speeds
        previous_positions = positions.copy()
        positions[active] += speeds * dt

        # Probes crossing a ramp leave with probability
        # (rho[r + 1] - rho[r]) / rho[r].
        for ramp, ramp_position in zip(ramp_index, ramp_positions):
            upstream_density = float(rho[t_idx, ramp])
            exit_probability = 0.0
            if upstream_density > 0.0:
                exit_probability = np.clip(
                    (rho[t_idx, ramp + 1] - upstream_density) / upstream_density,
                    0.0,
                    1.0,
                )
            crossing = active & (previous_positions < ramp_position) & (positions >= ramp_position)
            leaving = crossing & (rng.random(n_vehicles) < exit_probability)
            active &= ~leaving

        if periodic:
            positions[active] %= sim_params.lmax
        else:
            exited = positions >= sim_params.lmax
            active &= ~exited

    return dt, n_t, n_vehicles, all_positions, all_speeds


def trajectories_to_mask(
    all_positions: np.ndarray,
    n_cells: int,
    dx: float,
    stride: int = 1,
) -> np.ndarray:
    """Convert floating-car probe positions into a binary space-time observation mask.

    Parameters
    ----------
    all_positions:
        Array of shape ``(n_t_src, n_vehicles)`` where NaN encodes an inactive
        vehicle (not yet entered or already exited the road).
    n_cells:
        Number of spatial cells in the target grid.
    dx:
        Spatial cell width in metres.
    stride:
        Temporal downsampling factor.  Every ``stride`` consecutive source
        time steps are collapsed into one target time step; a cell is marked
        observed if any probe visited it during that interval.

    Returns
    -------
    mask : np.ndarray of shape ``(n_t_src // stride, n_cells)``, dtype bool.
    """
    n_t_src = all_positions.shape[0]
    n_t_tgt = n_t_src // stride

    # Truncate to a multiple of stride
    positions = all_positions[: n_t_tgt * stride]  # (n_t_tgt*stride, n_vehicles)

    valid = ~np.isnan(positions)

    with np.errstate(invalid="ignore"):
        cell_idx = np.where(
            valid,
            np.clip((positions / dx).astype(int), 0, n_cells - 1),
            0,  # dummy — masked out below
        ).astype(int)

    # Temporal index for each source step
    t_src = np.arange(n_t_tgt * stride)
    t_tgt_idx = np.broadcast_to((t_src // stride)[:, None], positions.shape)

    mask = np.zeros((n_t_tgt, n_cells), dtype=bool)
    mask[t_tgt_idx[valid], cell_idx[valid]] = True
    return mask


def trajectories_to_observations(
    all_positions: np.ndarray,
    probe_speeds: np.ndarray,
    n_cells_target: int,
    dx_target: float,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Build probe observations on a coarse grid using probe speed values.

    For each coarse (t, cell) that is visited by at least one probe, the
    observation value is the mean of the probe speed readings taken by all probes
    that fell in that coarse cell during that coarse time window.  Unobserved
    cells are set to zero and masked out.

    Parameters
    ----------
    all_positions:
        Array of shape ``(n_t_hd, n_vehicles)`` from
        :func:`simulate_probe_observations`.
    probe_speeds:
        Array of shape ``(n_t_hd, n_vehicles)`` from
        :func:`simulate_probe_observations`, containing the noisy per-vehicle
        speed at each HD time step.
    n_cells_target:
        Number of spatial cells in the coarse output grid.
    dx_target:
        Coarse spatial cell width in metres.
    stride:
        Temporal downsampling factor (HD steps → coarse steps).

    Returns
    -------
    observations : np.ndarray, shape ``(n_t_tgt, n_cells_target)``, float.
        Probe speed values averaged
        within each coarse cell/timestep bin.  Zero where unobserved.
    mask : np.ndarray, shape ``(n_t_tgt, n_cells_target)``, bool.
        True wherever at least one probe observation was recorded.
    """
    n_t_hd = all_positions.shape[0]
    n_t_tgt = n_t_hd // stride

    positions = all_positions[: n_t_tgt * stride]   # (n_t_tgt*stride, n_vehicles)
    speeds = probe_speeds[: n_t_tgt * stride]
    valid = (~np.isnan(positions)) & (~np.isnan(speeds))

    # Coarse target cell index
    with np.errstate(invalid="ignore"):
        tgt_cell = np.where(
            valid,
            np.clip((positions / dx_target).astype(int), 0, n_cells_target - 1),
            0,
        ).astype(int)

    # Coarse time index
    t_src = np.arange(n_t_tgt * stride)
    t_tgt_idx = np.broadcast_to((t_src // stride)[:, None], positions.shape)

    # Accumulate into coarse grid and count hits for averaging
    obs_sum   = np.zeros((n_t_tgt, n_cells_target), dtype=np.float64)
    obs_count = np.zeros((n_t_tgt, n_cells_target), dtype=np.int64)

    np.add.at(obs_sum,   (t_tgt_idx[valid], tgt_cell[valid]), speeds[valid])
    np.add.at(obs_count, (t_tgt_idx[valid], tgt_cell[valid]), 1)

    mask = obs_count > 0
    observations = np.where(mask, obs_sum / np.where(mask, obs_count, 1), 0.0)
    return observations.astype(np.float32), mask


def generate_arz_probe_observations(
    speed: np.ndarray,
    dx_meters: float,
    dt_seconds: float,
    scale_min: float,
    scale_max: float,
    n_ramps: int,
    rho_max: float,
    penetration_ratio: float,
    seed: int,
    speed_noise: float,
    periodic: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate normalized ARZ observations from noisy floating-car probes."""
    n_timesteps, n_cells = speed.shape
    sim_params = RoadSimulationParams(
        lmax=dx_meters * n_cells,
        tmax=dt_seconds * n_timesteps,
        dx_meters=dx_meters,
        dt_seconds=dt_seconds,
    )
    density = rho_max * np.clip(1.0 - speed / scale_max, 0.0, 1.0)
    ramp_index = np.linspace(0, speed.shape[1], n_ramps, endpoint=False, dtype=int)

    _, _, _, all_positions, all_speeds = simulate_probe_observations(
        sim_params=sim_params,
        v=speed,
        rho=density,
        ramp_index=ramp_index,
        penetration_ratio=penetration_ratio,
        rng=np.random.default_rng(seed=seed),
        speed_noise=speed_noise,
        periodic=periodic,
    )
    observations, mask = trajectories_to_observations(
        all_positions=all_positions,
        probe_speeds=all_speeds,
        n_cells_target=n_cells,
        dx_target=dx_meters,
        stride=1,
    )
    observations = 2.0 * (observations - scale_min) / (scale_max - scale_min) - 1.0
    observations *= mask
    return mask, observations.astype(np.float32)
