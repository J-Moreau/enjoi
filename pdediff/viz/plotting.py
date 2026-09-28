"""Helper function for plotting"""
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from numpy.typing import ArrayLike

from pdediff.mcs import KolmogorovFlow
from pdediff.sde import VPSDE
from pdediff.viz.kolmogorov import draw


def plot_1d_trajectories(trajectories):
    b, t, x = trajectories.squeeze(2).shape

    fig, axes = plt.subplots(b, 1, figsize=(5, 2 * trajectories.shape[0]))

    if b > 1:
        for i in range(b):
            axes[i].imshow(trajectories.squeeze(2)[i, :].T)
    else:
        axes.imshow(trajectories.squeeze(2)[0, :].T)

    return fig


def plot_probe_speed_and_trajectories(
    speed: np.ndarray,
    positions: np.ndarray,
    dt: float,
    lmax: float,
    *,
    vmax: float | None = None,
    title: str = "Speed + probe trajectories",
    trajectory_color: str = "white",
    trajectory_linewidth: float = 2,
    trajectory_alpha: float = 1.0,
    figsize: tuple[float, float] = (12.0, 5.0),
) -> plt.Figure:
    """Plot a speed field with probe trajectories split at periodic wraps."""
    n_t, n_vehicles = positions.shape
    t_axis = np.arange(n_t) * dt

    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(
        speed.T,
        aspect="auto",
        origin="lower",
        cmap="turbo_r",
        extent=[0.0, speed.shape[0] * dt, 0.0, lmax],
        vmin=0.0,
        vmax=vmax,
    )

    for vehicle_index in range(n_vehicles):
        vehicle_positions = positions[:, vehicle_index]
        finite = np.isfinite(vehicle_positions)
        discontinuity = (
            ~finite[:-1]
            | ~finite[1:]
            | (np.abs(np.diff(vehicle_positions)) > 0.5 * lmax)
        )
        segment_boundaries = np.flatnonzero(discontinuity) + 1
        segment_starts = np.concatenate(([0], segment_boundaries))
        segment_ends = np.concatenate((segment_boundaries, [n_t]))

        for start, end in zip(segment_starts, segment_ends):
            if end - start >= 2 and finite[start:end].all():
                ax.plot(
                    t_axis[start:end],
                    vehicle_positions[start:end],
                    color=trajectory_color,
                    linewidth=trajectory_linewidth,
                    alpha=trajectory_alpha,
                )

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (m)")
    fig.colorbar(image, ax=ax, label="Speed (m/s)")
    fig.tight_layout()
    return fig


def plot_mean_and_std(metrics, labels, title: str = "", xlabel: str = "", ylabel: str = "", figsize: tuple = (8, 8)):
    """
    Method to plot the mean and std of a metric.
    It assumes that metric has shape  [B, T] #todo: check this is valid always

    Args:
        metric: metric computed on different samples
        label: name of the method for which we are computing the metric (used in the legend)
        title: plot title
        xlabel: label for the x-axis
        ylabel: label for the y-axis
        figsize: size of the plot
    """
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    for i, metric in enumerate(metrics):
        assert (
            len(metric.shape) == 2
        ), "We are assuming that metric has shape batch x traj_length, if not either change or adapt the plotting method"

        # now we plot the average
        if torch.is_tensor(metric):
            metric_average = torch.mean(metric, 0)
            metric_std = torch.std(metric, 0)
        else:
            metric_average = np.mean(metric, 0)
            metric_std = np.std(metric, 0)

        axes.plot(np.arange(len(metric[0, :])), metric_average, c=f"C{i}", alpha=1, label=labels[i], zorder=10)
        axes.errorbar(np.arange(len(metric[0, :])), metric_average, yerr=metric_std, c=f"C{i}", alpha=0.3)

    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.legend()

    return fig


def plot_comparison_metric(
    metric: list[ArrayLike],
    labels: list[str],
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    figsize: tuple = (8, 8),
):
    """
    Method to plot the same metric of different methods.

    Args:
        metric: list containing different metric from different methods.
        labels: list containing different label associated with the metric inside the correlation list
        title: plot title
        xlabel: label for the x-axis
        ylabel: label for the y-axis
        figsize: size of the plot
    """

    fig, axes = plt.subplots(1, 1, figsize=figsize)

    for i in range(len(metric)):
        axes.plot(np.arange(len(metric[i])), metric[i], c=f"C{i}", label=labels[i], alpha=0.5)

    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.legend()

    return fig


def plot_metric_different_samples_and_average(
    metric: ArrayLike, label: str, title: str = "", xlabel: str = "", ylabel: str = "", figsize: tuple = (8, 8)
):
    """
    Method to plot the pearson correlation of a single method.
    It assumes that metric has sahpe  [B, T] #todo: check this is valid always

    Args:
        metric: metric computed on different samples
        label: name of the method for which we are computing the metric (used in the legend)
        title: plot title
        xlabel: label for the x-axis
        ylabel: label for the y-axis
        figsize: size of the plot
    """

    assert (
        len(metric.shape) == 2
    ), "We are assuming that metric has shape batch x traj_length, if not either change or adapt the plotting method"

    fig, axes = plt.subplots(1, 1, figsize=figsize)

    for i in range(metric.shape[0]):
        axes.plot(np.arange(len(metric[0, :])), metric[i, :], c="lightgrey", alpha=0.3)

    # now we plot the average
    if torch.is_tensor(metric):
        metric_average = torch.mean(metric, 0)
    else:
        metric_average = np.mean(metric, 0)

    axes.plot(np.arange(len(metric[0, :])), metric_average, c="firebrick", alpha=0.8, label=label, zorder=10)

    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.legend()

    return fig


def plot_mean_and_std(metrics, labels, title: str = "", xlabel: str = "", ylabel: str = "", figsize: tuple = (8, 8)):
    """
    Method to plot the mean and std of a metric.
    It assumes that metric has shape  [B, T] #todo: check this is valid always

    Args:
        metric: metric computed on different samples
        label: name of the method for which we are computing the metric (used in the legend)
        title: plot title
        xlabel: label for the x-axis
        ylabel: label for the y-axis
        figsize: size of the plot
    """
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    for i, metric in enumerate(metrics):
        assert (
            len(metric.shape) == 2
        ), "We are assuming that metric has shape batch x traj_length, if not either change or adapt the plotting method"

        # now we plot the average
        if torch.is_tensor(metric):
            metric_average = torch.mean(metric, 0)
            metric_std = torch.std(metric, 0)
        else:
            metric_average = np.mean(metric, 0)
            metric_std = np.std(metric, 0)

        axes.plot(np.arange(len(metric[0, :])), metric_average, c=f"C{i}", alpha=1, label=labels[i], zorder=10)
        axes.errorbar(np.arange(len(metric[0, :])), metric_average, yerr=metric_std, c=f"C{i}", alpha=0.3)

    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.legend()

    return fig

def plot_mean_and_std_err(
    metrics,
    labels,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    figsize: tuple = (8, 8),
    y_log_scale: bool = False,
):
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    for i, metric in enumerate(metrics):
        assert (
            len(metric.shape) == 2
        ), "We are assuming that metric has shape batch x traj_length, if not either change or adapt the plotting method"

        # now we plot the average
        if torch.is_tensor(metric):
            metric_average = torch.mean(metric, 0)
            metric_std_err = torch.std(metric, 0) / np.sqrt(metric.shape[0])
        else:
            metric_average = np.mean(metric, 0)
            metric_std_err = np.std(metric, 0) / np.sqrt(metric.shape[0])

        axes.plot(np.arange(len(metric[0, :])), metric_average, c=f"C{i}", alpha=0.6, label=labels[i], zorder=10)

        print(metric_average)
        print("---")
        axes.fill_between(
            np.arange(len(metric[0, :])),
            metric_average - metric_std_err,
            metric_average + metric_std_err,
            alpha=0.6,
            color=f"C{i}",
        )

    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    if y_log_scale:
        axes.set_yscale("log")
    axes.legend()

    return fig


def plot_kolmogorov_vorticity_trajectories(trajectories):
    # simple method to plot Kolmogorov trajectories
    # these are 64x64 states
    chain = KolmogorovFlow(size=256, dt=0.2)

    # First compute the vorticity of the trajectories
    vorticity = chain.vorticity(trajectories) # this results in batch, traj_length, state1, state2
    img = draw(vorticity, zoom=4)
    return img


def plot_spatiotemporal_reconstruction(
    gt: np.ndarray,
    mask: np.ndarray,
    y_obs: np.ndarray,
    samples: np.ndarray,
    cmap: str = "turbo_r",
    vmin: float | None = None,
    vmax: float | None = None,
    xlabel: str = "Time step",
    ylabel: str = "Spatial index",
    extent: list[float] | None = None,
    figsize: tuple[float, float] = (18, 10),
    sample_index: int = 0,
    channel_index: int = 0,
    data_type: Literal["arz", "ks", "kolmogorov"] = "ks",
) -> plt.Figure:
    """Plot a 2x3 grid of reconstruction results.

    Parameters
    ----------
    gt:
        Ground truth with shape ``(T, nx)``, ``(T, ny, nx)``, or
        ``(T, channels, ny, nx)``. For 2D states, the final timestep is shown.
    mask:
        Binary observation mask with the same state dimensions as ``gt``.
    samples:
        Posterior samples with a leading ensemble dimension.
    cmap:
        Legacy colormap fallback. ``data_type`` selects ``turbo_r`` for ARZ,
        ``RdBu_r`` for KS, and ``icefire`` for Kolmogorov.
    vmin, vmax:
        Color limits shared across GT / sample / mean panels.
    xlabel, ylabel:
        Axis labels.
    extent:
        ``[left, right, bottom, top]`` passed to ``imshow`` as ``extent``.
    figsize:
        Figure size ``(width, height)`` in inches.
    sample_index, channel_index:
        Ensemble sample and state channel to display.
    data_type:
        Dataset representation. Kolmogorov velocity fields are converted to
        vorticity and plotted with seaborn's ``icefire`` colormap.

    Returns
    -------
    matplotlib.figure.Figure
        The figure (suptitle is left to the caller).
    """
    n_samples = samples.shape[0]
    if data_type == "kolmogorov":
        gt_plot = KolmogorovFlow.vorticity(torch.as_tensor(gt[-1:])).detach().cpu().numpy()[0]
        y_obs_plot = KolmogorovFlow.vorticity(torch.as_tensor(y_obs[-1:])).detach().cpu().numpy()[0]
        samples_plot = KolmogorovFlow.vorticity(torch.as_tensor(samples[:, -1])).detach().cpu().numpy()
        mask_plot = mask[-1].all(axis=0)
        is_2d = True
        cmap = sns.color_palette("icefire", as_cmap=True)
        vmax = float(np.nanpercentile(np.abs(gt_plot), 99))
        vmin = -vmax
        plot_xlabel = "Space x"
        plot_ylabel = "Space y"
        channel_label = ""
    elif gt.ndim >= 3:
        is_2d = True
        if gt.ndim == 4:
            gt_plot = gt[-1, channel_index]
            mask_plot = mask[-1, channel_index]
            y_obs_plot = y_obs[-1, channel_index]
            samples_plot = samples[:, -1, channel_index]
        else:
            gt_plot = gt[-1]
            mask_plot = mask[-1]
            y_obs_plot = y_obs[-1]
            samples_plot = samples[:, -1]
        plot_xlabel = "Space x"
        plot_ylabel = "Space y"
        channel_label = f", channel {channel_index}" if gt.ndim == 4 else ""
    else:
        is_2d = False
        gt_plot = gt.T
        mask_plot = mask.T
        y_obs_plot = y_obs.T
        samples_plot = samples.transpose(0, 2, 1)
        plot_xlabel = xlabel
        plot_ylabel = ylabel
        channel_label = ""

    if data_type == "arz":
        cmap = "turbo_r"
    elif data_type == "ks":
        cmap = "RdBu_r"

    timestep_label = " (final timestep)" if is_2d else ""
    gen_mean = samples_plot.mean(axis=0)
    gen_std = samples_plot.std(axis=0)
    mean_err = gen_mean - gt_plot

    if vmin is None:
        vmin = float(gt_plot.min())
    if vmax is None:
        vmax = float(gt_plot.max())

    kw: dict = dict(aspect="equal" if is_2d else "auto", origin="lower", interpolation="none")
    if extent is not None:
        kw["extent"] = extent

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    axes[0].imshow(gt_plot, cmap=cmap, vmin=vmin, vmax=vmax, **kw)
    axes[0].set_title("Ground truth" + (f" (final timestep{channel_label})" if is_2d else ""))
    plt.colorbar(axes[0].images[0], ax=axes[0])

    axes[3].imshow(y_obs_plot, cmap=cmap, vmin=vmin, vmax=vmax, **kw)
    axes[3].imshow(1 - mask_plot, cmap="Greys", alpha=0.5, vmin=0, vmax=1, **kw)
    axes[3].set_title(f"Observations{timestep_label} ({mask_plot.mean():.1%} coverage)")
    plt.colorbar(axes[3].images[0], ax=axes[3])

    im2 = axes[1].imshow(samples_plot[sample_index], cmap=cmap, vmin=vmin, vmax=vmax, **kw)
    axes[1].set_title(f"Sample #{sample_index}{timestep_label}")
    plt.colorbar(im2, ax=axes[1])

    im3 = axes[2].imshow(gen_mean, cmap=cmap, vmin=vmin, vmax=vmax, **kw)
    axes[2].set_title(f"Mean over {n_samples} samples{timestep_label}")
    plt.colorbar(im3, ax=axes[2])

    im4 = axes[4].imshow(gen_std, cmap="viridis", **kw)
    axes[4].set_title(f"Std over {n_samples} samples{timestep_label}")
    plt.colorbar(im4, ax=axes[4])

    err_abs = float(np.abs(mean_err).max())
    rmse = float(np.sqrt((mean_err**2).mean()))
    im5 = axes[5].imshow(mean_err, cmap="RdBu_r", vmin=-err_abs, vmax=err_abs, **kw)
    axes[5].set_title(f"Error of the mean{timestep_label} ({n_samples} samples, RMSE={rmse:.3f})")
    plt.colorbar(im5, ax=axes[5])

    for ax in axes:
        ax.set_xlabel(plot_xlabel)
        ax.set_ylabel(plot_ylabel)

    fig.tight_layout()
    return fig


def plot_uncond_samples(sde: VPSDE, n_rows: int = 4, n_cols: int = 4, steps: int = 64) -> plt.Figure:
    """Sample n_rows*n_cols single-window trajectories unconditionally and plot
    them as a (n_rows x n_cols) grid of imshow subplots. Returns the figure."""
    n = n_rows * n_cols
    sde.eval()
    with torch.no_grad():
        samples = sde.sample(shape=(n,), steps=steps).cpu()  # (n, window*spatial, nx)

    sample = np.squeeze(samples[0].numpy())
    if sample.ndim >= 3:
        window = sde.net.window
        sample = sample.reshape(window, sample.shape[0] // window, *sample.shape[1:])
        sample = sample[:, 0]
        n_timesteps = sample.shape[0]
        fig, axes = plt.subplots(1, n_timesteps, figsize=(3 * n_timesteps, 3))
        for timestep, ax in enumerate(np.atleast_1d(axes)):
            ax.imshow(sample[timestep], aspect="auto", origin="lower", cmap="turbo_r")
            ax.set_title(f"t={timestep}")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        return fig

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    for idx, ax in enumerate(axes.flat):
        ax.imshow(np.squeeze(samples[idx].numpy()).T, aspect="auto", origin="lower", cmap="turbo_r")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return fig


def plot_cond_samples(sde: VPSDE, valid_data: torch.Tensor, num_observed: int, cond_indices: torch.Tensor,
                      n_rows: int = 4, n_cols: int = 4, steps: int = 64) -> plt.Figure:
    """Sample n_rows*n_cols single-window trajectories conditionally using
    fixed validation samples, and plot them as a grid. A dashed line marks the
    boundary between conditioned and predicted frames. Returns the figure."""
    n = n_rows * n_cols
    sde.eval()

    x = valid_data[cond_indices[:n]].cuda()
    mask = torch.zeros_like(x)
    mask[:, :num_observed] = 1.0
    x_condition = torch.cat([mask, mask * x], dim=1)
    sde.net.set_condition(x_condition)

    with torch.no_grad():
        samples = sde.sample(shape=(n,), steps=steps).cpu()

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    for idx, ax in enumerate(axes.flat):
        ax.imshow(samples[idx].numpy().T, aspect="auto", origin="lower", cmap="turbo_r")
        ax.axvline(x=num_observed - 0.5, color="white", linewidth=1, linestyle="--")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return fig

