import io

import numpy as np
import PIL.Image
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from numpy.typing import NDArray
from omegaconf import DictConfig

from pdediff.mcs import KolmogorovFlow
from pdediff.utils.sevir_cmap import VIL_COLORS, VIL_LEVELS, get_cmap


def plot_trajectory_and_observation(
    cfg: DictConfig,
    mask: torch.Tensor,
    y_obs: torch.Tensor,
    true_x: torch.Tensor,
) -> plt.Figure:
    n_timesteps = min(5, true_x.shape[1])
    timestep_indices = np.linspace(0, true_x.shape[1] - 1, n_timesteps, dtype=int)
    dataset_name = cfg.data.name.lower()

    if dataset_name in {"ks", "arz"}:
        ground_truth_frames = true_x[0, :, 0].T.detach().cpu().numpy()
        observation_frames = y_obs[0, :, 0].T.detach().cpu().numpy()
        observation_mask = mask[0, :, 0].T.detach().cpu().numpy().astype(bool)
        if dataset_name == "arz":
            cmap = "turbo_r"
            vmin = float(np.nanmin(ground_truth_frames))
            vmax = float(np.nanmax(ground_truth_frames))
            colorbar_label = "speed"
        else:
            cmap = "RdBu_r"
            vmax = float(np.nanpercentile(np.abs(ground_truth_frames), 99))
            vmin = -vmax
            colorbar_label = "state"

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(11, 5.5),
            constrained_layout=True,
            sharex=True,
            sharey=True,
        )
        for row, (frames, label) in enumerate(
            ((ground_truth_frames, "Ground truth"), (observation_frames, "Observation (mask)"))
        ):
            image = axes[row].imshow(
                frames,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
                origin="lower",
            )
            if row == 1:
                missing = np.ma.masked_where(observation_mask, np.ones_like(observation_mask))
                axes[row].imshow(
                    missing,
                    cmap="Greys",
                    vmin=0,
                    vmax=1,
                    aspect="auto",
                    origin="lower",
                )
            axes[row].set_ylabel(f"{label}\nspace")

        axes[-1].set_xlabel("timestep")
        fig.colorbar(image, ax=axes, shrink=0.9, label=colorbar_label)
    elif dataset_name in {"kolmogorov", "sevir"}:
        selected_timesteps = timestep_indices.tolist()
        if dataset_name == "kolmogorov":
            ground_truth_frames = KolmogorovFlow.vorticity(
                true_x[0, selected_timesteps]
            ).detach().cpu().numpy()
            observation_frames = (
                y_obs[0, selected_timesteps, 0].detach().cpu().numpy()
            )
            ground_truth_vmax = np.nanpercentile(np.abs(ground_truth_frames), 99)
            observation_vmax = np.nanpercentile(np.abs(observation_frames), 99)
            image_kwargs_by_row = (
                {
                    "cmap": sns.color_palette("icefire", as_cmap=True),
                    "vmin": -ground_truth_vmax,
                    "vmax": ground_truth_vmax,
                },
                {
                    "cmap": "RdBu_r",
                    "vmin": -observation_vmax,
                    "vmax": observation_vmax,
                },
            )
        else:
            # Previous standalone SEVIR loader, retained in case SEVIR is not loaded via cfg:
            # from pdediff.utils.misc import load_dataset
            # trajectory_i = 0
            # start_t = 0
            # sevir = load_dataset(
            #     "data/sevir/sevir_lr", data_type="vil", max_events=trajectory_i + 1
            # )
            # plot_frames = sevir["data"][
            #     trajectory_i, start_t : start_t + n_timesteps, 0
            # ].detach().cpu().numpy()
            # plot_frames = plot_frames * 255.0
            # cmap, norm, vmin, vmax = get_cmap("vil", encoded=True)
            # image_kwargs = {"cmap": cmap, "norm": norm, "vmin": vmin, "vmax": vmax}

            ground_truth_frames = (
                true_x[0, selected_timesteps, 0].detach().cpu().numpy() * 255.0
            )
            observation_frames = (
                y_obs[0, selected_timesteps, 0].detach().cpu().numpy() * 255.0
            )
            cmap, norm, _, _ = get_cmap("vil", encoded=True)
            image_kwargs = {"cmap": cmap, "norm": norm}
            image_kwargs_by_row = (image_kwargs, image_kwargs)

        observation_mask = mask[0, selected_timesteps, 0].detach().cpu().numpy().astype(bool)
        fig, axes = plt.subplots(
            2,
            n_timesteps,
            figsize=(2.8 * n_timesteps, 5.6),
            constrained_layout=True,
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        for panel, timestep in enumerate(timestep_indices):
            for row, frames in enumerate((ground_truth_frames, observation_frames)):
                image = axes[row, panel].imshow(
                    frames[panel],
                    origin="lower",
                    **image_kwargs_by_row[row],
                )
                if row == 1:
                    missing = np.ma.masked_where(
                        observation_mask[panel],
                        np.ones_like(observation_mask[panel]),
                    )
                    axes[row, panel].imshow(
                        missing,
                        cmap="Greys",
                        vmin=0,
                        vmax=1,
                        origin="lower",
                    )
                axes[row, panel].set_xticks([])
                axes[row, panel].set_yticks([])
            axes[0, panel].set_title(f"t = {timestep}")

        axes[0, 0].set_ylabel("Ground truth")
        axes[1, 0].set_ylabel("Observation (mask)")
        if dataset_name == "kolmogorov":
            fig.colorbar(
                axes[0, -1].images[0], ax=axes[0], shrink=0.82, label="vorticity"
            )
            fig.colorbar(
                axes[1, -1].images[0], ax=axes[1], shrink=0.82, label="channel 0"
            )
        else:
            fig.colorbar(image, ax=axes, shrink=0.82, label="VIL")
            legend_elements = [
                Patch(
                    facecolor=VIL_COLORS[level],
                    label=f"{int(VIL_LEVELS[level - 1])}-{int(VIL_LEVELS[level])}",
                )
                for level in range(1, len(VIL_LEVELS))
            ]
            axes[0, 0].legend(
                handles=legend_elements,
                loc="center left",
                bbox_to_anchor=(-0.9, 0.5),
                borderaxespad=0,
                frameon=False,
                fontsize=8,
            )
    else:
        raise ValueError(f"Unsupported dataset for plotting: {cfg.data.name!r}")

    fig.suptitle(f"{cfg.data.name} trajectory")
    return fig


def convert_fig_to_array(fig: plt.Figure) -> NDArray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = PIL.Image.open(buf)
    return np.array(img).transpose(2, 0, 1)


def rcparams(fraction=1, subplots=(1, 1), beamer=False, columns=1, ieee=False):
    width_pt = 396
    if ieee:
        if columns == 2:
            width_pt = 515.52
        elif columns == 1:
            width_pt = 216
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    if not beamer:
        return {
            "text.usetex": True,
            "font.family": "serif",
            "text.latex.preamble": "\\usepackage{times} ",
            "figure.figsize": (fig_width_in, fig_height_in),
            "figure.constrained_layout.use": True,
            "figure.autolayout": False,
            "savefig.pad_inches": 0.015,
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.titlesize": 8,
        }
    else:
        width_pt = 307
        fig_width_pt = width_pt * fraction
        # Convert from pt to inches
        inches_per_pt = 1 / 72.27
        golden_ratio = (5**0.5 - 1) / 2

        # Figure width in inches
        fig_width_in = fig_width_pt * inches_per_pt
        # Figure height in inches
        fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
        return {
            "text.usetex": True,
            "font.family": "sans-serif",
            "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}",
            "figure.figsize": (fig_width_in, fig_height_in),
            "figure.constrained_layout.use": True,
            "figure.autolayout": False,
            "savefig.pad_inches": 0.015,
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.titlesize": 8,
            "mathtext.fontset": "custom",
            "mathtext.rm": "DejaVu Sans",
            "mathtext.it": "DejaVu Sans:italic",
            "mathtext.bf": "DejaVu Sans:bold",
        }
