import os
from datetime import datetime
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf

from pdediff.ensemble import sample_ensemble
from pdediff.observations import generate_observations
from pdediff.sampler.utils import get_sampler
from pdediff.score import make_score
from pdediff.utils import cfg as sda_cfg  # noqa: F401
from pdediff.utils.misc import load_dataset, load_zarr_dataset
from pdediff.viz.plotting import plot_spatiotemporal_reconstruction


@hydra.main(config_path="config", config_name="main", version_base="1.1")
def main(cfg):
    project_dir = Path(get_original_cwd())
    ckpt_path = resolve_project_path(project_dir, cfg.ckpt_path)

    score = make_score(
        score=cfg.score,
        net=cfg.net,
        window=cfg.window,
        state_channels=cfg.data.state_shape[0],
        condition_dim=cfg.window * 2 if cfg.amortized else 0,
    )
    score.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    print(f"Loaded checkpoint from {ckpt_path}")

    # ── load test data ────────────────────────────────────────────────────────────
    if cfg.data.get("format", "h5") == "zarr":
        test_dataset = load_zarr_dataset(
            cfg.data.path,
            "test",
            cfg.data.zarr_subpath,
            cfg.data.scale_min,
            cfg.data.scale_max,
        )
        if cfg.data.name == "ARZ" and cfg.eval.n_eval_trajectories==10:
            # for ARZ, we remove free-flow trajectories from the test dataset to make it more challenging
            test_dataset["data"] = test_dataset["data"][[0,5,6,7,8,9,10,12,13,14]]
    else:
        test_dataset = load_dataset(Path(cfg.data.path) / "test.h5")

    trajectory_length = cfg.observation.trajectory_length
    for i in range(cfg.eval.n_eval_trajectories):
        cfg.eval.data_idx = i
        true_x = test_dataset["data"][i : i + 1, :trajectory_length]  # shape: (1, T, **)
        true_x = true_x.reshape(1, trajectory_length, *cfg.data.state_shape)
        print(f"true_x shape: {true_x.shape}")
        # ── build observation mask ────────────────────────────────────────────────────
        mask, y_obs = generate_observations(
            cfg,
            trajectory_length,
            true_x,
        )

        # ar_variant = cfg.eval.autoregressive.variant
        cfg.eval.forecast.trajectory_length = trajectory_length
        sampler = get_sampler(cfg)
        sampled_result = sample_ensemble(
            score=score,
            cfg=cfg,
            y=y_obs,
            mask=mask,
            sampler=sampler,
        )
        sampled_x = sampled_result.samples if hasattr(sampled_result, "samples") else sampled_result
    
        output_dir = make_output_dir(cfg)
        save_outputs(output_dir, sampled_x, cfg)
        plot_path = save_reconstruction_plot(output_dir, sampled_x, true_x, mask, y_obs, cfg)
        metrics = compute_metrics(sampled_x, true_x, mask)
        cfg_to_save = OmegaConf.to_container(cfg, resolve=True)
        log_to_wandb(cfg, cfg_to_save, metrics, output_dir, plot_path)

        if cfg.ensemble_eval_name == "default":
            log_default_single_sample_runs(
                cfg=cfg,
                sampled_x=sampled_x,
                true_x=true_x,
                mask=mask,
                output_dir=output_dir,
            )
        print(f"Saved ensemble output to {output_dir}")


def make_output_dir(cfg) -> Path:
    run_name = datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
    output_dir = Path(cfg.logdir).joinpath("ensemble_eval", run_name)
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def resolve_project_path(project_dir: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else project_dir.joinpath(path)


def save_outputs(output_dir: Path, sampled_x: torch.Tensor, cfg) -> None:
    np.save(output_dir.joinpath("sampled_x.npy"), sampled_x.detach().cpu().numpy())
    OmegaConf.save(config=cfg, f=output_dir.joinpath("config.yaml"), resolve=True)


def config_with_single_sample(cfg):
    single_sample_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    single_sample_cfg.eval.forecast.n_samples = 1
    return single_sample_cfg


def log_default_single_sample_runs(
    cfg,
    sampled_x: torch.Tensor,
    true_x: torch.Tensor,
    mask: torch.Tensor,
    output_dir: Path,
) -> None:
    single_sample_cfg = config_with_single_sample(cfg)
    single_sample_cfg_to_save = OmegaConf.to_container(single_sample_cfg, resolve=True)

    for sample_idx in range(sampled_x.shape[0]):
        single_sample_x = sampled_x[sample_idx : sample_idx + 1]
        sample_metrics = compute_metrics(single_sample_x, true_x, mask)
        sample_metrics["sample_index"] = sample_idx
        log_metrics_to_wandb(
            single_sample_cfg,
            single_sample_cfg_to_save,
            sample_metrics,
            output_dir,
            run_name=f"{cfg.name}_{cfg.ensemble_eval_name}_{output_dir.name}_sample_{sample_idx:03d}",
            group=f"{cfg.name}_{cfg.ensemble_eval_name}_single_sample",
        )


def save_reconstruction_plot(
    output_dir: Path,
    sampled_x: torch.Tensor,
    true_x: torch.Tensor,
    mask: torch.Tensor,
    y_obs: torch.Tensor,
    cfg,
) -> Path:
    gt = true_x.squeeze(2)[0].detach().cpu().numpy()
    mask_vis = mask.squeeze(2)[0].detach().cpu().numpy()
    y_obs_vis = y_obs.squeeze(2)[0].detach().cpu().numpy()
    samples = sampled_x.squeeze(2).detach().cpu().numpy()
    data_type = cfg.data.name.lower()

    if data_type == "arz":
        scale_min = float(cfg.data.scale_min)
        scale_max = float(cfg.data.scale_max)
        gt = (gt + 1.0) / 2.0 * (scale_max - scale_min) + scale_min
        y_obs_vis = (y_obs_vis + 1.0) / 2.0 * (scale_max - scale_min) + scale_min
        samples = (samples + 1.0) / 2.0 * (scale_max - scale_min) + scale_min
        vmin, vmax = scale_min, scale_max
    else:
        vmin, vmax = -3.0, 3.0

    fig = plot_spatiotemporal_reconstruction(
        gt=gt,
        mask=mask_vis,
        y_obs=y_obs_vis,
        samples=samples,
        cmap="turbo_r" if data_type == "arz" else "RdBu_r",
        vmin=vmin,
        vmax=vmax,
        xlabel="Time",
        ylabel="Position" if data_type == "arz" else "Space",
        extent=[0, 1, 0, 1],
        sample_index=min(4, samples.shape[0] - 1),
        data_type=data_type,
    )
    fig.suptitle(f"{cfg.eval.rollout_type} rollout - {cfg.ensemble_eval_name}", y=1.01)
    plot_path = output_dir.joinpath("reconstruction.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def compute_metrics(sampled_x: torch.Tensor, true_x: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    true_x = true_x.to(sampled_x.device)
    ensemble_mean = sampled_x.mean(dim=0, keepdim=True)
    skill = ((ensemble_mean - true_x) ** 2).mean().sqrt()
    spread = sampled_x.var(dim=0, unbiased=False).mean().sqrt()
    crps = empirical_crps(sampled_x, true_x)

    return {
        "skill": skill.item(),
        "spread": spread.item(),
        "spread_skill_ratio": (spread / skill.clamp_min(torch.finfo(skill.dtype).tiny)).item(),
        "crps": crps.item(),
        "observed_fraction": mask.mean().item(),
    }


def empirical_crps(sampled_x: torch.Tensor, true_x: torch.Tensor) -> torch.Tensor:
    ensemble_error = (sampled_x - true_x).abs().mean()
    pairwise_distance = (sampled_x.unsqueeze(0) - sampled_x.unsqueeze(1)).abs().mean()
    return ensemble_error - 0.5 * pairwise_distance


def log_to_wandb(
    cfg,
    cfg_to_save: dict,
    metrics: dict[str, float],
    output_dir: Path,
    plot_path: Path,
    run_name: str = None,
    group: str = None,
    artifact_name: str = None,
) -> None:
    try:
        import wandb
    except ModuleNotFoundError:
        print("wandb is not installed; skipping WandB logging")
        return

    wandb_cfg = cfg.logger.wandb
    if wandb_cfg.offline:
        os.environ["WANDB_MODE"] = "dryrun"

    run = wandb.init(
        project=wandb_cfg.project,
        name=run_name or f"{cfg.name}_{cfg.ensemble_eval_name}_{output_dir.name}",
        group=group or f"{cfg.name}_{cfg.ensemble_eval_name}",
        dir=str(output_dir),
        config=cfg_to_save,
    )
    run.log(
        {
            **metrics,
            "reconstruction": wandb.Image(str(plot_path), caption=f"{cfg.name} - {cfg.ensemble_eval_name}"),
        },
        step=cfg.eval.sampling.steps,
    )
    artifact = wandb.Artifact(name=artifact_name or f"ensemble_eval_{run.id}", type="ensemble_eval")
    artifact.add_file(str(output_dir.joinpath("sampled_x.npy")))
    artifact.add_file(str(output_dir.joinpath("config.yaml")))
    artifact.add_file(str(plot_path))
    run.log_artifact(artifact)
    run.finish()


def log_metrics_to_wandb(
    cfg,
    cfg_to_save: dict,
    metrics: dict[str, float],
    output_dir: Path,
    run_name: str,
    group: str,
) -> None:
    try:
        import wandb
    except ModuleNotFoundError:
        print("wandb is not installed; skipping WandB logging")
        return

    wandb_cfg = cfg.logger.wandb
    if wandb_cfg.offline:
        os.environ["WANDB_MODE"] = "dryrun"

    run = wandb.init(
        project=wandb_cfg.project,
        name=run_name,
        group=group,
        dir=str(output_dir),
        config=cfg_to_save,
    )
    run.log(metrics, step=cfg.eval.sampling.steps)
    run.finish()


if __name__ == "__main__":
    main()
