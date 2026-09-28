r"""Utilities for sampling conditional trajectory ensembles."""

from pathlib import Path
from typing import Optional

import torch
from hydra.utils import to_absolute_path

import pdediff.guidance as guidance
import pdediff.rollout as rollout
from pdediff.sampler.utils import get_sampler
from pdediff.score import load_score


def _get_guidance_term(cfg):
    guidance_type = cfg.eval.guidance.type
    if guidance_type == "SDA":
        return guidance.SDA
    if guidance_type == "scaled":
        return guidance.ScaledGuidance
    if guidance_type == "scaled_indep":
        return guidance.ScaledIndepGuidance
    if guidance_type == "scaled_norm":
        return guidance.ScaledNormGuidance
    if guidance_type == "DPS":
        return guidance.DPS
    if guidance_type == "VideoDiff":
        return guidance.VideoDiff
    if guidance_type == "PGDM":
        return guidance.PGDM
    if guidance_type in ("MMPS", "moment_matching"):
        return guidance.MomentMatchingGuidance
    raise ValueError(f"{guidance_type} is not supported")


def _get_ar_variant(cfg) -> str:
    return cfg.eval.autoregressive.variant


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else Path(to_absolute_path(str(path)))


def _load_marginal_score(cfg):
    empirical_variance = cfg.eval.autoregressive.empirical_variance
    if getattr(empirical_variance, "score", None) != "conditional":
        return None

    ckpt_path = empirical_variance.path_to_marginal_score_ckpt
    if ckpt_path is None:
        raise ValueError(
            "eval.autoregressive.empirical_variance.path_to_marginal_score_ckpt "
            "is required when score='conditional'."
        )

    return load_score(_resolve_path(ckpt_path))


def _normalize_single_trajectory(x: torch.Tensor, cfg, name: str) -> torch.Tensor:
    state_ndim = len(tuple(cfg.data.state_shape))
    if x.ndim == state_ndim + 1:
        x = x.unsqueeze(0)
    elif x.ndim != state_ndim + 2:
        raise ValueError(
            f"{name} should have shape (T, *state_shape) or (1, T, *state_shape), got {tuple(x.shape)}"
        )

    if x.shape[0] != 1:
        raise ValueError(
            f"{name} should contain a single trajectory before ensemble replication, got batch size {x.shape[0]}"
        )

    return x


def _repeat_ensemble(x: torch.Tensor, n_samples: int) -> torch.Tensor:
    repeats = [n_samples] + [1] * (x.ndim - 1)
    return x.repeat(*repeats)


def _build_amortized_initial_conditions(
    y_ensemble: torch.Tensor,
    mask_ensemble: torch.Tensor,
    cfg,
) -> torch.Tensor:
    initial_values = y_ensemble[:, : cfg.window].clone()
    initial_mask = mask_ensemble[:, : cfg.window].clone()
    initial_mask[:, cfg.eval.forecast.conditioned_frame :] = 0.0
    trailing_shape = tuple(cfg.data.state_shape)[1:]

    initial_values = initial_values.reshape(initial_values.shape[0], -1, *trailing_shape)
    initial_mask = initial_mask.reshape(initial_mask.shape[0], -1, *trailing_shape)

    return torch.cat((initial_mask, initial_mask * initial_values), dim=1)


def _resolve_gamma_pair(cfg) -> tuple[float, float]:
    gamma_obs = cfg.eval.guidance.gamma
    gamma_ar = cfg.eval.guidance.gamma1
    return gamma_obs, gamma_ar


def _sample_aao_ensemble(score, cfg, y, mask, n_samples: int, sampler=None, ensemble_batch_size: Optional[int] = None):
    sampler = get_sampler(cfg) if sampler is None else sampler

    def A(x, _mask):
        return x * _mask

    conditional_sampler = rollout.ConditionalRolloutAllAtOnce(
        unconditional_score=score,
        state_shape=tuple(cfg.data.state_shape),
        A=A,
        likelihoods=[guidance.Gaussian],
        guidance=_get_guidance_term(cfg),
        likelihood_stds=[cfg.eval.guidance.std],
        steps=cfg.eval.sampling.steps,
        corrections=cfg.eval.sampling.corrections,
        tau=cfg.eval.sampling.tau,
        gammas=[cfg.eval.guidance.gamma],
        sampler=sampler,
        model_type=getattr(cfg, "model_type", "noise"),
    )
    single_y = _normalize_single_trajectory(y, cfg, "y")
    single_mask = _normalize_single_trajectory(mask, cfg, "mask")

    batch_size = ensemble_batch_size
    if batch_size is None:
        batch_size = cfg.eval.forecast.test_batch_size
    batch_size = min(batch_size, n_samples)

    all_samples = []
    for i_batch, batch_start in enumerate(range(0, n_samples, batch_size)):
        batch_n = min(batch_size, n_samples - batch_start)
        batch_y = _repeat_ensemble(single_y, batch_n)
        batch_mask = _repeat_ensemble(single_mask, batch_n)
        batch_samples = conditional_sampler.sample_traj(
            trajectory_length=single_y.shape[1],
            y_conditioning=batch_y,
            mask=batch_mask,
            seed=cfg.seed + i_batch,
            batch_shape=(batch_n,),
        )
        all_samples.append(batch_samples)

    return torch.cat(all_samples, dim=0)


def _sample_ar_ensemble(score, cfg, y, mask, n_samples: int, sampler=None):
    sampler = get_sampler(cfg) if sampler is None else sampler
    gamma_obs, gamma_ar = _resolve_gamma_pair(cfg)

    def A(x, _mask):
        return x * _mask

    conditional_sampler = rollout.ConditionalARRollout(
        unconditional_score=score,
        state_shape=tuple(cfg.data.state_shape),
        A=A,
        likelihoods=[guidance.Gaussian, guidance.Gaussian],
        guidance=_get_guidance_term(cfg),
        likelihood_stds=[cfg.eval.guidance.std, cfg.eval.guidance.std_past],
        markov_blanket_window_size=cfg.window,
        steps=cfg.eval.sampling.steps,
        corrections=cfg.eval.sampling.corrections,
        tau=cfg.eval.sampling.tau,
        gammas=[gamma_obs, gamma_ar],
        sampler=sampler,
        model_type=cfg.model_type,
        task=cfg.eval.task,
    )
    y_ensemble = _repeat_ensemble(_normalize_single_trajectory(y, cfg, "y"), n_samples)
    mask_ensemble = _repeat_ensemble(_normalize_single_trajectory(mask, cfg, "mask"), n_samples)

    return conditional_sampler.sample_traj(
        trajectory_length=y_ensemble.shape[1],
        y_conditioning=y_ensemble,
        mask=mask_ensemble,
        n_conditioned_frame=cfg.eval.forecast.conditioned_frame,
        predictive_horizon=cfg.eval.forecast.predictive_horizon,
        seed=cfg.seed,
        batch_shape=(n_samples,),
    )


def _sample_amortized_ensemble(score, cfg, y, mask, n_samples: int, sampler=None):
    sampler = get_sampler(cfg) if sampler is None else sampler
    conditional_sampler = rollout.AmortizedRollout(
        score=score,
        state_shape=tuple(cfg.data.state_shape),
        steps=cfg.eval.sampling.steps,
        corrections=cfg.eval.sampling.corrections,
        tau=cfg.eval.sampling.tau,
        conditioned_frame=cfg.eval.forecast.conditioned_frame,
        predictive_horizon=cfg.eval.forecast.predictive_horizon,
        sampler=sampler,
        likelihood=guidance.Gaussian,
        guidance=_get_guidance_term(cfg),
        likelihood_std=cfg.eval.guidance.std,
        gamma=cfg.eval.guidance.gamma,
    )
    y_ensemble = _repeat_ensemble(_normalize_single_trajectory(y, cfg, "y"), n_samples)
    mask_ensemble = _repeat_ensemble(_normalize_single_trajectory(mask, cfg, "mask"), n_samples)
    initial_conditions = _build_amortized_initial_conditions(y_ensemble, mask_ensemble, cfg)
    # hide the initial conditions from the observations, since they are already provided to the amortized sampler
    y_ensemble[:, : cfg.eval.forecast.conditioned_frame] = 0.0
    mask_ensemble[:, : cfg.eval.forecast.conditioned_frame] = 0.0
    return conditional_sampler.sample_traj(
        trajectory_length=y_ensemble.shape[1],
        seed=cfg.seed,
        batch_shape=(n_samples,),
        conditions=initial_conditions,
        obs=y_ensemble,
        obs_mask=mask_ensemble,
    )


def _sample_ar_empirical_variance_ensemble(score, cfg, y, mask, n_samples: int, sampler=None):
    gamma_obs, gamma_ar = _resolve_gamma_pair(cfg)
    empirical_variance = cfg.eval.autoregressive.empirical_variance
    marginal_score = _load_marginal_score(cfg)

    def A(x, _mask):
        return x * _mask

    conditional_sampler = rollout.ConditionalARRolloutEmpiricalVariance(
        unconditional_score=score,
        state_shape=tuple(cfg.data.state_shape),
        A=A,
        likelihoods=[guidance.Gaussian, guidance.MultivariateGaussian if empirical_variance.full_covariance else guidance.Gaussian],
        guidance=_get_guidance_term(cfg),
        likelihood_stds=[cfg.eval.guidance.std],
        markov_blanket_window_size=cfg.window,
        steps=cfg.eval.sampling.steps,
        corrections=cfg.eval.sampling.corrections,
        tau=cfg.eval.sampling.tau,
        gammas=[gamma_obs, gamma_ar],
        sampler=sampler,
        model_type=cfg.model_type,
        task=cfg.eval.task,
        min_ar_std=empirical_variance.min_ar_std,
        ar_cov_localization_bandwidth=empirical_variance.ar_cov_localization_bandwidth,
        temporal_band=empirical_variance.temporal_band,
        hybrid_ratio=empirical_variance.hybrid_ratio,
        static_std=empirical_variance.static_std,
        use_sample_based_ar_conditioning=empirical_variance.use_sample_based_ar_conditioning,
        marginal_score=marginal_score,
    )
    y_ensemble = _repeat_ensemble(_normalize_single_trajectory(y, cfg, "y"), n_samples)
    mask_ensemble = _repeat_ensemble(_normalize_single_trajectory(mask, cfg, "mask"), n_samples)

    return conditional_sampler.sample_traj(
        trajectory_length=y_ensemble.shape[1],
        y_conditioning=y_ensemble,
        mask=mask_ensemble,
        n_conditioned_frame=cfg.eval.forecast.conditioned_frame,
        predictive_horizon=cfg.eval.forecast.predictive_horizon,
        seed=cfg.seed,
        batch_shape=(n_samples,),
    )


def _sample_particle_filter_ensemble(score, cfg, y, mask, n_samples: int, sampler=None):
    sampler = get_sampler(cfg) if sampler is None else sampler
    gamma_obs, gamma_ar = _resolve_gamma_pair(cfg)
    proposal_type = "amortized" if cfg.amortized else "joint"

    def A(x, _mask):
        return x * _mask

    conditional_sampler = rollout.FullyAdaptedAuxiliaryParticleFilterRollout(
        score=score,
        state_shape=tuple(cfg.data.state_shape),
        A=A,
        likelihoods=[guidance.Gaussian, guidance.Gaussian],
        guidance=_get_guidance_term(cfg),
        likelihood_stds=[cfg.eval.guidance.std, cfg.eval.guidance.std_past],
        markov_blanket_window_size=cfg.window,
        steps=cfg.eval.sampling.steps,
        corrections=cfg.eval.sampling.corrections,
        tau=cfg.eval.sampling.tau,
        gammas=[gamma_obs, gamma_ar],
        sampler=sampler,
        model_type=cfg.model_type,
        task=cfg.eval.task,
        proposal_type=proposal_type,
        alpha_init=cfg.eval.autoregressive.particle_filter.alpha_init,
        alpha_min=cfg.eval.autoregressive.particle_filter.alpha_min,
        alpha_max=cfg.eval.autoregressive.particle_filter.alpha_max,
        neff_min=cfg.eval.autoregressive.particle_filter.neff_min,
        neff_max=cfg.eval.autoregressive.particle_filter.neff_max,
        alpha_search_steps=cfg.eval.autoregressive.particle_filter.alpha_search_steps,
    )
    y_ensemble = _repeat_ensemble(_normalize_single_trajectory(y, cfg, "y"), n_samples)
    mask_ensemble = _repeat_ensemble(_normalize_single_trajectory(mask, cfg, "mask"), n_samples)

    return conditional_sampler.sample_traj(
        trajectory_length=y_ensemble.shape[1],
        y_conditioning=y_ensemble,
        mask=mask_ensemble,
        n_conditioned_frame=cfg.eval.forecast.conditioned_frame,
        predictive_horizon=cfg.eval.forecast.predictive_horizon,
        seed=cfg.seed,
        batch_shape=(n_samples,),
    )


def sample_ensemble(score, cfg, y, mask, n_samples: Optional[int] = None, sampler=None):
    n_samples = cfg.eval.forecast.n_samples if n_samples is None else n_samples
    ar_variant = _get_ar_variant(cfg)

    if cfg.eval.rollout_type == "all_at_once":
        return _sample_aao_ensemble(
            score=score,
            cfg=cfg,
            y=y,
            mask=mask,
            n_samples=n_samples,
            sampler=sampler,
            ensemble_batch_size=1,
        )

    if cfg.eval.rollout_type != "autoregressive":
        raise ValueError(
            f"Unsupported rollout_type {cfg.eval.rollout_type}; expected all_at_once or autoregressive"
        )

    if ar_variant == "particle_filter":
        return _sample_particle_filter_ensemble(
            score=score,
            cfg=cfg,
            y=y,
            mask=mask,
            n_samples=n_samples,
            sampler=sampler,
        )

    if cfg.amortized:
        return _sample_amortized_ensemble(
            score=score,
            cfg=cfg,
            y=y,
            mask=mask,
            n_samples=n_samples,
            sampler=sampler,
        )

    if ar_variant == "ensemble_var":
        return _sample_ar_empirical_variance_ensemble(
            score=score,
            cfg=cfg,
            y=y,
            mask=mask,
            n_samples=n_samples,
            sampler=sampler,
        )

    return _sample_ar_ensemble(
        score=score,
        cfg=cfg,
        y=y,
        mask=mask,
        n_samples=n_samples,
        sampler=sampler,
    )


__all__ = ["sample_ensemble"]