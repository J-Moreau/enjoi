from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Union

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import Size, Tensor, nn
from torch.distributions import Normal
from torch.func import grad, grad_and_value, vmap

from pdediff.covariance import (
    LinearSolver,
    covariance_matvec,
    solve_covariance_system,
    solve_linear_system,
)
from pdediff.sde import VPSDE
from pdediff.utils.data_preprocessing import append_zeros


@dataclass
class SDESampleHistory:
    x0: Tensor
    x_t: Tensor
    times: Tensor
    covariance: Optional[Tensor] = None
    std: Optional[Tensor] = None
    guidance_history: Optional[Tensor] = None

    def cpu(self) -> "SDESampleHistory":
        return self.to("cpu")

    def detach(self) -> "SDESampleHistory":
        return SDESampleHistory(
            x0=self.x0.detach(),
            x_t=self.x_t.detach(),
            times=self.times.detach(),
            covariance=self.covariance.detach() if hasattr(self.covariance, "detach") else self.covariance,
            std=self.std.detach() if isinstance(self.std, Tensor) else self.std,
            guidance_history=self.guidance_history.detach() if isinstance(self.guidance_history, Tensor) else self.guidance_history,
        )

    def to(self, *args, **kwargs) -> "SDESampleHistory":
        return SDESampleHistory(
            x0=self.x0.to(*args, **kwargs),
            x_t=self.x_t.to(*args, **kwargs),
            times=self.times.to(*args, **kwargs),
            covariance=self.covariance.to(*args, **kwargs) if hasattr(self.covariance, "to") else self.covariance,
            std=self.std.to(*args, **kwargs) if isinstance(self.std, Tensor) else self.std,
            guidance_history=self.guidance_history.to(*args, **kwargs) if isinstance(self.guidance_history, Tensor) else self.guidance_history,
        )

######################################################
##################  Likelihood  ######################
######################################################


class Likelihood(ABC, nn.Module):
    """p(y|x_0) = p(y|A(x))"""

    def __init__(self, y: Tensor, A: Callable[[Tensor], Tensor], mask: Tensor = None):
        super().__init__()
        self.register_buffer("y", y)
        self.A = A
        self.mask = mask

    def err(self, x: Tensor) -> Tensor:
        if self.mask is not None:
            obs = self.A(x, self.mask.to(x.device))
        else:
            obs = self.A(x)

        assert obs.shape == self.y.shape, f"Obs shape {obs.shape} different to y shape {self.y.shape}"
        return self.y.to(x.device) - obs

    def set_observation(self, y: Tensor, mask: Tensor = None):
        self.y.copy_(y)
        if mask is not None:
            self.mask = mask



class Gaussian(Likelihood):
    """p(y|x_0) = N(y|A(x), std^2)"""

    def __init__(self, y: Tensor, A: Callable[[Tensor], Tensor], std: float = 1.0, mask: Tensor = None):
        super().__init__(y, A, mask)
        self.std = std

    def sample(self, shape=()) -> Tensor:
        return Normal(self.y, self.std).rsample(shape)


class MultivariateGaussian(Likelihood):
    """p(y|x_0) = N(y|A(x), cov)"""

    def __init__(
        self,
        y: Tensor,
        A: Callable[[Tensor], Tensor],
        cov: Tensor,
        mask: Tensor = None,
    ):
        super().__init__(y, A, mask)
        if isinstance(cov, Tensor):
            self.register_buffer("cov", cov)
        else:
            self.cov = cov


def _likelihood_precision_residual(
    likelihood: Likelihood,
    err: Tensor,
    nugget: Union[float, Tensor] = 0.0,
    linear_solver: LinearSolver = "cg",
    max_iter: int = 20,
    tol: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    if isinstance(likelihood, MultivariateGaussian):
        err_flat = err.reshape(err.shape[0], -1)
        precision_residual = solve_covariance_system(
            likelihood.cov,
            err_flat,
            err.shape,
            nugget=nugget,
            solver=linear_solver,
            max_iter=max_iter,
            tol=tol,
        ).reshape_as(err)
        log_prob = -(err * precision_residual).sum() / 2
        return precision_residual, log_prob

    _std = likelihood.std.to(err.device) if isinstance(likelihood.std, Tensor) else likelihood.std
    var = _std ** 2 + nugget
    precision_residual = err / var
    log_prob = -(err * precision_residual).sum() / 2
    return precision_residual, log_prob


def _likelihood_observation_score(
    likelihood: Likelihood,
    err: Tensor,
    x: Tensor,
    nugget: Union[float, Tensor] = 0.0,
    linear_solver: LinearSolver = "cg",
    max_iter: int = 20,
    tol: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    precision_residual, log_prob = _likelihood_precision_residual(
        likelihood,
        err,
        nugget=nugget,
        linear_solver=linear_solver,
        max_iter=max_iter,
        tol=tol,
    )
    observation = likelihood.y.to(device=err.device, dtype=err.dtype) - err
    score, = torch.autograd.grad(
        (observation * precision_residual.detach()).sum(),
        x,
        retain_graph=True,
    )
    return score, log_prob


######################################################
################## Guidance term #####################
######################################################


class GuidedScore(ABC, nn.Module):
    def __init__(self, sde: VPSDE, likelihoods: Likelihood):
        super().__init__()
        self.likelihoods = likelihoods
        self.sde = sde

    def get_sde(self, shape) -> VPSDE:
        return self.sde.__class__(self, shape=shape)

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass


class SDA(GuidedScore):
    def __init__(
        self,
        sde: VPSDE,
        likelihoods: List[Likelihood],
        gammas: List[float],
    ):
        super().__init__(sde, likelihoods)
        self.gammas = gammas
        assert len(likelihoods) == len(gammas), (
            f"Number of specified likelihoods {len(likelihoods)} is different "
            f"from the number of specified gammas {len(gammas)}. Please specify "
            "a guidance strength for each likelihood."
        )
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)

        with torch.enable_grad():
            x = x.detach().requires_grad_(True)

            eps = self.sde.noise_prediction_fn(x, t)
            x_ = (x - sigma * eps) / mu

            log_probs = []
            for likelihood, gamma in zip(self.likelihoods, self.gammas):
                err = likelihood.err(x_)
                _std = likelihood.std.to(x.device) if isinstance(likelihood.std, Tensor) else likelihood.std
                var = _std ** 2 + gamma * (sigma / mu) ** 2
                log_probs.append(-(err ** 2 / var).sum() / 2)

        for log_prob in log_probs:
            s, = torch.autograd.grad(log_prob, x, retain_graph=True)
            eps = eps - sigma * s

        return eps
    
class ScaledGuidance(GuidedScore):
    def __init__(
        self,
        sde: VPSDE,
        likelihoods: List[Likelihood],
        gammas: List[float],
        linear_solver: LinearSolver = "cg",
        max_iter: int = 3,
        tol: float = 1e-3,
    ):
        """
        Guided score that normalizes the combined guidance signal by its RMS.
        """
        super().__init__(sde, likelihoods)
        self.gammas = gammas
        self.linear_solver = linear_solver
        self.max_iter = max_iter
        self.tol = tol
        self.guidance_history: List[Tensor] = []
        assert len(likelihoods) == len(gammas), (
            f"Number of specified likelihoods {len(likelihoods)} is different "
            f"from the number of specified gammas {len(gammas)}. Please specify "
            "a guidance strength for each likelihood."
        )

    def _score_term(
        self,
        likelihood: Likelihood,
        err: Tensor,
        x: Tensor,
        nugget: Union[float, Tensor] = 0.0,
    ) -> tuple[Tensor, Tensor]:
        return _likelihood_observation_score(
            likelihood,
            err,
            x,
            nugget=nugget,
            linear_solver=self.linear_solver,
            max_iter=self.max_iter,
            tol=self.tol,
        )

    def _guidance_norm(self, guidance_signal: Tensor) -> Tensor:
        dims = tuple(range(1, guidance_signal.ndim))
        return guidance_signal.square().mean(dim=dims, keepdim=True).sqrt().clamp_min(torch.finfo(guidance_signal.dtype).tiny)

    def _guidance_signal(self, score_terms: List[Tensor]) -> Tensor:
        guidance_signal = sum(score_term for score_term in score_terms)
        return self.gammas[0] * guidance_signal / self._guidance_norm(guidance_signal)

    def eps_and_guidance(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)

        with torch.enable_grad():
            x = x.detach().requires_grad_(True)

            eps = self.sde.noise_prediction_fn(x, t)
            x_ = (x - sigma * eps) / mu

            score_terms = []
            for likelihood in self.likelihoods:
                err = likelihood.err(x_)
                score_term, _ = self._score_term(likelihood, err, x)
                score_terms.append(score_term)

            guidance_signal = self._guidance_signal(score_terms)

        return eps, guidance_signal

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        eps, guidance_signal = self.eps_and_guidance(x, t)
        eps = eps - self.sde.sigma(t) * guidance_signal

        self.guidance_history.append(guidance_signal.detach().clone())
        return eps


class ScaledIndepGuidance(ScaledGuidance):
    def _guidance_signal(self, score_terms: List[Tensor]) -> Tensor:
        return sum(
            gamma * score_term / self._guidance_norm(score_term)
            for gamma, score_term in zip(self.gammas, score_terms)
        )


class ScaledNormGuidance(ScaledGuidance):
    """
    Guidance terms are scaled by the error norm
    """
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)

        with torch.enable_grad():
            x = x.detach().requires_grad_(True)

            eps = self.sde.noise_prediction_fn(x, t)
            x_ = (x - sigma * eps) / mu

            score_terms = []
            errors = []
            for likelihood in self.likelihoods:
                err = likelihood.err(x_)
                score_term, log_prob = self._score_term(likelihood, err, x)
                score_terms.append(score_term)
                errors.append(log_prob)

        guidance_signal = self.gammas[0] * sum(score_terms) / (-sum(errors)).sqrt() * mu / sigma
        eps = eps - sigma * guidance_signal

        self.guidance_history.append(guidance_signal.detach().clone())
        return eps

class ConditionalGuidance(GuidedScore):
    r"""Conditional guidance from joint and marginal reconstructions.

    Denote the conditioned past by ``k`` and the predicted future by ``kp``.
    This guidance uses three guidance terms: two evaluated on the joint
    reconstruction ``(xkt, xkpt)`` and one evaluated on the marginal past
    reconstruction ``xkt``. The past likelihood is reused for the joint-past
    and marginal-past guidance terms.

    The target factorization is

    ``p(xkkpt|yonek,ykp) = p(\ykp|\xkkpt)p(\xkpt|\xkt)p(\yonek|\xkkpt) p(\xkt|\yonek)``.

    In score/noise form the guided conditional score is

    ``grad_xkt p(xkkpt|y) = s(xkt,xkpt)[k] - s(xkt) + guid(ykp|xkt,xkpt)[k] + guid(yk|xkt,xkpt)[k] + guid(yk|xkt)``

    and

    ``grad_xkpt p(xkkpt|y) = s(xkt,xkpt)[kp] + guid(ykp|xkt,xkpt)[kp] + guid(yk|xkt,xkpt)[kp]``.
    """

    def __init__(
        self,
        sde: VPSDE,
        past_likelihood: Likelihood,
        obs_likelihood: Likelihood,
        marginal_score: nn.Module,
        n_conditioned_frame: int,
        gammas: List[float],
        model_type: str = "noise",
        linear_solver: LinearSolver = "cg",
        max_iter: int = 20,
        tol: float = 1e-8,
    ):
        likelihoods = [obs_likelihood, past_likelihood]
        super().__init__(sde, likelihoods)
        self.gammas = gammas
        self.marginal_sde = VPSDE(marginal_score, model_type=model_type)
        self.n_conditioned_frame = n_conditioned_frame
        self.linear_solver = linear_solver
        self.max_iter = max_iter
        self.tol = tol
        assert len(gammas) == 2, (
            f"Conditional guidance expects two gammas, one for future observations "
            f"and one for past conditioning. Got {len(gammas)}."
        )

    def _score_term(
        self,
        likelihood: Likelihood,
        err: Tensor,
        x: Tensor,
        gamma: float,
        mu: Tensor,
        sigma: Tensor,
    ) -> tuple[Tensor, Tensor]:
        gamma_term = gamma * (sigma / mu) ** 2
        return _likelihood_observation_score(
            likelihood,
            err,
            x,
            nugget=gamma_term,
            linear_solver=self.linear_solver,
            max_iter=self.max_iter,
            tol=self.tol,
        )

    def _log_prob_gamma(self, gamma: float) -> float:
        return gamma

    def _combine_guidance_terms(self, score_terms: List[Tensor], gamma: float) -> Tensor:
        return sum(score_term for score_term in score_terms)

    def _scale_guidance_terms(
        self,
        past_guidance: Tensor,
        future_guidance: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return past_guidance, future_guidance

    def _store_guidance_history(
        self,
        x: Tensor,
        past_guidance: Tensor,
        future_guidance: Tensor,
    ) -> None:
        pass

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)
        n = self.n_conditioned_frame

        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            x_cond = x[:, :n, ...]

            eps_joint = self.sde.noise_prediction_fn(x, t)
            eps_marginal = self.marginal_sde.noise_prediction_fn(x_cond, t)

            x_denoised = (x - sigma * eps_joint) / mu
            x_cond_denoised = (x_cond - sigma * eps_marginal) / mu

            obs_likelihood, past_likelihood = self.likelihoods
            s_obs, _ = self._score_term(
                obs_likelihood,
                obs_likelihood.err(x_denoised),
                x,
                self._log_prob_gamma(self.gammas[0]),
                mu,
                sigma,
            )
            s_joint_past, _ = self._score_term(
                past_likelihood,
                past_likelihood.err(x_denoised),
                x,
                self._log_prob_gamma(self.gammas[1]),
                mu,
                sigma,
            )
            s_marginal_past, _ = self._score_term(
                past_likelihood,
                past_likelihood.err(x_cond_denoised),
                x_cond,
                self._log_prob_gamma(self.gammas[1]),
                mu,
                sigma,
            )

        past_guidance = self._combine_guidance_terms(
            [s_obs[:, :n, ...], s_joint_past[:, :n, ...], s_marginal_past],
            self.gammas[1],
        )
        future_guidance = self._combine_guidance_terms(
            [s_obs[:, n:, ...], s_joint_past[:, n:, ...]],
            self.gammas[0],
        )
        past_guidance, future_guidance = self._scale_guidance_terms(past_guidance, future_guidance)

        eps = eps_joint.clone()
        eps[:, :n, ...] = eps_joint[:, :n, ...] - eps_marginal - sigma * past_guidance
        eps[:, n:, ...] = eps_joint[:, n:, ...] - sigma * future_guidance

        self._store_guidance_history(x, past_guidance, future_guidance)
        return eps


class ConditionalSDA(ConditionalGuidance):
    pass


class ConditionalScaledGuidance(ConditionalGuidance):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.guidance_history: List[Tensor] = []

    def _log_prob_gamma(self, gamma: float) -> float:
        return 0.0

    def _combine_guidance_terms(self, score_terms: List[Tensor], gamma: float) -> Tensor:
        return sum(score_term for score_term in score_terms)

    def _scale_guidance_terms(
        self,
        past_guidance: Tensor,
        future_guidance: Tensor,
    ) -> tuple[Tensor, Tensor]:
        total_guidance = torch.cat(
            [
                past_guidance.reshape(past_guidance.shape[0], -1),
                future_guidance.reshape(future_guidance.shape[0], -1),
            ],
            dim=1,
        )
        norm = total_guidance.square().mean(dim=1).sqrt().clamp_min(torch.finfo(total_guidance.dtype).tiny)
        norm = norm.reshape(norm.shape[0], *((1,) * (past_guidance.ndim - 1)))
        return self.gammas[1] * past_guidance / norm, self.gammas[0] * future_guidance / norm

    def _store_guidance_history(
        self,
        x: Tensor,
        past_guidance: Tensor,
        future_guidance: Tensor,
    ) -> None:
        n = self.n_conditioned_frame
        guidance_signal = torch.zeros_like(x)
        guidance_signal[:, :n, ...] = past_guidance
        guidance_signal[:, n:, ...] = future_guidance
        self.guidance_history.append(guidance_signal.detach().clone())

class DPS(GuidedScore):
    def __init__(
        self,
        sde: VPSDE,
        likelihoods: List[Likelihood],
        gammas: List[float],
    ):
        super().__init__(sde, likelihoods)
        self.gammas = gammas
        assert len(likelihoods) == len(gammas), (
            f"Number of specified likelihoods {len(likelihoods)} is different "
            f"from the number of specified gammas {len(gammas)}. Please specify "
            "a guidance strength for each likelihood."
        )
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)

        with torch.enable_grad():
            x = x.detach().requires_grad_(True)

            eps = self.sde.noise_prediction_fn(x, t)
            x_ = (x - sigma * eps) / mu

            errs = []
            for likelihood in zip(self.likelihoods):
                errs.append((likelihood.err(x_)).square().sum())

        for err, gamma in zip(errs, self.gammas):
            s, = torch.autograd.grad(err, x, retain_graph=True)
            s = -s / gamma / err.sqrt()
            eps = eps - sigma * s

        return eps

class VideoDiff(GuidedScore):
    def __init__(
        self,
        sde: VPSDE,
        likelihoods: List[Likelihood],
        gammas: List[float],
    ):
        super().__init__(sde, likelihoods)
        self.gammas = gammas
        assert len(likelihoods) == len(gammas), (
            f"Number of specified likelihoods {len(likelihoods)} is different "
            f"from the number of specified gammas {len(gammas)}. Please specify "
            "a guidance strength for each likelihood."
        )
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)

        with torch.enable_grad():
            x = x.detach().requires_grad_(True)

            eps = self.sde.noise_prediction_fn(x, t)
            x_ = (x - sigma * eps) / mu

            log_probs = []
            for likelihood, gamma in zip(self.likelihoods, self.gammas):
                err = likelihood.err(x_)
                var = gamma * (sigma / mu) ** 2
                log_probs.append(-(err ** 2 / var).sum() / 2)

        for log_prob in log_probs:
            s, = torch.autograd.grad(log_prob, x, retain_graph=True)
            eps = eps - sigma * s

        return eps

class PGDM(GuidedScore):
    def __init__(
        self,
        sde: VPSDE,
        likelihoods: List[Likelihood],
        gammas: List[float],
    ):
        super().__init__(sde, likelihoods)
        self.gammas = gammas
        assert len(likelihoods) == len(gammas), (
            f"Number of specified likelihoods {len(likelihoods)} is different "
            f"from the number of specified gammas {len(gammas)}. Please specify "
            "a guidance strength for each likelihood."
        )
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)

        with torch.enable_grad():
            x = x.detach().requires_grad_(True)

            eps = self.sde.noise_prediction_fn(x, t)
            x_ = (x - sigma * eps) / mu

            log_probs = []
            for likelihood, gamma in zip(self.likelihoods, self.gammas):
                err = likelihood.err(x_)
                var = likelihood.std ** 2 + (sigma) ** 2 / (mu**2 + sigma**2)
                log_probs.append(-(err ** 2 / var).sum() / 2)

        for log_prob in log_probs:
            s, = torch.autograd.grad(log_prob, x, retain_graph=True)
            eps = eps - sigma * s

        return eps


class MomentMatchingGuidance(GuidedScore):
    r"""Moment-matching posterior sampling guidance.

    This is the PyTorch analogue of the MMPS correction used in Rozet et al. and
    the FA-APF ConditionalDenoiser: linearize the denoiser and observation map at
    the current noisy state, solve in observation space, and correct the denoised
    estimate by ``sigma_t^2 J_f^T H^T v``.

    Translated with ChatGPT from the original JAX implementation
    """

    def __init__(
        self,
        sde: VPSDE,
        likelihoods: List[Likelihood],
        gammas: List[float],
        max_iter: int = 1,
        tol: float = 1e-3,
        linear_solver: Literal["bicgstab", "cg"] = "cg",
    ):
        super().__init__(sde, likelihoods)
        self.gammas = gammas
        self.max_iter = max_iter
        self.tol = tol
        self.linear_solver = linear_solver
        self.guidance_history: List[Tensor] = []
        assert len(likelihoods) == len(gammas), (
            f"Number of specified likelihoods {len(likelihoods)} is different "
            f"from the number of specified gammas {len(gammas)}. Please specify "
            "a guidance strength for each likelihood."
        )
        
    def _expand_to(self, value: Tensor, target: Tensor) -> Tensor:
        while value.ndim < target.ndim:
            value = value.unsqueeze(-1)
        return value

    def _observation(self, likelihood: Likelihood, denoised: Tensor) -> Tensor:
        if likelihood.mask is not None:
            return likelihood.A(denoised, likelihood.mask.to(denoised.device))
        return likelihood.A(denoised)

    def _covariance_matvec(self, likelihood: Likelihood, vector: Tensor, observation_shape: torch.Size) -> Tensor:
        if isinstance(likelihood, MultivariateGaussian):
            return covariance_matvec(likelihood.cov, vector, observation_shape)

        std = likelihood.std.to(vector.device) if isinstance(likelihood.std, Tensor) else likelihood.std
        if isinstance(std, Tensor):
            variance = std.to(dtype=vector.dtype) ** 2
            return (vector.reshape(observation_shape) * variance).reshape_as(vector)
        return vector * (std ** 2)

    def _solve_linear_system(self, matvec: Callable[[Tensor], Tensor], rhs: Tensor) -> Tensor:
        return solve_linear_system(
            matvec,
            rhs,
            solver=self.linear_solver,
            max_iter=self.max_iter,
            tol=self.tol,
        )

    def _likelihood_score(
        self,
        x: Tensor,
        t: Tensor,
        likelihood: Likelihood,
        mu: Tensor,
        sigma: Tensor,
    ) -> Tensor:
        def denoise(noisy: Tensor) -> Tensor:
            eps = self.sde.noise_prediction_fn(noisy, t)
            return (noisy - sigma * eps) / mu

        x0 = denoise(x)
        yhat = self._observation(likelihood, x0)
        residual = likelihood.y.to(x.device, dtype=x.dtype) - yhat
        residual_flat = residual.reshape(residual.shape[0], -1)

        def observation_vjp(vector_flat: Tensor) -> Tensor:
            vector = vector_flat.reshape_as(yhat).detach()
            gradient, = torch.autograd.grad(
                (yhat * vector).sum(),
                x,
                retain_graph=True,
            )
            return gradient

        def observation_jvp(state_tangent: Tensor) -> Tensor:
            _, jacobian_vector_product = torch.autograd.functional.jvp(
                lambda state: self._observation(likelihood, state),
                (x0,),
                (state_tangent,),
                create_graph=False,
                strict=False,
            )
            return jacobian_vector_product

        def matvec(vector_flat: Tensor) -> Tensor:
            vector_jacobian_product = observation_vjp(vector_flat)
            covariance_part = self._covariance_matvec(likelihood, vector_flat, yhat.shape)
            diffusion_part = (sigma**2/mu * observation_jvp(vector_jacobian_product)).reshape_as(vector_flat)
            return covariance_part + diffusion_part

        observation_space_solution = self._solve_linear_system(matvec, residual_flat)
        return observation_vjp(observation_space_solution)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)

        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            eps = self.sde.noise_prediction_fn(x, t)
            denoised = (x - sigma * eps) / mu

            correction = torch.zeros_like(x)
            for likelihood, gamma in zip(self.likelihoods, self.gammas):
                correction = correction + gamma * self._likelihood_score(x, t, likelihood, mu, sigma)

        corrected_denoised = denoised + (sigma ** 2) * correction
        corrected_eps = (x - mu * corrected_denoised) / sigma

        self.guidance_history.append(correction.detach().clone())
        return corrected_eps

    
class AlternatingVPSDE(VPSDE):
    """
    Alternates between two noise estimators, one for the integration in time and one for the corrector step.
    """
    def __init__(
        self,
        net: nn.Module,
        likelihoods: List[nn.Module],
        gammas: List[float],
        n_conditioned_frame: int,
        shape: Size = (),
        alpha: str = "cos",
        eta: float = 1e-3,
        model_type: str = "noise",
    ):
        super().__init__(net, shape, alpha, eta, model_type)
        self.likelihoods = likelihoods
        self.gammas = gammas
        self.n_conditioned_frame = n_conditioned_frame


    # def _log_prob(self, likelihood, gamma, err, x, mu, sigma):
    #     gamma_term = gamma * (sigma / mu) ** 2
    #     _std = likelihood.std.to(x.device) if isinstance(likelihood.std, Tensor) else likelihood.std
    #     var = _std ** 2 + gamma_term
    #     return -(err ** 2 / var).sum() / 2

    def _log_prob(self, likelihood: Likelihood, gamma: float, err: Tensor, x: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
        gamma_term = gamma * (sigma / mu) ** 2

        if isinstance(likelihood, MultivariateGaussian):
            cov = likelihood.cov.to(x.device) if isinstance(likelihood.cov, Tensor) else likelihood.cov
            if not isinstance(cov, Tensor):
                cov = torch.as_tensor(cov, dtype=x.dtype, device=x.device)
            else:
                cov = cov.to(dtype=x.dtype)

            if cov.ndim >= 2 and cov.shape[-2:] == (err[0].numel(), err[0].numel()):
                err_flat = err.reshape(err.shape[0], -1)
                identity = torch.eye(
                    err_flat.shape[-1],
                    dtype=x.dtype,
                    device=x.device,
                )
                C = cov + gamma_term * identity
                solution = torch.linalg.solve(C, err_flat.unsqueeze(-1)).squeeze(-1)
                norm = (err_flat * solution).sum()
            else:
                C = cov + gamma_term
                norm = (err ** 2 / C).sum()

            return -norm / 2

        _std = likelihood.std.to(x.device) if isinstance(likelihood.std, Tensor) else likelihood.std
        var = _std ** 2 + gamma_term
        return -(err ** 2 / var).sum() / 2

    def eps_predict_scaled(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Past : (Obs Guidance + Past Guidance)_scaled + Score
        Future : Obs Guidance + Past Guidance + Score
        """
        mu, sigma = self.mu(t), self.sigma(t)
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)

            eps = self.noise_prediction_fn(x, t)
            x_ = (x - sigma * eps) / mu

            errors = []
            for likelihood, gamma in zip(self.likelihoods, self.gammas):
                err = likelihood.err(x_)
                # errors.append(self._error(likelihood, gamma, err, x, mu, sigma))
                errors.append(-(err ** 2).sum() / 2)
            
        observations_error, past_error = errors
        obs_likelihood, past_likelihood = self.likelihoods
        past_gamma, future_gamma = self.gammas
        s_obs, = torch.autograd.grad(observations_error, x, retain_graph=True)
        s_past, = torch.autograd.grad(past_error, x, retain_graph=True)

        # scaled update
        obs_std = obs_likelihood.std.to(x.device) if isinstance(obs_likelihood.std, Tensor) else obs_likelihood.std
        past_std = past_likelihood.std.to(x.device) if isinstance(past_likelihood.std, Tensor) else past_likelihood.std
        guidance = s_obs[:, :]/obs_std**2 + s_past[:, :]/past_std**2
        guidance_norm = guidance.square().mean(dim=(1,2,3), keepdim=True).sqrt()
        guidance_stepsize = self.gammas[0]
        eps[:,:] = eps[:,:] - sigma * guidance_stepsize * guidance / guidance_norm

        # unscaled update on x_{k+1} (future)
        # eps[:,n:] = eps[:,n:] - sigma * s_obs[:,n:]/ (obs_std**2 + future_gamma*((sigma / mu) ** 2))
        # eps[:,n:] = eps[:,n:] - sigma * s_past[:,n:]/ (past_std**2 + past_gamma*((sigma / mu) ** 2))

        return eps

    def eps_correct(self, x: Tensor, t: Tensor) -> Tensor:
        return self.eps_predict_scaled(x, t)

    def _guidance_uncertainty(self) -> tuple[Optional[Tensor], Optional[Tensor]]:
        covariance = None
        std = None
        likelihood = self.likelihoods[-1]
        if isinstance(likelihood, MultivariateGaussian):
            covariance = likelihood.cov
        elif isinstance(likelihood, Gaussian):
            std = likelihood.std

        covariance = covariance.detach() if hasattr(covariance, "detach") else covariance
        std = std.detach() if isinstance(std, Tensor) else std
        return covariance, std
    
    def sample(
        self,
        shape: Size = (),
        steps: int = 64,
        corrections: int = 0,
        tau: float = 1.0,
    ) -> SDESampleHistory:
        r"""Samples from :math:`p(x(0))`.

        Arguments:
            shape: The batch shape.
            steps: The number of discrete time steps.
            corrections: The number of Langevin corrections per time steps.
            tau: The amplitude of Langevin steps.
        """
        x = torch.randn(shape + self.shape).to(self.device).cuda()
        x = x.reshape(-1, *self.shape)

        time = torch.linspace(1, 0, steps + 1).to(self.device).cuda()
        dt = 1 / steps

        with torch.no_grad():
            x_t_history = [x.detach().clone()]
            x0_history = []
            for t in (time[:-1]):
                # Predictor
                r = self.mu(t - dt) / self.mu(t)
                eps_predict = self.eps_predict_scaled(x, t)
                x0_history.append(((x - self.sigma(t) * eps_predict)/self.mu(t)).detach().clone())
                x = r * x + (self.sigma(t - dt) - r * self.sigma(t)) * eps_predict

                # Corrector
                for _ in range(corrections):
                    eps = torch.randn_like(x)
                    # NOTE: since we are parametrizing eps(x(t),t) = - sigma(t) * score(x(t),t)
                    # if we need to use score(x(t),t) we have to always compute
                    # score(x(t),t) = - eps(x(t),t) / sigma(t)
                    s = -self.eps_correct(x, t - dt) / self.sigma(t - dt)
                    delta = tau / s.square().mean(dim=self.dims, keepdim=True)

                    x = x + delta * s + torch.sqrt(2 * delta) * eps

                x_t_history.append(x.detach().clone())

        covariance, std = self._guidance_uncertainty()
        return SDESampleHistory(
            x0=torch.stack(x0_history).reshape((steps,) + shape + self.shape),
            x_t=torch.stack(x_t_history).reshape((steps + 1,) + shape + self.shape),
            times=time.detach().clone(),
            covariance=covariance,
            std=std,
        ).detach()