r"""Fast Sampling of Diffusion Models with Exponential Integrator"""

import sys

sys.path.append("../")
from typing import Optional, Tuple

import torch
from torch import Size, Tensor
from tqdm import tqdm

from pdediff.guidance import (
    DPS,
    SDA,
    Gaussian,
    MultivariateGaussian,
    ScaledGuidance,
    SDESampleHistory,
)
from pdediff.sampler.helper import expand_dims
from pdediff.sampler.sampler import Sampler
from pdediff.sde import VPSDE


class ExpInt(Sampler):
    def __init__(
            self,
            steps: int,
            corrections: int,
            tau: float,
            skip_type: str = "linear",
            correcting_x0_fn = None,
            thresholding_max_val: float = 1.,
            dynamic_thresholding_ratio: float = 0.995,
            denoise_to_zero = False,
    ):
        super().__init__(steps, corrections)
        self.tau = tau
        self.skip_type = skip_type
        if skip_type == "t":
            self.sampling_eps = 1e-3
            self.sampling_T = 1 - 1e-3
        elif skip_type == "linear":
            self.sampling_eps = 1e-3
            self.sampling_T = 1
        elif skip_type == "logSNR":
            self.sampling_eps = 1e-3
            self.sampling_T = 1
        else:
            raise ValueError("Unsupported skip_type {}, need to be 'logSNR' or 'linear' or 't'".format(skip_type))
        if correcting_x0_fn == "dynamic_thresholding":
            self.correcting_x0_fn = self.dynamic_thresholding_fn
        elif correcting_x0_fn == "None":
            self.correcting_x0_fn = None
        else:
            self.correcting_x0_fn = correcting_x0_fn
        self.thresholding_max_val = thresholding_max_val
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio 
        self.denoise_to_zero = denoise_to_zero

    def get_time_steps(self, sde, skip_type, t_T, t_0, N):
        """Compute the intermediate time steps for sampling.

        Args:
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps.
                - 'linear': uniform time for the time steps. (**Recommended for high-resolutional data**.)
                - 't': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            N: A `int`. The total number of the spacing of the time steps.
            device: A torch device.
        Returns:
            A pytorch tensor of the time steps, with the shape (N + 1,).
        """
        if skip_type == 'logSNR':
            lambda_T = sde.marginal_lambda(torch.tensor(t_T).to(sde.device))
            lambda_0 = sde.marginal_lambda(torch.tensor(t_0).to(sde.device))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(sde.device)
            return sde.inverse_lambda(logSNR_steps)
        elif skip_type == 'linear':
            return torch.linspace(t_T, t_0, N + 1).to(sde.device)
        elif skip_type == 't':
            t_order = 2
            t = torch.linspace(t_T ** (1. / t_order), t_0 ** (1. / t_order), N + 1).pow(t_order).to(sde.device)
            return t
        else:
            raise ValueError("Unsupported skip_type {}, need to be 'logSNR' or 'linear' or 't'".format(skip_type))

    def dynamic_thresholding_fn(self, x0, t):
        """
        The dynamic thresholding method.
        """
        dims = x0.dim()
        p = self.dynamic_thresholding_ratio
        s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
        s = expand_dims(torch.maximum(s, self.thresholding_max_val * torch.ones_like(s).to(s.device)), dims)
        x0 = torch.clamp(x0, -s, s) / s
        return x0
    
    def denoise_to_zero_fn(self, sde: VPSDE, x, s):
        """
        Denoise at the final step, which is equivalent to solve the ODE from lambda_s to infty by first-order discretization.
        """
        x0 = sde.data_prediction_fn(x, s)
        if self.correcting_x0_fn is not None:
            return self.correcting_x0_fn(x0, s)
        else:
            return x0
        
    def sample(
            self,
            sde: VPSDE,
            shape: Size = (),
    ) -> Tensor:
        r"""Samples from :math:`p(x(0))`.

        Arguments:
            shape: The batch shape.
            steps: The number of discrete time steps.
            corrections: The number of Langevin corrections per time steps.
            tau: The amplitude of Langevin steps.
        """
        x = torch.randn(shape + sde.shape).to(sde.device)
        x = x.reshape(-1, *sde.shape)

        # time = get_rev_ts_th(sde, self.steps, self.ts_order, sampling_eps = self.sampling_eps, sampling_T = self.sampling_T, ts_phase = self.ts_phase)
        time = self.get_time_steps(sde, skip_type=self.skip_type, t_T=self.sampling_T, t_0=self.sampling_eps,
                                   N=self.steps)
        dts = time[:-1] - time[1:]

        with torch.no_grad():
            for dt, t in tqdm(zip(dts, (time[:-1])), total=len(dts)):
                # for t in time[:-1]:
                # Predictor
                r = sde.mu(t - dt) / sde.mu(t)
                noise_pred = sde.noise_prediction_fn(x, t)
                x = r * x + (sde.sigma(t - dt) - r * sde.sigma(t)) * noise_pred
                # Corrector
                for _ in range(self.corrections):
                    eps = torch.randn_like(x)
                    # NOTE: since we are parametrizing eps(x(t),t) = - sigma(t) * score(x(t),t)
                    # if we need to use score(x(t),t) we have to always compute
                    # score(x(t),t) = - eps(x(t),t) / sigma(t)
                    noise_pred = sde.noise_prediction_fn(x, t - dt)
                    s = -noise_pred / sde.sigma(t - dt)
                    delta = self.tau / s.square().mean(dim=sde.dims, keepdim=True)

                    x = x + delta * s + torch.sqrt(2 * delta) * eps

            if self.denoise_to_zero:
                t = torch.ones((1,)).to(sde.device) * self.sampling_eps
                x = self.denoise_to_zero_fn(sde, x, t)
        return x.reshape(shape + sde.shape)

class DeterministicExpInt(ExpInt):
    def sample(
            self,
            sde: VPSDE,
            shape: Size = (),
    ) -> Tensor:
        r"""Samples from :math:`p(x(0))`.

        Arguments:
            shape: The batch shape.
            steps: The number of discrete time steps.
            corrections: The number of Langevin corrections per time steps.
            tau: The amplitude of Langevin steps.
        """
        x = torch.randn(shape + sde.shape).to(sde.device)
        x = x.reshape(-1, *sde.shape)

        # time = get_rev_ts_th(sde, self.steps, self.ts_order, sampling_eps = self.sampling_eps, sampling_T = self.sampling_T, ts_phase = self.ts_phase)
        time = self.get_time_steps(sde, skip_type=self.skip_type, t_T=self.sampling_T, t_0=self.sampling_eps,
                                   N=self.steps)
        dts = time[:-1] - time[1:]

        with torch.no_grad():
            for dt, t in tqdm(zip(dts, (time[:-1])), total=len(dts)):
                # for t in time[:-1]:
                # Predictor
                r = sde.mu(t - dt) / sde.mu(t)
                noise_pred = sde.noise_prediction_fn(x, t)
                x = r * x + (sde.sigma(t - dt) - r * sde.sigma(t)) * noise_pred
                # Corrector
                for _ in range(self.corrections):
                    # NOTE: since we are parametrizing eps(x(t),t) = - sigma(t) * score(x(t),t)
                    # if we need to use score(x(t),t) we have to always compute
                    # score(x(t),t) = - eps(x(t),t) / sigma(t)
                    noise_pred = sde.noise_prediction_fn(x, t - dt)
                    s = -noise_pred / sde.sigma(t - dt)
                    delta = self.tau / s.square().mean(dim=sde.dims, keepdim=True)
                    x = x + delta * s 

            if self.denoise_to_zero:
                t = torch.ones((1,)).to(sde.device) * self.sampling_eps
                x = self.denoise_to_zero_fn(sde, x, t)
        return x.reshape(shape + sde.shape)

class DSGExpInt(ExpInt):
    r"""
    Diffusion Spherical Guidance ()

    Guidance with Spherical Gaussian Constraint for Conditional Diffusion, Yang et al. 2024
    """
    def __init__(
            self,
            steps: int,
            corrections: int,
            tau: float,
            skip_type: str = "linear",
            correcting_x0_fn = None,
            thresholding_max_val: float = 1.,
            dynamic_thresholding_ratio: float = 0.995,
            denoise_to_zero = False,
            eta = 1.0,
            guidance_ratio = 0.2,
    ):
        super().__init__(steps, corrections, tau, skip_type, correcting_x0_fn, thresholding_max_val, dynamic_thresholding_ratio, denoise_to_zero)
        self.eta = eta
        self.guidance_ratio = guidance_ratio

    def sample(
                self,
                sde: VPSDE,
                shape: Size = (),
        ) -> Tensor:
        assert isinstance(sde.net, ScaledGuidance)

        sde, guided_sde = sde.net.sde, sde # extract unguided sde

        x = torch.randn(shape + sde.shape).to(sde.device)
        x = x.reshape(-1, *sde.shape)

        # time = get_rev_ts_th(sde, self.steps, self.ts_order, sampling_eps = self.sampling_eps, sampling_T = self.sampling_T, ts_phase = self.ts_phase)
        time = self.get_time_steps(sde, skip_type=self.skip_type, t_T=self.sampling_T, t_0=self.sampling_eps,
                                    N=self.steps)
        dts = time[:-1] - time[1:]

        with torch.no_grad():
            for dt, t in tqdm(zip(dts, (time[:-1])), total=len(dts)):
                # for t in time[:-1]:
                a_t = sde.mu(t)
                b_t = sde.sigma(t)
                a_tm = sde.mu(t - dt)
                b_tm = sde.sigma(t - dt)
                sigma_t = self.eta*(b_tm/b_t).sqrt() * (1 - (a_t/a_tm)**2).sqrt()
                r = a_tm / a_t

                noise_pred, guidance = guided_sde.eps_and_guidance(x, t)
                # Compute E[x_tm|x_t]
                x_mean = r * x + ((1 - a_tm ** 2 - sigma_t**2).sqrt() - r * b_t) * noise_pred 
                # Projection to the sphere of radius sqrt(n)*sigma_t around E[x_tm | x_t]
                d_proj = sigma_t * guidance / guidance.square().mean(dim=tuple(range(1, x.ndim)), keepdim=True).sqrt()
                # Compute Euler SDE step with random noise of amplitude sigma_t
                eps = torch.randn_like(x)
                d_random = sigma_t * eps
                # Mix the two steps with guidance ratio
                gr = self.guidance_ratio
                d = (1 - gr) * d_random + gr * d_proj
                # Project back to the sphere of radius sqrt(n)*sigma_t around E[x_tm | x_t]
                x = x_mean + sigma_t * d / d.square().mean(dim=tuple(range(1, x.ndim)), keepdim=True).sqrt()

                # Corrector
                for _ in range(self.corrections):
                    eps = torch.randn_like(x)
                    # NOTE: since we are parametrizing eps(x(t),t) = - sigma(t) * score(x(t),t)
                    # if we need to use score(x(t),t) we have to always compute
                    # score(x(t),t) = - eps(x(t),t) / sigma(t)
                    noise_pred = guided_sde.noise_prediction_fn(x, t - dt)
                    s = -noise_pred / guided_sde.sigma(t - dt)
                    delta = self.tau / s.square().mean(dim=guided_sde.dims, keepdim=True)

                    x = x + delta * s + torch.sqrt(2 * delta) * eps

            if self.denoise_to_zero:
                t = torch.ones((1,)).to(guided_sde.device) * self.sampling_eps
                x = self.denoise_to_zero_fn(guided_sde, x, t)
        return x.reshape(shape + guided_sde.shape)


class DMAPExpInt(ExpInt):
    """
    Rethinking Diffusion Posterior Sampling: From Conditional Score Estimator to Maximizing a Posterior, Xu et al. 2025
    """
    def __init__(
        self,
        steps: int,
        corrections: int,
        tau: float,
        skip_type: str = "linear",
        correcting_x0_fn = None,
        thresholding_max_val: float = 1.,
        dynamic_thresholding_ratio: float = 0.995,
        denoise_to_zero = False,
        eta = 1.0,
        ):
        super().__init__(steps, corrections, tau, skip_type, correcting_x0_fn, thresholding_max_val, dynamic_thresholding_ratio, denoise_to_zero)
        self.eta = eta

    def sample(
            self,
            sde: VPSDE,
            shape: Size = (),
        ) -> Tensor:
        # Assert that the score is already guided
        assert isinstance(sde.net, ScaledGuidance)

        sde, guided_sde = sde.net.sde, sde # extract unguided sde

        x = torch.randn(shape + sde.shape).to(sde.device)
        x = x.reshape(-1, *sde.shape)

        # time = get_rev_ts_th(sde, self.steps, self.ts_order, sampling_eps = self.sampling_eps, sampling_T = self.sampling_T, ts_phase = self.ts_phase)
        time = self.get_time_steps(sde, skip_type=self.skip_type, t_T=self.sampling_T, t_0=self.sampling_eps,
                                    N=self.steps)
        dts = time[:-1] - time[1:]

        with torch.no_grad():
            for dt, t in tqdm(zip(dts, (time[:-1])), total=len(dts)):
                # for t in time[:-1]:
                a_t = sde.mu(t)
                b_t = sde.sigma(t)
                a_tm = sde.mu(t - dt)
                b_tm = sde.sigma(t - dt)
                sigma_t = self.eta*(b_tm/b_t).sqrt() * (1 - (a_t/a_tm)**2).sqrt()
                r = a_tm / a_t

                # DDPM step
                eps = torch.randn_like(x)
                noise_pred = sde.noise_prediction_fn(x, t)
                # Compute E[x_tm|x_t]
                x_tm_mean = r * x + ((b_tm**2 - sigma_t**2).clamp_min(0).sqrt() - r * b_t) * noise_pred
                # Compute Euler SDE step with random noise of amplitude sigma_t
                x_tm = x_tm_mean + sigma_t * eps
                for _ in range(1 + self.corrections):
                    # Guidance step
                    _, guidance = guided_sde.net.eps_and_guidance(x_tm, t-dt)
                    x_tm = x_tm - (b_tm - r * b_t) * b_t * guidance

                    # Projection to the sphere of radius sqrt(n)*sigma_t around E[x_tm | x_t]
                    d_proj = sigma_t * (x_tm - x_tm_mean) / (x_tm - x_tm_mean).square().mean(dim=tuple(range(1, x_tm.ndim)), keepdim=True).sqrt()
                    x = x_tm_mean + d_proj

            if self.denoise_to_zero:
                t = torch.ones((1,)).to(sde.device) * self.sampling_eps
                x = self.denoise_to_zero_fn(sde, x, t)
        return x.reshape(shape + sde.shape)


    
class HistoryExpInt(ExpInt):
    def _guidance_uncertainty(self, sde: VPSDE) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        covariance = None
        std = None
        likelihoods = getattr(sde.net, "likelihoods", None)
        if likelihoods is not None:
            likelihood = likelihoods[-1]
            if isinstance(likelihood, MultivariateGaussian):
                covariance = likelihood.cov
            elif isinstance(likelihood, Gaussian):
                std = likelihood.std

        covariance = covariance.detach() if hasattr(covariance, "detach") else covariance
        std = std.detach() if isinstance(std, Tensor) else std
        return covariance, std

    def _guidance_history(self, sde: VPSDE) -> Optional[Tensor]:
        guidance_history = getattr(sde.net, "guidance_history", None)
        if guidance_history is None:
            return None
        if isinstance(guidance_history, Tensor):
            return guidance_history.detach().clone()
        if len(guidance_history) == 0:
            return None
        return torch.stack(guidance_history).detach().clone()

    def sample(
            self,
            sde: VPSDE,
            shape: Size = (),
    ) -> SDESampleHistory:
        r"""Samples from :math:`p(x(0))` and stores the denoising history."""
        x = torch.randn(shape + sde.shape).to(sde.device)
        x = x.reshape(-1, *sde.shape)

        if hasattr(sde.net, "guidance_history"):
            sde.net.guidance_history = []

        time = self.get_time_steps(
            sde,
            skip_type=self.skip_type,
            t_T=self.sampling_T,
            t_0=self.sampling_eps,
            N=self.steps,
        )
        dts = time[:-1] - time[1:]

        with torch.no_grad():
            x_t_history = [x.detach().clone()]
            x0_history = []
            for dt, t in tqdm(zip(dts, time[:-1]), total=len(dts)):
                noise_pred = sde.noise_prediction_fn(x, t)
                x0_history.append(((x - sde.sigma(t) * noise_pred) / sde.mu(t)).detach().clone())

                r = sde.mu(t - dt) / sde.mu(t)
                x = r * x + (sde.sigma(t - dt) - r * sde.sigma(t)) * noise_pred

                for _ in range(self.corrections):
                    eps = torch.randn_like(x)
                    noise_pred = sde.noise_prediction_fn(x, t - dt)
                    s = -noise_pred / sde.sigma(t - dt)
                    delta = self.tau / s.square().mean(dim=sde.dims, keepdim=True)

                    x = x + delta * s + torch.sqrt(2 * delta) * eps

                x_t_history.append(x.detach().clone())

            if self.denoise_to_zero:
                t = torch.ones((1,), dtype=x.dtype, device=x.device) * self.sampling_eps
                x = self.denoise_to_zero_fn(sde, x, t)
                x0_history[-1] = x.detach().clone()

        covariance, std = self._guidance_uncertainty(sde)
        guidance_history = self._guidance_history(sde)
        return SDESampleHistory(
            x0=torch.stack(x0_history).reshape((self.steps,) + shape + sde.shape),
            x_t=torch.stack(x_t_history).reshape((self.steps + 1,) + shape + sde.shape),
            times=time.detach().clone(),
            covariance=covariance,
            std=std,
            guidance_history=guidance_history,
        ).detach()


HistoryExpint = HistoryExpInt
