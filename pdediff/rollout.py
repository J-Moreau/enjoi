r"""
Rollout algorithms for forecasting

In the rollout, I'm using an overlapping sliding window.
For fairness, I want the output of the task to be either
"filtering" (predict current state with current observations, don't
overwrite previously predicted state when new information comes in)
or "smoothing" (predict current state, possibly overwriting
the past with new information). It doesn't matter what the algorithm
 is doing in its internal state, it could be doing smoothing the past
 inside the window. If I select "filtering", the overlapping past states
 shouldn't be rewritten in the output
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from torch.func import vmap

from pdediff.covariance import empirical_localized_covariance_operator
from pdediff.guidance import (
    SDA,
    VPSDE,
    AlternatingVPSDE,
    ConditionalSDA,
    GuidedScore,
    Likelihood,
    MultivariateGaussian,
    SDESampleHistory,
    Tensor,
)
from pdediff.mcs import *
from pdediff.sampler.expint import HistoryExpInt
from pdediff.sampler.sampler import Sampler
from pdediff.score import *


@dataclass
class AutoregressiveHistory:
    samples: Tensor
    windows: list[SDESampleHistory]
    window_starts: list[int]
    n_conditioned_frame: int
    predictive_horizon: int
    guidance_history: Optional[list[Tensor]] = None
    weights: Optional[list[Tensor]] = None

    def cpu(self) -> "AutoregressiveHistory":
        return self.to("cpu")

    def detach(self) -> "AutoregressiveHistory":
        return AutoregressiveHistory(
            samples=self.samples.detach(),
            windows=[window.detach() for window in self.windows],
            window_starts=self.window_starts,
            n_conditioned_frame=self.n_conditioned_frame,
            predictive_horizon=self.predictive_horizon,
            guidance_history=[history.detach() for history in self.guidance_history]
            if self.guidance_history is not None
            else None,
            weights=[weights.detach() for weights in self.weights]
            if self.weights is not None
            else None,
        )

    def to(self, *args, **kwargs) -> "AutoregressiveHistory":
        return AutoregressiveHistory(
            samples=self.samples.to(*args, **kwargs),
            windows=[window.to(*args, **kwargs) for window in self.windows],
            window_starts=self.window_starts,
            n_conditioned_frame=self.n_conditioned_frame,
            predictive_horizon=self.predictive_horizon,
            guidance_history=[
                history.to(*args, **kwargs) for history in self.guidance_history
            ]
            if self.guidance_history is not None
            else None,
            weights=[weights.to(*args, **kwargs) for weights in self.weights]
            if self.weights is not None
            else None,
        )


#######################################################
##################    Rollout    ######################
#######################################################


def set_seed(seed):
    # force reproducibility
    torch.backends.cudnn.deterministic = True

    # set the seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _pad_time_axis_to_length(x: Tensor, target_length: int) -> Tensor:
    current_length = x.shape[1]
    if current_length >= target_length:
        return x

    pad_shape = list(x.shape)
    pad_shape[1] = target_length - current_length
    padding = torch.zeros(*pad_shape, dtype=x.dtype, device=x.device)
    return torch.cat((x, padding), dim=1)


class Rollout(ABC):
    # each sampler should have an unconditional score and a state space info
    # plus all the sampling parameters like predictive steps, correction steps, tau
    # that you get from **cfg.eval.sampling
    def __init__(
        self,
        unconditional_score: MCScoreNet,
        state_shape: tuple,
        steps: int,
        corrections: int,
        tau: float,
        sampler: Sampler = None,
    ):
        super().__init__()
        """
        Abstract sampler class.

        Args:
            unconditional_score: unconditional or prior score. Should be a MCScoreNet class.
            state_shape: Event shape, i.e. shape of a single trajectory state.
            steps: Number of diffusion steps, i.e. how many discretization steps we use in solving the reverse SDE.
            corrections: Number of Langevin corrections steps for each diffusion step.
            tau: parameters that approximates the SNR during the Langevin corrections. If corrections=0, this parameter has 0 influence.
            sampler: type of sampler used (Expint/DPM).
        """

        self.unconditional_score = unconditional_score
        self.state_shape = state_shape
        self.steps = steps
        self.corrections = corrections
        self.tau = tau
        self.sampler = sampler

    @abstractmethod
    def sample_traj(self, trajectory_length: int) -> Tensor:
        pass

    @abstractmethod
    def sample_traj_conditioning(self, *args, **kwargs) -> Tensor:
        pass

    def sample_traj(
        self,
        trajectory_length,
        y_conditioning,
        mask,
        seed=0,
        batch_shape=(),
        *args,
        **kwargs,
    ) -> Tensor:
        sampled_trajectories = []
        for y_cond_batch, mask_batch in zip(
            y_conditioning.split(batch_shape[0]), mask.split(batch_shape[0])
        ):
            traj_batch = self.sample_traj_conditioning(
                trajectory_length,
                y_cond_batch,
                mask_batch,
                seed=seed,
                batch_shape=(y_cond_batch.shape[0],),
                *args,
                **kwargs,
            )
            sampled_trajectories.append(traj_batch)

        sampled_trajectories = torch.cat(sampled_trajectories)

        return sampled_trajectories


class ConditionalRolloutAllAtOnce(Rollout):
    def __init__(
        self,
        unconditional_score: MCScoreNet,
        state_shape: tuple,
        A,
        likelihoods: List[Likelihood],
        guidance: GuidedScore,
        likelihood_stds: List[float],
        steps: int,
        corrections: int,
        tau: float,
        gammas: float = None,
        sampler: Sampler = None,
        model_type: str = "noise",
    ):
        super().__init__(
            unconditional_score, state_shape, steps, corrections, tau, sampler
        )

        """
        Sampling all at once from the conditional score.

        Args:
            unconditional_score: unconditional or prior score. Should be a MCScoreNet class.
            state_shape: Event shape, i.e. shape of a single trajectory state.
            A: linear or non-linear mapping that from x gets the observatin y.
            likelihood: observation likelihood, i.e. p(y|x_0) = p(y|A(x)). Example can be Gaussian, i.e. p(y|x_0) = N(y|A(x), std^2) 
            guidance: guidance term, for now either SDA or DPS
            likelihood_std: standard deviation of the likelihood
            steps: Number of diffusion steps, i.e. how many discretization steps we use in solving the reverse SDE.
            corrections: Number of Langevin corrections steps for each diffusion step.
            tau: parameters that approximates the SNR during the Langevin corrections. If corrections=0, this parameter has 0 influence.
            gamma: approximation for the scaling the covariance of the guidance, when using SDA
            sampler: type of sampler used (Expint/DPM).
            model_type: the output of the network (noise/x_start/v_prediction).
        """

        self.A = A
        self.likelihoods = likelihoods
        self.guidance = guidance
        self.likelihood_stds = likelihood_stds

        assert len(likelihoods) == 1, (
            f"For AAO rollout you should specify one likelihood for the conditioning "
            f"on observations, but number of specified likelihoods is {len(likelihoods)}."
        )
        assert len(likelihood_stds) == 1, (
            f"For AAO rollout you should specify one likelihood for the conditioning "
            f"on observations, but number of specified likelihood stds is {len(likelihood_stds)}."
        )
        assert len(gammas) == 1, (
            f"For AAO rollout you should specify one likelihood for the conditioning "
            f"on observations, but number of specified gammas is {len(gammas)}."
        )
        self.gammas = gammas
        self.model_type = model_type

    def sample_traj_conditioning(
        self,
        trajectory_length,
        y_conditioning,
        mask,
        seed,
        batch_shape=(),
    ):
        """
        Args:
           trajectory_length: length of the trajectory generated, i.e. number of states
           y_conditioning: observations we are conditioning on
           seed: seed used for reproducibility
           batch_shape: tuple stating how many trajectories we want to sample. For example: () or (1,) indicates one, (5,) indicates 5 and so on
        """
        set_seed(seed)

        likelihood = self.likelihoods[0](
            y=y_conditioning * mask,
            A=self.A,
            std=self.likelihood_stds[0],
            mask=mask,
        )

        guided_score = self.guidance(
            VPSDE(
                self.unconditional_score,
                shape=(trajectory_length, *self.state_shape),
                model_type=self.model_type,
            ),
            likelihoods=[likelihood],
            gammas=[self.gammas[0]],
        )

        # create reverse sde with guided score
        if torch.cuda.is_available():
            sde = VPSDE(
                guided_score,
                shape=(trajectory_length, *self.state_shape),
                model_type=self.model_type,
            ).cuda()
        else:
            sde = VPSDE(
                guided_score,
                shape=(trajectory_length, *self.state_shape),
                model_type=self.model_type,
            )

        sde.eval()

        if self.sampler is None:
            x = sde.sample(
                steps=self.steps,
                corrections=self.corrections,
                tau=self.tau,
                shape=batch_shape,
            ).cpu()
        else:
            x = self.sampler.sample(
                sde,
                shape=batch_shape,
            ).cpu()

        return x


class ConditionalARRollout(Rollout):
    def __init__(
        self,
        unconditional_score: MCScoreNet,
        state_shape: tuple,
        A,
        likelihoods: List[Likelihood],
        guidance: GuidedScore,
        likelihood_stds: List[float],
        markov_blanket_window_size: int,
        steps: int,
        corrections: int,
        tau: float,
        gammas: float = None,
        sampler: Sampler = None,
        model_type: str = "noise",
        task: str = "forecast",
    ):
        super().__init__(
            unconditional_score, state_shape, steps, corrections, tau, sampler
        )

        """
        Sampling autoregressively with reconstruction guidance. At the first step we condition on the observation ys, but for all the other
        steps we are conditioning on the latest generated states.

        Args:
            unconditional_score: unconditional or prior score. Should be a MCScoreNet class.
            state_shape: Event shape, i.e. shape of a single trajectory state.
            A: linear or non-linear mapping that from x gets the observatin y.
            likelihoods: list of observation likelihoods, i.e. p(y|x_0) = p(y|A(x)). Example can be Gaussian, i.e. p(y|x_0) = N(y|A(x), std^2). Note this is the likelihood we use only at the first step (t=0).
            guidance: guidance term, for now either SDA or DPS. We need this because at the first step we generate unconditionally, but then we condition on the previously generated states.
            likelihood_stds: list of standard deviations of the likelihoods.
            markov_blanket_window_size: size of the markov blanket, i.e. 2*order + 1
            steps: Number of diffusion steps, i.e. how many discretization steps we use in solving the reverse SDE.
            corrections: Number of Langevin corrections steps for each diffusion step.
            tau: parameters that approximates the SNR during the Langevin corrections. If corrections=0, this parameter has 0 influence.
            gammas: approximation for the scaling the covariance of the guidance, when using SDA.
            sampler: type of sampler used (Expint/DPM).
            model_type: the output of the network (noise/x_start/v_prediction).
            task: task to solve (forecast or data_assimilation).
        """
        self.markov_blanket_window_size = markov_blanket_window_size
        self.A = A
        self.likelihoods = likelihoods
        self.guidance = guidance
        self.likelihood_stds = likelihood_stds

        assert len(likelihoods) == 2, (
            f"For AR rollout you should specify two likelihoods - "
            f"one for the conditioning on observations, and one for the AR conditioning, "
            f"but number of specified likelihoods is {len(likelihoods)}."
        )
        assert len(likelihood_stds) == 2, (
            f"For AR rollout you should only specify one likelihood std - "
            f"one for the conditioning on observations, and one for the AR conditioning, "
            f"but number of specified likelihood stds is {len(likelihood_stds)}."
        )
        assert len(gammas) == 2, (
            f"For AR rollout you should only specify one guidance strength - "
            f"one for the conditioning on observations, and one for the AR conditioning, "
            f"but number of specified gammas is {len(gammas)}."
        )

        self.gammas = gammas
        self.model_type = model_type
        self.task = task

    def sample_traj(
        self,
        trajectory_length: int,
        y_conditioning: Tensor,
        mask: Tensor,
        seed: int = 0,
        batch_shape: tuple = (),
        *args,
        **kwargs,
    ) -> AutoregressiveHistory:
        sampled_trajectories = []
        diagnostic_windows = []
        diagnostic_window_starts = []
        guidance_history = [] if isinstance(self.sampler, HistoryExpInt) else None
        n_conditioned_frame = kwargs.get("n_conditioned_frame")
        predictive_horizon = kwargs.get("predictive_horizon")

        for y_cond_batch, mask_batch in zip(
            y_conditioning.split(batch_shape[0]), mask.split(batch_shape[0])
        ):
            traj_batch = self.sample_traj_conditioning(
                trajectory_length,
                y_cond_batch,
                mask_batch,
                seed=seed,
                batch_shape=(y_cond_batch.shape[0],),
                *args,
                **kwargs,
            )
            sampled_trajectories.append(traj_batch.samples)
            diagnostic_windows.extend(traj_batch.windows)
            diagnostic_window_starts.extend(traj_batch.window_starts)
            if guidance_history is not None and traj_batch.guidance_history is not None:
                guidance_history.extend(traj_batch.guidance_history)
            n_conditioned_frame = traj_batch.n_conditioned_frame
            predictive_horizon = traj_batch.predictive_horizon

        return AutoregressiveHistory(
            samples=torch.cat(sampled_trajectories),
            windows=diagnostic_windows,
            window_starts=diagnostic_window_starts,
            n_conditioned_frame=n_conditioned_frame,
            predictive_horizon=predictive_horizon,
            guidance_history=guidance_history,
        )

    def sample_traj_conditioning(
        self,
        trajectory_length: int,
        y_conditioning: Tensor,
        mask_conditioning: Tensor,
        n_conditioned_frame: int,
        predictive_horizon: int,
        seed: int,
        batch_shape: tuple = (),
    ) -> AutoregressiveHistory:
        """
        Function that generate a trajectory conditioned on some observation y. The function works as follows:
            First iter: generate (n_conditioned_frame + predictive_horizon) states conditioned on y_conditioning
            all the other iter: generate (n_conditioned_frame + predictive_horizon) states conditioned on previously n_conditioned_frame generated states

        Args:
            trajectory_length: length of the trajectory generated, i.e. number of states.
            y_conditioning: observations we are conditioning on to generate the first subtrajectory.
            n_conditioned_frame: how many states we are conditioning to generate predictive_horizon state each step.
            predictive_horizon: number of states generated at each step.
            seed: seed used for reproducibility
            batch_shape: tuple stating how many trajectories we want to sample. For example: () or (1,) indicates one, (5,) indicates 5 and so on
        """

        assert (
            n_conditioned_frame + predictive_horizon >= self.markov_blanket_window_size
        ), (
            "We do not allow n_conditioned_frame + predictive_horizon < markov_blanket_window_size."
        )

        # reproducibility
        set_seed(seed)

        stride = predictive_horizon
        idx_start = 0
        subtrajectory_length = n_conditioned_frame + predictive_horizon
        idx_end = subtrajectory_length

        def A_autoregressive_step(x):
            return x[:, :n_conditioned_frame, ...]

        sampled_traj = None
        diagnostic_windows = []
        diagnostic_window_starts = []
        guidance_history = [] if isinstance(self.sampler, HistoryExpInt) else None

        # starting the generation
        while idx_start + n_conditioned_frame < trajectory_length:
            sub_y = y_conditioning[:, idx_start:idx_end, :]
            sub_mask = mask_conditioning[:, idx_start:idx_end, :].clone()
            sub_y = _pad_time_axis_to_length(sub_y, subtrajectory_length)
            sub_mask = _pad_time_axis_to_length(sub_mask, subtrajectory_length)

            # At first step we only condition on observations
            if idx_start == 0:
                cond_on_obs_likelihood = self.likelihoods[0](
                    y=sub_y * sub_mask,
                    A=self.A,
                    std=0.01,
                    mask=sub_mask,
                )

                guided_score = self.guidance(
                    VPSDE(
                        self.unconditional_score,
                        shape=(subtrajectory_length, *self.state_shape),
                        model_type=self.model_type,
                    ),
                    likelihoods=[cond_on_obs_likelihood],
                    gammas=[self.gammas[0]],
                )

                if torch.cuda.is_available():
                    sde = VPSDE(
                        guided_score,
                        shape=(subtrajectory_length, *self.state_shape),
                        model_type=self.model_type,
                    ).cuda()
                else:
                    sde = VPSDE(
                        guided_score,
                        shape=(subtrajectory_length, *self.state_shape),
                        model_type=self.model_type,
                    )

                sde.eval()

                if self.sampler is None:
                    sample = sde.sample(
                        steps=self.steps,
                        corrections=self.corrections,
                        tau=self.tau,
                        shape=batch_shape,
                    ).cpu()
                else:
                    sample = self.sampler.sample(
                        sde,
                        shape=batch_shape,
                    ).cpu()

                if isinstance(sample, SDESampleHistory):
                    diagnostic_windows.append(sample)
                    diagnostic_window_starts.append(idx_start)
                    if (
                        guidance_history is not None
                        and sample.guidance_history is not None
                    ):
                        guidance_history.append(sample.guidance_history)
                    x = sample.x0[-1]
                else:
                    x = sample

                # at the first step we keep all the generated states
                sampled_traj = x.clone()
                print("First sampled traj", sampled_traj.shape)

            # At the next steps we condition on observations and previously generated states
            else:
                sub_mask[:, :n_conditioned_frame, :] = 0
                x_star = A_autoregressive_step(sampled_traj[:, idx_start:idx_end, ...])

                cond_on_obs_likelihood = self.likelihoods[0](
                    y=sub_y * sub_mask,
                    A=self.A,
                    std=self.likelihood_stds[0],
                    mask=sub_mask,
                )

                cond_AR_likelihood = self.likelihoods[1](
                    y=x_star,
                    A=A_autoregressive_step,
                    std=self.likelihood_stds[1],
                    mask=None,
                )

                # We now create a guided score using the two likelihoods
                guided_score = self.guidance(
                    VPSDE(
                        self.unconditional_score,
                        shape=(subtrajectory_length, *self.state_shape),
                        model_type=self.model_type,
                    ),
                    likelihoods=[cond_on_obs_likelihood, cond_AR_likelihood],
                    gammas=self.gammas,
                )

                if torch.cuda.is_available():
                    sde = VPSDE(
                        guided_score,
                        shape=(subtrajectory_length, *self.state_shape),
                        model_type=self.model_type,
                    ).cuda()
                else:
                    sde = VPSDE(
                        guided_score,
                        shape=(subtrajectory_length, *self.state_shape),
                        model_type=self.model_type,
                    )

                sde.eval()

                if self.sampler is None:
                    sample = sde.sample(
                        steps=self.steps,
                        corrections=self.corrections,
                        tau=self.tau,
                        shape=batch_shape,
                    ).cpu()
                else:
                    sample = self.sampler.sample(
                        sde,
                        shape=batch_shape,
                    ).cpu()

                if isinstance(sample, SDESampleHistory):
                    diagnostic_windows.append(sample)
                    diagnostic_window_starts.append(idx_start)
                    if (
                        guidance_history is not None
                        and sample.guidance_history is not None
                    ):
                        guidance_history.append(sample.guidance_history)
                    x = sample.x0[-1]
                else:
                    x = sample

                # At the next step we either don't resample the overlapping states (forecast)
                # Or we resample them (data assimilation)
                if self.task == "forecast":
                    sampled_traj = torch.cat(
                        (
                            sampled_traj,
                            x[:, (subtrajectory_length - predictive_horizon) :, ...],
                        ),
                        dim=1,
                    )
                elif self.task == "data_assimilation":
                    sampled_traj = torch.cat(
                        (
                            sampled_traj[:, :-n_conditioned_frame, :],
                            x,
                        ),
                        dim=1,
                    )
                else:
                    raise ValueError(f"{self.task} is not supported")
                print("Sampled traj shape", sampled_traj.shape)

            idx_start += stride
            idx_end += stride

        sampled_traj = sampled_traj[:, :trajectory_length, ...]

        return AutoregressiveHistory(
            samples=sampled_traj,
            windows=diagnostic_windows,
            window_starts=diagnostic_window_starts,
            n_conditioned_frame=n_conditioned_frame,
            predictive_horizon=predictive_horizon,
            guidance_history=guidance_history,
        )


class AmortizedRollout(Rollout):
    """Sampling from the amortized model."""

    def __init__(
        self,
        score: MCScoreNet,
        state_shape: tuple,
        steps: int,
        corrections: int,
        tau: float,
        conditioned_frame: int,
        predictive_horizon: int,
        sampler: Optional[Sampler] = None,
        likelihood: Optional[Likelihood] = None,
        guidance: Optional[GuidedScore] = None,
        likelihood_std: Optional[float] = None,
        gamma: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            unconditional_score=score,
            state_shape=state_shape,
            steps=steps,
            corrections=corrections,
            tau=tau,
            sampler=sampler,
        )

        self.conditioned_frame = conditioned_frame
        self.predictive_horizon = predictive_horizon
        self.window = conditioned_frame + predictive_horizon
        input_dim = self.window
        self.shape = (input_dim * state_shape[0], *state_shape[1:])
        self.state_shape = state_shape
        self.multiplier = state_shape[0]
        self.sampler = sampler
        self.likelihood = likelihood
        self.guidance = guidance
        self.likelihood_std = likelihood_std
        self.gamma = gamma

    def sample_traj(
        self,
        trajectory_length: int,
        seed: int,
        batch_shape: tuple = (),
        conditions: Optional[Tensor] = None,
        obs: Optional[Tensor] = None,
        obs_mask: Optional[Tensor] = None,
    ):
        """
        Method that creates the reverse SDE using the unconditinal score and sample a batch of trajectories.

        Args:
           trajectory_length: length of the trajectory generated, i.e. number of states
           seed: seed used for reproducibility
           batch_shape: tuple stating how many trajectories we want to sample.
                        For example: () or (1,) indicates one, (5,) indicates 5 and so on
           conditions: condition initial states
        """
        gen_sequence = self.sample_traj_fixed_initial_state(
            trajectory_length=trajectory_length,
            seed=seed,
            batch_shape=(len(conditions),),
            conditions=conditions,
            obs=obs,
            obs_mask=obs_mask,
        )
        return gen_sequence.reshape((gen_sequence.shape[0], -1, *self.state_shape))

    def sample_traj_conditioning(self, *args, **kwargs):
        return self.sample_traj(*args, **kwargs)

    def sample_traj_fixed_initial_state(
        self,
        trajectory_length: int,
        seed: int,
        batch_shape: tuple = (),
        conditions: Tensor = None,
        obs: Optional[Tensor] = None,
        obs_mask: Optional[Tensor] = None,
    ):
        set_seed(seed)
        sde = VPSDE(self.unconditional_score.kernel, shape=self.shape)
        if obs is not None:
            obs = obs.reshape((obs.shape[0], -1, *self.state_shape[1:]))
            obs_mask = obs_mask.reshape((obs.shape[0], -1, *self.state_shape[1:]))

            def A(x, mask):
                return x * mask

            likelihood = self.likelihood(
                y=obs[:, : self.multiplier * self.window],
                A=A,
                std=self.likelihood_std,
                mask=obs_mask[:, : self.multiplier * self.window],
            )
            guided_score = self.guidance(sde, [likelihood], [self.gamma])
            sde = VPSDE(
                guided_score,
                shape=self.shape,
            )

        if torch.cuda.is_available():
            sde = sde.cuda()

        if self.sampler is None:
            sampling_function = lambda: sde.sample(
                steps=self.steps,
                corrections=self.corrections,
                tau=self.tau,
                shape=batch_shape,
            )
        else:
            sampling_function = lambda: self.sampler.sample(sde, shape=batch_shape)

        generated_length = 0
        generated_sequence = None

        x_condition = conditions

        if x_condition is not None and torch.cuda.is_available():
            x_condition = x_condition.float().cuda()

        zero_mask_ph = torch.zeros_like(x_condition)
        if obs is not None:
            current_obs = obs[:, : self.multiplier * self.window]
            current_mask = obs_mask[:, : self.multiplier * self.window]

        while generated_length < trajectory_length:
            if obs is None:
                sde.net.set_condition(x_condition)
            else:
                sde.net.sde.net.set_condition(x_condition)
                likelihood.set_observation(current_obs, current_mask)
            x = sampling_function()
            if generated_sequence is not None:
                generated_sequence = torch.cat(
                    [
                        generated_sequence,
                        x[:, -self.predictive_horizon * self.multiplier :],
                    ],
                    dim=1,
                )
                generated_length += self.predictive_horizon
            else:
                generated_sequence = x
                generated_length += self.window

            x_condition = generated_sequence[
                :, -self.conditioned_frame * self.multiplier :
            ]
            mask = torch.ones_like(x_condition)

            x_condition = torch.cat(
                [
                    x_condition,
                    zero_mask_ph[:, : self.predictive_horizon * self.multiplier],
                ],
                dim=1,
            )
            mask = torch.cat(
                [mask, zero_mask_ph[:, : self.predictive_horizon * self.multiplier]],
                dim=1,
            )

            x_condition = torch.cat([mask, x_condition], dim=1)

            if obs is not None:
                current_obs = obs[
                    :,
                    self.multiplier
                    * (generated_length - self.conditioned_frame) : self.multiplier
                    * (generated_length + self.predictive_horizon),
                ]
                current_mask = obs_mask[
                    :,
                    self.multiplier
                    * (generated_length - self.conditioned_frame) : self.multiplier
                    * (generated_length + self.predictive_horizon),
                ]
                target_obs_length = self.multiplier * self.window
                current_obs = _pad_time_axis_to_length(current_obs, target_obs_length)
                current_mask = _pad_time_axis_to_length(current_mask, target_obs_length)

        return generated_sequence[:, : trajectory_length * self.multiplier].cpu()


class ConditionalARRolloutEmpiricalVariance(Rollout):
    r"""AR rollout where the past-conditioning likelihood uses empirical
    uncertainty computed over the sample batch.

    At each AR step after the first:

    * :math:`\bar{x}^a` = mean of the generated past across samples (optional AR target)
    * :math:`\bar{\Sigma}^a` = diagonal variance or localized covariance across samples (AR uncertainty)

    Two guidance terms with separate gammas:

    1. **Observation guidance** :math:`\mathcal{N}(y^b \mid H \hat{x}_0^b, R)` with ``gamma_obs``
    2. **AR guidance** :math:`\mathcal{N}(x^a \mid \hat{x}_0^a, \bar{\Sigma}^a)` or
       :math:`\mathcal{N}(\bar{x}^a \mid \hat{x}_0^a, \bar{\Sigma}^a)` with ``gamma_ar``
    """

    def __init__(
        self,
        unconditional_score: MCScoreNet,
        state_shape: tuple,
        A,
        likelihoods: List[Likelihood],
        guidance: GuidedScore,
        likelihood_stds: List[float],
        markov_blanket_window_size: int,
        steps: int,
        corrections: int,
        tau: float,
        gammas: List[float] = None,
        sampler: Sampler = None,
        model_type: str = "noise",
        task: str = "data_assimilation",
        min_ar_std: float = 1e-3,
        ar_cov_localization_bandwidth: int = 0,
        temporal_band: Optional[int] = 0,
        hybrid_ratio: float = 0.0,
        static_std: float = 1.0,
        use_sample_based_ar_conditioning: bool = True,
        marginal_score: MCScoreNet = None,
    ):
        super().__init__(
            unconditional_score, state_shape, steps, corrections, tau, sampler
        )

        self.markov_blanket_window_size = markov_blanket_window_size
        self.A = A
        self.likelihoods = likelihoods
        self.guidance = guidance
        self.likelihood_stds = likelihood_stds

        assert len(likelihoods) == 2, (
            f"Expected 2 likelihood classes (obs + AR), got {len(likelihoods)}."
        )
        assert len(gammas) == 2, (
            f"Expected 2 gammas [gamma_obs, gamma_ar], got {len(gammas)}."
        )
        assert len(likelihood_stds) == 1, (
            "Expected 1 likelihood std (for observations only); "
            "AR std is computed from the empirical variance. "
            f"Got {len(likelihood_stds)}."
        )

        self.gammas = gammas
        self.model_type = model_type
        self.task = task
        self.min_ar_std = min_ar_std
        self.ar_cov_localization_bandwidth = ar_cov_localization_bandwidth
        self.temporal_band = temporal_band
        self.hybrid_ratio = hybrid_ratio
        self.static_std = static_std
        self.use_sample_based_ar_conditioning = use_sample_based_ar_conditioning
        self.marginal_score = marginal_score

    def sample_traj(
        self,
        trajectory_length,
        y_conditioning,
        mask,
        seed=0,
        batch_shape=(),
        *args,
        **kwargs,
    ):
        sampled_trajectories = []
        diagnostic_windows = []
        diagnostic_window_starts = []
        guidance_history = [] if isinstance(self.sampler, HistoryExpInt) else None
        n_conditioned_frame = kwargs.get("n_conditioned_frame")
        predictive_horizon = kwargs.get("predictive_horizon")

        for y_cond_batch, mask_batch in zip(
            y_conditioning.split(batch_shape[0]), mask.split(batch_shape[0])
        ):
            traj_batch = self.sample_traj_conditioning(
                trajectory_length,
                y_cond_batch,
                mask_batch,
                seed=seed,
                batch_shape=(y_cond_batch.shape[0],),
                *args,
                **kwargs,
            )
            if isinstance(traj_batch, AutoregressiveHistory):
                sampled_trajectories.append(traj_batch.samples)
                diagnostic_windows.extend(traj_batch.windows)
                diagnostic_window_starts.extend(traj_batch.window_starts)
                if (
                    guidance_history is not None
                    and traj_batch.guidance_history is not None
                ):
                    guidance_history.extend(traj_batch.guidance_history)
                n_conditioned_frame = traj_batch.n_conditioned_frame
                predictive_horizon = traj_batch.predictive_horizon
            else:
                sampled_trajectories.append(traj_batch)

        samples = torch.cat(sampled_trajectories)
        return AutoregressiveHistory(
            samples=samples,
            windows=diagnostic_windows,
            window_starts=diagnostic_window_starts,
            n_conditioned_frame=n_conditioned_frame,
            predictive_horizon=predictive_horizon,
            guidance_history=guidance_history,
        )

    def sample_traj_conditioning(
        self,
        trajectory_length,
        y_conditioning,
        mask_conditioning,
        n_conditioned_frame,
        predictive_horizon,
        seed,
        batch_shape=(),
    ) -> AutoregressiveHistory:
        r"""Generate a trajectory with empirical-variance AR conditioning.

        * First window: condition on observations only.
        * Subsequent windows: condition on observations **and** on the
                    previously generated states, using either each sample's own past
                    values or the empirical mean across the sample batch.
        """
        assert (
            n_conditioned_frame + predictive_horizon >= self.markov_blanket_window_size
        ), "n_conditioned_frame + predictive_horizon < markov_blanket_window_size."

        set_seed(seed)

        stride = predictive_horizon
        subtrajectory_length = n_conditioned_frame + predictive_horizon
        idx_start = 0
        idx_end = subtrajectory_length

        def A_past(x):
            """Forward operator: extract the first *n_conditioned_frame* states."""
            return x[:, :n_conditioned_frame, ...]

        sampled_traj = None
        diagnostic_windows = []
        diagnostic_window_starts = []
        guidance_history = [] if isinstance(self.sampler, HistoryExpInt) else None

        while idx_start + n_conditioned_frame < trajectory_length:
            sub_y = y_conditioning[:, idx_start:idx_end, :]
            sub_mask = mask_conditioning[:, idx_start:idx_end, :].clone()
            sub_y = _pad_time_axis_to_length(sub_y, subtrajectory_length)
            sub_mask = _pad_time_axis_to_length(sub_mask, subtrajectory_length)

            # ── first window: observation guidance only ────────────────────
            if idx_start == 0:
                obs_likelihood = self.likelihoods[0](
                    y=sub_y * sub_mask,
                    A=self.A,
                    std=0.01,
                    mask=sub_mask,
                )

                guided_score = SDA(
                    VPSDE(
                        self.unconditional_score,
                        shape=(subtrajectory_length, *self.state_shape),
                        model_type=self.model_type,
                    ),
                    likelihoods=[obs_likelihood],
                    gammas=[0.1],
                )
                sde = VPSDE(
                    guided_score,
                    shape=(subtrajectory_length, *self.state_shape),
                    model_type=self.model_type,
                )

            # ── subsequent windows: obs + empirical-variance AR guidance ──
            else:
                # Zero out the obs mask on the overlapping past states
                sub_mask[:, :n_conditioned_frame, :] = 0

                # Empirical mean and std across the sample dimension
                past_samples = sampled_traj[
                    :, idx_start : idx_start + n_conditioned_frame, ...
                ]
                x_bar = past_samples.mean(dim=0, keepdim=True).expand_as(past_samples)

                x_target = (
                    past_samples if self.use_sample_based_ar_conditioning else x_bar
                )

                obs_likelihood = self.likelihoods[0](
                    y=sub_y * sub_mask,
                    A=self.A,
                    std=self.likelihood_stds[0],
                    mask=sub_mask,
                )

                ar_likelihood_cls = self.likelihoods[1]
                if issubclass(ar_likelihood_cls, MultivariateGaussian):
                    x_cov = empirical_localized_covariance_operator(
                        past_samples,
                        bandwidth=self.ar_cov_localization_bandwidth,
                        temporal_band=self.temporal_band,
                        min_std=self.min_ar_std,
                        hybrid_ratio=self.hybrid_ratio,
                        static_std=self.static_std,
                        compute_inverse=False,
                    )
                    ar_likelihood = ar_likelihood_cls(
                        y=x_target,
                        A=A_past,
                        cov=x_cov,
                        mask=None,
                    )
                else:
                    x_std = (
                        past_samples.std(dim=0, keepdim=True)
                        .clamp(min=self.min_ar_std)
                        .expand_as(past_samples)
                    )
                    ar_likelihood = ar_likelihood_cls(
                        y=x_target,
                        A=A_past,
                        std=x_std,
                        mask=None,
                    )

                # ── build reverse SDE and sample ──────────────────────────────
                joint_sde = VPSDE(
                    self.unconditional_score,
                    shape=(subtrajectory_length, *self.state_shape),
                    model_type=self.model_type,
                )
                if self.marginal_score is not None:
                    guided_score = ConditionalSDA(
                        sde=joint_sde,
                        past_likelihood=ar_likelihood,
                        obs_likelihood=obs_likelihood,
                        marginal_score=self.marginal_score,
                        n_conditioned_frame=n_conditioned_frame,
                        gammas=self.gammas,
                        model_type=self.model_type,
                    )
                else:
                    guided_score = self.guidance(
                        joint_sde,
                        likelihoods=[obs_likelihood, ar_likelihood],
                        gammas=self.gammas,
                    )
                sde = VPSDE(
                    guided_score,
                    shape=(subtrajectory_length, *self.state_shape),
                    model_type=self.model_type,
                )

            if torch.cuda.is_available():
                sde = sde.cuda()
            sde.eval()

            if self.sampler is None:
                sample = sde.sample(
                    steps=self.steps,
                    corrections=self.corrections,
                    tau=self.tau,
                    shape=batch_shape,
                ).cpu()
            else:
                sample = self.sampler.sample(sde, shape=batch_shape).cpu()

            if isinstance(sample, SDESampleHistory):
                diagnostic_windows.append(sample)
                diagnostic_window_starts.append(idx_start)
                if guidance_history is not None and sample.guidance_history is not None:
                    guidance_history.append(sample.guidance_history)
                x = sample.x0[-1]
            else:
                x = sample

            # ── accumulate trajectory ─────────────────────────────────────
            if idx_start == 0:
                sampled_traj = x.clone()
                print("First sampled traj", sampled_traj.shape)
            else:
                if self.task == "forecast":
                    # don't overwrite the overlapping past states; keep the predictive_horizon states
                    sampled_traj = torch.cat(
                        (
                            sampled_traj,
                            x[:, (subtrajectory_length - predictive_horizon) :, ...],
                        ),
                        dim=1,
                    )
                elif self.task == "data_assimilation":
                    # overwrite the overlapping past states with the newly "smoothed" ones
                    sampled_traj = torch.cat(
                        (sampled_traj[:, :-n_conditioned_frame, :], x),
                        dim=1,
                    )
                else:
                    raise ValueError(f"{self.task} is not supported")
                print("Sampled traj shape", sampled_traj.shape)

            idx_start += stride
            idx_end += stride

        return AutoregressiveHistory(
            samples=sampled_traj[:, :trajectory_length, ...],
            windows=diagnostic_windows,
            window_starts=diagnostic_window_starts,
            n_conditioned_frame=n_conditioned_frame,
            predictive_horizon=predictive_horizon,
            guidance_history=guidance_history,
        )


class FullyAdaptedAuxiliaryParticleFilterRollout(Rollout):
    r"""Autoregressive rollout with a fully-adapted auxiliary particle filter.

    The proposal :math:`p(x^{k+1} | x^k, y^{k+1})` is obtained either from the
    amortized rollout or from joint reconstruction guidance.
    """

    def __init__(
        self,
        score: MCScoreNet,
        state_shape: tuple,
        A,
        likelihoods: List[Likelihood],
        guidance: GuidedScore,
        likelihood_stds: List[float],
        markov_blanket_window_size: int,
        steps: int,
        corrections: int,
        tau: float,
        gammas: List[float] = None,
        sampler: Sampler = None,
        model_type: str = "noise",
        task: str = "forecast",
        proposal_type: str = "joint",
        alpha_init: float = 1.0,
        alpha_min: float = 1e-12,
        alpha_max: float = 1.0,
        neff_min: float = 0.5,
        neff_max: float = 0.9,
        alpha_search_steps: int = 20,
    ):
        super().__init__(score, state_shape, steps, corrections, tau, sampler)

        self.markov_blanket_window_size = markov_blanket_window_size
        self.A = A
        self.likelihoods = likelihoods
        self.guidance = guidance
        self.likelihood_stds = likelihood_stds
        self.gammas = gammas
        self.model_type = model_type
        self.task = task
        self.proposal_type = proposal_type
        self.alpha_init = alpha_init
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.neff_min = neff_min
        self.neff_max = neff_max
        self.alpha_search_steps = alpha_search_steps

        assert len(likelihoods) >= 1, "Expected at least one likelihood class."
        assert len(likelihood_stds) >= 1, "Expected at least one likelihood std."
        assert len(gammas) >= 1, "Expected at least one gamma."
        if proposal_type not in {"joint", "amortized"}:
            raise ValueError(
                "proposal_type must be either 'joint' or 'amortized', "
                f"got {proposal_type}."
            )
        if proposal_type == "joint" and len(likelihoods) < 2:
            raise ValueError("Joint PF proposals require two likelihood classes.")
        if proposal_type == "joint" and len(likelihood_stds) < 2:
            raise ValueError("Joint PF proposals require two likelihood stds.")
        if proposal_type == "joint" and len(gammas) < 2:
            raise ValueError("Joint PF proposals require two gammas.")
        if task != "forecast":
            raise ValueError(
                "FullyAdaptedAuxiliaryParticleFilterRollout currently supports forecast only."
            )

    def sample_traj(
        self,
        trajectory_length,
        y_conditioning,
        mask,
        seed=0,
        batch_shape=(),
        *args,
        **kwargs,
    ) -> AutoregressiveHistory:
        return self.sample_traj_conditioning(
            trajectory_length,
            y_conditioning,
            mask,
            seed=seed,
            batch_shape=batch_shape,
            *args,
            **kwargs,
        )

    def _resolve_neff_threshold(self, threshold: float, n_particles: int) -> float:
        if threshold <= 1.0:
            return threshold * n_particles
        return threshold

    def _weights_from_log_likelihood(
        self,
        log_likelihood: Tensor,
        alpha: float,
    ) -> tuple[Tensor, float]:
        log_likelihood = torch.nan_to_num(
            log_likelihood,
            nan=torch.finfo(log_likelihood.dtype).min,
            posinf=torch.finfo(log_likelihood.dtype).min,
            neginf=torch.finfo(log_likelihood.dtype).min,
        )
        shifted = log_likelihood - log_likelihood.max()
        weights = torch.softmax(alpha * shifted, dim=0)
        neff = 1.0 / weights.square().sum().item()
        return weights, neff

    def _adapt_inflation(
        self,
        log_likelihood: Tensor,
        alpha: float,
        n_particles: int,
    ) -> tuple[float, Tensor, float]:
        neff_min = self._resolve_neff_threshold(self.neff_min, n_particles)
        neff_max = self._resolve_neff_threshold(self.neff_max, n_particles)

        alpha_min = self.alpha_min
        alpha_max = self.alpha_max
        alpha = min(max(float(alpha), alpha_min), alpha_max)

        for _ in range(self.alpha_search_steps):
            weights, neff = self._weights_from_log_likelihood(log_likelihood, alpha)

            if neff > neff_max:
                alpha_min = alpha
                alpha = 0.5 * (alpha_min + alpha_max)
            elif neff < neff_min:
                alpha_max = alpha
                alpha = 0.5 * (alpha_min + alpha_max)
            else:
                return alpha, weights, neff

        weights, neff = self._weights_from_log_likelihood(log_likelihood, alpha)
        return alpha, weights, neff

    def _masked_gaussian_log_likelihood(
        self,
        predicted: Tensor,
        y: Tensor,
        mask: Tensor,
    ) -> Tensor:
        mask = mask.expand_as(predicted)
        y = y.expand_as(predicted)
        err = (y - predicted) * mask
        std = self.likelihood_stds[0]
        variance = std.square() if isinstance(std, Tensor) else std**2
        return -0.5 * err.reshape(err.shape[0], -1).square().sum(dim=1) / variance

    def _build_joint_proposal(
        self,
        past_states: Tensor,
        future_y: Tensor,
        future_mask: Tensor,
        n_conditioned_frame: int,
        predictive_horizon: int,
        seed: int,
    ) -> Tensor:
        def A_past(x):
            return x[:, :n_conditioned_frame, ...]

        batch_size = past_states.shape[0]
        future_y = future_y.expand(batch_size, *future_y.shape[1:])
        future_mask = future_mask.expand(batch_size, *future_mask.shape[1:])

        proposal_y = torch.cat((torch.zeros_like(past_states), future_y), dim=1)
        proposal_mask = torch.cat((torch.zeros_like(past_states), future_mask), dim=1)

        obs_likelihood = self.likelihoods[0](
            y=proposal_y * proposal_mask,
            A=self.A,
            std=self.likelihood_stds[0],
            mask=proposal_mask,
        )
        past_likelihood = self.likelihoods[1](
            y=past_states,
            A=A_past,
            std=self.likelihood_stds[1],
            mask=None,
        )

        guided_score = self.guidance(
            VPSDE(
                self.unconditional_score,
                shape=(n_conditioned_frame + predictive_horizon, *self.state_shape),
                model_type=self.model_type,
            ),
            likelihoods=[obs_likelihood, past_likelihood],
            gammas=self.gammas,
        )

        sde = VPSDE(
            guided_score,
            shape=(n_conditioned_frame + predictive_horizon, *self.state_shape),
            model_type=self.model_type,
        )
        if torch.cuda.is_available():
            sde = sde.cuda()
        sde.eval()

        set_seed(seed)
        if self.sampler is None:
            return sde.sample(
                steps=self.steps,
                corrections=self.corrections,
                tau=self.tau,
                shape=(batch_size,),
            ).cpu()
        return self.sampler.sample(sde, shape=(batch_size,)).cpu()

    def _build_amortized_proposal(
        self,
        past_states: Tensor,
        future_y: Tensor,
        future_mask: Tensor,
        n_conditioned_frame: int,
        predictive_horizon: int,
        seed: int,
    ) -> Tensor:
        batch_size = past_states.shape[0]
        future_y = future_y.expand(batch_size, *future_y.shape[1:])
        future_mask = future_mask.expand(batch_size, *future_mask.shape[1:])

        zero_future = torch.zeros_like(future_y)
        cond_mask = torch.cat((torch.ones_like(past_states), zero_future), dim=1)
        cond_values = torch.cat((past_states, zero_future), dim=1)
        trailing_shape = self.state_shape[1:]
        conditions = torch.cat(
            (
                cond_mask.reshape(batch_size, -1, *trailing_shape),
                (cond_mask * cond_values).reshape(batch_size, -1, *trailing_shape),
            ),
            dim=1,
        )

        proposal_obs = torch.cat((torch.zeros_like(past_states), future_y), dim=1)
        proposal_mask = torch.cat((torch.zeros_like(past_states), future_mask), dim=1)

        amortized_rollout = AmortizedRollout(
            score=self.unconditional_score,
            state_shape=self.state_shape,
            steps=self.steps,
            corrections=self.corrections,
            tau=self.tau,
            conditioned_frame=n_conditioned_frame,
            predictive_horizon=predictive_horizon,
            sampler=self.sampler,
            likelihood=self.likelihoods[0],
            guidance=self.guidance,
            likelihood_std=self.likelihood_stds[0],
            gamma=self.gammas[0],
        )
        return amortized_rollout.sample_traj(
            trajectory_length=n_conditioned_frame + predictive_horizon,
            seed=seed,
            batch_shape=(batch_size,),
            conditions=conditions,
            obs=proposal_obs,
            obs_mask=proposal_mask,
        )

    def _sample_proposal(
        self,
        past_states: Tensor,
        future_y: Tensor,
        future_mask: Tensor,
        n_conditioned_frame: int,
        predictive_horizon: int,
        seed: int,
    ) -> Tensor:
        if self.proposal_type == "amortized":
            return self._build_amortized_proposal(
                past_states,
                future_y,
                future_mask,
                n_conditioned_frame,
                predictive_horizon,
                seed,
            )
        return self._build_joint_proposal(
            past_states,
            future_y,
            future_mask,
            n_conditioned_frame,
            predictive_horizon,
            seed,
        )

    def tweedie_expectation(
        self,
        past_states: Tensor,
        predictive_horizon: int,
        seed: int,
    ) -> Tensor:
        """
        Computes the Tweedie expectation (one-shot denoising from t=1 to t=0)
        E[x_{k+1}(t=0) | x_{k+1}(t=1), x_k(t=0)]
        because the model is conditioned on the past states x_k(t=0),
        this should result in a blurry but reasonable prediction of the future states
        """
        assert self.proposal_type == "amortized", "Tweedie expectation is only implemented for amortized proposal."
        batch_size = past_states.shape[0]
        window_length = past_states.shape[1] + predictive_horizon
        trailing_shape = self.state_shape[1:]
        sample_shape = (window_length * self.state_shape[0], *trailing_shape)
        device = next(self.unconditional_score.parameters()).device
        sde = VPSDE(
            self.unconditional_score.kernel,
            shape=sample_shape,
            model_type=self.model_type,
        ).to(device)
        past_states = past_states.to(device)
        sde.eval()
        set_seed(seed)
        eps_1 = sde.base_sample(
            torch.empty(
                batch_size,
                *sample_shape,
                dtype=past_states.dtype,
                device=device,
            )
        )

        # build the conditioning tensor for the amortized model
        zero_future = torch.zeros(
            batch_size,
            predictive_horizon,
            *self.state_shape,
            dtype=past_states.dtype,
            device=device,
        )
        cond_mask = torch.cat((torch.ones_like(past_states), zero_future), dim=1)
        cond_values = torch.cat((past_states, zero_future), dim=1)
        conditions = torch.cat(
            (
                cond_mask.reshape(batch_size, -1, *trailing_shape),
                (cond_mask * cond_values).reshape(batch_size, -1, *trailing_shape),
            ),
            dim=1,
        )
        sde.net.set_condition(conditions)

        # compute Tweedie denoising expectation
        t = torch.tensor(1.0, dtype=eps_1.dtype, device=device)
        with torch.no_grad():
            eps = sde.noise_prediction_fn(x=eps_1, t=t)
            x_tweedie = (eps_1 - sde.sigma(t) * eps) / sde.mu(t)

        return x_tweedie.reshape(
            batch_size,
            window_length,
            *self.state_shape,
        ).cpu()

    def sample_traj_conditioning(
        self,
        trajectory_length,
        y_conditioning,
        mask_conditioning,
        n_conditioned_frame,
        predictive_horizon,
        seed,
        batch_shape=(),
    ) -> AutoregressiveHistory:
        assert (
            n_conditioned_frame + predictive_horizon >= self.markov_blanket_window_size
        ), "n_conditioned_frame + predictive_horizon < markov_blanket_window_size."

        n_particles = batch_shape[0] if batch_shape else y_conditioning.shape[0]
        single_y = y_conditioning[:1]
        single_mask = mask_conditioning[:1]
        initial_state = (
            single_y[:, :n_conditioned_frame] * single_mask[:, :n_conditioned_frame]
        ).repeat(
            n_particles,
            *([1] * (single_y.ndim - 1)),
        )

        sampled_traj = initial_state.clone()
        idx_start = 0
        idx_end = n_conditioned_frame + predictive_horizon
        alpha = self.alpha_init
        particle_weights = []

        while idx_start + n_conditioned_frame < trajectory_length:
            future_y = single_y[:, idx_start + n_conditioned_frame : idx_end, ...]
            future_mask = single_mask[:, idx_start + n_conditioned_frame : idx_end, ...]
            future_y = _pad_time_axis_to_length(future_y, predictive_horizon)
            future_mask = _pad_time_axis_to_length(future_mask, predictive_horizon)

            current_past = sampled_traj[
                :, idx_start : idx_start + n_conditioned_frame, ...
            ]
            has_observation = future_mask.bool().any().item()

            if has_observation:
                # use one shot denoising from pure noise at time t=1 when using amortized proposal
                if self.proposal_type == "amortized":
                    lookahead = self.tweedie_expectation(
                        current_past,
                        predictive_horizon,
                        seed,
                    )
                # for joint proposal, we can't do one-shot guidance, so we do a full on integration
                else:
                    lookahead = self._sample_proposal(
                        current_past,
                        future_y,
                        torch.zeros_like(future_mask),
                        n_conditioned_frame,
                        predictive_horizon,
                        seed,
                    )
                lookahead_future = lookahead[:, n_conditioned_frame:, ...]
                log_like = self._masked_gaussian_log_likelihood(
                    lookahead_future,
                    future_y,
                    future_mask,
                )
                alpha, weights, _ = self._adapt_inflation(log_like, alpha, n_particles)
                particle_weights.append(weights.detach().clone())
                ancestor_idx = torch.distributions.Categorical(probs=weights).sample(
                    (n_particles,)
                )
            else:
                # No observation update: keep particle identities instead of resampling uniformly.
                ancestor_idx = torch.arange(n_particles, device=current_past.device)
            ancestor_past = current_past[ancestor_idx]
            propagated = self._sample_proposal(
                ancestor_past,
                future_y,
                future_mask,
                n_conditioned_frame,
                predictive_horizon,
                seed,
            )

            sampled_traj = torch.cat(
                (sampled_traj, propagated[:, n_conditioned_frame:, ...]),
                dim=1,
            )
            idx_start += predictive_horizon
            idx_end += predictive_horizon

        return AutoregressiveHistory(
            samples=sampled_traj[:, :trajectory_length, ...],
            windows=[],
            window_starts=[],
            n_conditioned_frame=n_conditioned_frame,
            predictive_horizon=predictive_horizon,
            weights=particle_weights,
        )
