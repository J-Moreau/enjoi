from pdediff.sampler.dpm import DPM
from pdediff.sampler.expint import DMAPExpInt, ExpInt, HistoryExpInt


def get_sampler(cfg):
    # Sampler
    if cfg.sampler.name == 'expint':
        sampler = ExpInt(
            steps=cfg.eval.sampling.steps, 
            corrections=cfg.eval.sampling.corrections, 
            tau=cfg.eval.sampling.tau, 
            skip_type=cfg.sampler.skip_type,
            correcting_x0_fn=cfg.sampler.correcting_x0_fn,
            denoise_to_zero=cfg.sampler.denoise_to_zero)
    elif cfg.sampler.name in {'history_expint', 'HistoryExpint'}:
        sampler = HistoryExpInt(
            steps=cfg.eval.sampling.steps,
            corrections=cfg.eval.sampling.corrections,
            tau=cfg.eval.sampling.tau,
            skip_type=cfg.sampler.skip_type,
            correcting_x0_fn=cfg.sampler.correcting_x0_fn,
            denoise_to_zero=cfg.sampler.denoise_to_zero)
    elif cfg.sampler.name == 'dpm':
        sampler = DPM(
            steps=cfg.eval.sampling.steps, 
            corrections=cfg.eval.sampling.corrections, 
            tau=cfg.eval.sampling.tau, 
            skip_type=cfg.sampler.skip_type, 
            order=cfg.sampler.order, 
            correcting_x0_fn=cfg.sampler.correcting_x0_fn, 
            algorithm_type=cfg.sampler.alg_type, 
            denoise_to_zero=cfg.sampler.denoise_to_zero)
    elif cfg.sampler.name == 'dmap-expint':
        sampler = DMAPExpInt(
            steps=cfg.eval.sampling.steps, 
            corrections=cfg.eval.sampling.corrections, 
            tau=cfg.eval.sampling.tau, 
            skip_type=cfg.sampler.skip_type, 
            correcting_x0_fn=cfg.sampler.correcting_x0_fn, 
            denoise_to_zero=cfg.sampler.denoise_to_zero,
            eta=cfg.sampler.eta,
    )
    else:
        print("Using ExpInt sampler with skip_type: linear")
        sampler = None # Default is ExpInt
    return sampler