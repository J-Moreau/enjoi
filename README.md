# Data assimilation with Generative models

This repository is forked from the [github](https://github.com/cambridge-mlg/pdediff) of 'On conditional diffusion models for PDE simulations' Shysheya et al. NeurIPS 2024

## Install

We use [uv](https://docs.astral.sh/uv/) for managing dependencies, install the requirements with:
```sh
uv sync 
```

## Train

We use `wandb` for logging, authenticate first
```
uv run python main.py -m seed=0 mode=train experiment=KS_joint_SDA window=9
```

## Inference

```
uv run python run_ensemble.py -m experiment=KS_joint_SDA ensemble_eval=default observation=exponential_full ckpt_path='results/KS_joint_SDA/window=9/0/ckpt/latest.pth'
```
