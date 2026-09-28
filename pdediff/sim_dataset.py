from pathlib import Path

# import lightning as L
import numpy as np
import torch
import zarr
from torch.utils.data import DataLoader, Dataset


class MaskedDataset1D(Dataset):
    def __init__(
        self,
        zarr_path: str | Path,
        split: str,
        resolution: str,
        n_context_timesteps: int,
        n_label_cells: int,
        return_traj: bool = False,
    ) -> None:
        store = zarr.open_group(str(zarr_path), mode="r")
        group = store[f"{resolution}/{split}"]
        self.rho: zarr.Array = group["rho"]  # type: ignore[assignment]
        self.v: zarr.Array = group["v"]  # type: ignore[assignment]
        self.n_samples: int = self.rho.shape[0]
        self.nt: int = self.rho.shape[1]
        self.nx: int = self.rho.shape[2]
        self.n_context_timesteps = n_context_timesteps
        self.n_label_cells = n_label_cells
        self.return_traj = return_traj

        # Spatial window: centered middle W cells
        pad = (self.nx - self.n_label_cells) // 2
        self.x_start = pad
        self.x_end = pad + self.n_label_cells

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        rho = np.array(self.rho[i])  # (nt, nx)
        v = np.array(self.v[i])
        # Stack rho and v as channels → (nt, nx, 2)
        full = np.stack([rho, v], axis=-1)
        full_tensor = torch.from_numpy(full)

        # Input: first C timesteps, full spatial extent
        u_input = full_tensor[: self.n_context_timesteps]  # (C, nx, 2)

        if self.return_traj:
            # Label: all timesteps, middle W spatial cells
            u_label = full_tensor[:, self.x_start : self.x_end]  # (nt, W, 2)
        else:
            rand_idx = np.random.randint(0, self.nt - 1)
            u_label = full_tensor[rand_idx + 1, self.x_start : self.x_end]  # (W, 2)

        return {"input": u_input, "label": u_label}


# class SimPDEDataModule(L.LightningDataModule):
#     def __init__(
#         self,
#         zarr_path: str | Path,
#         resolution: str,
#         n_context_timesteps: int,
#         n_label_cells: int,
#         batch_size: int = 32,
#         num_workers: int = 4,
#     ) -> None:
#         super().__init__()
#         self.zarr_path = zarr_path
#         self.resolution = resolution
#         self.n_context_timesteps = n_context_timesteps
#         self.n_label_cells = n_label_cells
#         self.batch_size = batch_size
#         self.num_workers = num_workers

#     def setup(self, stage: str | None = None) -> None:
#         common = dict(
#             zarr_path=self.zarr_path,
#             resolution=self.resolution,
#             n_context_timesteps=self.n_context_timesteps,
#             n_label_cells=self.n_label_cells,
#         )
#         if stage in ("fit", None):
#             self.train_dataset = MaskedDataset1D(**common, split="train")
#             self.val_dataset = MaskedDataset1D(**common, split="valid", return_traj=True)
#         if stage == "test":
#             self.test_dataset = MaskedDataset1D(**common, split="test", return_traj=True)

#     def train_dataloader(self) -> DataLoader:
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             shuffle=True,
#             num_workers=self.num_workers,
#             pin_memory=self.num_workers > 0,
#         )

#     def val_dataloader(self) -> DataLoader:
#         return DataLoader(
#             self.val_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#         )

#     def test_dataloader(self) -> DataLoader:
#         return DataLoader(
#             self.test_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#         )
