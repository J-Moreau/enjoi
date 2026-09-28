"""SEVIR dataloader utilities.

Adapted from FlowDAS' weather forecasting SEVIR dataloader:
https://github.com/umjiayx/FlowDAS/blob/main/experiments/weather_forecasting/sevir_dataloader.py
"""

import datetime
import os
from math import ceil
from pathlib import Path
from typing import Any, Callable, Dict, Sequence, Union

import h5py
import numpy as np
import numpy.random as nprand
import pandas as pd
import torch
from einops import rearrange
from torch.nn.functional import avg_pool2d

SEVIR_DATA_TYPES = ["vis", "ir069", "ir107", "vil", "lght"]
SEVIR_RAW_DTYPES = {
    "vis": np.int16,
    "ir069": np.int16,
    "ir107": np.int16,
    "vil": np.uint8,
    "lght": np.int16,
}
LIGHTING_FRAME_TIMES = np.arange(-120.0, 125.0, 5) * 60
SEVIR_DATA_SHAPE = {"lght": (48, 48)}

PREPROCESS_SCALE_SEVIR = {
    "vis": 1,
    "ir069": 1 / 1174.68,
    "ir107": 1 / 2562.43,
    "vil": 1 / 47.54,
    "lght": 1 / 0.60517,
}
PREPROCESS_OFFSET_SEVIR = {
    "vis": 0,
    "ir069": 3683.58,
    "ir107": 1552.80,
    "vil": -33.44,
    "lght": -0.02990,
}
PREPROCESS_SCALE_01 = {
    "vis": 1,
    "ir069": 1,
    "ir107": 1,
    "vil": 1 / 255,
    "lght": 1,
}
PREPROCESS_OFFSET_01 = {
    "vis": 0,
    "ir069": 0,
    "ir107": 0,
    "vil": 0,
    "lght": 0,
}

SEVIR_RAW_SEQ_LEN = 49
SEVIR_LR_RAW_SEQ_LEN = 25


def change_layout(
    data: Union[np.ndarray, torch.Tensor],
    in_layout: str = "NHWT",
    out_layout: str = "NHWT",
    ret_contiguous: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    in_layout = " ".join(in_layout.replace("C", "1"))
    out_layout = " ".join(out_layout.replace("C", "1"))
    data = rearrange(data, f"{in_layout} -> {out_layout}")
    if ret_contiguous:
        if isinstance(data, np.ndarray):
            data = np.ascontiguousarray(data)
        elif isinstance(data, torch.Tensor):
            data = data.contiguous()
    return data


def _default_sevir_paths(path: Union[str, Path]) -> tuple[Path, Path, int]:
    root = Path(path)
    if (root / "CATALOG.csv").exists():
        dataset_root = root
    elif (root / "sevir_lr" / "CATALOG.csv").exists():
        dataset_root = root / "sevir_lr"
    else:
        dataset_root = root

    raw_seq_len = SEVIR_LR_RAW_SEQ_LEN if dataset_root.name == "sevir_lr" else SEVIR_RAW_SEQ_LEN
    return dataset_root / "CATALOG.csv", dataset_root / "data", raw_seq_len


class SEVIRDataLoader:
    r"""DataLoader that loads SEVIR sequences and splits events into segments."""

    def __init__(
        self,
        data_types: Sequence[str] = ("vil",),
        seq_len: int = 49,
        raw_seq_len: int = 49,
        sample_mode: str = "sequent",
        stride: int = 12,
        batch_size: int = 1,
        layout: str = "NHWT",
        num_shard: int = 1,
        rank: int = 0,
        split_mode: str = "uneven",
        sevir_catalog: Union[str, Path, pd.DataFrame] = None,
        sevir_data_dir: Union[str, Path] = None,
        start_date: datetime.datetime = None,
        end_date: datetime.datetime = None,
        datetime_filter: Callable[[pd.Series], pd.Series] = None,
        catalog_filter: Union[str, Callable[[pd.DataFrame], pd.Series]] = "default",
        shuffle: bool = False,
        shuffle_seed: int = 1,
        output_type: Any = np.float32,
        preprocess: bool = True,
        rescale_method: str = "01",
        downsample_dict: Dict[str, Sequence[int]] = None,
        verbose: bool = False,
    ) -> None:
        self._dtypes = SEVIR_RAW_DTYPES
        self.lght_frame_times = LIGHTING_FRAME_TIMES
        self.data_shape = SEVIR_DATA_SHAPE
        self.raw_seq_len = raw_seq_len
        self.seq_len = seq_len
        self.sample_mode = sample_mode
        self.stride = stride
        self.batch_size = batch_size
        self.layout = layout
        self.num_shard = num_shard
        self.rank = rank
        self.split_mode = split_mode
        self._samples = None
        self._hdf_files: Dict[str, h5py.File] = {}
        self.data_types = list(data_types)
        self.sevir_data_dir = Path(sevir_data_dir)
        self.datetime_filter = datetime_filter
        self.catalog_filter = catalog_filter
        self.start_date = start_date
        self.end_date = end_date
        self.shuffle = shuffle
        self.shuffle_seed = int(shuffle_seed)
        self.output_type = output_type
        self.preprocess = preprocess
        self.downsample_dict = downsample_dict
        self.rescale_method = rescale_method
        self.verbose = verbose

        if isinstance(sevir_catalog, pd.DataFrame):
            self.catalog = sevir_catalog
        else:
            self.catalog = pd.read_csv(sevir_catalog, parse_dates=["time_utc"], low_memory=False)

        if self.start_date is not None:
            self.catalog = self.catalog[self.catalog.time_utc > self.start_date]
        if self.end_date is not None:
            self.catalog = self.catalog[self.catalog.time_utc <= self.end_date]
        if self.datetime_filter:
            self.catalog = self.catalog[self.datetime_filter(self.catalog.time_utc)]

        if self.catalog_filter is not None:
            if self.catalog_filter == "default":
                self.catalog_filter = lambda catalog: catalog.pct_missing == 0
            self.catalog = self.catalog[self.catalog_filter(self.catalog)]

        self._compute_samples()
        self._open_files(verbose=self.verbose)
        self.reset()

    def _compute_samples(self) -> None:
        imgts = set(self.data_types)
        filtcat = self.catalog[np.logical_or.reduce([self.catalog.img_type == imgt for imgt in self.data_types])]
        filtcat = filtcat[filtcat.file_name.map(lambda file_name: (self.sevir_data_dir / file_name).exists())]
        filtcat = filtcat.groupby("id").filter(lambda rows: imgts.issubset(set(rows["img_type"])))
        filtcat = filtcat.groupby("id").filter(lambda rows: rows.shape[0] == len(self.data_types))
        self._samples = filtcat.groupby("id").apply(lambda rows: self._df_to_series(rows, self.data_types))
        if self.shuffle:
            self.shuffle_samples()

    def shuffle_samples(self) -> None:
        self._samples = self._samples.sample(frac=1, random_state=self.shuffle_seed)

    def _df_to_series(self, df: pd.DataFrame, imgts: Sequence[str]) -> pd.DataFrame:
        row_dict = {}
        df = df.set_index("img_type")
        for imgt in imgts:
            row = df.loc[imgt]
            index = row.file_index if imgt != "lght" else row.id
            row_dict.update({f"{imgt}_filename": [row.file_name], f"{imgt}_index": [index]})
        return pd.DataFrame(row_dict)

    def _open_files(self, verbose: bool = True) -> None:
        hdf_filenames = []
        for imgt in self.data_types:
            hdf_filenames += list(np.unique(self._samples[f"{imgt}_filename"].values))
        self._hdf_files = {}
        for filename in hdf_filenames:
            if verbose:
                print("Opening HDF5 file for reading", filename)
            self._hdf_files[filename] = h5py.File(self.sevir_data_dir / filename, "r")

    def close(self) -> None:
        for filename in self._hdf_files:
            self._hdf_files[filename].close()
        self._hdf_files = {}

    @property
    def num_seq_per_event(self) -> int:
        return 1 + (self.raw_seq_len - self.seq_len) // self.stride

    @property
    def total_num_seq(self) -> int:
        return int(self.num_seq_per_event * self.num_event)

    @property
    def total_num_event(self) -> int:
        return int(self._samples.shape[0])

    @property
    def start_event_idx(self) -> int:
        return self.total_num_event // self.num_shard * self.rank

    @property
    def end_event_idx(self) -> int:
        if self.split_mode == "ceil":
            last_start_event_idx = self.total_num_event // self.num_shard * (self.num_shard - 1)
            num_event = self.total_num_event - last_start_event_idx
            return self.start_event_idx + num_event
        if self.split_mode == "floor":
            return self.total_num_event // self.num_shard * (self.rank + 1)
        if self.rank == self.num_shard - 1:
            return self.total_num_event
        return self.total_num_event // self.num_shard * (self.rank + 1)

    @property
    def num_event(self) -> int:
        return self.end_event_idx - self.start_event_idx

    def _read_data(self, row: pd.Series, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        imgts = np.unique([key.split("_")[0] for key in list(row.keys())])
        for imgt in imgts:
            filename = row[f"{imgt}_filename"]
            index = row[f"{imgt}_index"]
            t_slice = slice(0, None)
            if imgt == "lght":
                lght_data = self._hdf_files[filename][index][:]
                data_i = self._lght_to_grid(lght_data, t_slice)
            else:
                data_i = self._hdf_files[filename][imgt][index : index + 1, :, :, t_slice]
            data[imgt] = np.concatenate((data[imgt], data_i), axis=0) if imgt in data else data_i
        return data

    def _lght_to_grid(self, data: np.ndarray, t_slice: slice = slice(0, None)) -> np.ndarray:
        out_size = (*self.data_shape["lght"], len(self.lght_frame_times)) if t_slice.stop is None else (*self.data_shape["lght"], 1)
        if data.shape[0] == 0:
            return np.zeros((1,) + out_size, dtype=np.float32)

        x, y = data[:, 3], data[:, 4]
        mask = np.logical_and.reduce([x >= 0, x < out_size[0], y >= 0, y < out_size[1]])
        data = data[mask, :]
        if data.shape[0] == 0:
            return np.zeros((1,) + out_size, dtype=np.float32)

        times = data[:, 0]
        if t_slice.stop is not None:
            if t_slice.stop > 0:
                if t_slice.stop < len(self.lght_frame_times):
                    time_mask = np.logical_and(times >= self.lght_frame_times[t_slice.stop - 1], times < self.lght_frame_times[t_slice.stop])
                else:
                    time_mask = times >= self.lght_frame_times[-1]
            else:
                time_mask = np.logical_and(times >= self.lght_frame_times[0], times < self.lght_frame_times[1])
            data = data[time_mask, :]
            z = np.zeros(data.shape[0], dtype=np.int64)
        else:
            z = np.digitize(times, self.lght_frame_times) - 1
            z[z == -1] = 0

        x = data[:, 3].astype(np.int64)
        y = data[:, 4].astype(np.int64)
        indices = np.ravel_multi_index(np.array([y, x, z]), out_size)
        counts = np.bincount(indices, minlength=np.prod(out_size))
        return np.reshape(counts, out_size).astype(np.int16)[np.newaxis, :]

    @property
    def sample_count(self) -> int:
        return self._sample_count

    def inc_sample_count(self) -> None:
        self._sample_count += 1

    def reset(self, shuffle: bool = None) -> None:
        self._curr_event_idx = self.start_event_idx
        self._curr_seq_idx = 0
        self._sample_count = 0
        if shuffle is None:
            shuffle = self.shuffle
        if shuffle:
            self.shuffle_samples()

    def __len__(self) -> int:
        return self.total_num_seq // self.batch_size

    @property
    def use_up(self) -> bool:
        if self.sample_mode == "random":
            return False
        curr_event_remain_seq = self.num_seq_per_event - self._curr_seq_idx
        all_remain_seq = curr_event_remain_seq + (self.end_event_idx - self._curr_event_idx - 1) * self.num_seq_per_event
        if self.split_mode == "floor":
            return all_remain_seq < self.batch_size
        return all_remain_seq <= 0

    def _load_event_batch(self, event_idx: int, event_batch_size: int) -> list[np.ndarray]:
        event_idx_slice_end = event_idx + event_batch_size
        pad_size = 0
        if event_idx_slice_end > self.end_event_idx:
            pad_size = event_idx_slice_end - self.end_event_idx
            event_idx_slice_end = self.end_event_idx
        pd_batch = self._samples.iloc[event_idx:event_idx_slice_end]
        data = {}
        for _, row in pd_batch.iterrows():
            data = self._read_data(row, data)
        event_batch = []
        for imgt in self.data_types:
            if pad_size > 0:
                pad_shape = [pad_size] + list(data[imgt].shape[1:])
                event_batch.append(np.concatenate((data[imgt].astype(self.output_type), np.zeros(pad_shape, dtype=self.output_type)), axis=0))
            else:
                event_batch.append(data[imgt].astype(self.output_type))
        return event_batch

    def __iter__(self) -> "SEVIRDataLoader":
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        if self.sample_mode == "random":
            self.inc_sample_count()
            ret_dict = self._random_sample()
        else:
            if self.use_up:
                raise StopIteration
            self.inc_sample_count()
            ret_dict = self._sequent_sample()
        ret_dict = self.data_dict_to_tensor(data_dict=ret_dict, data_types=self.data_types)
        if self.preprocess:
            ret_dict = self.preprocess_data_dict(data_dict=ret_dict, data_types=self.data_types, layout=self.layout, rescale=self.rescale_method)
        if self.downsample_dict is not None:
            ret_dict = self.downsample_data_dict(data_dict=ret_dict, data_types=self.data_types, factors_dict=self.downsample_dict, layout=self.layout)
        return ret_dict

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        data_dict = self._idx_sample(index=index)
        data_dict = self.data_dict_to_tensor(data_dict=data_dict, data_types=self.data_types)
        if self.preprocess:
            data_dict = self.preprocess_data_dict(data_dict=data_dict, data_types=self.data_types, layout=self.layout, rescale=self.rescale_method)
        if self.downsample_dict is not None:
            data_dict = self.downsample_data_dict(data_dict=data_dict, data_types=self.data_types, factors_dict=self.downsample_dict, layout=self.layout)
        return data_dict

    @staticmethod
    def preprocess_data_dict(
        data_dict: Dict[str, Union[np.ndarray, torch.Tensor]],
        data_types: Sequence[str] = None,
        layout: str = "NHWT",
        rescale: str = "01",
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        if rescale == "sevir":
            scale_dict = PREPROCESS_SCALE_SEVIR
            offset_dict = PREPROCESS_OFFSET_SEVIR
        elif rescale == "01":
            scale_dict = PREPROCESS_SCALE_01
            offset_dict = PREPROCESS_OFFSET_01
        else:
            raise ValueError(f"Invalid rescale option: {rescale}.")
        if data_types is None:
            data_types = data_dict.keys()
        for key, data in data_dict.items():
            if key in data_types:
                if isinstance(data, np.ndarray):
                    data = data.astype(np.float32)
                elif isinstance(data, torch.Tensor):
                    data = data.float()
                else:
                    raise TypeError
                data_dict[key] = change_layout(data=scale_dict[key] * (data + offset_dict[key]), in_layout="NHWT", out_layout=layout)
        return data_dict

    @staticmethod
    def process_data_dict_back(
        data_dict: Dict[str, torch.Tensor],
        data_types: Sequence[str] = None,
        rescale: str = "01",
    ) -> Dict[str, torch.Tensor]:
        if rescale == "sevir":
            scale_dict = PREPROCESS_SCALE_SEVIR
            offset_dict = PREPROCESS_OFFSET_SEVIR
        elif rescale == "01":
            scale_dict = PREPROCESS_SCALE_01
            offset_dict = PREPROCESS_OFFSET_01
        else:
            raise ValueError(f"Invalid rescale option: {rescale}.")
        if data_types is None:
            data_types = data_dict.keys()
        for key in data_types:
            data_dict[key] = data_dict[key].float() / scale_dict[key] - offset_dict[key]
        return data_dict

    @staticmethod
    def data_dict_to_tensor(
        data_dict: Dict[str, Union[np.ndarray, torch.Tensor]],
        data_types: Sequence[str] = None,
    ) -> Dict[str, torch.Tensor]:
        ret_dict = {}
        if data_types is None:
            data_types = data_dict.keys()
        for key, data in data_dict.items():
            if key in data_types:
                if isinstance(data, torch.Tensor):
                    ret_dict[key] = data.detach().clone()
                elif isinstance(data, np.ndarray):
                    ret_dict[key] = torch.from_numpy(data)
                else:
                    raise ValueError(f"Invalid data type: {type(data)}. Should be torch.Tensor or np.ndarray")
            else:
                ret_dict[key] = data
        return ret_dict

    @staticmethod
    def downsample_data_dict(
        data_dict: Dict[str, Union[np.ndarray, torch.Tensor]],
        data_types: Sequence[str] = None,
        factors_dict: Dict[str, Sequence[int]] = None,
        layout: str = "NHWT",
    ) -> Dict[str, torch.Tensor]:
        if factors_dict is None:
            factors_dict = {}
        if data_types is None:
            data_types = data_dict.keys()
        downsampled_data_dict = SEVIRDataLoader.data_dict_to_tensor(data_dict=data_dict, data_types=data_types)
        for key, data in data_dict.items():
            factors = factors_dict.get(key, None)
            if factors is not None:
                downsampled_data_dict[key] = change_layout(data=downsampled_data_dict[key], in_layout=layout, out_layout="NTHW")
                t_slice = [slice(None, None)] * 4
                t_slice[1] = slice(None, None, factors[0])
                downsampled_data_dict[key] = downsampled_data_dict[key][tuple(t_slice)]
                downsampled_data_dict[key] = avg_pool2d(input=downsampled_data_dict[key], kernel_size=(factors[1], factors[2]))
                downsampled_data_dict[key] = change_layout(data=downsampled_data_dict[key], in_layout="NTHW", out_layout=layout)
        return downsampled_data_dict

    def _random_sample(self) -> Dict[str, np.ndarray]:
        num_sampled = 0
        event_idx_list = nprand.randint(low=self.start_event_idx, high=self.end_event_idx, size=self.batch_size)
        seq_idx_list = nprand.randint(low=0, high=self.num_seq_per_event, size=self.batch_size)
        seq_slice_list = [slice(seq_idx * self.stride, seq_idx * self.stride + self.seq_len) for seq_idx in seq_idx_list]
        ret_dict = {}
        while num_sampled < self.batch_size:
            event = self._load_event_batch(event_idx=event_idx_list[num_sampled], event_batch_size=1)
            for imgt_idx, imgt in enumerate(self.data_types):
                sampled_seq = event[imgt_idx][[0], :, :, seq_slice_list[num_sampled]]
                ret_dict[imgt] = np.concatenate((ret_dict[imgt], sampled_seq), axis=0) if imgt in ret_dict else sampled_seq
            num_sampled += 1
        return ret_dict

    def _sequent_sample(self) -> Dict[str, np.ndarray]:
        event_idx = self._curr_event_idx
        seq_idx = self._curr_seq_idx
        num_sampled = 0
        sampled_idx_list = []
        while num_sampled < self.batch_size:
            sampled_idx_list.append({"event_idx": event_idx, "seq_idx": seq_idx})
            seq_idx += 1
            if seq_idx >= self.num_seq_per_event:
                event_idx += 1
                seq_idx = 0
            num_sampled += 1

        start_event_idx = sampled_idx_list[0]["event_idx"]
        event_batch_size = sampled_idx_list[-1]["event_idx"] - start_event_idx + 1
        event_batch = self._load_event_batch(event_idx=start_event_idx, event_batch_size=event_batch_size)
        ret_dict = {"mask": []}
        all_no_pad_flag = True
        for sampled_idx in sampled_idx_list:
            batch_slice = [sampled_idx["event_idx"] - start_event_idx]
            seq_slice = slice(sampled_idx["seq_idx"] * self.stride, sampled_idx["seq_idx"] * self.stride + self.seq_len)
            for imgt_idx, imgt in enumerate(self.data_types):
                sampled_seq = event_batch[imgt_idx][batch_slice, :, :, seq_slice]
                ret_dict[imgt] = np.concatenate((ret_dict[imgt], sampled_seq), axis=0) if imgt in ret_dict else sampled_seq
            no_pad_flag = sampled_idx["event_idx"] < self.end_event_idx
            if not no_pad_flag:
                all_no_pad_flag = False
            ret_dict["mask"].append(no_pad_flag)
        if all_no_pad_flag:
            ret_dict["mask"] = None
        self._curr_event_idx = event_idx
        self._curr_seq_idx = seq_idx
        return ret_dict

    def _idx_sample(self, index: int) -> Dict[str, np.ndarray]:
        event_idx = (index * self.batch_size) // self.num_seq_per_event
        seq_idx = (index * self.batch_size) % self.num_seq_per_event
        num_sampled = 0
        sampled_idx_list = []
        while num_sampled < self.batch_size:
            sampled_idx_list.append({"event_idx": event_idx, "seq_idx": seq_idx})
            seq_idx += 1
            if seq_idx >= self.num_seq_per_event:
                event_idx += 1
                seq_idx = 0
            num_sampled += 1

        start_event_idx = sampled_idx_list[0]["event_idx"]
        event_batch_size = sampled_idx_list[-1]["event_idx"] - start_event_idx + 1
        event_batch = self._load_event_batch(event_idx=start_event_idx, event_batch_size=event_batch_size)
        ret_dict = {}
        for sampled_idx in sampled_idx_list:
            batch_slice = [sampled_idx["event_idx"] - start_event_idx]
            seq_slice = slice(sampled_idx["seq_idx"] * self.stride, sampled_idx["seq_idx"] * self.stride + self.seq_len)
            for imgt_idx, imgt in enumerate(self.data_types):
                sampled_seq = event_batch[imgt_idx][batch_slice, :, :, seq_slice]
                ret_dict[imgt] = np.concatenate((ret_dict[imgt], sampled_seq), axis=0) if imgt in ret_dict else sampled_seq
        return ret_dict


def load_sevir_dataset(
    path: Union[str, Path],
    data_type: str = "vil",
    seq_len: int = None,
    stride: int = 12,
    max_events: int = 64,
    layout: str = "NTCHW",
    rescale_method: str = "01",
) -> Dict[str, Any]:
    catalog_path, data_dir, raw_seq_len = _default_sevir_paths(path)
    if seq_len is None:
        seq_len = raw_seq_len
    loader = SEVIRDataLoader(
        data_types=(data_type,),
        seq_len=seq_len,
        raw_seq_len=raw_seq_len,
        stride=stride,
        batch_size=1,
        layout=layout,
        sevir_catalog=catalog_path,
        sevir_data_dir=data_dir,
        preprocess=True,
        rescale_method=rescale_method,
    )
    try:
        n_events = min(max_events, len(loader)) if max_events is not None else len(loader)
        data = torch.stack([loader[index][data_type][0] for index in range(n_events)])
    finally:
        loader.close()
    return {"data": data.float(), "catalog": loader._samples.iloc[:n_events], "data_type": data_type}