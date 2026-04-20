import math
import numpy as np
from typing import Iterator, Optional
import torch
from torch.utils.data.dataloader import _BaseDataLoaderIter
from torch.utils.data import Dataset, _DatasetKind
from torch.utils.data.distributed import DistributedSampler
from operator import itemgetter
import torch.distributed as dist
import warnings
import time
import logging

__all__ = ['InfoBatch']

logger = logging.getLogger(__name__)


def info_hack_indices(self):
    with torch.autograd.profiler.record_function(self._profile_name):
        if self._sampler_iter is None:
            # TODO(https://github.com/pytorch/pytorch/issues/76750)
            self._reset()  # type: ignore[call-arg]
        if isinstance(self._dataset, InfoBatch):
            indices, data = self._next_data()
        else:
            data = self._next_data()
        self._num_yielded += 1
        if self._dataset_kind == _DatasetKind.Iterable and \
                self._IterableDataset_len_called is not None and \
                self._num_yielded > self._IterableDataset_len_called:
            warn_msg = ("Length of IterableDataset {} was reported to be {} (when accessing len(dataloader)), but {} "
                        "samples have been fetched. ").format(self._dataset, self._IterableDataset_len_called,
                                                              self._num_yielded)
            if self._num_workers > 0:
                warn_msg += ("For multiprocessing data-loading, this could be caused by not properly configuring the "
                             "IterableDataset replica at each worker. Please see "
                             "https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset for examples.")
            warnings.warn(warn_msg)
        if isinstance(self._dataset, InfoBatch):
            self._dataset.set_active_indices(indices)
        return data


_BaseDataLoaderIter.__next__ = info_hack_indices


@torch.no_grad()
def concat_all_gather(tensor, dim=0):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=dim)
    return output


@torch.no_grad()
def _safe_concat_all_gather_1d(tensor: torch.Tensor):
    """
    All-gather a 1D tensor across ranks even when each rank has a different length.

    Strategy:
    - First all_gather the local lengths.
    - Pad each local tensor to the global max length so shapes match.
    - all_gather the padded tensors.
    - Slice each gathered chunk back to its true length and concatenate.

    This avoids shape mismatch crashes in distributed all_gather for variable batch sizes
    (e.g., last batch, drop_last=False) and prevents downstream negative-dimension errors.
    """
    assert tensor.ndim == 1, "_safe_concat_all_gather_1d expects a 1D tensor"
    if not (dist.is_available() and dist.is_initialized()):
        return tensor

    world_size = dist.get_world_size()
    device = tensor.device

    # Gather lengths
    local_len = torch.tensor([tensor.numel()], device=device, dtype=torch.long)
    len_list = [torch.zeros_like(local_len) for _ in range(world_size)]
    dist.all_gather(len_list, local_len)
    lengths = torch.cat(len_list, dim=0).cpu().tolist()
    max_len = max(lengths) if lengths else 0
    if max_len == 0:
        return torch.empty(0, dtype=tensor.dtype, device=device)

    # Pad to max_len
    cur_len = tensor.numel()
    if cur_len < max_len:
        if tensor.dtype.is_floating_point or tensor.dtype in (torch.float16, torch.bfloat16):
            pad_val = 0.0
        else:
            pad_val = 0
        pad = torch.full((max_len - cur_len,), pad_val, dtype=tensor.dtype, device=device)
        tensor_padded = torch.cat([tensor, pad], dim=0)
    else:
        tensor_padded = tensor

    # All-gather padded tensors (same shape across ranks)
    gather_list = [torch.empty_like(tensor_padded) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor_padded)

    # Slice back to true lengths and concat
    chunks = [chunk[:ln] for chunk, ln in zip(gather_list, lengths)]
    return torch.cat(chunks, dim=0)


class InfoBatch(Dataset):
    """
    InfoBatch aims to achieve lossless training speed up by randomly prunes a portion of less informative samples
    based on the loss distribution and rescales the gradients of the remaining samples to approximate the original
    gradient. See https://arxiv.org/pdf/2303.04947.pdf

    .. note::.
        Dataset is assumed to be of constant size.

    Args:
        dataset: Dataset used for training.
        num_epochs (int): The number of epochs for pruning.
        prune_ratio (float, optional): The proportion of samples being pruned during training.
        delta (float, optional): The first delta * num_epochs the pruning process is conducted. It should be close to 1. Defaults to 0.875.
    """

    def __init__(self, dataset: Dataset, num_epochs: int,
                 prune_ratio: float = 0.5, delta: float = 0.875):
        self.dataset = dataset
        self.keep_ratio = min(1.0, max(1e-1, 1.0 - prune_ratio))
        self.num_epochs = num_epochs
        self.delta = delta
        # self.scores stores the loss value of each sample. Note that smaller value indicates the sample is better learned by the network.
        self.scores = torch.ones(len(self.dataset)) * 3
        self.weights = torch.ones(len(self.dataset))
        self.num_pruned_samples = 0
        self.cur_batch_index = None

    def set_active_indices(self, cur_batch_indices: torch.Tensor):
        # Normalize to 1D torch.LongTensor to avoid dtype/shape mismatches later
        if not torch.is_tensor(cur_batch_indices):
            cur_batch_indices = torch.tensor(cur_batch_indices, dtype=torch.long)
        else:
            cur_batch_indices = cur_batch_indices.to(dtype=torch.long)
        if cur_batch_indices.ndim > 1:
            cur_batch_indices = cur_batch_indices.view(-1)
        self.cur_batch_index = cur_batch_indices

    def update(self, values):
        assert isinstance(values, torch.Tensor)
        # print(values.shape)
        # exit()
        batch_size = values.shape[0]
        assert self.cur_batch_index is not None and len(self.cur_batch_index) == batch_size, 'not enough index'
        device = values.device
        weights = self.weights[self.cur_batch_index].to(device)
        indices = self.cur_batch_index.to(device)
        loss_val = values.detach().clone()
        self.cur_batch_index = None

        # Basic validation: indices must be in range and non-negative
        idx_min = 0
        idx_max = len(self.dataset)
        valid_mask = (indices >= idx_min) & (indices < idx_max)
        if (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0):
            invalid = (~valid_mask).sum().item()
            if invalid > 0:
                logger.warning(f"[InfoBatch] update: filtered {invalid} invalid indices out of {indices.numel()}")
        if valid_mask.sum().item() < indices.numel():
            indices = indices[valid_mask]
            loss_val = loss_val[valid_mask]

        if dist.is_available() and dist.is_initialized():
            # Variable-length safe all_gather for indices and loss values
            indices = _safe_concat_all_gather_1d(indices.view(-1))
            loss_val = _safe_concat_all_gather_1d(loss_val.view(-1))

        # Deduplicate indices if any (keep last occurrence)
        if indices.numel() > 0:
            # Use CPU dict to keep last value per index for simplicity and robustness
            kv = {}
            for i in range(indices.numel()):
                kv[int(indices[i].item())] = float(loss_val[i].item())
            if len(kv) != indices.numel():
                # Rebuild tensors after dedup
                k_list = list(kv.keys())
                v_list = list(kv.values())
                indices = torch.tensor(k_list, dtype=torch.long, device='cpu')
                loss_val = torch.tensor(v_list, dtype=values.dtype, device='cpu')
            else:
                indices = indices.cpu().long()
                loss_val = loss_val.cpu()
        if indices.numel() > 0:
            self.scores[indices] = loss_val
        values.mul_(weights)
        return values.mean()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # self.cur_batch_index.append(index)
        return index, self.dataset[index]  # , index
        # return self.dataset[index], index, self.scores[index]

    def prune(self):
        # Prune samples that are well learned, rebalance the weight by scaling up remaining
        # well learned samples' learning rate to keep estimation about the same
        # for the next version, also consider new class balance
        t0 = time.perf_counter()
        well_learned_mask = (self.scores < self.scores.mean()).numpy()
        well_learned_indices = np.where(well_learned_mask)[0]
        remained_indices = np.where(~well_learned_mask)[0].tolist()
        # print('#well learned samples %d, #remained samples %d, len(dataset) = %d' % (np.sum(well_learned_mask), np.sum(~well_learned_mask), len(self.dataset)))
        selected_indices = np.random.choice(well_learned_indices, int(
            self.keep_ratio * len(well_learned_indices)), replace=False)
        self.reset_weights()
        if len(selected_indices) > 0:
            self.weights[selected_indices] = 1 / self.keep_ratio
            remained_indices.extend(selected_indices)
        self.num_pruned_samples += len(self.dataset) - len(remained_indices)
        np.random.shuffle(remained_indices)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0):
            logger.info(
                f"[InfoBatch-Timing] prune: well_learned={int(np.sum(well_learned_mask))}, "
                f"selected={int(len(selected_indices))}, remained={len(remained_indices)}, "
                f"pruned_round={len(self.dataset) - len(remained_indices)}, time={dt_ms:.1f} ms"
            )
        return remained_indices

    @property
    def sampler(self):
        sampler = IBSampler(self)
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedIBSampler(sampler)
        return sampler

    def no_prune(self):
        t0 = time.perf_counter()
        samples_indices = list(range(len(self)))
        np.random.shuffle(samples_indices)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0):
            logger.info(
                f"[InfoBatch-Timing] no_prune: remained={len(samples_indices)}, time={dt_ms:.1f} ms"
            )
        return samples_indices

    def mean_score(self):
        return self.scores.mean()

    def get_weights(self, indexes):
        return self.weights[indexes]

    def get_pruned_count(self):
        return self.num_pruned_samples

    @property
    def stop_prune(self):
        return self.num_epochs * self.delta

    def reset_weights(self):
        self.weights[:] = 1


class IBSampler(object):
    def __init__(self, dataset: InfoBatch):
        self.dataset = dataset
        self.stop_prune = dataset.stop_prune
        self.iterations = 0
        self.sample_indices = None
        self.iter_obj = None
        self.reset()

    def __getitem__(self, idx):
        return self.sample_indices[idx]

    def reset(self):
        t_total0 = time.perf_counter()
        np.random.seed(self.iterations)
        if self.iterations > self.stop_prune:
            # print('we are going to stop prune, #stop prune %d, #cur iterations %d' % (self.iterations, self.stop_prune))
            if self.iterations == self.stop_prune + 1:
                self.dataset.reset_weights()
            phase = "no_prune"
            t_phase0 = time.perf_counter()
            self.sample_indices = self.dataset.no_prune()
            phase_dt_ms = (time.perf_counter() - t_phase0) * 1000.0
        else:
            # print('we are going to continue pruning, #stop prune %d, #cur iterations %d' % (self.iterations, self.stop_prune))
            phase = "prune"
            t_phase0 = time.perf_counter()
            self.sample_indices = self.dataset.prune()
            phase_dt_ms = (time.perf_counter() - t_phase0) * 1000.0
        self.iter_obj = iter(self.sample_indices)
        total_dt_ms = (time.perf_counter() - t_total0) * 1000.0

        if (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0):
            logger.info(
                f"[InfoBatch-Timing] IBSampler.reset: iter={self.iterations}, phase={phase}, "
                f"samples={len(self.sample_indices)}, phase_time={phase_dt_ms:.1f} ms, "
                f"total_time={total_dt_ms:.1f} ms"
            )
        self.iterations += 1

    def __next__(self):
        return next(self.iter_obj)  # may raise StopIteration

    def __len__(self):
        return len(self.sample_indices)

    def __iter__(self):
        self.reset()
        return self


class DistributedIBSampler(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler can change size during training.
    """

    class DatasetFromSampler(Dataset):
        def __init__(self, sampler: IBSampler):
            self.dataset = sampler
            # self.indices = None

        def reset(self, ):
            self.indices = None
            self.dataset.reset()

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index: int):
            """Gets element of the dataset.
            Args:
                index: index of the element in the dataset
            Returns:
                Single element by index
            """
            # if self.indices is None:
            #    self.indices = list(self.dataset)
            return self.dataset[index]

    def __init__(self, dataset: IBSampler, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = True) -> None:
        sampler = self.DatasetFromSampler(dataset)
        super(DistributedIBSampler, self).__init__(
            sampler, num_replicas, rank, shuffle, seed, drop_last)
        self.sampler = sampler
        self.dataset = sampler.dataset.dataset  # the real dataset.
        self.iter_obj = None

    def __iter__(self) -> Iterator[int]:
        """
        Notes self.dataset is actually an instance of IBSampler rather than InfoBatch.
        """
        t_total0 = time.perf_counter()
        t_reset0 = time.perf_counter()
        self.sampler.reset()
        reset_dt_ms = (time.perf_counter() - t_reset0) * 1000.0
        if self.drop_last and len(self.sampler) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.sampler) - self.num_replicas) /
                self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(
                len(self.sampler) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        t_build0 = time.perf_counter()
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # type: ignore[arg-type]
            indices = torch.randperm(len(self.sampler), generator=g).tolist()
        else:
            indices = list(range(len(self.sampler)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size /
                                                len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size
        indices = indices[self.rank:self.total_size:self.num_replicas]
        # print('distribute iter is called')
        self.iter_obj = iter(itemgetter(*indices)(self.sampler))
        build_dt_ms = (time.perf_counter() - t_build0) * 1000.0
        total_dt_ms = (time.perf_counter() - t_total0) * 1000.0
        if (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0):
            logger.info(
                f"[InfoBatch-Timing] DistributedIBSampler.__iter__: reset={reset_dt_ms:.1f} ms, "
                f"build_indices={build_dt_ms:.1f} ms, total={total_dt_ms:.1f} ms, "
                f"world_size={self.num_replicas}, num_samples={self.num_samples}, "
                f"total_size={self.total_size}, drop_last={self.drop_last}"
            )
        return self.iter_obj
