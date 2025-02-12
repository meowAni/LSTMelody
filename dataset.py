# defines custom dataset class to handle data stored in shards

import torch
from torch.utils.data import Dataset
import random
from multiprocessing import Manager
import threading

from parse_config import Config

class ShardedDataset(Dataset):
  def __init__(self, paths, rel_idxs, cache_size=3):
    self.paths = paths
    self.rel_idxs = rel_idxs
    self.n_songs = rel_idxs[-1] + 1
    self.cache_size = cache_size

    self.manager = Manager()
    self.cache = self.manager.dict()
    self.lock = threading.Lock()

  def __len__(self):
    return self.n_songs * Config.get("preprocessing")['samples_per_file']

  def _get_idxs(self, idx):
    idx = idx // Config.get("preprocessing")['samples_per_file']

    # binary search on shard to find shard
    low, high = 0, len(self.rel_idxs) - 1
    while (low < high):
      mid = (low + high) // 2
      if (self.rel_idxs[mid] < idx):
        low = mid + 1
      else:
        high = mid
    shard_idx = low
    
    if (shard_idx == 0):
      abs_idx = idx
    else:
      abs_idx = idx - self.rel_idxs[shard_idx - 1] - 1
    return shard_idx, abs_idx

  def _load_shard(self, shard_idx):
    shard_path = self.paths[shard_idx]
    if shard_path in self.cache:
      return self.cache[shard_path]

    shard = torch.load(shard_path)
    with self.lock:
      if len(self.cache) >= self.cache_size:
          self.cache.pop(next(iter(self.cache)))  # Remove oldest item
      self.cache[shard_path] = shard

    return shard

  def __getitem__(self, idx):
    shard_idx, abs_idx = self._get_idxs(idx)
    shard = self._load_shard(shard_idx)

    seq_len = Config.get("design")['sequence_length']
    sample_len = shard[abs_idx].shape[0]

    if (sample_len - 1 - seq_len <= 0):
      return self.__getitem__(random.randint(0, self.n_songs * Config.get("preprocessing")['samples_per_file']))
    
    sub_idx = random.randint(0, sample_len - 1 - seq_len)
    return shard[abs_idx][sub_idx: sub_idx + seq_len], shard[abs_idx][sub_idx + seq_len]