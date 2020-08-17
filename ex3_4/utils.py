from typing import Dict, List

import random
import json
import os
import os.path as osp

import numpy as np
import torch
from torch import nn


def save_model(save_dir: str,
               net: nn.Module,
               final_model_metrics: Dict[str, float],
               training_metrics: Dict[str, List[float]]
               ) -> None:
    os.makedirs(save_dir, exist_ok=True)

    net_path = osp.join(save_dir, "net_state_dict.pth")
    torch.save(net.state_dict(), net_path)

    final_model_metrics_path = osp.join(save_dir, "final_model_metrics.json")
    with open(final_model_metrics_path, 'w') as f:
        json.dump(final_model_metrics, f)

    training_metrics_path = osp.join(save_dir, "training_metrics.json")
    with open(training_metrics_path, 'w') as f:
        json.dump(training_metrics, f)


class RandomStateContextManager:
    def __init__(self, seed: int):
        self.seed = seed
        self.orig_random_state = random.getstate()
        self.orig_numpy_state = np.random.get_state()
        self.orig_torch_state = torch.get_rng_state()

    def __enter__(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        if self.seed is not None:
            torch.manual_seed(self.seed)
        else:
            torch.seed()

    def __exit__(self, exc_type, exc_val, exc_tb):
        random.setstate(self.orig_random_state)
        np.random.set_state(self.orig_numpy_state)
        torch.set_rng_state(self.orig_torch_state)


def _test_random_state_context_manager():
    def print_randoms():
        print(random.randint(0, 1000), ',',
              np.random.randint(1000), ',',
              torch.randint(1000, size=(1,)).item())

    def with_manager():
        print("\nwith context manager:")
        with RandomStateContextManager(34):
            print_randoms()

    def with_None_manager():
        print("\nwith None-seed context manager:")
        with RandomStateContextManager(None):
            print_randoms()

    def without_manager():
        print("\nwithout context manager")
        print_randoms()

    without_manager()
    with_manager()
    without_manager()
    with_manager()
    with_None_manager()
    with_None_manager()


if __name__ == '__main__':
    _test_random_state_context_manager()
