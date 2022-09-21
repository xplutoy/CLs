import numpy as np
from typing import List, Tuple
import torch
import random


class ReservoirBuffer:
    def __init__(self, buffer_size: int,
                 attributes: List[str],
                 device
                 ) -> None:

        assert len(attributes) != 0, 'get emtpy attributes'
        self.attr_list = attributes
        self.buffer_size = buffer_size
        self.device = device
        self.nb_observed_samples = 0

    def get(self, size):
        """随机选取size大小样本

        Args:
            size (_type_): _description_
        """
        rst = []
        real_get_size = min(size, self.nb_observed_samples)
        current_buffer_size = min(self.buffer_size, self.nb_observed_samples)
        idxs = random.sample(range(current_buffer_size), real_get_size)
        for i in range(len(self.attr_list)):
            attr_name = self._get_attr_name(i)
            attr = getattr(self, attr_name)
            rst.append(attr[idxs])
        return rst

    def add(self, experiment):
        attr_name = self._get_attr_name(0)
        if not hasattr(self, attr_name):
            self._init_tensors(experiment)

        batch_size = experiment[0].shape[0]
        for i in range(batch_size):
            self.nb_observed_samples += 1
            index = self._reservior()
            if index != -1:
                for t in range(len(self.attr_list)):
                    attr_t = getattr(self, self._get_attr_name(t))
                    attr_t[index] = experiment[t][i].to(self.device)

    def _get_attr_name(self, i):
        assert 0 <= i and i < len(self.attr_list)
        return self.attr_list[i].split(':')[0]

    def _init_tensors(self, experiment: Tuple) -> None:
        assert len(self.attr_list) != 0, 'len(attr_list) should not be zero'
        assert len(self.attr_list) == len(experiment), 'need equal length!'

        for idx, attr_str in enumerate(self.attr_list):
            attr_name, attr_type = attr_str.split(':')
            if not hasattr(self, attr_str):
                setattr(self, attr_name, torch.zeros((self.buffer_size,
                        *list(experiment[idx].shape[1:])), dtype=eval(attr_type), device=self.device))

    def _reservior(self) -> int:

        if self.nb_observed_samples < self.buffer_size:
            return self.nb_observed_samples

        rand = np.random.randint(self.nb_observed_samples+1)
        if rand < self.buffer_size:
            return rand

        return -1

    def __len__(self):
        return min(self.nb_observed_samples, self.buffer_size)
