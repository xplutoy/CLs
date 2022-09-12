import random
from continuum import rehearsal
import numpy as np
from typing import Tuple


def rand_batch_get(memory: rehearsal.RehearsalMemory, batch_size) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """随机从memory中选取一batch数据

    Args:
        memory (rehearsal.RehearsalMemory):
        batch_size (_type_): 
    """
    x, y, t = memory.get()
    idx = random.sample(range(len(y)), batch_size)
    return x[idx], y[idx], t[idx]
