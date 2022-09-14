from argparse import Namespace
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from continuum import rehearsal,Logger
from utils.buffer import rand_batch_get
from utils.weight import init_weights
from tqdm import tqdm


class Er(nn.Module):
    def __init__(self,
                 net: nn.Module,
                 buffer_size: int,
                 lr: float,
                 batch_size: int,
                 minibatch_size: int,
                 metric_log: Logger,
                 device: torch.device = 'cpu',
                 ) -> None:
        super().__init__()

        self.net = net.to(device)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.device = device
        self.metric_log = metric_log

        self.buffer = rehearsal.RehearsalMemory(
            memory_size=buffer_size,
            herding_method="random"
        )
        init_weights(self.net)

    def observe(self, taskset):
        self.net.train()
        task_loader = DataLoader(
            taskset, batch_size=self.batch_size, shuffle=True)

        for x, y, t in task_loader:

            if len(self.buffer) != 0:
                _x, _y, _t = rand_batch_get(
                    self.buffer, self.minibatch_size)
                x = torch.cat((x, torch.from_numpy(np.expand_dims(_x, 1))))
                y = torch.cat((y, torch.from_numpy(_y)))
                t = torch.cat((t, torch.from_numpy(_t)))

            x = x.to(self.device)
            y = y.to(self.device)

            logits = self.net(x)
            loss = self.criterion(logits, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            preds = torch.argmax(logits, dim=-1)
            # acc = (preds == y).to(torch.float64).mean().item()
            # tqdm.write(
            #     f"task_id={self.log.nb_tasks}, online_acc = {acc}, loss = {loss.item()}")

            self.metric_log.add([preds, y, t], subset='train')

        self.buffer.add(*taskset.get_raw_samples(), None)

    @torch.no_grad()
    def eval(self, taskset):
        self.net.eval()
        task_loader = DataLoader(
            taskset, batch_size=self.batch_size, shuffle=False)

        for x, y, t in task_loader:

            x = x.to(self.device)
            y = y.to(self.device)

            logits = self.net(x)
            preds = torch.argmax(logits, dim=-1)
            self.metric_log.add([preds, y, t], subset='test')
