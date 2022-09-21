from argparse import Namespace
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import rand_batch_get
from utils import init_weights


class Er(nn.Module):
    def __init__(self,
                net: nn.Module,
                memory,
                lr: float,
                batch_size: int,
                minibatch_size: int,
                device: torch.device = 'cpu'
                ) -> None:
        super().__init__()

        self.net = net.to(device)
        self.buffer = memory
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=0.9)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.device = device

        init_weights(self.net)

    def observe(self, taskset, log=None):
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
            if log != None:
                log.add([preds, y, t], subset='train')
                log.add(loss.item(), 'loss', 'train')

        self.buffer.add(*taskset.get_raw_samples(), None)

    @torch.no_grad()
    def eval(self, taskset, log=None):
        self.net.eval()
        task_loader = DataLoader(
            taskset, batch_size=self.batch_size, shuffle=False)

        for x, y, t in task_loader:

            x = x.to(self.device)
            y = y.to(self.device)

            logits = self.net(x)
            preds = torch.argmax(logits, dim=-1)
            loss = self.criterion(logits, y)
            if log != None:
                log.add([preds, y, t], subset='test')
                log.add(loss.item(), 'loss', 'test')
