import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import init_weights
from buffer import ReservoirBuffer


class ErNative(nn.Module):
    def __init__(self,
                 net: nn.Module,
                 buffer_size: int,
                 lr: float,
                 batch_size: int,
                 minibatch_size: int,
                 device: torch.device = 'cpu'
                 ) -> None:
        super().__init__()

        self.net = net.to(device)
        self.buffer = ReservoirBuffer(
            buffer_size, ['x:torch.float32', 'y:torch.int'], device)
        self.optimizer = torch.optim.SGD(
            self.net.parameters(), lr=lr, momentum=0.9)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.device = device

        init_weights(self.net)

    def observe(self, taskset, metric_log=None):
        self.net.train()
        task_loader = DataLoader(
            taskset, batch_size=self.batch_size, shuffle=True)

        for x, y, t in task_loader:
            _x = x.to(self.device)
            _y = y.to(self.device)

            if len(self.buffer) != 0:
                experiment = self.buffer.get(self.minibatch_size)
                _x = torch.cat((_x, experiment[0]))
                _y = torch.cat((_y, experiment[1]))

            logits = self.net(_x)
            loss = self.criterion(logits, _y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.buffer.add((x, y))

            preds = torch.argmax(logits, dim=-1)
            if metric_log != None:
                metric_log.add([preds, _y, t], subset='train')
                metric_log.add(loss.item(), 'loss', 'train')

        # self.buffer.add(*taskset.get_raw_samples(), None)

    @torch.no_grad()
    def eval(self, taskset, metric_log=None):
        self.net.eval()
        task_loader = DataLoader(
            taskset, batch_size=self.batch_size, shuffle=False)

        for x, y, t in task_loader:

            x = x.to(self.device)
            y = y.to(self.device)

            logits = self.net(x)
            preds = torch.argmax(logits, dim=-1)
            loss = self.criterion(logits, y)
            if metric_log != None:
                metric_log.add([preds, y, t], subset='test')
                metric_log.add(loss.item(), 'loss', 'test')
