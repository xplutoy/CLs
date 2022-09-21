import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import init_weights
from buffer import ReservoirBuffer


class Der(nn.Module):
    def __init__(self,
                 net: nn.Module,
                 buffer_size: int,
                 lr: float,
                 batch_size: int,
                 minibatch_size: int,
                 alpha: float,
                 beta: float,
                 device: torch.device = 'cpu'
                 ) -> None:
        super().__init__()

        self.net = net.to(device)
        self.buffer = ReservoirBuffer(
            buffer_size, ['x:torch.float32', 'y:torch.int64', 'logits:torch.float32'], device)
        self.optimizer = torch.optim.SGD(
            self.net.parameters(), lr=lr, momentum=0.9)
        self.ce_criterion = torch.nn.CrossEntropyLoss()
        self.mse_criterion = torch.nn.MSELoss()
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.alpha = alpha
        self.beta = beta
        self.device = device

        init_weights(self.net)

    def observe(self, taskset, metric_log=None):
        self.net.train()
        task_loader = DataLoader(
            taskset, batch_size=self.batch_size, shuffle=True)

        for x, y, t in task_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            logits = self.net(x)
            loss = self.ce_criterion(logits, y)

            if len(self.buffer) != 0:
                experiment = self.buffer.get(self.minibatch_size)
                _x, _y, _logits = experiment[0], experiment[1], experiment[2]
                loss += self.alpha * self.mse_criterion(self.net(_x), _logits)
                experiment = self.buffer.get(self.minibatch_size)
                _x, _y = experiment[0], experiment[1]
                loss += self.beta * self.ce_criterion(self.net(_x), _y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.buffer.add((x, y, logits.data))
            preds = torch.argmax(logits, dim=-1)
            if metric_log != None:
                metric_log.add([preds, y, t], subset='train')
                metric_log.add(loss.item(), 'loss', 'train')

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
            loss = self.ce_criterion(logits, y)
            if metric_log != None:
                metric_log.add([preds, y, t], subset='test')
                metric_log.add(loss.item(), 'loss', 'test')
