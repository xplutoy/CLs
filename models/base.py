from argparse import Namespace
from turtle import forward
import torch
import torch.nn as nn
from continuum import TaskSet, Logger


class ContinualModel(nn.Module):

    def __init__(self, backbone: nn.Module, device: torch.device, args: Namespace) -> None:
        super().__init__()
        self.net = backbone.to(device)
        self.device = device
        self.args = args

        self.log = Logger()
        self.opt = torch.optim.SGD(self.net.parameters(), lr=self.args.lr)

    def forward(self, x):
        return self.net(x)

    def observe(self, taskset: TaskSet):
        """模型每次接受一个任务的数据处理

        Args:
            taskset (TaskSet): 
        """
        pass
