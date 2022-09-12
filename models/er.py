import numpy as np
import torch
from torch.utils.data import DataLoader
from continuum import rehearsal
from utils.buffer import rand_batch_get
from utils.weight import init_weights
from models.base import ContinualModel
from tqdm import tqdm


class Er(ContinualModel):
    def __init__(self, backbone, device, args) -> None:
        super().__init__(backbone, device, args)

        self.loss_ce = torch.nn.CrossEntropyLoss()
        self.buffer = rehearsal.RehearsalMemory(
            memory_size=args.buffer_size,
            herding_method="random"
        )
        init_weights(self.net)

    def observe(self, taskset):
        task_loader = DataLoader(
            taskset, batch_size=self.args.batch_size, shuffle=True)

        for x, y, t in task_loader:

            if len(self.buffer) != 0:
                _x, _y, _t = rand_batch_get(
                    self.buffer, self.args.minibatch_size)
                x = torch.cat((x, torch.from_numpy(np.expand_dims(_x, 1))))
                y = torch.cat((y, torch.from_numpy(_y)))
                t = torch.cat((t, torch.from_numpy(_t)))

            x = x.to(self.device)
            y = y.to(self.device)

            logits = self.net(x)
            loss = self.loss_ce(logits, y)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            preds = torch.argmax(logits, dim=-1)
            # acc = (preds == y).to(torch.float64).mean().item()
            # tqdm.write(
            #     f"task_id={self.log.nb_tasks}, online_acc = {acc}, loss = {loss.item()}")

            self.log.add([preds, y, t], subset='train')

        self.buffer.add(*taskset.get_raw_samples(), None)
