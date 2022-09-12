from argparse import ArgumentParser
from scenarios.classics import split_mnist_scenarios, permut_mnist_scenarios
import torch
from torch.utils.data import DataLoader

from models.er import Er
from utils.args import add_rehearsal_args, add_experiment_args
from backbones.simple_mlp import SimpleMLP
from tqdm import tqdm, trange


def main():
    # 解析命令行
    parser = ArgumentParser(description='cls')
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    args = parser.parse_known_args()[0]

    # gpu
    gpu_ids = [i for i in range(torch.cuda.device_count())]
    device = torch.device(
        f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')

    mlp = SimpleMLP()
    model = Er(mlp, device,args)
    n_tasks = permut_mnist_scenarios['train'].nb_tasks

    # train & test
    with trange(n_tasks, position=0) as tt:
        for task_id, taskset in enumerate(permut_mnist_scenarios["train"]):
            tt.set_description(f'Task {task_id} Processing')

            # Continuum库Log的一个bug，end_task
            if task_id != 0:
                model.log.end_task()

            # task level train
            with trange(args.n_epochs, position=1, leave=False) as et:
                for epoch in range(args.n_epochs):
                    et.set_description(f'Task {task_id} Epoch {epoch}')

                    model.observe(taskset)
                    model.log.end_epoch()

                    et.update()
            tt.update()

            tqdm.write(
                f"Task id={task_id}, acc={model.log.online_accuracy}, avg_acc={model.log.online_cumulative_performance}")
            # test


if __name__ == '__main__':
    main()
