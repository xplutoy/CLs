from argparse import ArgumentParser
from scenarios.classics import split_mnist_scenarios, permut_mnist_scenarios
import torch
from torch.utils.data import DataLoader

from models.er import Er
from utils.args import add_rehearsal_args, add_experiment_args
from backbones.simple_mlp import SimpleMLP
from tqdm import tqdm, trange
from continuum import Logger


def main():
    # 解析命令行
    parser = ArgumentParser(description='cls')
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    args = parser.parse_known_args()[0]

    # metric
    metric_log = Logger()

    # gpu
    gpu_ids = [i for i in range(torch.cuda.device_count())]
    device = torch.device(
        f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')

    mlp = SimpleMLP()
    model = Er(mlp,
               buffer_size=args.buffer_size,
               lr=args.lr,
               batch_size=args.batch_size,
               minibatch_size=args.minibatch_size,
               device=device,
               metric_log=metric_log
               )

    n_tasks = permut_mnist_scenarios['train'].nb_tasks
    train_scenario = permut_mnist_scenarios['train']
    test_scenario = permut_mnist_scenarios['test']

    # train & test
    with trange(n_tasks, position=0) as tt:
        for task_id, taskset in enumerate(train_scenario):
            tt.set_description(f'Task {task_id} Processing')

            # task level train
            with trange(args.n_epochs, position=1, leave=False) as et:
                for epoch in range(args.n_epochs):
                    et.set_description(f'Task {task_id} Epoch {epoch}')

                    model.observe(taskset)

                    et.update()

            tt.update()

            # task level test 当前模型对当前任务
            for taskset in test_scenario:
                model.eval(test_scenario[task_id])

            tqdm.write(
                f'Test : Model {task_id} acc={round(metric_log.accuracy, 4)}, avg_acc={round(metric_log.average_incremental_accuracy,4)}'
            )

            metric_log.end_task()


if __name__ == '__main__':
    main()
