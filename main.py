from argparse import ArgumentParser
from scenarios.classics import permut_mnist_scenarios
import torch

from models.er import Er
from utils.args import add_rehearsal_args, add_experiment_args
from backbones.simple_mlp import SimpleMLP
from tqdm import tqdm, trange
from continuum import Logger
from torch.utils.tensorboard import SummaryWriter


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
    model = Er(mlp,
               buffer_size=args.buffer_size,
               lr=args.lr,
               batch_size=args.batch_size,
               minibatch_size=args.minibatch_size,
               device=device
               )

    n_tasks = permut_mnist_scenarios['train'].nb_tasks
    train_scenario = permut_mnist_scenarios['train']
    test_scenario = permut_mnist_scenarios['test']

    # train & test
    train_log = Logger(['performance', 'loss'], ['train'])
    with trange(n_tasks, position=0) as tt:
        for task_id, taskset in enumerate(train_scenario):
            tt.set_description(f'Task {task_id} Processing')

            # task level train
            if task_id != 0:
                train_log.end_task()
            with trange(args.n_epochs, position=1, leave=False) as et:
                for epoch in range(args.n_epochs):
                    et.set_description(f'Task {task_id} Epoch {epoch}')

                    model.observe(taskset, train_log)

                    et.update()
                    train_log.end_epoch()
            tt.update()

            # 一个任务训练完后，用已经观察到的任务(包含当前任务，所以遍历任务+1)测试数据测试当前模型
            test_metric_log = Logger(['performance', 'loss'], ['test'])
            for current_task_id in range(task_id+1):
                if current_task_id != 0:
                    test_metric_log.end_task()
                # 传入已经观察到的任务的测试数据（后面的一些metric需要这种训练方式）
                for observed_task_id in range(current_task_id+1):
                    model.eval(test_scenario[observed_task_id], test_metric_log)

            tqdm.write(
                f'Test : Model {task_id} train_acc= {train_log.online_accuracy:.4f}' +
                f' test_acc= {test_metric_log.accuracy:.4f}' +
                f' test_avg_acc_A= {test_metric_log.accuracy_A:.4f}' +
                f' backward_transfer= {test_metric_log.backward_transfer:.4f}' +
                f' forward_transfer= {test_metric_log.forward_transfer:.4f}' +
                f' positive_backward_transfer= {test_metric_log.positive_backward_transfer:.4f}' +
                f' remembering= {test_metric_log.remembering:.4f}' +
                f' forgetting= {test_metric_log.forgetting:.4f}' +
                f' test_per_task_acc={str(test_metric_log.accuracy_per_task)}'
            )

    # 用tensorboard查看训练loss和测试loss的变化
    writer = SummaryWriter('runs/er')
    train_losses = train_log.get_logs('loss', 'train')
    test_losses = test_metric_log.get_logs('loss', 'test')
    train_global_step, test_global_step = 0, 0
    for task_id in range(n_tasks):
        for epoch_id in range(args.n_epochs):
            task_epoch_losses = train_losses[task_id][epoch_id]
            for loss in task_epoch_losses:
                writer.add_scalar('train_loss', loss, train_global_step)
                train_global_step = train_global_step + 1
        test_epoch_losses = test_losses[task_id][-1]
        for loss in test_epoch_losses:
            writer.add_scalar('test_loss', loss, test_global_step)
            test_global_step = test_global_step + 1


if __name__ == '__main__':
    main()
