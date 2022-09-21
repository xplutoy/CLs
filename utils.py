import torch.nn as nn
from argparse import ArgumentParser
from continuum import Logger
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--experiment_id', type=str, default='cl')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--scenario', type=str, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--n_epochs', type=int, required=True)


def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--buffer_size', type=int, required=True,
                        help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int, required=True,
                        help='The batch size of the memory buffer.')


def init_weights(module: nn.Module):
    """https://adityassrana.github.io/blog/theory/2020/08/26/Weight-Init.html

    Args:
        module (nn.Module): 
    """
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def run(model, train_scenario, test_scenario, n_epochs):
    # train & test
    train_metric_log = Logger(['performance', 'loss'], ['train'])
    test_metric_log = Logger(['performance', 'loss'], ['test'])
    nb_tasks = train_scenario.nb_tasks
    with trange(nb_tasks, position=0) as tt:
        for task_id, taskset in enumerate(train_scenario):
            tt.set_description(f'Task {task_id} Processing')

            # task level train
            if task_id != 0:
                train_metric_log.end_task()
            with trange(n_epochs, position=1, leave=False) as et:
                for epoch in range(n_epochs):
                    et.set_description(f'Task {task_id} Epoch {epoch}')

                    if epoch != 0:
                        train_metric_log.end_epoch()
                    model.observe(taskset, train_metric_log)

                    et.update()
            tt.update()

            # 一个任务训练完后，用已经观察到的任务(包含当前任务，所以遍历任务+1)测试数据测试当前模型
            # 传入已经观察到的任务的测试数据（后面的一些metric需要这种训练方式）
            if task_id != 0:
                test_metric_log.end_task()
            for observed_task_id in range(task_id+1):
                model.eval(test_scenario[observed_task_id], test_metric_log)

            str_acc_per_task = [
                f'{acc:.4f}'for acc in test_metric_log.accuracy_per_task]
            tqdm.write(
                f'Test : Model {task_id} train_acc= {train_metric_log.online_accuracy:.4f}' +
                f' test_avg_acc= {test_metric_log.accuracy:.4f}' +
                f' test_avg_acc_A= {test_metric_log.accuracy_A:.4f}' +
                f' backward_transfer= {test_metric_log.backward_transfer:.4f}' +
                f' forward_transfer= {test_metric_log.forward_transfer:.4f}' +
                f' positive_backward_transfer= {test_metric_log.positive_backward_transfer:.4f}' +
                f' remembering= {test_metric_log.remembering:.4f}' +
                f' forgetting= {test_metric_log.forgetting:.4f}' +
                f' test_per_task_acc = {str_acc_per_task}'
            )

    # 用tensorboard查看训练loss和测试loss的变化
    writer = SummaryWriter(f'runs/er')
    train_losses = train_metric_log.get_logs('loss', 'train')
    test_losses = test_metric_log.get_logs('loss', 'test')
    train_global_step, test_global_step = 0, 0
    for task_id in range(nb_tasks):
        for epoch_id in range(n_epochs):
            task_epoch_losses = train_losses[task_id][epoch_id]
            for loss in task_epoch_losses:
                writer.add_scalar('train_loss', loss, train_global_step)
                train_global_step = train_global_step + 1
        test_epoch_losses = test_losses[task_id][-1]
        for loss in test_epoch_losses:
            writer.add_scalar('test_loss', loss, test_global_step)
            test_global_step = test_global_step + 1
