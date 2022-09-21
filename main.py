from argparse import ArgumentParser
import torch
from continuum import rehearsal

from data import select_scenarios
from utils import add_rehearsal_args, add_experiment_args
from backbones.simple_mlp import SimpleMLP
from models.er import Er
from utils import run



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
    
    scenario_name = args.scenario
    scenario_dict = select_scenarios(scenario_name)
    n_tasks = scenario_dict['train'].nb_tasks
    nb_classes = scenario_dict['train'].nb_classes

    if scenario_name == 'split_mnist':
        memory = rehearsal.RehearsalMemory(
                memory_size=args.buffer_size,
                herding_method="random",
                fixed_memory=True,
                nb_total_classes=nb_classes
                )
    else:
        raise ValueError('not support scenario')

    mlp = SimpleMLP()
    model = Er(mlp,
                memory= memory,
                lr=args.lr,
                batch_size=args.batch_size,
                minibatch_size=args.minibatch_size,
                device=device
               )

    run(model,scenario_dict['train'],scenario_dict['test'],n_tasks,args.n_epochs)


if __name__ == '__main__':
    main()
