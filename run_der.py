from argparse import ArgumentParser
import torch

from data import select_scenarios
from utils import add_rehearsal_args, add_experiment_args
from backbones.simple_mlp import SimpleMLP
from models.der import Der
from utils import run


def main():
    # 解析命令行
    parser = ArgumentParser()
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('alpha', type=float, default=0.5)
    parser.add_argument('beta', type=float, default=0.5)

    args = parser.parse_known_args()[0]

    # gpu
    device = torch.device(
        f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    scenario_name = args.scenario
    scenario_dict, net = select_scenarios(scenario_name)

    model = Der(
        net=net,
        buffer_size=args.buffer_size,
        lr=args.lr,
        batch_size=args.batch_size,
        minibatch_size=args.minibatch_size,
        alpha=args.alpha,
        beta=args.beta,
        device=device
    )

    run(model, scenario_dict['train'], scenario_dict['test'], args.n_epochs)


if __name__ == '__main__':
    main()
