from continuum import ClassIncremental, Permutations
from continuum.datasets import MNIST, CIFAR100, CIFAR10
from backbones.simple_mlp import SimpleMLP
from backbones.simple_cnn import SimpleCNN
from torchvision import transforms

split_mnist_scenarios = {
    "train": ClassIncremental(
        MNIST(data_path="../dataset", download=True, train=True),
        increment=2
    ),
    "test": ClassIncremental(
        MNIST(data_path="../dataset", download=True, train=True),
        increment=2
    )
}

permut_mnist_scenarios = {
    "train": Permutations(
        MNIST(data_path="../dataset", download=True, train=True),
        nb_tasks=10,
        seed=0,
        shared_label_space=True
    ),
    "test": Permutations(
        MNIST(data_path="../dataset", download=True, train=False),
        nb_tasks=10,
        seed=0,
        shared_label_space=True
    )
}

cifar10_scenarios = {
    'train': ClassIncremental(
        CIFAR10(data_path="../dataset", download=True, train=True),
        increment=2,
        transformations=[
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)), # https://github.com/kuangliu/pytorch-cifar/issues/19
        ]
    ),
    'test': ClassIncremental(
        CIFAR10(data_path="../dataset", download=True, train=False),
        increment=2,
        transformations=[
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
}

cifar100_scenarios = {
    'train': ClassIncremental(
        CIFAR100(data_path="../dataset", download=True, train=True),
        increment=10,
        transformations=[
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    ),
    'test': ClassIncremental(
        CIFAR100(data_path="../dataset", download=True, train=False),
        increment=10,
        transformations=[
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )
}


def select_scenarios(witch):
    """根据场景描述字符串，选择默认的场景和网络
    """
    if witch == 'permut_mnist':
        return permut_mnist_scenarios, SimpleMLP()
    elif witch == 'split_mnist':
        return split_mnist_scenarios, SimpleMLP()
    elif witch == 'cifar10':
        return cifar10_scenarios, SimpleCNN(num_classes=10)
    elif witch == 'cifar100':
        return cifar100_scenarios, SimpleCNN(num_classes=100)
    else:
        raise ValueError('only support permut_mnist | split_mnist')
