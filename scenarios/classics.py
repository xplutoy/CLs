from continuum import ClassIncremental, Permutations
from continuum.datasets import MNIST, CIFAR10, CIFAR100

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
