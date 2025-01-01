import torchvision

from ab.nn.util import Const


def loader(transform=None):
    train_set = torchvision.datasets.CIFAR10(root=Const.data_dir_global, train=True, transform=transform, download=True)
    test_set = torchvision.datasets.CIFAR10(root=Const.data_dir_global, train=False, transform=transform, download=True)
    return (10,), train_set, test_set