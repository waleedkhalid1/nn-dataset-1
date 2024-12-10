import torchvision


def loader(transform=None):
    train_set = torchvision.datasets.CIFAR10(root='data', train=True, transform=transform, download=True)
    test_set = torchvision.datasets.CIFAR10(root='data', train=False, transform=transform, download=True)
    return train_set, test_set