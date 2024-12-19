import torchvision.transforms as transforms
import torch
import numpy as np

class Reshape:
    """ 
    Custom transformation to reshape the input to (3, 32, 32).
    """
    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x = x.reshape(3, 32, 32)
        elif isinstance(x, torch.Tensor):
            x = x.reshape(3, 32, 32)
        else:
            x = np.array(x).reshape(3, 32, 32)
        return x

class NormalizeToFloat:
    """ 
    Custom transformation to normalize image data to float32 and scale to [0, 1].
    """
    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x = x.astype(np.float32) / 255.0
        elif isinstance(x, torch.Tensor):
            x = x.float() / 255.0
        else:
            x = np.array(x).astype(np.float32) / 255.0
        return x

class ToComplex64:
    """ 
    Custom transformation to convert PyTorch tensor to torch.complex64.
    """
    def __call__(self, x):
        """
        Transform method to convert the tensor to torch.complex64.
        :param x: The input tensor, usually a NumPy array, PIL Image, or PyTorch tensor.
        :return: Tensor converted to torch.complex64.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.type(torch.complex64)  # to torch.complex64
        return x
    
def transform():
    """
    Define the transformation to be applied to the CIFAR-10 dataset.
    :return: A transformation pipeline using torchvision.transforms.
    """
    return transforms.Compose([
        Reshape(),
        NormalizeToFloat(),
        transforms.ToTensor(),
        ToComplex64()])