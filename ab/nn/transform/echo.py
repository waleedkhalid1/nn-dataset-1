import torchvision.transforms as transforms

def transform():
    return transforms.Compose([
        transforms.ToTensor()])