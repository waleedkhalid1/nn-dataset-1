import torchvision.transforms as transforms

def transform(**kwargs):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform