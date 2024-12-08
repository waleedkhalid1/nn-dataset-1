from typing import Dict, List

from torch import nn

from .code import ResNet, Bottleneck, FCNHead, VGG, make_layers, vgg_cfgs

backbones: Dict[str, List[nn.Module]] = {
    "ResNet50": [ResNet(Bottleneck, [3, 4, 6, 3], num_classes = 100, replace_stride_with_dilation=[False, True, True]),FCNHead(2048, 21)],
    "ResNet101": [ResNet(Bottleneck, [3, 4, 23, 3], num_classes = 100, replace_stride_with_dilation=[False, True, True]),FCNHead(2048, 21)],
    "VGG16": [VGG(make_layers(vgg_cfgs["D"]),num_classes=100),FCNHead(512, 21)],
}
args = [*backbones["ResNet50"]]