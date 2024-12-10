from typing import Dict, List, Optional, Union
from ab.nn.dataset.DeepLabV3.code import Bottleneck, FCNHead, ResNet, DeepLabHead

from torch import nn


backbones: Dict[str, dict] = {
    "ResNet50": [ResNet(Bottleneck, [3, 4, 6, 3], num_classes=100, replace_stride_with_dilation=[False, True, True]),DeepLabHead(2048,21),None],
    "ResNet101": [ResNet(Bottleneck, [3, 4, 23, 3], num_classes=100, replace_stride_with_dilation=[False, True, True]),DeepLabHead(2048,21),None],
}
args = [*backbones["ResNet50"]]