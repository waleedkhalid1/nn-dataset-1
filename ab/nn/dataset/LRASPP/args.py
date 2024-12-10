from ab.nn.dataset.LRASPP.code import Net, MobileNetV3, _mobilenet_v3_conf

args = [MobileNetV3(*_mobilenet_v3_conf("mobilenet_v3_large"), num_classes = 100), None, None, 21]