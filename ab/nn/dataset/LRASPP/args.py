from .code import MobileNetV3, _mobilenet_v3_conf  # Relative import, or will cause error by location of trainer

args = [MobileNetV3(*_mobilenet_v3_conf("mobilenet_v3_large"), num_classes = 100), None, None, 21]