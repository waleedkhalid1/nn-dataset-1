from ab.nn.util.Const import nn_module

def nn_mod(*nms):
    return ".".join((nn_module,) + nms)